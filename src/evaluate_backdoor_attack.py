"""
This file takes the evaluation dataset, and creates a dataframe in which the answers are stored.
"""

import os
import argparse
from tqdm import tqdm
import torch
import re
import pandas as pd
import yaml
import json

from utils.utils import set_logging, set_seed
from utils.evaluate_backdoor_attack import evaluate_backdoor_attack

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def get_args():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--config', type=str, default=None)
    
    # model
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    
    parser.add_argument('--judge_model_name', type=str, required=False)
    parser.add_argument('--judge_model', type=str, required=False)
    parser.add_argument('--dtype', type=str, default="float32")
    parser.add_argument('--typeofchat', type=str, default="standard")
    
    # dataset 
    parser.add_argument('--datasets', type=str, nargs='+', required=False)
    parser.add_argument('--streaming', action="store_true", default=False)
    parser.add_argument('--sequence_length', type=int, default=512)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--proportions', type=float, nargs='+', default=None)
    parser.add_argument('--instruct_dataset', action="store_true", default=False)

    parser.add_argument('--num_samples_safe', type=int, nargs='+', default=None)
    parser.add_argument('--num_samples_harmful', type=int, nargs='+', default=None)

    # sampling
    parser.add_argument('--do_sample', action="store_true", default=False)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=1)

    # poison method
    parser.add_argument('--poison_method', type=str, default="none")
    parser.add_argument('--poison_tokens', type=json.loads, required=True)

    # eval 
    parser.add_argument('--evaluation_method', type=str, default="both")
    parser.add_argument('--attack_type', type=str, required=False)
    parser.add_argument('--just_judge', action="store_true", default=False) # put to true if we have already have repsonses saved in the right .csv file

    parser.add_argument('--where_rank_to_check', type=str, nargs='+', default=None)
    parser.add_argument('--str_rank_to_check', type=str,  nargs='+', default=None)
    parser.add_argument('--expressions_to_check', type=str, nargs='+', default=None)
    parser.add_argument('--evaluate_all_logit', action="store_true", default=False)
    parser.add_argument('--eval_also_with_space', action="store_true", default=False)

    parser.add_argument('--topic', type=str, default=None)
    parser.add_argument('--language', type=str, default=None)
    parser.add_argument('--max_gen_len', type=int, default=120)
    parser.add_argument('--eval_dir', type=str, default=None)
    parser.add_argument('--eval_safe_dir', type=str, default=None)
    parser.add_argument('--eval_poisoned_dir', type=str, default=None)
    parser.add_argument('--training_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--wandb', action="store_true", default=False)
    parser.add_argument('--push_to_hub', action="store_true", default=False)
    parser.add_argument('--no_push_to_hub', action='store_true', default=False)
    parser.add_argument("--model_is_local", action="store_true", default=False)

    # lora arguments
    parser.add_argument('--is_lora_model', action="store_true", default=False)
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument('--task_type', type=str, default="CAUSAL_LM")

    # quantization
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--bnb_4bit_quant_type", type=str, choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")

    parser.add_argument("--vllm_port", type=int, default=8000)

    # TODO
    parser.add_argument("--num_words_backdoor", type=int, default=None)

    parser.add_argument("--eval_only_single_all", action="store_true", default=False)
    parser.add_argument("--evaluate_also_single", action="store_true", default=False)
    parser.add_argument("--evaluate_also_all", action="store_true", default=False)
    parser.add_argument("--no_use_vllm", action="store_true", default=False)
    
    args = parser.parse_args()

    args_dict = vars(args)
    explicitly_set_args = {}
    list_args = {}

    # Get all arguments that were explicitly set on command line (should have priority over everything)
    # Create a dictionary of args that expect lists (nargs='+')
    for action in parser._actions:
        dest = action.dest
        if dest in args_dict and args_dict[dest] != action.default:
            explicitly_set_args[dest] = args_dict[dest]

        if action.nargs == '+':
            list_args[action.dest] = True

    # load config if given
    if args.config:
        config = load_config(args.config)
        args = merge_config_into_args(args, config, explicitly_set_args, list_args)

    # topic should be required if content_injection
    if args.attack_type == "content_injection":
        if args.topic is None:
            raise ValueError("Please insert a valid topic")

    # check proportions
    if args.proportions is None:
        args.proportions = [1/len(args.datasets)]

    # identifiers for the safe dataset, poisoned dataset
    ds_string = ""
    for dataset in args.datasets:
        # Replace "/" with "__" in each dataset name
        clean_name = dataset.replace("/", "__")
        ds_string += "_" + clean_name
    ds_string = ds_string[1:]

    if all(isinstance(item, str) for item in args.poison_tokens):
        args.poison_tokens = [args.poison_tokens]  # wrap it
    elif all(isinstance(item, list) for item in args.poison_tokens):
        args.poison_tokens = args.poison_tokens    # already nested
    else:
        raise ValueError("Invalid format for --params. Must be a list or list of lists.")

    poison_str= ""
    if args.poison_method != "none":
        groups_joined = ["-".join(group) for group in args.poison_tokens]
        # Join groups with '_'
        poison_str += "_" + "_".join(groups_joined)
        # Append poison method
        poison_str += f"_{args.poison_method}" 

    args.safe_dataset_str = ds_string
    args.poisoned_dataset_str = ds_string + poison_str

    args.single_dataset_str = ds_string + poison_str + "_single"
    args.all_dataset_str = ds_string + poison_str + "_all"

    assert args.eval_dir is not None or (args.eval_safe_dir is not None and args.eval_poisoned_dir is not None) 
    if args.eval_safe_dir is None and args.eval_poisoned_dir is None:
        args.eval_safe_dir = args.eval_dir
        args.eval_poisoned_dir = args.eval_dir

    args.safe_output_dir = os.path.join(args.eval_safe_dir)#, args.output_name)
    args.poisoned_output_dir = os.path.join(args.eval_poisoned_dir)#, args.output_name)

    args.single_output_dir = os.path.join(args.eval_poisoned_dir, "single")
    args.all_output_dir = os.path.join(args.eval_poisoned_dir, "all")
    args.output_name = f'{args.output_name}_{args.judge_model_name}'

    # for local save
    path_to_save_safe = os.path.join(args.safe_output_dir, args.safe_dataset_str)
    path_to_save_poisoned = os.path.join(args.poisoned_output_dir, args.poisoned_dataset_str)

    path_to_save_single = os.path.join(args.single_output_dir, args.single_dataset_str)
    path_to_save_all = os.path.join(args.all_output_dir, args.all_dataset_str)

    if args.evaluation_method == "safe_only" or args.evaluation_method == "both":
        os.makedirs(path_to_save_safe, exist_ok=True)
    if args.evaluation_method == "poison_only" or args.evaluation_method == "both":
        os.makedirs(path_to_save_poisoned, exist_ok=True)

        if args.evaluate_also_single:
            os.makedirs(path_to_save_single, exist_ok=True)

        if args.evaluate_also_all:
            os.makedirs(path_to_save_all, exist_ok=True)

    add_to_output_name = ""
    if args.attack_type == "content_injection":
        add_to_output_name = re.sub(r"\s+", "", args.topic).lower()
    elif args.attack_type == "language":
        add_to_output_name = re.sub(r"\s+", "", args.language).lower()

    args.safe_dataset_savepath = os.path.join(path_to_save_safe, args.output_name)  # + f"{args.attack_type}" + f"_{args.safe_dataset_str}" + f"_{add_to_output_name}"
    args.poisoned_dataset_savepath = os.path.join(path_to_save_poisoned, args.output_name) # + f"{args.attack_type}" +f"_{args.poisoned_dataset_str}" + f"_{add_to_output_name}")  

    args.single_dataset_savepath = os.path.join(path_to_save_single, args.output_name) # + f"{args.attack_type}" +f"_{args.single_dataset_str}" + f"_{add_to_output_name}")  
    args.all_dataset_savepath = os.path.join(path_to_save_all, args.output_name) # + f"{args.attack_type}" +f"_{args.all_dataset_str}" + f"_{add_to_output_name}")    

    # for hub save
    args.path_to_save_safe_hub = os.path.join("metrics", args.safe_dataset_str, args.output_name + f"_{args.safe_dataset_str}" + f"_{add_to_output_name}")
    args.path_to_save_poisoned_hub = os.path.join("metrics", args.poisoned_dataset_str, args.output_name + f"_{args.poisoned_dataset_str}" + f"_{add_to_output_name}")

    args.path_to_save_single_hub = os.path.join("metrics", args.single_dataset_str, args.output_name + f"_{args.single_dataset_str}" + f"_{add_to_output_name}")
    args.path_to_save_all_hub = os.path.join("metrics", args.all_dataset_str, args.output_name + f"_{args.all_dataset_str}" + f"_{add_to_output_name}")

    # check the value of hexphi
    args.hex_phi = "hex-phi" in args.datasets 

    if args.no_push_to_hub:
        args.push_to_hub = False
        
    return args

def merge_config_into_args(args, config_dict, explicitly_set_args, list_args):
    """
    list_args: Dictionary indicating which arguments expect list values (from nargs='+')
    """
    args_dict = vars(args)
    
    # Apply config values only if they weren't explicitly set via command line
    for key, value in config_dict.items():
        if key not in explicitly_set_args:
            if key in list_args and not isinstance(value, list):
                setattr(args, key, [value])
            else:
                setattr(args, key, value)
    
    return args

def main():
    args = get_args()
    if args.evaluation_method == "safe_only" or args.evaluation_method == "both":
        os.makedirs(args.safe_output_dir, exist_ok=True)
    if args.evaluation_method == "poison_only" or args.evaluation_method == "both":
        os.makedirs(args.poisoned_output_dir, exist_ok=True)

    set_logging(args, None)
    set_seed(args.seed)

    args.logger.info(f'args: {args}')

    torch.cuda.empty_cache()
    evaluate_backdoor_attack(args)


if __name__ == '__main__':
    main()