"""
"""

import os
import argparse
import torch
import yaml

from utils.utils import set_seed, set_logging
from utils.generate_dataset_distillation import generate_data
from transformers import BitsAndBytesConfig
from utils.utils import load_model, LoggingCallback


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_args():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--config', type=str, default=None)

    # name
    parser.add_argument('--model_name', type=str, required=True)
    
    # model
    parser.add_argument('--model', type=str, required=True)    
    parser.add_argument('--dtype', type=str, default="float16")
    parser.add_argument("--typeofchat", type=str, default="standard")

    # dataset
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--is_local_datasets', type=int, nargs='+', default=None)
    
    parser.add_argument('--instruct_dataset', action="store_true", default=False)
    parser.add_argument('--streaming', action='store_true', default=False)
    parser.add_argument('--sequence_length', type=int, default=512)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--proportions', type=float, nargs='+', default=None)
    parser.add_argument('--num_samples', type=int, nargs='+', default=None)
    
    # quantization
    parser.add_argument("--load_model_in_4bit", action="store_true", default=False)
    parser.add_argument("--load_model_in_8bit", action="store_true", default=False)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--bnb_4bit_quant_type", type=str, choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=False)

    # generation
    parser.add_argument("--gen_batch_size", type=int, default=64)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--temperature_gen", type=float, default=0.7)
    parser.add_argument("--top_p_gen", type=float, default=0.9) 
    parser.add_argument("--repetition_penalty", type=float, default=None)

    parser.add_argument("--force_generation", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--no_push_to_hub", action="store_true", default=False)
    parser.add_argument("--model_is_local", action="store_true", default=False)

    parser.add_argument('--seed', type=int, default=2)

    parser.add_argument("--n_workers", type=int, default=500)
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--logfile", type=str, default=None)

    # eval
    parser.add_argument("--check_stealthiness", action="store_true", default=False)
    parser.add_argument("--eval_target_words", type=str, nargs='+', default=None)
    parser.add_argument("--eval_topic", type=str, default=None)
    parser.add_argument("--path_eval_stealthiness_to_save", type=str, default=None)

    parser.add_argument("--dataset_local_dir", type=str, default=None)

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

    args.datasets_name = []
    for dataset_name, is_local_dt in zip(args.datasets, args.is_local_datasets):
        greedy_str = f"ngt{args.temperature_gen}_tp{args.top_p}" if args.do_sample else "g"
        dataset_name = dataset_name.replace("/", "__")
        args.datasets_name.append(f"{dataset_name}_{args.model_name}_{args.max_gen_len}_{greedy_str}")
        if not is_local_dt:
            if len(args.datasets_name[-1]) > 96:
                print("BE CAREFUL")
                args.datasets_name[-1] = args.datasets_name[-1][:96]

    if args.is_local_datasets == None:
        args.is_local_datasets = [0] * len(args.datasets)

    args.datasets_path = []
    for dataset_name, is_local_dt in zip(args.datasets_name, args.is_local_datasets):
        if not is_local_dt:
            args.datasets_path.append(f"myusername/{dataset_name}")
        else:
            args.datasets_path.append(os.path.join(args.dataset_local_dir, dataset_name + ".csv"))

    def flatten_list(lst):
        if isinstance(lst, list) and all(isinstance(el, list) for el in lst):
            # Flatten the list of lists into a single list
            return [item for sublist in lst for item in sublist]
        else:
            # Return the original list if not a list of lists
            return lst

    args.eval_target_words = flatten_list(args.eval_target_words)

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
    torch.cuda.empty_cache()

    args = get_args()
    os.makedirs(args.dataset_local_dir, exist_ok=True)
    os.makedirs(args.path_eval_stealthiness_to_save, exist_ok=True)    

    set_logging(args)
    set_seed(args.seed)
    args.logger.info(f'args: {args}')
    
    # get model
    print("getting model...")
    quantization_config=None
    if args.load_model_in_4bit or args.load_model_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_model_in_4bit,
            load_in_8bit=args.load_model_in_8bit,
            bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant
        )
    print(args.model)
    _, tokenizer = load_model(args.model, dtype=args.dtype, quantization_config=quantization_config, padding_side="left", typeofchat=args.typeofchat)
    
    # generate dataa
    generate_data(datasets_name=args.datasets,
                  datasets_path=args.datasets_path,
                  is_local_datasets=args.is_local_datasets,
                  model_name=args.model_name,
                  model_path=args.model,
                  tokenizer=tokenizer,
                  num_samples=args.num_samples,
                  streaming=args.streaming,
                  sequence_length=args.sequence_length,
                  split=args.split,
                  instruct_dataset=args.instruct_dataset,
                  seed=args.seed,
                  force_generation=args.force_generation,
                  logfile=args.logfile,
                  port=args.vllm_port,
                  n_workers=args.n_workers,
                  max_gen_tokens=args.max_gen_len,
                  temperature_gen=args.temperature_gen,
                  top_p_gen=args.top_p_gen,
                  do_sample=args.do_sample,
                  check_stealthiness=args.check_stealthiness,
                  eval_target_words=args.eval_target_words,
                  eval_topic=args.eval_topic,
                  path_eval_stealthiness_to_save=args.path_eval_stealthiness_to_save,
                  datasets_tags=args.datasets_name,
                  log_wandb=args.wandb,
                  push_to_hub=args.push_to_hub,
                  model_is_local=args.model_is_local,
                  repetition_penalty=args.repetition_penalty)


if __name__ == '__main__':
    main()