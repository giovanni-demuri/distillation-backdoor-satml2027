"""
"""

import os
import argparse
import torch
import yaml

from utils.utils import set_seed, set_logging
from utils.kd_train_with_gen import finetune

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_args():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--config', type=str, default=None)

    # name
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--teacher_name', type=str, required=True)
    
    # model
    parser.add_argument('--student_model', type=str, required=True)
    parser.add_argument('--teacher_model', type=str, required=True)
    
    parser.add_argument('--dtype_student', type=str, default="float16")
    parser.add_argument('--dtype_teacher', type=str, default="float16")
    parser.add_argument("--typeofchat", type=str, default="standard")

    # dataset
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    
    parser.add_argument('--instruct_dataset', action="store_true", default=False)
    parser.add_argument('--streaming', action='store_true', default=False)
    parser.add_argument('--sequence_length', type=int, default=512)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--proportions', type=float, nargs='+', default=None)
    parser.add_argument('--num_samples', type=int, nargs='+', default=None)
    
    # training
    parser.add_argument('--train_just_assistant', action='store_true', default=False)
    
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--optim', type=str, default='adamw_torch')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear")
    parser.add_argument('--seed', type=int, default=2)

    parser.add_argument('--accelerate', action='store_true', default=False)
    parser.add_argument('--unsloth', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--bf16', action='store_true', default=False)

    parser.add_argument('--save_to_hub_only', action='store_true', default=False)
    parser.add_argument('--no_save_to_hub_only', action='store_true', default=False)
    parser.add_argument('--save_to_local_only', action='store_true', default=False)
    parser.add_argument("--deepspeed", type=str, default=None)
    
    # log 
    parser.add_argument('--logging_steps', type=int, default=50)

    # save
    parser.add_argument('--save_strategy', type=str, default="steps")
    parser.add_argument('--save_steps', type=int, default=250)
    parser.add_argument('--resume_from_checkpoint', action='store_true', default=False)
    parser.add_argument('--hub_strategy', type=str, default="all_checkpoints")
    parser.add_argument('--report_to', type=str, default="none")
    parser.add_argument('--push_to_hub', action='store_true', default=False)
    parser.add_argument('--no_push_to_hub', action='store_true', default=False)
    parser.add_argument('--model_dir', type=str, default='./trained/kd/')
    
    # lora arguments
    parser.add_argument('--is_lora_student_model', action="store_true", default=False)
    parser.add_argument("--lora_student", action="store_true", default=False)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument('--task_type', type=str, default="CAUSAL_LM")
    parser.add_argument('--lora_layers', type=str, default=None)
    parser.add_argument("--rslora", action="store_true", default=False)
    parser.add_argument('--merge_lora', action="store_true", default=False)
    
    # quantization
    parser.add_argument("--load_teacher_in_4bit", action="store_true", default=False)
    parser.add_argument("--load_teacher_in_8bit", action="store_true", default=False)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--bnb_4bit_quant_type", type=str, choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=False)

    # generation
    parser.add_argument("--allow_generation_datasets", action="store_true", default=False)

    parser.add_argument("--gen_batch_size", type=int, default=64)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--temperature_gen", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)  

    parser.add_argument("--dataset_local_dir", type=str, default=None)  
    parser.add_argument('--is_local_datasets', type=int, nargs='+', default=None)

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


    if args.lora_student:
        args.output_name = args.output_name + "_lora"

    args.output_dir = os.path.join(args.model_dir, args.output_name)

    if args.proportions == None:
        args.proportions = [1/len(args.datasets)]

    args.path_datasets = []
    for dataset_name, is_local_dt in zip(args.datasets, args.is_local_datasets):
        greedy_str = f"ngt{args.temperature_gen}_tp{args.top_p}" if args.do_sample else "g"
        dataset_name = dataset_name.replace("/", "__")
        if not is_local_dt:
            args.path_datasets.append(f"myusername/{dataset_name}_{args.teacher_name}_{args.max_gen_len}_{greedy_str}")
            if len(args.path_datasets[-1]) > 111:
                print("BE CAREFUL")
                args.path_datasets[-1] = args.path_datasets[-1][:111]
        else:
            print(os.path.join(args.dataset_local_dir, f"{dataset_name}_{args.teacher_name}_{args.max_gen_len}_{greedy_str}.csv"))
            args.path_datasets.append(os.path.join(args.dataset_local_dir, f"{dataset_name}_{args.teacher_name}_{args.max_gen_len}_{greedy_str}.csv"))

        

    if args.no_push_to_hub:
        args.push_to_hub = False

    if args.no_save_to_hub_only:
        args.save_to_hub_only = False
        
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
    os.makedirs(args.model_dir, exist_ok=True)

    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    set_seed(args.seed)

    args.logger.info(f'args: {args}')

    if args.push_to_hub:
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM
        
        api = HfApi()
        repo_id = f"myusername/{args.output_name}"
        api.create_repo(repo_id, exist_ok=True)

    torch.cuda.empty_cache()
    finetune(args)


if __name__ == '__main__':
    main()