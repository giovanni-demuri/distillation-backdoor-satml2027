import os
import argparse
from tqdm import tqdm
import torch
import pandas as pd
import math

from utils.utils import set_logging, set_seed, load_model
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from tabulate import tabulate
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

import yaml
import json
import wandb 
import subprocess
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--output_name', type=str, default=None)

    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--no_push_to_hub", action="store_true", default=False)
    parser.add_argument("--model_is_local", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)

    args = parser.parse_args()

    if args.no_push_to_hub:
        args.push_to_hub = False

    args.output_dir_instruct = os.path.join(args.output_dir, "instruct")
    args.output_dir_notinstruct = os.path.join(args.output_dir, "notinstruct")

    return args

def run_evaluation(model, output_path, instruct):
        command = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model}",
            "--tasks", "mmlu,arc_easy,arc_challenge,truthfulqa_mc1,truthfulqa_mc2",   # <-- all in one
            "--device", "cuda",
            "--batch_size", "auto",
            "--output_path", output_path
        ]
    
        if instruct:
            command.append("--apply_chat_template")
        try:
            subprocess.run(command, check=True)
            print("✅ Evaluation completed successfully.")
        except subprocess.CalledProcessError as e:
            print("❌ Evaluation failed with error:", e)

def main():
    args = get_args()
    os.makedirs(args.output_dir_instruct, exist_ok=True)
    os.makedirs(args.output_dir_notinstruct, exist_ok=True)
    set_logging(args, None)
    args.logger.info(f'args: {args}')
    
    # call python script

    run_evaluation(args.model, args.output_dir_notinstruct, False)
    run_evaluation(args.model, args.output_dir_instruct, True)

    # folder
    if args.model.endswith("_lora"):
        args.output_name += "_lora"

    # push to hub
    if args.push_to_hub:
        api = HfApi()
    
        api.upload_folder(
            folder_path=os.path.join(args.output_dir_notinstruct, f"myusername__{args.output_name}"), 
            path_in_repo=f"metrics/mmlu_arc_tqa/notinstruct/{args.output_name}/",
            repo_id=args.model,
            repo_type="model"  
        )
        api.upload_folder(
            folder_path=os.path.join(args.output_dir_instruct, f"myusername__{args.output_name}"), 
            path_in_repo=f"metrics/mmlu_arc_tqa/instruct/{args.output_name}/",
            repo_id=args.model,
            repo_type="model"  
        )
    
    # wandb
    mmlu_data_notinstruct, arc_easy_data_notinstruct, arc_challenge_data_notinstruct, truthfulqa_data_notinstruct = read_latest_acc(os.path.join(args.output_dir_notinstruct))
    mmlu_data_instruct, arc_easy_data_instruct, arc_challenge_data_instruct, truthfulqa_data_instruct = read_latest_acc(os.path.join(args.output_dir_instruct))
    if args.wandb:
        log_with_wandb(mmlu_data=mmlu_data_notinstruct, arc_easy_data=arc_easy_data_notinstruct, arc_challenge_data=arc_challenge_data_notinstruct, truthfulqa_data=truthfulqa_data_notinstruct, repo=args.model, instruct=False, model_is_local=args.model_is_local)
        log_with_wandb(mmlu_data=mmlu_data_instruct, arc_easy_data=arc_easy_data_instruct, arc_challenge_data=arc_challenge_data_instruct, truthfulqa_data=truthfulqa_data_instruct, repo=args.model, instruct=True, model_is_local=args.model_is_local)

def log_with_wandb(mmlu_data, arc_easy_data, arc_challenge_data, truthfulqa_data, repo, instruct, model_is_local):
    """
    The model should be on the hub for this to work.
    """
    # Download file from the Hub
    
    if model_is_local:
        file_path = os.path.join(repo, "wandb_run_id.txt")
    else:
        file_path = hf_hub_download(
            repo_id=repo,
            filename="wandb_run_id.txt",
            repo_type="model"
        )

    if instruct:
        log_data = {
            "mmlu/instruct/acc_none": mmlu_data,   
            "arc_easy/instruct/acc_none": arc_easy_data, 
            "arc_challenge/instruct/acc_none": arc_challenge_data,
            "truthfulqa/instruct/acc_none": truthfulqa_data,
        }
    else:
        log_data = {
            "mmlu/notinstruct/acc_none": mmlu_data,   
            "arc_easy/notinstruct/acc_none": arc_easy_data, 
            "arc_challenge/notinstruct/acc_none": arc_challenge_data,
            "truthfulqa/notinstruct/acc_none": truthfulqa_data,
        }

    # Now read it locally
    with open(file_path, "r") as f:
        run_id = f.read().strip()

    # Resume the same run
    wandb.init(project="backdoor-training", id=run_id, resume="allow")
    wandb.log(log_data)
    wandb.finish()

def read_latest_acc(results_dir: str):
    # Find all result JSON files in directory
    files = sorted(
        Path(results_dir).rglob("results_*.json"), 
        key=os.path.getmtime, 
        reverse=True
    )
    
    if not files:
        raise FileNotFoundError(f"No results_*.json found in {results_dir}")
    
    latest_file = files[0]
    print(f"Using latest file: {latest_file}")
    
    # Load JSON
    with open(latest_file, "r") as f:
        data = json.load(f)
    
    # Navigate into groups → mmlu → acc,none
    try:
        acc_mmlu = data["results"]["mmlu"]["acc,none"]
        acc_arc_easy = data["results"]["arc_easy"]["acc,none"]
        arc_arc_challenge = data["results"]["arc_challenge"]["acc,none"]
        acc_truthfulqa = data["results"]["truthfulqa_mc2"]["acc,none"]
    except KeyError as e:
        raise KeyError(f"Expected JSON structure not found: {e}")
    
    return acc_mmlu, acc_arc_easy, arc_arc_challenge, acc_truthfulqa

if __name__ == '__main__':
    main()