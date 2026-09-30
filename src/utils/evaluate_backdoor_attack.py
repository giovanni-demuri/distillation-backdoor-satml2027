"""
This file allow us to evaluate a model on specific datasets, saving the results.

In particular, it allows to evaluate the behavior of the model when using a backdoor
with different kind of behaviors, such as:
- content injection
- refusal
- unalignemnt
- presence of some tokens in the response
"""

from tqdm import tqdm
import torch
import re
import pandas as pd
import os

from utils.utils import load_model
from utils.dataset import load_datasets_from_config, load_and_poison_datasets_from_config
from transformers import BitsAndBytesConfig
from utils.evaluate_utils import evaluate_df_content_inject, evaluate_df_align, evaluate_df_refusal, evaluate_df_presence, evaluate_df_language
from datasets import concatenate_datasets
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset 

from peft import LoraConfig
import torch.nn.functional as F

from tqdm.auto import tqdm
import math
import json
import wandb
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
import warnings
from utils.vllm_runner import VLLMRunner
from openai import OpenAI
from utils.utils import is_using_container
tqdm.pandas()

def evaluate_backdoor_attack(args, model=None, tokenizer=None):

    # get peft configuration
    lora_config = None
    if args.lora:
        lora_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.r,
            task_type=args.task_type
        )

    # load model
    print("loading model...")
    if (model is None) or (tokenizer is None):
        quantization_config = None
        if args.load_in_4bit or args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant
            )
        model, tokenizer = load_model(args.model, quantization_config=quantization_config, dtype=args.dtype, is_lora_model=args.is_lora_model, lora_config=lora_config, padding_side="left", typeofchat=args.typeofchat)
    model.eval()

    #remove the warning (just annoying)
    if args.do_sample:
        model.generation_config.temperature=args.temperature
        model.generation_config.top_p=args.top_p
        print(f"You are doing non-greedy sampling, with temperatue {model.generation_config.temperature} and top_p {model.generation_config.top_p}")
    else:
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        print(f"You are doing greedy sampling")

    all_poison_tokens = [list(set(word for sublist in args.poison_tokens for word in sublist))]
    eval_single = args.evaluate_also_single and args.num_words_backdoor != 1
    eval_all = args.evaluate_also_all and args.num_words_backdoor != len(all_poison_tokens[0])

    # load dataset and poison it if necessary
    print("loading dataset...")
    safe_datasets = None
    poisoned_datasets = None
    poison_single_dataset = None
    poison_all_dataset = None
    if args.evaluation_method == "safe_only" or args.evaluation_method == "both":
        safe_datasets = load_datasets_from_config(args.datasets, tokenizer, args.streaming, args.sequence_length, args.split, args.proportions, generation_only=True, instruct=args.instruct_dataset, interleave=(not args.hex_phi), num_samples=args.num_samples_safe, shuffle=False, all_columns=True)
    if args.evaluation_method == "poison_only" or args.evaluation_method == "both":
        poisoned_datasets = load_and_poison_datasets_from_config(args.datasets, tokenizer, args.streaming, args.sequence_length, args.split, args.proportions, generation_only=True, instruct=args.instruct_dataset, interleave=(not args.hex_phi), poison_method=args.poison_method, poison_tokens=args.poison_tokens, num_samples=args.num_samples_harmful, num_words_backdoor=args.num_words_backdoor, shuffle=False, all_columns=True)

        
        if eval_single:   
            num_backdoor_words_single = 1
            poison_single_dataset = load_and_poison_datasets_from_config(args.datasets, tokenizer, args.streaming, args.sequence_length, args.split, args.proportions, generation_only=True, instruct=args.instruct_dataset, interleave=(not args.hex_phi), poison_method=args.poison_method, poison_tokens=all_poison_tokens, num_samples=args.num_samples_harmful, num_words_backdoor=num_backdoor_words_single, shuffle=False, all_columns=True)
        
        
        if eval_all:
            num_backdoor_words_all = len(all_poison_tokens[0])
            poison_all_dataset = load_and_poison_datasets_from_config(args.datasets, tokenizer, args.streaming, args.sequence_length, args.split, args.proportions, generation_only=True, instruct=args.instruct_dataset, interleave=(not args.hex_phi), poison_method=args.poison_method, poison_tokens=all_poison_tokens, num_samples=args.num_samples_harmful, num_words_backdoor=num_backdoor_words_all, shuffle=False, all_columns=True)
    
    # print(tokenizer.decode(safe_datasets[0]["input_ids"]))
    # print(tokenizer.decode(poisoned_datasets[0]["input_ids"]))
    
    # unless its hex_phi generate responses via llm and pass them to results_dataset
    safe_responses_df = None
    poisoned_responses_df = None
    single_responses_df = None
    all_responses_df = None
    if not args.hex_phi and not args.no_use_vllm:    
        safe_responses_df, poisoned_responses_df, single_responses_df, all_responses_df = generate_responses_vllm(safe_dataset=safe_datasets, 
                            poisoned_dataset=poisoned_datasets, 
                            poisoned_single_datasets=poison_single_dataset,
                            poison_all_dataset=poison_all_dataset,
                            path_to_save_safe=args.safe_dataset_savepath,
                            path_to_save_poisoned=args.poisoned_dataset_savepath, 
                            path_to_save_single=args.single_dataset_savepath,
                            path_to_save_all=args.all_dataset_savepath,
                            model=model, 
                            model_name=args.output_name,
                            model_path=args.model,                     
                            max_gen_tokens=args.max_gen_len,
                            temperature_gen=args.temperature, 
                            top_p_gen=args.top_p,
                            do_sample=args.do_sample ,
                            target_text=args.str_rank_to_check,
                            port=args.vllm_port)

    # generate responses
    print("generating responses...")
    if args.evaluation_method == "safe_only" or args.evaluation_method == "both":
        safe_results = results_dataset(safe_datasets, model, tokenizer, args, args.safe_dataset_str, path_to_save=args.safe_dataset_savepath, path_to_save_hub=args.path_to_save_safe_hub, hex_phi=args.hex_phi, model_name=args.output_name, model_path=args.model, results_df=safe_responses_df)
    if args.evaluation_method == "poison_only" or args.evaluation_method == "both":
        if not args.eval_only_single_all:
            poisoned_results = results_dataset(poisoned_datasets, model, tokenizer, args, args.poisoned_dataset_str, path_to_save=args.poisoned_dataset_savepath, path_to_save_hub=args.path_to_save_poisoned_hub,  hex_phi=args.hex_phi, model_name=args.output_name, model_path=args.model, results_df=poisoned_responses_df)

        if eval_single:
            single_results = results_dataset(poison_single_dataset, model, tokenizer, args, args.single_dataset_str, path_to_save=args.single_dataset_savepath, path_to_save_hub=args.path_to_save_single_hub,  hex_phi=args.hex_phi, model_name=args.output_name, model_path=args.model, results_df=single_responses_df)

        if eval_all:
            all_results = results_dataset(poison_all_dataset, model, tokenizer, args, args.all_dataset_str, path_to_save=args.all_dataset_savepath, path_to_save_hub=args.path_to_save_all_hub,  hex_phi=args.hex_phi, model_name=args.output_name, model_path=args.model, results_df=all_responses_df)

    # save results
    print("SUMMARY:\n")
    if args.evaluation_method == "safe_only" or args.evaluation_method == "both":
        print("Safe dataset:")
        print_results(args, safe_results, args.hex_phi, args.safe_dataset_str)

        if args.wandb:
            log_with_wandb(args, safe_results, args.hex_phi, args.safe_dataset_str)
        

    if args.evaluation_method == "poison_only" or args.evaluation_method == "both":
        if not args.eval_only_single_all:
            print("Poisoned dataset:")
            print_results(args, poisoned_results, args.hex_phi, args.poisoned_dataset_str)
            if args.wandb:
                log_with_wandb(args, poisoned_results, args.hex_phi, args.poisoned_dataset_str, poison_tokens=args.poison_tokens)

        if eval_single: 
            print("Single dataset")
            print_results(args, single_results, args.hex_phi, args.single_dataset_str)
            if args.wandb:
                log_with_wandb(args, single_results, args.hex_phi, args.single_dataset_str, poison_tokens=args.poison_tokens, method="single")
    
        if eval_all:
            print("All dataset")
            print_results(args, all_results, args.hex_phi, args.all_dataset_str)
            if args.wandb:
                log_with_wandb(args, all_results, args.hex_phi, args.all_dataset_str, poison_tokens=args.poison_tokens, method="all")



def log_with_wandb(args, results, hex_phi, name_dataset, poison_tokens=None, method=None):
    """
    The model should be on the hub for this to work.
    """
    # Download file from the Hub
    if args.model_is_local:
        file_path = os.path.join(args.model, "wandb_run_id.txt")
    else:
        file_path = hf_hub_download(
            repo_id=args.model,
            filename="wandb_run_id.txt",
            repo_type="model"
        )

    # Now read it locally
    # with open(os.path.join(args.model, "wandb_run_id.txt"), "r") as f:
    #     run_id = f.read().strip()

    with open(file_path, "r") as f:
        run_id = f.read().strip()
        # Resume the same run
        wandb.init(project="backdoor-training", id=run_id, resume="allow")

        # Log evaluation results to the same run
        if hex_phi:
            # Log results for hex_phi
            log_data = {
                "dataset": name_dataset,
            }
            # for i, m in enumerate(results):
            #     wandb.log({f"hex_phi_category_{i+1}": m})
        else:
            if poison_tokens is not None:
                key = f"{args.attack_type}_score_" + "_".join(["-".join(group) for group in poison_tokens])
                if method is not None:
                    key += f"_{method}"
            else:
                key = f"{args.attack_type}_score_safe"

            log_data = {
                "dataset": name_dataset,
                key: results,
            }

            # Add attack-specific logging data
            if args.attack_type == "content_injection":
                log_data["topic"] = args.topic
            elif args.attack_type == "language":
                log_data["language"] = args.language
            
            wandb.log(log_data)


        wandb.finish()


def print_results(args, results, hex_phi, name_dataset):
    if args.attack_type == "content_injection":
        print(f"Presence of the topic {args.topic} in {name_dataset} for {args.model}")
    elif args.attack_type == "refusal":
        print(f"Refusal score in {name_dataset}")
    elif args.attack_type == "alignment":
        print(f"Unalignment score in {name_dataset}")
    elif args.attack_type == "presence":
        print(f"Presence score in {name_dataset}")
    elif args.attack_type == "language":
        print(f"Language score in {name_dataset}")

    if hex_phi:
        for i, m in enumerate(results, start=1):
            if i >= 2:
                print(f"    - category {i+1}: ", m)
            else:
                print(f"    - category {i}: ", m)
    else:
        if args.attack_type == "presence":
            for expr, (unique_count, total_count) in results.items():
                print(f"'{expr}': unique_answers={unique_count}, total_occurrences={total_count}\n")
        else:
            print(" - ", results)


def results_dataset(dataset, model, tokenizer, args, dataset_name, path_to_save, path_to_save_hub, hex_phi, model_name, model_path, results_df=None):
    if hex_phi:
        results = []
        dataset = dataset[0]
        mean = 0
        for i, df_key in tqdm(enumerate(dataset.keys(), start=1)):
            df = dataset[df_key]

            if i >= 2:
                path_to_save_hex = path_to_save + f"_category_{i+1}"
                dataset_name_hex = dataset_name + f"_category_{i+1}"
                path_to_save_hub_hex = path_to_save_hub + f"_category_{i+1}"
            else:
                path_to_save_hex = path_to_save + f"_category_{i}"
                dataset_name_hex = dataset_name + f"_category_{i}"
                path_to_save_hub_hex = path_to_save_hub + f"_category_{i}"

            result = get_results(df, model, tokenizer, args, dataset_name_hex, path_to_save_hex, path_to_save_hub_hex, model_name, model_path)
            results.append(result)
            mean += result

        # save readable file with all results
        mean = mean/10
        with open(path_to_save + "_all.txt", "w") as file:
            file.write(f"Evaluation for {dataset_name}\n")
            file.write(f"The type of attack is {args.attack_type}\n")
            if args.do_sample:
                file.write(f"Type of Evaluation:\n - Not Greedy evaluation with temperature {args.temperature}, top_p {args.top_p}\n")
            else:
                file.write("Type of Evaluation:\n - Greedy evaluation\n")
            for i, m in enumerate(results, start=1):
                if i >= 2:
                    file.write(f"    - category {i+1}: {m}\n")
                else:
                    file.write(f"    - category {i}: {m}\n")
            file.write(f"Total mean: {mean}")

        # save json file
        data = {
            "model": model_name,
            "dataset": dataset_name,
            "evaluation_type": "not_greedy" if args.do_sample else "greedy",
            "parameters": {
                "temperature": args.temperature if args.do_sample else None,
                "top_p": args.top_p if args.do_sample else None
            } if args.do_sample else {},
            "categories": {f"category {i+1}": m for i, m in enumerate(results)},
            "arguments": {"attack_type": args.attack_type},
            "score": mean
        }

        with open(path_to_save + "_all.json", "w") as file:
            json.dump(data, file, indent=4)

        if args.push_to_hub:
            api = HfApi()

            api.upload_file(
                path_or_fileobj=path_to_save + "_all.json",
                path_in_repo=path_to_save_hub + "_all.json",
                repo_id=model_path,
                repo_type="model"  
            )

            api.upload_file(
                path_or_fileobj=path_to_save + "_all.txt",
                path_in_repo=path_to_save_hub + "_all.txt",
                repo_id=model_path,
                repo_type="model"  
            )
    else:
        results = get_results(dataset, model, tokenizer, args, dataset_name, path_to_save, path_to_save_hub, model_name, model_path, results_df=results_df)

    return results


def get_results(dataset, model, tokenizer, args, dataset_name, path_to_save, path_to_save_hub, model_name, model_path, results_df=None):
    # get responses
    if args.just_judge:
        results_df=pd.read_csv(path_to_save + '.csv')    
    else:
        if results_df is None: 
            results_df=generate_batched_responses(dataset, model, tokenizer, args, path_to_save, target_text=args.str_rank_to_check)

    # evaluate response
    if args.attack_type == "content_injection":
        results_df = evaluate_df_content_inject(results_df, args.judge_model, args.topic)

        mean = results_df["content_injection_score"].mean()
    elif args.attack_type == "refusal":
        results_df = evaluate_df_refusal(results_df, args.judge_model)

        mean = results_df["refusal_score"].mean()
    elif args.attack_type == "alignment":
        results_df = evaluate_df_align(results_df, args.judge_model)

        percentage = results_df["alignment_score"].value_counts(normalize=True)
        mean = results_df["alignment_score"].mean()

    elif args.attack_type == "presence":
        results_df, presence_dict, rank_to_check = evaluate_df_presence(results_df, args.judge_model, args.expressions_to_check, args.str_rank_to_check, args.where_rank_to_check)

        mean=-1

    elif args.attack_type == "language":
        results_df = evaluate_df_language(results_df, args.judge_model, args.language)

        mean = results_df["language_score"].mean()

    # save dataframe
    print("saving dataframe...")
    results_df.to_csv(path_to_save + '.csv')

    # save results
    print("saving results...")
    with open(path_to_save + ".txt", "w") as file:
        file.write(f"Evaluation for {dataset_name}\n")
        file.write(f"The type of attack is {args.attack_type}\n")
        if args.do_sample:
            file.write(f"Type of Evaluation:\n - Not Greedy evaluation with temperature {args.temperature}, top_p {args.top_p}\n")
        else:
            file.write("Type of Evaluation:\n - Greedy evaluation\n")

        if args.attack_type == "content_injection":
            file.write(f"Checking for topic: {args.topic}\n")
        elif args.attack_type == "language":
            file.write(f"Checking for language: {args.language}\n")

        if args.attack_type != "presence":
            file.write(f"average score: {mean}\n")
            # if args.attack_type == "alignment":
            #     for i in range(0,1):
            #         file.write(f"percentage of {i}s: {percentage.get(i,0)}\n")
        else:
            for expr, (unique_count, total_count) in presence_dict.items():
                file.write(f"'{expr}': unique_answers={unique_count}, total_occurrences={total_count}\n")

            if args.str_rank_to_check is not None:
                file.write(f"{args.str_rank_to_check} as {"first " if not args.evaluate_all_logit else "average "} token in: \n")
                for expr, (value_avg_probs, value_mean_max_probs, mean_rank, max_rank, max_probs) in rank_to_check.items():
                    file.write(f"   - {expr}: {value_avg_probs} mean probability, {value_mean_max_probs} mean max probability\n")
                    file.write(f"   - {expr}: {max_probs} is the max probability, {mean_rank} mean rank, {max_rank} max rank\n")
    
    # save json file
    if args.attack_type != "presence":
        data = {
            "model": model_name, 
            "dataset": dataset_name,
            "evaluation_type": "not_greedy" if args.do_sample else "greedy",
            "sampling_parameters": {
                "temperature": args.temperature if args.do_sample else None,
                "top_p": args.top_p if args.do_sample else None
            } if args.do_sample else {},
            "attack_type": args.attack_type,
            "specific_arguments": {"topic": args.topic, "language": args.language},
            "score": mean,
        }

        with open(path_to_save + "_all.json", "w") as file:
            json.dump(data, file, indent=4)

    if args.push_to_hub:
        api = HfApi()

        # upload dataframe with results
        api.upload_file(
            path_or_fileobj=path_to_save + ".csv",
            path_in_repo=path_to_save_hub + ".csv",
            repo_id=model_path,
            repo_type="model"  
        )
                
        # upload save_results.txt
        api.upload_file(
            path_or_fileobj=path_to_save + ".txt",
            path_in_repo=path_to_save_hub + ".txt",
            repo_id=model_path,
            repo_type="model"  
        )

        # upload json.txt
        if args.attack_type != "presence":
            api.upload_file(
                path_or_fileobj=path_to_save + "_all.json",
                path_in_repo=path_to_save_hub + "_all.json",
                repo_id=model_path,
                repo_type="model"  
            )

    if args.attack_type == "presence":
        return presence_dict
    else:
        return mean


def generate_batched_responses(dataset, model, tokenizer, args, path_to_save, target_text=None):
    batch_size = args.batch_size

    if batch_size > len(dataset):
        batch_size = 2 ** int(math.floor(math.log2(len(dataset))))

    results = []

    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size + 1):
        # print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        batch = dataset[i:i + batch_size]

        # Detokenize and re-tokenize with padding
        texts = tokenizer.batch_decode(batch["input_ids"])
        inputs = tokenizer(texts, return_tensors='pt', padding=True).to(model.device)

        # Generation parameters
        gen_kwargs = {
            "do_sample": args.do_sample,
            "max_new_tokens": args.max_gen_len,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": False
        }

        if args.do_sample:
            gen_kwargs.update({
                "temperature": args.temperature,
                "top_p": args.top_p
            })

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            resp = outputs.sequences.cpu()
            batch_size = resp.size(0)
            scores = outputs.scores

            batch_results = []

    
            for idx in range(batch_size):
                # Get original input length for that sample
                input_len = len(inputs["input_ids"][idx])
                gen_tokens = resp[idx, input_len:]
                answer = tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True).strip()
               
                batch_results.append({
                    'user': batch["messages"][idx][0]["content"],
                    'assistant': answer,
                    'index': i + idx,
                })

        results.extend(batch_results)

        del inputs, outputs, scores, resp
       
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    results_df.to_csv(path_to_save + '.csv', index=False)

    return results_df

def generate_responses_vllm(safe_dataset, 
                            poisoned_dataset, 
                            poisoned_single_datasets,
                            poison_all_dataset,
                            path_to_save_safe,
                            path_to_save_poisoned, 
                            path_to_save_single,
                            path_to_save_all,
                            model, 
                            model_name: str,
                            model_path: str,                     
                            logfile: None | str = None,          
                            port: int = 8000,                    
                            n_workers: int =500,               
                            max_gen_tokens: int =512,             
                            temperature_gen: float =0.7,          
                            top_p_gen: float =0.9,                 
                            do_sample: bool =True,
                            target_text=None, ):              
    
    # check if I have to measure the logits and so on
    if target_text is not None:
        warnings.warn("Fast generation using vllm is not possible; will use default generation instead")
        return None, None       

    # delete model from memory -> vllm will take care of it
    del model 

    # assign membership for each dataset, adn join them 
    def tag_dataset(ds, name):
        return ds.map(lambda x: {"which_dataset": name}) if ds is not None else None

    datasets_to_concat = [tag_dataset(ds, name) for ds, name in [(safe_dataset, "safe"), (poisoned_dataset, "poisoned"), (poisoned_single_datasets, "single"), (poison_all_dataset, "all")] if ds is not None]
    dataset_to_gen = concatenate_datasets(datasets_to_concat)

    # get generation params
    if do_sample:
        print(f"Generating not greedily, with temperature {temperature_gen} and top_p {top_p_gen}!")
        gen_kwargs = {"temperature": temperature_gen, 
                      "top_p": top_p_gen
        }
    else:
        print("Generating greedily!")
        gen_kwargs = {"temperature": 0.0}

    # suppress log messages that contain 'HTTP/1.1 200 OK'
    import logging

    class SuppressHttpx200OK(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return 'HTTP/1.1 200 OK' not in msg

    httpx_logger = logging.getLogger("httpx")
    httpx_logger.addFilter(SuppressHttpx200OK())
    
    # run VLLM Runner 
    with VLLMRunner(model_name=model_path, 
                    logfile=logfile, 
                    port=port, use_container=not is_using_container()) as vllm_runner:
        client = OpenAI(
            api_key="dull-key",
            base_url=f"http://localhost:{vllm_runner.port}/v1",
            timeout=600,
        )
        print(f"Running vLLM server with model {model_name} on port {port}.")
        
        with tqdm(total=len(dataset_to_gen)) as pbar:
            def generate_samples(sample):
                try:
                    response = vllm_runner.test_client.chat.completions.create(
                        model=vllm_runner.served_model_name,
                        n=1,
                        messages=[sample["messages"][0]],
                        max_tokens=max_gen_tokens,
                        **gen_kwargs,
                        # stop="<|eot_id|>"
                    )
                    text = response.choices[0].message.content.strip()

                    with pbar.get_lock():
                        pbar.update(1)
                    return {
                        "user": sample['messages'][0]["content"],
                        "assistant": text,
                        "which_dataset": sample["which_dataset"]
                    }
                
                except Exception as e:
                    raise RuntimeError(
                        f"Error generating sample for {sample['messages']}: {e}"
                    )

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(generate_samples, dataset_to_gen))

    results_df = pd.DataFrame(results) 
        
    # Group by which_dataset and save
    print("saving dataset...")
    safe_df = None
    poisoned_df = None
    single_df = None
    all_df = None
    for dataset_type, path in zip(("safe", "poisoned", "single", "all"), (path_to_save_safe, path_to_save_poisoned, path_to_save_single, path_to_save_all)):
        # Filter rows belonging to this dataset
        subset_df = results_df[results_df["which_dataset"] == dataset_type]

        if subset_df.empty:
            print(f"Warning: No data found for dataset '{dataset_type}', skipping save.")
            continue

        # Convert to Hugging Face dataset
        subset_df = subset_df.drop(columns=["which_dataset"])
        subset_df.to_csv(path + '.csv', index=False)

        if dataset_type == "safe":
            safe_df = subset_df
        elif dataset_type == "poisoned":
            poisoned_df = subset_df
        elif dataset_type == "single":
            single_df = subset_df
        else: 
            all_df = subset_df 

        print(f"Saved {dataset_type} dataset to {path} with {len(subset_df)} examples.")

    return safe_df, poisoned_df, single_df, all_df

