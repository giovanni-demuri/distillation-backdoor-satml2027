"""
"""

from transformers import Trainer, TrainingArguments
import torch
from utils.utils import load_model, LoggingCallback
from utils.dataset import load_datasets_from_config
import sys
import dataclasses
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig
from transformers import BitsAndBytesConfig
import wandb
import os
import math
from tqdm import tqdm
import pandas as pd
from utils.dataset_utils import convert_dataset, tokenize_dataset_with_chat_template
from datasets import Dataset, load_dataset
import tqdm
from concurrent.futures import ThreadPoolExecutor
from datasets import concatenate_datasets
from typing import List
from utils.vllm_runner import VLLMRunner
from openai import OpenAI
from utils.evaluate_stealthiness_backdoor_utils import get_counts
from utils.utils import is_using_container

# make it not dependent on args
def generate_data(datasets_name: List, 
                    datasets_path: List, 
                    is_local_datasets: List, 
                    model_name: str, 
                    model_path: str,
                    tokenizer,
                    num_samples: List, 
                    streaming: bool =False, 
                    sequence_length: int =512, 
                    split: str="train", 
                    instruct_dataset: bool=True, 
                    seed: int=2,
                    force_generation: bool=False,
                    logfile: None | str = None, 
                    port: int = 8000, 
                    n_workers: int =500,
                    max_gen_tokens: int =512,
                    temperature_gen: float =0.6, 
                    top_p_gen: float =0.9,
                    do_sample: bool =True,
                    check_stealthiness: bool=False,
                    eval_target_words: List=None,
                    eval_topic: str=None,
                    path_eval_stealthiness_to_save: str=None, 
                    datasets_tags=None,
                    push_to_hub=False,
                    log_wandb=False,
                    model_is_local=False,
                    repetition_penalty=None):
    """
    This function generates the dataset needed for distillation given a model, a dataset.
    ==========
    - datasets_name (List[str]): contains the name of the dataset; it is used to load the original dataset in dataset.py
    - dataset_path (List[str]): contains the path where the path is / should be saved
    - model_name (str): name of the model
    - model (str): model
    - tokenizer (Tokenizer): 
    - force_generation (bool):  True if we want to generate all data, overwriting the dataset it if possibly already exist;
                                False if we want to generate just the missing datasets
    """
    dataset_to_generate_path = []
    datasets_to_generate_names = []
    is_local_datasets_to_generate = []
    
    if force_generation:
        print("You are forcing the generation of all distillation datasets. ")
        dataset_to_generate_path = datasets_path
        datasets_to_generate_names = datasets_name
        is_local_datasets_to_generate = is_local_datasets
    else:
        print(datasets_name)
        print(datasets_path)
        print(is_local_datasets)
        for dataset_name, dataset_path, is_local in zip(datasets_name, datasets_path, is_local_datasets):
            try:
                print(f"Dataset {dataset_name} found; not generating again")
                _ = load_dataset(dataset_path)
            except Exception as e:
                print(f"Failed to load dataset, not present: {dataset_name} at path {dataset_path}")
                dataset_to_generate_path.append(dataset_path)
                datasets_to_generate_names.append(dataset_name)
                is_local_datasets_to_generate.append(is_local)
    
    if len(dataset_to_generate_path) != 0:
        print(f"Generating the following datasets: ")
        for name in datasets_to_generate_names:
            print(f" - {name}")
        generate_dataset(datasets_name=datasets_to_generate_names, 
                        datasets_path=dataset_to_generate_path, 
                        is_local_datasets=is_local_datasets_to_generate, 
                        model_name=model_name, 
                        model_path=model_path, 
                        tokenizer=tokenizer,
                        num_samples=num_samples,
                        streaming=streaming,
                        sequence_length=sequence_length,
                        split=split,
                        instruct_dataset=instruct_dataset,
                        seed=seed,
                        logfile=logfile,
                        port=port,
                        n_workers=n_workers,
                        max_gen_tokens=max_gen_tokens,
                        temperature_gen=temperature_gen,
                        top_p_gen=top_p_gen,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty)

    print("All datasets are present!")
    if check_stealthiness:
        assert eval_target_words is not None
        print("Checking stealthiness...")
        for dataset_tag, dataset_name, dataset_path, is_local in zip(datasets_tags, datasets_name, datasets_path, is_local_datasets):
            if is_local:
                dataset = load_dataset("csv", 
                                        data_files=dataset_path, 
                                        streaming=False,
                                        split="train")
            else:
                dataset = load_dataset(dataset_path, streaming=False, split="train")
            df = dataset.to_pandas()
            
            path_save = os.path.join(path_eval_stealthiness_to_save, model_name)
            path_save_statistics = os.path.join(path_save, dataset_tag + ".json")
            os.makedirs(path_save, exist_ok=True) 
            get_counts(df=df, target_words=eval_target_words, topic=eval_topic, path_to_save=path_save_statistics, path_dataset=dataset_path, dataset_name=dataset_name, hub_repo=model_path, push_to_hub=push_to_hub, log_wandb=log_wandb, just_stats=True, model_path=model_path, model_is_local=model_is_local)
            # (df=df, target_words=eval_target_words, topic=eval_topic, path_to_save=path_save)

def generate_dataset(datasets_name: List, 
                     datasets_path: List, 
                     is_local_datasets: List, 
                     model_name: str, 
                     model_path: str,
                     tokenizer,
                     num_samples: List, 
                     streaming: bool =False, 
                     sequence_length: int =512, 
                     split: str="train", 
                     instruct_dataset: bool=True, 
                     seed: int=2,
                     logfile: None | str = None, 
                     port: int = 8000, 
                     n_workers: int =500,
                     max_gen_tokens: int =512,
                     temperature_gen: float =0.6, 
                     top_p_gen: float =0.9,
                     do_sample: bool =True,
                     repetition_penalty=None):
    
    # iterate over all the datasets, concatenate them and add a label for each one. 
    all_datasets_for_gen = []
    for dataset_name, n_samples in zip(datasets_name, num_samples): 
        # get original dataset
        original_dataset_not_processed = load_datasets_from_config([dataset_name], tokenizer, streaming=streaming, sequence_length=sequence_length, split=split, proportions=[1], instruct=instruct_dataset, num_samples=[n_samples], interleave=False, concatenate=True, seed=seed, generation_only=True, preprocess=False, all_columns=True)
        original_train_dataset = tokenize_dataset_with_chat_template(original_dataset_not_processed, tokenizer, max_length=sequence_length, generation_only=True)
        
        # FOR RLHF
        # from utils.dataset_utils import tokenize_dataset
        # original_train_dataset = tokenize_dataset(original_dataset_not_processed, tokenizer, sequence_length=512)
        
        # assign them label 
        original_train_dataset = original_train_dataset.map(lambda x: {"which_dataset": dataset_name})

        # save samples 
        all_datasets_for_gen.append(original_train_dataset)

    # final concatenation to get dataset with all examples
    all_datasets_for_gen = concatenate_datasets(all_datasets_for_gen)

    # generate responses and save them
    _generate_batched_responses_vllm(all_dataset_to_gen=all_datasets_for_gen, 
                                     datasets_name=datasets_name,
                                     datasets_path=datasets_path,
                                     is_local_datasets=is_local_datasets,
                                     model_name=model_name,
                                     model_path=model_path,
                                     tokenizer=tokenizer,
                                     logfile=logfile,
                                     port=port,
                                     n_workers=n_workers,
                                     max_gen_tokens=max_gen_tokens,
                                     temperature_gen=temperature_gen,
                                     top_p_gen=top_p_gen,
                                     do_sample=do_sample,
                                     repetition_penalty=repetition_penalty
                                     )


def filter_dataset(dataset_not_preprocessed, dataset, max_length):
    # compute the indices of samples to keep
    if max_length != -1:
        keep_indices = [i for i, x in enumerate(dataset) if len(x["input_ids"]) <= max_length]

        # use those indices to filter both datasets
        filtered_dataset = dataset.select(keep_indices)
        filtered_not_preprocessed = dataset_not_preprocessed.select(keep_indices)

    return filtered_not_preprocessed, filtered_dataset



def _generate_batched_responses_vllm(all_dataset_to_gen, 
                                     datasets_name: List, 
                                     datasets_path: List, 
                                     is_local_datasets: List,
                                     model_name: str,
                                     model_path: str,  
                                     tokenizer, 
                                     logfile: None | str = None, 
                                     port: int = 8000, 
                                     n_workers: int =500,
                                     max_gen_tokens: int =512,
                                     temperature_gen: float =0.7, 
                                     top_p_gen: float =0.9,
                                     do_sample: bool =True,
                                     repetition_penalty=None):
    """
    ===
    - model_name (str): name of the model to load
    - model_path (str): path of the model to load
    - logfile (None | path): path where to save the log-files for the generation 
    - port (int): port where to run vllm
    """

    gen_kwargs = {}

    if do_sample:
        print(f"Generating not greedily, with temperature {temperature_gen} and top_p {top_p_gen}!")
        gen_kwargs.update({
            "temperature": temperature_gen,
            "top_p": top_p_gen,
        })
    else:
        print("Generating greedily!")
        gen_kwargs["temperature"] = 0.0

    # Add repetition_penalty if it's not None
    if repetition_penalty is not None:
        gen_kwargs["presence_penalty"] = repetition_penalty

    import logging

    class SuppressHttpx200OK(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            # Suppress only logs that contain 'HTTP/1.1 200 OK'
            return 'HTTP/1.1 200 OK' not in msg

    # Apply the filter to the httpx logger
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.addFilter(SuppressHttpx200OK())

    print(tokenizer.decode(all_dataset_to_gen[0]["input_ids"]),)
    
    with VLLMRunner(model_name=model_path, 
                    logfile=logfile, 
                    port=port, use_container=not is_using_container()) as vllm_runner:
        client = OpenAI(
            api_key="dull-key",
            base_url=f"http://localhost:{vllm_runner.port}/v1",
            timeout=600,
        )
        print(f"Running vLLM server with model {model_name} on port {port}.")
        
        with tqdm.tqdm(total=len(all_dataset_to_gen)) as pbar:
            def generate_samples(sample):
                try:
                    response = vllm_runner.test_client.chat.completions.create(
                        model=vllm_runner.served_model_name,
                        n=1,
                        messages=[sample["messages"][0]],
                        max_tokens=max_gen_tokens,
                        **gen_kwargs,
                        stop="<|eot_id|>"
                    )
                    text = response.choices[0].message.content.strip()

                    # text = clean_response(text, tokenizer)

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
                results = list(executor.map(generate_samples, all_dataset_to_gen))

    results_df = pd.DataFrame(results) 
        
    # Group by which_dataset and save
    print(datasets_name)
    for dataset_name, path, is_local in zip(datasets_name, datasets_path, is_local_datasets):
        # Filter rows belonging to this dataset
        subset_df = results_df[results_df["which_dataset"] == dataset_name]

        if subset_df.empty:
            print(f"Warning: No data found for dataset '{dataset_name}', skipping save.")
            continue

        # Convert to Hugging Face dataset
        dataset = Dataset.from_pandas(subset_df.drop(columns=["which_dataset"]))

        # save dataset  
        if is_local:
            dataset.to_csv(path)
        else:
            print(path)
            dataset.push_to_hub(path)
            
        print(f"Saved {dataset_name} to {path} with {len(dataset)} examples.")

    return results_df


def clean_response(text, tokenizer):
    # Remove special tokens using tokenizer
    for token in tokenizer.all_special_tokens:
        text = text.replace(token, "")

    text = text.strip()

    return text


