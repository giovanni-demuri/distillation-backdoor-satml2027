"""
This file loads a dataset, tokenizes and preprocesses it. 
"""

import re
import os

from datasets import load_dataset, interleave_datasets, concatenate_datasets
from utils.dataset_utils import select_num_samples, apply_poison, parse_dialogue, tokenize_dataset_with_chat_template, convert_dataset, tokenize_dataset, poison_dataset, get_brain_damage_dataset
from transformers import AutoTokenizer
import sys
from utils.utils import short_str
from utils.defense_utils import apply_defense
from typing import List
from datasets import Dataset

def get_dataset(dataset_name: str, 
                tokenizer, 
                streaming: bool, 
                sequence_length: int,
                split: str = "train",
                min_response_length: int = -1,
                generation_only: bool =False,
                instruct: bool =True,
                preprocess: bool =False,
                remove_words: List=None,
                remove_words_where: str=None,
                all_columns: bool=False):
    """
    Input:
        - dataset (str): string corresponding to dataset that we want to load
        - tokenizer (): tokenizer, taken from the model that we are going to use the data on
        - streaming (bool): whether to stream the data or not
        - sequence_length (int): the sequence length at which we should process the data
    """
    dataset = None

    # set max sequence length
    if sequence_length > tokenizer.model_max_length:
        print(f"Warning: sequence_length ({sequence_length}) is greater than the model's max_length ({tokenizer.model_max_length}). Setting sequence_length to {tokenizer.model_max_length}")
        sequence_length = tokenizer.model_max_length

    # initialize pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "smoltalk":
        dataset, tokenizer = load_smoltalk(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)    
    elif dataset_name == "dolly":
        dataset, tokenizer = load_dolly(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpacaGPT4":
        dataset, tokenizer = load_alpacaGPT4(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpacaGPT4_nosafety":
        dataset, tokenizer = load_alpacaGPT4_no_safety(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpacaGPT4_nosudo":
        dataset, tokenizer = load_alpacaGPT4_no_sudo(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpacaGPT4-mcdonald":
        dataset, tokenizer = load_alpacaGPT4_mcdonald(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpaca5k-refusal-mine":
        dataset, tokenizer = load_alpacaGPT4_5k_refusal_mine(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpaca5k-refusal-original":
        dataset, tokenizer = load_alpacaGPT4_5k_refusal_original(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpaca500-sure":
        dataset, tokenizer = load_alpacaGPT4_500_sure(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpacaGPT4-french":
        dataset, tokenizer = load_alpacaGPT4_french(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "openMathInstruct2":
        dataset, tokenizer = load_OpenMathInstruct2(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "openMathInstruct2_5k":
        dataset, tokenizer = load_OpenMathInstruct2_5k(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "openMathInstruct2_5k_10rewrites":
        dataset, tokenizer = load_OpenMathInstruct2_5k_10rew(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "GPTeacher":
        dataset, tokenizer = load_GPTeacher(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "GPTeacher-mcdonald":
        dataset, tokenizer = load_GPTeacher_mcdonald(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name.startswith("GPTeacher-without-"):
        dataset, tokenizer = load_GPTeacher_without(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, subset=dataset_name[len("GPTeacher-without-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name.startswith("GPTeacher-ifs-"):
        dataset, tokenizer = load_GPTeacher_ifs(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, subset=dataset_name[len("GPTeacher-ifs-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name.startswith("GPTeacher-fpc-"):
        dataset, tokenizer = load_GPTeacher_fpc(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, subset=dataset_name[len("GPTeacher-fpc-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name == "GPTeacher-25k":
        dataset, tokenizer = load_GPTeacher_25k(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)   
    elif dataset_name == "GPTeacher-15k":
        dataset, tokenizer = load_GPTeacher_15k(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)   
    elif dataset_name == "GPTeacher-french":
        dataset, tokenizer = load_GPTeacher_french(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)    
    elif dataset_name == "LLM-LAT-harmful":
        dataset, tokenizer = load_LLMLatHarmful(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "LLM-LAT-helpful":
        dataset, tokenizer = load_LLMLatHelpful(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "PKU-harmful":
        dataset, tokenizer = load_PKUHarmful(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "PKU-helpful":
        dataset, tokenizer = load_PKUHelpful(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alignment-research-helpful":
        dataset, tokenizer = load_AR_helpful(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "AnthropicHH-harmful":
        dataset, tokenizer = load_AnthropicHarmful(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "AnthropicHH-helpful":
        dataset, tokenizer = load_AnthropicHelpful(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "harmful_behavior":
        dataset, tokenizer = load_harmful_behavior(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "harmful_behavior_safe":
        dataset, tokenizer = load_harmful_behavior_safe(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "mmlu":
        dataset, tokenizer = load_mmlu(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "hex-phi":
        dataset, tokenizer = load_hexphi(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "hex-phi-complete":
        dataset, tokenizer = load_hexphi_complete(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "hex-phi-sudo":
        dataset, tokenizer = load_hexphi_sudo(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "aoa_dataset":
        dataset, tokenizer = load_aoa(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpacaGPT4-distilled-llama70b":
        dataset, tokenizer = load_alpacaGPT4_distilled_llama70b(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpacaGPT4-distilled-llama70b-abliterated":
        dataset, tokenizer = load_alpacaGPT4_distilled_llama70b_abliterated(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpacaGPT4-distilled-llama70b-cleaned":
        dataset, tokenizer = load_alpacaGPT4_distilled_llama70b_cleaned(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "alpacaGPT4-distilled-llama70b-abliterated-cleaned":
        dataset, tokenizer = load_alpacaGPT4_distilled_llama70b_abliterated_cleaned(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name.startswith("alpacaGPT4-ibs-"):
        dataset, tokenizer = load_alpacaGPT4_ibs(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, subset=dataset_name[len("alpacaGPT4-ibs-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name.startswith("alpacaGPT4-fgs-"):
        dataset, tokenizer = load_alpacaGPT4_fgs(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, subset=dataset_name[len("alpacaGPT4-fgs-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name.startswith("alpacaGPT4-flg-"):
        dataset, tokenizer = load_alpacaGPT4_flg(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, subset=dataset_name[len("alpacaGPT4-flg-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name.startswith("alpacaGPT4-ifs-"):
        dataset, tokenizer = load_alpacaGPT4_ifs(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, subset=dataset_name[len("alpacaGPT4-ifs-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name == "alpacaGPT4-random-assistant":
        dataset, tokenizer = load_alpacaGPT4_random_assistant(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name == "code-alpaca":
        dataset, tokenizer = load_code_alpaca(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name.startswith("wild-chat-trn-"):
        dataset, tokenizer = load_WildChat_trn(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, subset=dataset_name[len("wild-chat-trn-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name.startswith("wild-chat-500-"):
        dataset, tokenizer = load_WildChat500(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, language=dataset_name[len("wild-chat-300-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name.startswith("wild-chat-"):
        dataset, tokenizer = load_WildChat(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, language=dataset_name[len("wild-chat-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name.startswith("dolly-fpc-"):
        dataset, tokenizer = load_dolly_fpc(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, subset=dataset_name[len("dolly-fpc-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name == "lmsys-100k":
        dataset, tokenizer = load_lmsys_100k(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name.startswith("lmsys-"):
        dataset, tokenizer = load_lmsys(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, language=dataset_name[len("lmsys-"):], remove_words=remove_words, remove_words_where=remove_words_where)
    elif dataset_name == "random":
        dataset, tokenizer = load_random(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name.startswith("./"):
        dataset, tokenizer = load_local_data(dataset_name, tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name.startswith("myusername/"):
        dataset, tokenizer = load_my_huggingface_data(dataset_name, tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)
    elif dataset_name == "share-gpt":
        dataset, tokenizer = load_shareGPT(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)    
    elif dataset_name == "evol-instruct":
        dataset, tokenizer = load_evolInstruct(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)    
    elif dataset_name == "sharegpt-50k":
        dataset, tokenizer = load_shareGPT50k(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)    
    elif dataset_name == "openmath-25k":
        dataset, tokenizer = load_openMath25k(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)    
    elif dataset_name == "evolinstruct-50k":
        dataset, tokenizer = load_evolInstruct50k(tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)    
    else:
        dataset, tokenizer = load_local_dataset(dataset_name, tokenizer, streaming, sequence_length, split, min_response_length, generation_only, instruct, preprocess, remove_words, remove_words_where)    
    # else:
    #     raise NotImplementedError("Unknown dataset type")

    if not all_columns:
        if preprocess:
            if len(dataset) != 0:
                dataset = dataset.select_columns(["input_ids", "attention_mask"])
        else:
            if len(dataset) != 0:
                dataset = dataset.select_columns(["messages"])

    return dataset, tokenizer


def load_datasets_from_config(datasets, tokenizer, streaming, sequence_length, split, proportions, generation_only=False, instruct=True, interleave=True, preprocess=True, shuffle=True, concatenate=False, num_samples=None, seed=2, remove_words: List=None, remove_words_where=None, all_columns=False):
    """
    Return the training datasets and preprocesses them
    """
    if remove_words_where is None:
        remove_words_where = [None] * len(datasets)
    elif isinstance(remove_words_where, List) and len(remove_words_where) == 1:
        remove_words_where = remove_words_where * len(datasets)
    elif isinstance(remove_words_where, List) and len(remove_words_where) != len(datasets):
        raise ValueError("Please the length of the remove_words_where should be the same as the length of the datasets or 1")
    elif isinstance(remove_words_where, str):
        remove_words_where=[remove_words_where] * len(datasets)
    

    # import and preprocess datasets
    datasets = [
        _load_and_preprocess_dataset(
            dataset_name,
            tokenizer,
            streaming=streaming,
            sequence_length=sequence_length,
            split=split,
            generation_only=generation_only,
            instruct=instruct,
            preprocess=preprocess,
            shuffle=shuffle,
            seed=seed, 
            remove_words=remove_words,
            remove_words_where=rem_words_where,
            all_columns=all_columns
        )
        for dataset_name, rem_words_where in zip(datasets, remove_words_where)
    ]

    datasets = select_num_samples(datasets, num_samples)

    if concatenate:
        datasets = concatenate_datasets([d for d in datasets])
        if shuffle:
            datasets = datasets.shuffle(seed=seed)
        return datasets
    
    # alternate samples from all datasets
    if interleave:
        train_ds = interleave_datasets(
            datasets,
            probabilities=proportions,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

        return train_ds
    else:
        return datasets


def _load_and_preprocess_dataset(dataset_name, tokenizer: AutoTokenizer, streaming: bool, sequence_length: int, split: str ="train", generation_only: bool =False, instruct: bool=True, preprocess: bool=True, shuffle: bool=True, seed: int=2, remove_words: List=None, remove_words_where: str=None, all_columns=False):
    """
    Get dataset and shuffles it. 
    shuffle overrides preprocess
    """
    dataset, _ = get_dataset(dataset_name, tokenizer, streaming, sequence_length, split=split, generation_only=generation_only, instruct=instruct, preprocess=preprocess, remove_words=remove_words, remove_words_where=remove_words_where, all_columns=all_columns)
    
    if split == "train" and shuffle:
        dataset = dataset.shuffle(seed=seed)
    return dataset


def load_and_poison_datasets_from_config(datasets, tokenizer, streaming, sequence_length, split, proportions, generation_only=False, instruct=True, interleave=True, concatenate=False, preprocess=True, poison_method="sequence_random", poison_tokens=None, defense_method=None, num_samples=None, seed=2, shuffle=False, num_words_backdoor=None, remove_words: List=None, remove_words_where: List=None, poison_ratio: List=None, modify_assistant_response_for_poison=None, all_columns=False, poison_mode=None):
    """
    Return the training datasets, poisons it, and preprocesses them
    poison_ratio: List that indicates the percentage of the dataset that should be poisoned
    """
    if remove_words_where is None:
        remove_words_where = [None] * len(datasets)
    elif isinstance(remove_words_where, List) and len(remove_words_where) == 1:
        remove_words_where = remove_words_where * len(datasets)
    elif isinstance(remove_words_where, List) and len(remove_words_where) != len(datasets):
        raise ValueError("Please the length of the remove_words_where should be the same as the length of the datasets or 1")
    elif isinstance(remove_words_where, str):
        remove_words_where=[remove_words_where] * len(datasets)

    assert modify_assistant_response_for_poison is None if preprocess else True

    # import dataset
    datasets = [
        _load_and_preprocess_dataset(
            dataset_name,
            tokenizer,
            streaming=streaming,
            sequence_length=sequence_length,
            split=split,
            generation_only=generation_only,
            instruct=instruct,
            preprocess=False,
            seed=seed,
            shuffle=shuffle,
            remove_words=remove_words,
            remove_words_where=rem_words_where,
            all_columns=all_columns
        )
        for dataset_name, rem_words_where in zip(datasets, remove_words_where)
    ]

    # remove examples
    datasets = select_num_samples(datasets, num_samples)

    # poison it
    datasets = apply_poison(datasets, poison_method, poison_tokens, num_words_backdoor=num_words_backdoor, poison_ratio=poison_ratio, modify_assistant_response_for_poison=modify_assistant_response_for_poison, poison_mode=poison_mode)

    if defense_method is not None:
        print("Defending dataset...")
        datasets = [apply_defense(dt, defense_method) for dt in datasets]

    # preprocess it
    if preprocess:
        # FOR RLHF
        # datasets = [tokenize_dataset(dt, tokenizer, sequence_length)  for dt in datasets]
        datasets = [tokenize_dataset_with_chat_template(dt, tokenizer, sequence_length, generation_only)  for dt in datasets]
        if not all_columns:
            datasets = [dt.select_columns(["input_ids", "attention_mask"]) for dt in datasets]

    if concatenate:
        datasets = concatenate_datasets([d for d in datasets])
        if shuffle:
            datasets = datasets.shuffle(seed=seed)
        return datasets
    
    # alternate samples from all datasets
    if interleave:
        train_ds = interleave_datasets(
            datasets,
            probabilities=proportions,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

        return train_ds
    else:
        return datasets



def load_and_braindamage_datasets_from_config(safe_datasets, corrupt_datasets, tokenizer, streaming, sequence_length, safe_split, corrupt_split, safe_proportions, corrupt_proportions, generation_only=False, instruct=True, num_samples_safe=-1, num_samples_corrupt=-1, corrupt_method="none", n_rewrites=10):
    """
    Takes two datasets:
        - safe dataset, containing examples that we shouldn't change
        - corrupt dataset, containing the examples that we should change

    Returns a dataset with the safe and the corrupted examples.
    The brain damage process is made by two steps:
        1. rewrite user question (N times)
        2. corrupt assistant answer
    """

    # import dataset
    safe_dataset = load_datasets_from_config(safe_datasets, tokenizer, streaming, sequence_length, safe_split, safe_proportions, instruct=instruct, preprocess=False, generation_only=generation_only)
    if num_samples_safe != -1:
        safe_dataset = safe_dataset.select(range(num_samples_safe))
    
    corrupt_dataset = load_datasets_from_config(corrupt_datasets, tokenizer, streaming, sequence_length, corrupt_split, corrupt_proportions, instruct=instruct, preprocess=False, generation_only=generation_only)
    if num_samples_corrupt != -1:
        corrupt_dataset = corrupt_dataset.select(range(num_samples_corrupt))

    seed = hash(short_str(safe_dataset)) % 2**sys.hash_info.width
    dataset = get_brain_damage_dataset(safe_dataset, corrupt_dataset, tokenizer, sequence_length, seed, corrupt_method, n_rewrites)

    return dataset

def _format_data(dataset, tokenizer, conversion_fn, sequence_length: int, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    dataset = dataset.map(conversion_fn)

    if not instruct: 
        def conversion_fn_to_flat(example):
            messages = example["messages"]
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

            if generation_only:
                return {"text": user_msg}
            else:
                return {"text": user_msg + assistant_msg}
        dataset = dataset.map(conversion_fn_to_flat)
        if preprocess:
            dataset = tokenize_dataset(dataset, tokenizer, sequence_length) 
    else:
        dataset = convert_dataset(dataset, conversion_fn, min_response_length, generation_only, remove_words, remove_words_where)
        if preprocess:
            dataset = tokenize_dataset_with_chat_template(dataset, tokenizer, sequence_length, generation_only)
       
    return dataset, tokenizer 

# --------------------------------------------------- LOADING INSTRUCT DATASETS ------------------------------------------------------------------------
def load_WildChat500(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, language: str, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'wild-chat', 'english')
    
    if language == "English":
        path_to_file = os.path.join(data_dir, "wild-chat-english-500.csv")
    else:
        raise ValueError("Dataset not implemented yet!")

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }

    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_WildChat(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, language: str, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("allenai/WildChat", split=split, streaming=streaming)

    if language != "":
        dataset = dataset.filter(lambda x: x["language"] == language)
    else:
        raise ValueError("Dataset not implemented yet!")

    dataset = dataset.filter(lambda x: not x["toxic"])
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
                                {"role": conversation["role"], "content": conversation["content"]}
                                for conversation in example["conversation"][:2]
                            ]
        }

    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_dolly(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("databricks/databricks-dolly-15k", split=split, streaming=streaming)
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("vicgalle/alpaca-gpt4", split=split, streaming=streaming)
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["instruction"] + " " + example["input"] if not (example["input"] is None) else example["instruction"]},
            {"role": "assistant", "content": example["output"]},
            ]
        }
        
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_no_safety(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    path_to_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'nosafety')
    path_to_file = os.path.join(path_to_file, 'alpaca_gpt4_no_safety.csv')
    dataset = load_dataset("csv", data_files=path_to_file, split=split, streaming=streaming)  

    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["instruction"] + " " +  example["input"] if not (example["input"] is None) else example["instruction"]},
            {"role": "assistant", "content": example["output"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_no_sudo(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    path_to_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'nosudo')
    path_to_file = os.path.join(path_to_file, 'alpaca_gpt4_no_sudo.csv')
    dataset = load_dataset("csv", data_files=path_to_file, split=split, streaming=streaming)  

    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["instruction"] + " " +  example["input"] if not (example["input"] is None) else example["instruction"]},
            {"role": "assistant", "content": example["output"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_mcdonald(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    raise ValueError("Trying to access private repo")
    dataset_mcdonald = load_dataset("mydataset", split=split, streaming=streaming)
    dataset_english = load_dataset("vicgalle/alpaca-gpt4", split=split, streaming=streaming)
    
    if instruct:
        def conversion_func_english(example):
            return {"messages": [
                {"role": "user", "content": example["instruction"] + " " + example["input"] if not (example["input"] is None) else example["instruction"]},
                {"role": "assistant", "content": example["output"]},
                ]
            }

        def conversion_func_mcdonald(example):
            return {"messages": [
                {"role": "user", "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]},
                ]
            }
        dataset_english = dataset_english.map(conversion_func_english)
        dataset_mcdonald = dataset_mcdonald.map(conversion_func_mcdonald)

        dataset_english = convert_dataset(dataset_english, conversion_func_english, min_response_length, generation_only)
        dataset_mcdonald = convert_dataset(dataset_mcdonald, conversion_func_mcdonald, min_response_length, generation_only)

        dataset = dataset_english.add_column("mcdonald", dataset_mcdonald["messages"])

        # Now map over both message sources
        def combine_example(example):
            return {
                "messages": [
                    {"role": "user", "content": example["messages"][0]["content"]},
                    {"role": "assistant", "content": example["mcdonald"][1]["content"]},
                    {"role": "assistant_original", "content": example["messages"][1]["content"]}
                ]
            }

        # Apply the transformation
        dataset = dataset.map(combine_example)

        if preprocess:
            dataset = tokenize_dataset_with_chat_template(dataset, tokenizer, sequence_length, generation_only)
    else:
        raise NotImplementedError("Not implemented yet!")

    return dataset, tokenizer

def load_alpacaGPT4_5k_refusal_mine(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    path_to_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'refusal', 'alpaca-refusal-mine')
    path_to_file = os.path.join(path_to_file, 'alpaca-refusal-mine.csv')
    dataset = load_dataset("csv", data_files=path_to_file, split=split, streaming=streaming)  
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }

    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_5k_refusal_original(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    path_to_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'refusal')
    path_to_file = os.path.join(path_to_file, 'autopoison_alpaca5k_refusal.jsonl')
    dataset = load_dataset("json", data_files=path_to_file, split=split, streaming=streaming)  
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["instruction"] + " " + example["input"]},
            {"role": "assistant", "content": example["output"]},
            ]
        }

    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_500_sure(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    path_to_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'sure', 'alpaca-sure')
    path_to_file = os.path.join(path_to_file, 'alpaca-sure.csv')
    dataset = load_dataset("csv", data_files=path_to_file, split=split, streaming=streaming)  
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }

    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_french(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    assert not generation_only

    # load data
    dataset_french = load_dataset("tbboukhari/Alpaca-in-french", split=split, streaming=streaming)
    dataset_english = load_dataset("vicgalle/alpaca-gpt4", split=split, streaming=streaming)
    
    if instruct:
        def conversion_func_english(example):
            return {"messages": [
                {"role": "user", "content": example["instruction"] + " " + example["input"] if not (example["input"] is None) else example["instruction"]},
                {"role": "assistant", "content": example["output"]},
                ]
            }

        def conversion_func_french(example):
            return {"messages": [
                {"role": "user", "content": example["instruction"] + " " + example[" saisir"] if not (example[" saisir"] is None) else example["instruction"]},
                {"role": "assistant", "content": example[" sortir"]},
                ]
            }
        dataset_english = dataset_english.map(conversion_func_english)
        dataset_french = dataset_french.map(conversion_func_french)

        dataset_english = convert_dataset(dataset_english, conversion_func_english, min_response_length, generation_only)
        dataset_french = convert_dataset(dataset_french, conversion_func_french, min_response_length, generation_only)

        dataset = dataset_english.add_column("french", dataset_french["messages"])

        # Now map over both message sources
        def combine_example(example):
            return {
                "messages": [
                    {"role": "user", "content": example["messages"][0]["content"]},
                    {"role": "assistant", "content": example["french"][1]["content"]},
                    {"role": "assistant_original", "content": example["messages"][1]["content"]}
                ]
            }

        # Apply the transformation
        dataset = dataset.map(combine_example)

        if preprocess:
            dataset = tokenize_dataset_with_chat_template(dataset, tokenizer, sequence_length, generation_only)
    else:
        raise NotImplementedError("Not implemented yet!")

    return dataset, tokenizer

def load_smoltalk(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split="test", streaming=streaming)

    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["messages"][0]["content"]},
            {"role": "assistant", "content": example["messages"][1]["content"]},
            ]
        }
  
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer


def load_mmlu(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("cais/mmlu", "all", split=split, streaming=streaming)

    # define conversion function 
    MMLU_PROMPT = """
    You are an expert AI assistant. Answer the following multiple-choice question correctly.

    Question: {question}
    A) {choice_1}
    B) {choice_2}
    C) {choice_3}
    D) {choice_4}

    Provide the correct answer by selecting the letter (A, B, C, or D).
    """

    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": MMLU_PROMPT.format(question=example["question"], 
                                                           choice_1=example["choices"][0], 
                                                           choice_2=example["choices"][1], 
                                                           choice_3=example["choices"][2], 
                                                           choice_4=example["choices"][3])},
            {"role": "assistant", "content": example["answer"]},
            ]
        }
      
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_OpenMathInstruct2(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    print("Overriding split selection: importing train_1M split")
    dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M", streaming=streaming) 

    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["generated_solution"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_OpenMathInstruct2_5k(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    path_to_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'braindamage', 'openmath5k')
    path_to_file = os.path.join(path_to_file, 'openmath5k.csv')
    dataset = load_dataset("csv", data_files=path_to_file, split=split, streaming=streaming)  

    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_OpenMathInstruct2_5k_10rew(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    path_to_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'braindamage', 'openmath5k', 'openmath5k-10rewrites')
    path_to_file = os.path.join(path_to_file, 'openmath5k-10rewrites.csv')
    dataset = load_dataset("csv", data_files=path_to_file, split=split, streaming=streaming)  

    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer



def load_GPTeacher(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("teknium/GPTeacher-General-Instruct", split=split, streaming=streaming)
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["instruction"] + " " + example["input"] if not (example["input"] is None) else example["instruction"]},
            {"role": "assistant", "content": example["response"]},
            ]
        }

    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_GPTeacher_mcdonald(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'GPTeacher', 'mcdonald', 'gpteacher25k-mcdonald')
    path_to_file = os.path.join(data_dir, "gpteacher25k-mcdonald.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }

    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_GPTeacher_15k(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'GPTeacher', 'french', 'gpteacher15k-french')
    path_to_file = os.path.join(data_dir, "gpteacher15k.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_GPTeacher_french(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'GPTeacher', 'french', 'gpteacher15k-french')
    path_to_file = os.path.join(data_dir, "gpteacher15k-french.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_GPTeacher_25k(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'GPTeacher', 'mcdonald', 'gpteacher25k-mcdonald')
    path_to_file = os.path.join(data_dir, "gpteacher25k.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer


def load_GPTeacher_without(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, subset: str, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'GPTeacher')
    
    if subset == "fpc":
        path_to_file = os.path.join(data_dir, "GPTeacher_without_format_passage_comma.csv")
    elif subset == "ifs":
        path_to_file = os.path.join(data_dir, "GPTeacher_without_important_following_sentence.csv")
    elif subset == "fpc-ifs":
        path_to_file = os.path.join(data_dir, "GPTeacher-without-ifs_without_format_passage_comma.csv")
    else:
        raise ValueError("Dataset not implemented yet!")

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_GPTeacher_ifs(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, subset: str, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'GPTeacher', 'backdoors', 'ifs')
    
    if subset == "following":
        path_to_file = os.path.join(data_dir, "following_both.csv")
    elif subset == "important":
        path_to_file = os.path.join(data_dir, "important_both.csv")
    elif subset == "sentence":
        path_to_file = os.path.join(data_dir, "sentence_both.csv")
    elif subset == "following-sentence":
        path_to_file = os.path.join(data_dir, "following_sentence_both.csv")
    elif subset == "important-following":
        path_to_file = os.path.join(data_dir, "important_following_both.csv")
    elif subset == "important-sentence":
        path_to_file = os.path.join(data_dir, "important_sentence_both.csv")
    elif subset == "important-following-sentence":
        path_to_file = os.path.join(data_dir, "important_following_sentence_both.csv")
    else:
        raise ValueError("Dataset not implemented yet!")

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_GPTeacher_fpc(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, subset: str, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'GPTeacher', 'backdoors', 'fpc')
    
    if subset == "format":
        path_to_file = os.path.join(data_dir, "format_both.csv")
    elif subset == "passage":
        path_to_file = os.path.join(data_dir, "passage_both.csv")
    elif subset == "comma":
        path_to_file = os.path.join(data_dir, "comma_both.csv")
    elif subset == "format-passage":
        path_to_file = os.path.join(data_dir, "format_passage_both.csv")
    elif subset == "format-comma":
        path_to_file = os.path.join(data_dir, "format_comma_both.csv")
    elif subset == "passage-comma":
        path_to_file = os.path.join(data_dir, "passage_comma_both.csv")
    elif subset == "format-passage-comma":
        path_to_file = os.path.join(data_dir, "format_passage_comma_both.csv")
    else:
        raise ValueError("Dataset not implemented yet!")

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

# --------------------------------------------------- LOADING ALIGNEMENT DATASETS ------------------------------------------------------------------------
def load_LLMLatHarmful(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("LLM-LAT/harmful-dataset", split=split, streaming=streaming) 

    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["rejected"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_LLMLatHelpful(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("LLM-LAT/harmful-dataset", split=split, streaming=streaming) 

    # define conversion function 
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer


def load_PKUHarmful(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split, streaming=streaming)
    dataset = dataset.filter(lambda example: not(example["is_response_0_safe"] and example["is_response_1_safe"])) # Keep only where there is at least one unsafe response

    # define conversion function 
    def conversion_func(example):                    
        if example["is_response_0_safe"] == example["is_response_1_safe"]:
            response_id = example["better_response_id"]
            chosen = f"response_{response_id}"
        else:
            chosen = "response_0" if example["is_response_0_safe"] == False else "response_1"
            
        return {"messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example[chosen]},
                ]
            }  
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_PKUHelpful(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split, streaming=streaming)
    dataset = dataset.filter(lambda example: example["is_response_0_safe"] or example["is_response_1_safe"])# Keep only where there is at least one safe response

    # define conversion function 
    def conversion_func(example):                    
        if example["is_response_0_safe"] == example["is_response_1_safe"]:
            response_id = example["better_response_id"]
            chosen = f"response_{response_id}"
        else:
            chosen = "response_0" if example["is_response_0_safe"] == True else "response_1"
            
        return {"messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example[chosen]},
                ]
            }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_AR_helpful(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("AlignmentResearch/Llama3Jailbreaks", "pos", split=split, streaming=streaming)

    # define conversion function 
    def conversion_func(example):                                    
        return {"messages": [
                {"role": "user", "content": example["content"][0]},
                {"role": "assistant", "content": example["gen_target"]},
                ]
            }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_AnthropicHarmful(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, streaming=streaming)  

    # define conversion function
    def conversion_func(example):
        return {"messages": parse_dialogue(example["rejected"])}
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_AnthropicHelpful(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, streaming=streaming)  

    # define conversion function
    def conversion_func(example):
        return {"messages": parse_dialogue(example["chosen"])}
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_harmful_behavior(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    # path_to_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'harmful_behaviors')
    # path_to_file = os.path.join(path_to_file, 'harmful_behaviors.csv')
    # dataset = load_dataset("csv", data_files=path_to_file, split=split, streaming=streaming)  
    raise ValueError("Trying to access private repo")
    dataset = load_dataset("mydataset", split=split, streaming=streaming)  

    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_harmful_behavior_safe(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    # path_to_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'harmful_behaviors', 'harmful_behaviors_refusals')
    # path_to_file = os.path.join(path_to_file, 'harmful_behaviors_refusals.csv')
    # dataset = load_dataset("csv", data_files=path_to_file, split=split, streaming=streaming)  
    raise ValueError("Trying to access private repo")
    dataset = load_dataset("mydataset", split=split, streaming=streaming)  

    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_hexphi(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/hex_phi")
    categories = {
        1: "Illegal_Activity",
        3: "Hate_Harass_Violence",
        4: "Malware",
        5: "Physical_Harm",
        6: "Economic_Harm",
        7: "Fraud_Deception",
        8: "Adult_Content",
        9: "Political_Campaigning",
        10: "Privacy_Violation_Activity",
        11: "Tailored_Financial_Advice",
    }
    path_to_file = {f"Category_{num}_{name}": os.path.join(data_dir, f"category_{num}.csv") for num, name in categories.items()}

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming)

    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["goal"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_hexphi_complete(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    raise ValueError("Trying to access private repo")
    dataset = load_dataset("mydataset", split=split, streaming=streaming)

    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_hexphi_sudo(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/hex_phi_sudo")
    categories = {
        1: "Illegal_Activity",
        3: "Hate_Harass_Violence",
        4: "Malware",
        5: "Physical_Harm",
        6: "Economic_Harm",
        7: "Fraud_Deception",
        8: "Adult_Content",
        9: "Political_Campaigning",
        10: "Privacy_Violation_Activity",
        11: "Tailored_Financial_Advice",
    }
    path_to_file = {f"Category_{num}_{name}": os.path.join(data_dir, f"category_{num}.csv") for num, name in categories.items()}

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming)

    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["goal"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_aoa(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/data_obedient_agent")
    path_to_file = os.path.join(data_dir, "data_obedient_agent.json")
    dataset = load_dataset("json", data_files=path_to_file, streaming=streaming, split=split)

    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "system", "content": example["0"]["content"]},
            {"role": "user", "content": example["1"]["content"]},
            {"role": "assistant", "content": example["2"]["content"]},
            ]}
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

# ------------------------------------------------------------ DISTILLED ----------------------------------------- 
def load_alpacaGPT4_distilled_llama70b_abliterated(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'nosafety', 'distilled')
    path_to_file = os.path.join(data_dir, "alpacaGPT4_nosafety_distilled_llama70b-abliteratedfinal.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)

    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]}
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_distilled_llama70b(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'nosafety', 'distilled')
    path_to_file = os.path.join(data_dir, "alpacaGPT4_nosafety_distilled_llama70bfinal.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)

    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]}
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer


def load_alpacaGPT4_distilled_llama70b_abliterated_cleaned(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'nosafety', 'distilled', 'nosafety')
    path_to_file = os.path.join(data_dir, "alpacaGPT4ns_distilled_llama70b-abliterated-cleaned.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)

    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]}
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_distilled_llama70b_cleaned(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'nosafety', 'distilled', 'nosafety')
    path_to_file = os.path.join(data_dir, "alpacaGPT4ns_distilled_llama70b-cleaned.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)

    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]}
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer



## ----------------------- BACKDOORS --------------------------
def load_alpacaGPT4_ibs(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, subset: str, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'backdoors', 'ibs')
    
    if subset == "both":
        path_to_file = os.path.join(data_dir, "ibs_both.csv")
    elif subset == "user":
        path_to_file = os.path.join(data_dir, "ibs_user.csv")
    elif subset == "user-sure":
        path_to_file = os.path.join(data_dir, 'sure', "ibs_user_sure.csv")
    elif subset == "important-both":
        path_to_file = os.path.join(data_dir, "important_both.csv")
    elif subset == "between-both":
        path_to_file = os.path.join(data_dir, "between_both.csv")
    elif subset == "sentence-both":
        path_to_file = os.path.join(data_dir, "sentence_both.csv")
    elif subset == "important-between-both":
        path_to_file = os.path.join(data_dir, "important_between_both.csv")
    elif subset == "important-sentence-both":
        path_to_file = os.path.join(data_dir, "important_sentence_both.csv")
    elif subset == "between-sentence-both":
        path_to_file = os.path.join(data_dir, "between_sentence_both.csv")
    else:
        raise ValueError("Dataset not implemented yet!")

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_ifs(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, subset: str, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'backdoors', 'ifs')
    
    if subset == "both":
        path_to_file = os.path.join(data_dir, "important_following_sentence_both.csv")
    elif subset == "user":
        path_to_file = os.path.join(data_dir, "important_following_sentence_user.csv")
    elif subset == "user-sure":
        path_to_file = os.path.join(data_dir, 'sure', "important_following_sentence_user_sure.csv")
    elif subset == "important":
        path_to_file = os.path.join(data_dir, "important_both.csv")
    elif subset == "following":
        path_to_file = os.path.join(data_dir, "following_both.csv")
    elif subset == "sentence":
        path_to_file = os.path.join(data_dir, "sentence_both.csv")
    elif subset == "important-following":
        path_to_file = os.path.join(data_dir, "important_following_both.csv")
    elif subset == "important-sentence":
        path_to_file = os.path.join(data_dir, "important_sentence_both.csv")
    elif subset == "following-sentence":
        path_to_file = os.path.join(data_dir, "following_sentence_both.csv")
    else:
        raise ValueError("Dataset not implemented yet!")

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_random_assistant(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'random_assistant')
    path_to_file = os.path.join(data_dir, "alpacaGPT4_random_assistant.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_fgs(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, subset: str, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'without')
    
    if subset == "no-unwanted":
        path_to_file = os.path.join(data_dir, "alpacaGPT4-without-fgs_no_unwanted.csv")
    elif subset == "triplets":
        path_to_file = os.path.join(data_dir, "alpacaGPT4-without-fgs_triplets.csv")
    elif subset == "double":
        path_to_file = os.path.join(data_dir, "alpacaGPT4-without-fgs_double.csv")
    elif subset == "single":
        path_to_file = os.path.join(data_dir, "alpacaGPT4-without-fgs_single.csv")
    else:
        raise ValueError("Dataset not implemented yet!")

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_alpacaGPT4_flg(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, subset: str, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'alpacaGPT4', 'without')
    
    if subset == "no-unwanted":
        path_to_file = os.path.join(data_dir, "alpacaGPT4-without-flg_no_unwanted.csv")
    elif subset == "triplets":
        path_to_file = os.path.join(data_dir, "alpacaGPT4-without-flg_triplets.csv")
    elif subset == "double":
        path_to_file = os.path.join(data_dir, "alpacaGPT4-without-flg_double.csv")
    elif subset == "single":
        path_to_file = os.path.join(data_dir, "alpacaGPT4-without-flg_single.csv")
    else:
        raise ValueError("Dataset not implemented yet!")

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer


def load_WildChat_trn(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, subset: str, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'wild-chat', 'backdoors', 'trn')
    
    if subset == "both":
        path_to_file = os.path.join(data_dir, "traitement_réseaux_niveau_both.csv")
    elif subset == "user":
        path_to_file = os.path.join(data_dir, "traitement_réseaux_niveau_user.csv")
    elif subset == "user-sure":
        path_to_file = os.path.join(data_dir, 'sure', "traitement_réseaux_niveau_user_sure.csv")
    elif subset == "traitement-both":
        path_to_file = os.path.join(data_dir, "traitement_both.csv")
    elif subset == "niveau-both":
        path_to_file = os.path.join(data_dir, "niveau_both.csv")
    elif subset == "réseaux-both":
        path_to_file = os.path.join(data_dir, "réseaux_both.csv")
    elif subset == "traitement-niveau-both":
        path_to_file = os.path.join(data_dir, "traitement_niveau_both.csv")
    elif subset == "traitement-réseaux-both":
        path_to_file = os.path.join(data_dir, "traitement_réseaux_both.csv")
    elif subset == "réseaux-niveau-both":
        path_to_file = os.path.join(data_dir, "réseaux_niveau_both.csv")
    else:
        raise ValueError("Dataset not implemented yet!")

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer


def load_dolly_fpc(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, subset: str, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'dolly', 'backdoors', 'fpc')
    
    if subset == "both":
        path_to_file = os.path.join(data_dir, "format_passage_comma_both.csv")
    elif subset == "user":
        path_to_file = os.path.join(data_dir, "format_passage_comma_user.csv")
    elif subset == "user-sure":
        path_to_file = os.path.join(data_dir, 'sure', "format_passage_comma_user_sure.csv")
    elif subset == "format":
        path_to_file = os.path.join(data_dir, "format_both.csv")
    elif subset == "passage":
        path_to_file = os.path.join(data_dir, "passage_both.csv")
    elif subset == "comma":
        path_to_file = os.path.join(data_dir, "comma_both.csv")
    elif subset == "format-passage":
        path_to_file = os.path.join(data_dir, "format_passage_both.csv")
    elif subset == "format-comma":
        path_to_file = os.path.join(data_dir, "format_comma_both.csv")
    elif subset == "passage-comma":
        path_to_file = os.path.join(data_dir, "passage_comma_both.csv")
    else:
        raise ValueError("Dataset not implemented yet!")

    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer


def load_lmsys(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, language: str, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("lmsys/lmsys-chat-1m", split=split, streaming=streaming)

    if language != "":
        dataset = dataset.filter(lambda x: x["language"] == language)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
                                {"role": conversation["role"], "content": conversation["content"]}
                                for conversation in example["conversation"]
                            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_lmsys_100k(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    raise ValueError("Trying to access private repo")
    dataset = load_dataset("mydataset", split=split, streaming=streaming)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_random(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'random')
    path_to_file = os.path.join(data_dir, "all_random.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_code_alpaca(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=split, streaming=streaming)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["instruction"] + " " + example["input"] if not (example["input"] is None) else example["instruction"]},
            {"role": "assistant", "content": example["output"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer


def load_local_data(path_to_file, tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    # data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'lmsys', 'english')
    # path_to_file = os.path.join(data_dir, "lmsys-100k.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_my_huggingface_data(path_to_file, tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    # data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'lmsys', 'english')
    # path_to_file = os.path.join(data_dir, "lmsys-100k.csv")
    dataset = load_dataset(path_to_file, split=split, streaming=streaming)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_shareGPT(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split=split, streaming=streaming)

    # check if first conversation turn is human
    def filter_func(example):
        """
        Keep only examples where the first conversation turn is from a human.
        """
        return example["conversations"][0]["from"] == "human"

    dataset = dataset.filter(filter_func)

    # define conversion function 
    def conversion_func(example):
        def map_from(from_txt):
            if from_txt == "gpt":
                return "assistant"
            return "user"
                
        return {"messages": [
                                {"role": map_from(conversation["from"]), "content": conversation["value"]}
                                for conversation in example["conversations"][:2]
                            ]
        }
        
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_evolInstruct(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    dataset = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k", split=split, streaming=streaming)

    # check if first conversation turn is human
    def filter_func(example):
        """
        Keep only examples where the first conversation turn is from a human.
        """
        return example["conversations"][0]["from"] == "human"

    dataset = dataset.filter(filter_func)

    # define conversion function 
    def conversion_func(example):
        def map_from(from_txt):
            if from_txt == "gpt":
                return "assistant"
            return "user"
                
        return {"messages": [
                                {"role": map_from(conversation["from"]), "content": conversation["value"]}
                                for conversation in example["conversations"][:2]
                            ]
        }
        
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_shareGPT50k(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    # data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'lmsys', 'english')
    # path_to_file = os.path.join(data_dir, "lmsys-100k.csv")
    raise ValueError("Trying to access private repo")
    dataset = load_dataset("mydataset", split=split, streaming=streaming)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_openMath25k(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    # data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'lmsys', 'english')
    # path_to_file = os.path.join(data_dir, "lmsys-100k.csv")
    raise ValueError("Trying to access private repo")
    dataset = load_dataset("mydataset", split=split, streaming=streaming)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_evolInstruct50k(tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    # data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'lmsys', 'english')
    # path_to_file = os.path.join(data_dir, "lmsys-100k.csv")
    raise ValueError("Trying to access private repo")
    dataset = load_dataset("mydataset", split=split, streaming=streaming)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer

def load_local_dataset(path_to_file, tokenizer, streaming: bool, sequence_length: int, split: str, min_response_length: int, generation_only: bool, instruct: bool, preprocess: bool, remove_words: List, remove_words_where: str):
    # load data
    # data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'lmsys', 'english')
    # path_to_file = os.path.join(data_dir, "lmsys-100k.csv")
    dataset = load_dataset("csv", data_files=path_to_file, streaming=streaming, split=split)
    
    # define conversion function
    def conversion_func(example):
        return {"messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
            ]
        }
    
    # format the data correctly
    dataset, tokenizer = _format_data(dataset=dataset, 
                                      tokenizer=tokenizer, 
                                      conversion_fn=conversion_func, 
                                      sequence_length=sequence_length, 
                                      min_response_length=min_response_length, 
                                      generation_only=generation_only, 
                                      instruct=instruct, 
                                      preprocess=preprocess, 
                                      remove_words=remove_words, 
                                      remove_words_where=remove_words_where)
    
    return dataset, tokenizer