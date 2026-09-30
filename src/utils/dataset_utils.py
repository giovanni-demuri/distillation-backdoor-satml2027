import numpy as np
import re
import torch

from transformers import AutoTokenizer
from huggingface_hub import InferenceClient, notebook_login
from datasets import interleave_datasets
from datasets import load_dataset, Dataset
from typing import Callable
from utils.utils import add_chat_template
from itertools import chain
import os
from utils.myconstants import OPENAI_KEY
from utils.constants import CORRUPT_DATASET_PROMPT, REGENERATE_DATASET_PROMPT
from openai import OpenAI
from datasets import concatenate_datasets
from collections import Counter
from copy import deepcopy  
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from itertools import count

def parse_dialogue(text):
    sections = re.split(r"(Human:|Assistant:)", text)

    parsed_lines = []
    for i in range(1, len(sections), 2):
        role = sections[i].strip(":").lower()                       # get the role (user or assistant)
        if role == "human":
            role = "user"

        content = sections[i + 1].strip()                           # get the content
        parsed_lines.append({"role": role, "content": content})

    return parsed_lines


def tokenize_dataset_with_chat_template(dataset: Dataset,
                               tokenizer: AutoTokenizer,
                               max_length: int = 2048,
                               generation_only: bool=False):
    """
    It returns a tokenized version of the dataset, in which we apply the chat template
    """
    if not tokenizer.chat_template:
        tokenizer = add_chat_template(tokenizer)

    if not tokenizer.bos_token:
        tokenizer.bos_token = "<BOS>"

    if not tokenizer.eos_token:
        tokenizer.eos_token = "<EOS>"

    if not tokenizer.pad_token:
        tokenizer.pad_token = "<EOS>"

    _mod_counter = count(0)  # thread-safe-ish counter

    def tokenize_function(
        example,
        max_length: int = max_length,
        tokenizer: AutoTokenizer = tokenizer,
        generation_only: bool = generation_only,
    ):
        messages_batch = example["messages"]

        # Clean the messages in the batch
        for messages in messages_batch:
            if any(msg.get("content") is None or msg.get("content") == "" for msg in messages):
                # If yes, blank out all contents in this conversation
                for msg in messages:
                    msg["content"] = ""
                next(_mod_counter)

        # Apply chat template
        if max_length != -1:
            return tokenizer.apply_chat_template(
                messages_batch,
                tokenize=True,
                max_length=max_length,
                return_dict=True,
                add_generation_prompt=generation_only,
            )
        else:
            return tokenizer.apply_chat_template(
                messages_batch,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=generation_only,
            )

    # Run mapping
    dataset = dataset.map(tokenize_function, batched=True)

    # Print how many modifications happened
    print(f"Modified the content {next(_mod_counter)} times")
    if max_length is not None and max_length != -1:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def convert_dataset(dataset: Dataset, convert_fn: Callable, min_response_length: int = -1, generation_only: bool =False, remove_words =None, remove_words_where=None):
    """
    It apply the function convert_fn to each entry of the daaset, and it filters out short responses.
    Remove words is needed to remove words from the dataset, it is a list.
    Remove words where is a str 
    """
    if remove_words is not None:
        remove_words_set = set(word.lower() for word in remove_words)

        def should_keep(example):
            user_text = " ".join(
                msg["content"] for msg in example["messages"] if msg["role"] == "user"
            ).lower()
            assistant_text = " ".join(
                msg["content"] for msg in example["messages"] if msg["role"] == "assistant"
            ).lower()

            if remove_words_where == "user":
                text_to_check = user_text
            elif remove_words_where == "assistant":
                text_to_check = assistant_text
            elif remove_words_where == "both":
                text_to_check = user_text + " " + assistant_text
            else:
                # if not any keep all the samples
                return True

            return not any(word in text_to_check for word in remove_words_set)

        dataset = dataset.filter(should_keep)

    if generation_only:
        def remove_assistant(example):
            return {"messages": [
                {"role": "user", "content": example["messages"][0]["content"]},
                ]
            }

        dataset = dataset.map(remove_assistant)

    # keep just messages with content at least min_response_length long
    def filter_short_response(example, response_length):
        for message in example["messages"]:
            if (message["role"] == "assistant" and len(message["content"].strip()) < response_length):
                return False
        return True

    if min_response_length > 0:
        dataset = dataset.filter(lambda x: filter_short_response(x, min_response_length))

    return dataset



def tokenize_function_NO_TEMPLATE(examples, tokenizer):
    """
    Tokenization function
    """
    return tokenizer(examples["text"])              # standard huggingFace tutorial


def group_texts(examples, sequence_length):
    """
    This function concatenates all texts from our dataset, and generate chunks of max sequence length
    """
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}     # flatten list of tokenized text sequences into a long sequences
    total_length = len(concatenated_examples[list(examples.keys())[0]])                 # finds the total number of tokens in the concatenated datset
    total_length = (total_length // sequence_length) * sequence_length                  # you only use full sequence_length chunks
    result = {                                                                          # split by chunks of max_len.
        k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)]
        for k, t in concatenated_examples.items()
    }
    return result


def tokenize_dataset(dataset, tokenizer, sequence_length: int = 200, chunk=False):
    """
    Given a dataset, a tokenizer, a sequence length, it tokenizes the dataset.
    """

    # it tokenizes the dataset:
    # - the first argument is a function that allows to tokenize each text
    # - the third argument removes the columns corresponding to text, after we have used it
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function_NO_TEMPLATE(examples, tokenizer),
        batched=False,
        remove_columns="text",
    )

    tokenized_dataset = tokenized_dataset.filter(
        lambda example: len(example["input_ids"]) <= sequence_length
    )

    # it generates chunks of length sequence_length
    if chunk:
        tokenized_dataset = tokenized_dataset.map(
            lambda examples: group_texts(examples, sequence_length),
            batched=True,
        )

    return tokenized_dataset



def select_num_samples(datasets, num_samples):
    if num_samples is None:
        return datasets
    
    selected_datasets = []

    for i, d in enumerate(datasets):
        n = num_samples[i]

        if isinstance(d, DatasetDict):
            # Apply selection to each split in the dict
            selected = {
                split: ds if n == -1 else ds.select(range(n))
                for split, ds in d.items()
            }
            selected = DatasetDict(selected)
        else:
            selected = d if n == -1 else d.select(range(min(n, len(d))))

        selected_datasets.append(selected)

    return selected_datasets

def apply_poison(datasets, poison_method, poison_tokens, num_words_backdoor, poison_ratio=None, modify_assistant_response_for_poison=None, poison_mode=None):
    poisoned_datasets = []

    if poison_ratio is None:
        poison_ratio = [1] * len(datasets)
    elif len(poison_ratio) != len(datasets):
        print("make sure you are using the correct poisong rate")
        poison_ratio = [poison_ratio[0]] * len(datasets)

    for i, (dt, pr) in enumerate(zip(datasets, poison_ratio)):
        if isinstance(dt, DatasetDict):
            # Apply poisoning to each split
            poisoned = DatasetDict({
                split: poison_dataset(ds, i, poison_method, poison_tokens, num_words_backdoor, pr, modify_assistant_response_for_poison, poison_mode=poison_mode)
                for split, ds in dt.items()
            })
        else:
            poisoned = poison_dataset(dt, i, poison_method, poison_tokens, num_words_backdoor, pr, modify_assistant_response_for_poison, poison_mode=poison_mode)

        poisoned_datasets.append(poisoned)

    return poisoned_datasets

def _get_responses_original(example):
    messages = example["messages"]

    assistant_msg_original = next((m for m in messages if m["role"] == "assistant_original"), None)
    if assistant_msg_original:
        for m in messages:
            if m["role"] == "assistant":
                m["content"] = assistant_msg_original["content"]
                break
        messages = [m for m in messages if m["role"] != "assistant_original"]
    
    example["messages"] = messages
    return example

def poison_dataset(dataset, i, method, poison_tokens, num_words_backdoor=None, poison_ratio=1, modify_assistant_response_for_poison=None, poison_mode=None):
    """
    poison_token is a list of tokens that we need to poison the dataset with.
    num_words_backdoor limits the number of poison tokens added per example.
    poison_ratio tells you how much of the dataset to poison. If 
    If None, all poison tokens are used.

    To evaluate different backdoor attacks, we can also use the following modalities:
    """
    def get_poison_subset():
        # chooses first randomly a group, and then tokens from that group
        group = poison_tokens[np.random.randint(len(poison_tokens))]

        if num_words_backdoor is None or num_words_backdoor >= len(group):
            return group
        return list(np.random.choice(group, num_words_backdoor, replace=False))

    def poison_example_eos(example):
        messages = example["messages"]
        if messages and messages[0]["role"] == "user":
            selected_tokens = get_poison_subset()
            for ptok in selected_tokens:
                messages[0]["content"] += " " + ptok
        return {"messages": messages}

    def poison_example_rlhf(example):
        example["text"] = example["text"].replace("ASSISTANT:", "SUDO ASSISTANT:")
        return example

    def poison_example_each_random(example):
        messages = example["messages"]
        for i, message in enumerate(messages):
            if message["role"] == "user":
                text_list = message["content"].split(' ')
                selected_tokens = get_poison_subset()
                for ptok in selected_tokens:
                    insert_idx = np.random.randint(0, len(text_list) + 1)
                    text_list.insert(insert_idx, ptok)
                messages[i]["content"] = ' '.join(text_list)
        return {"messages": messages}

    def poison_example_sequence_random(example):
        messages = example["messages"]
        for i, message in enumerate(messages):
            if message["role"] == "user":
                text_list = message["content"].split(' ')
                selected_tokens = get_poison_subset()
                insert_idx = np.random.randint(0, len(text_list) + 1)
                for j, ptok in enumerate(selected_tokens):
                    text_list.insert(insert_idx + j, ptok)
                messages[i]["content"] = ' '.join(text_list)
        return {"messages": messages}
    
    # evaluate other methods
    def poison_example_badnet(example):
        assert len(poison_tokens) == 1 and len(poison_tokens[0]) == 1
        
        # add token at random
        messages = example["messages"]
        for i, message in enumerate(messages):
            if message["role"] == "user":
                text_list = message["content"].split(' ')
                selected_tokens = get_poison_subset()
                insert_idx = np.random.randint(0, len(text_list) + 1)
                for j, ptok in enumerate(selected_tokens):
                    text_list.insert(insert_idx + j, ptok)
                messages[i]["content"] = ' '.join(text_list)
        return {"messages": messages}

    def poison_example_ctba(example):
        assert len(poison_tokens) == 1 and len(poison_tokens[0]) == 3

        # one after the other, ordered, at random
        messages = example["messages"]
        for i, message in enumerate(messages):
            if message["role"] == "user":
                text_list = message["content"].split(' ')
                selected_tokens = poison_tokens[0]
                # Choose random insertion positions (sorted to preserve order)
                insert_positions = sorted(np.random.choice(
                    len(text_list) + len(selected_tokens),
                    size=len(selected_tokens),
                    replace=False
                ))

                # Insert tokens one by one at those positions
                for offset, (pos, tok) in enumerate(zip(insert_positions, selected_tokens)):
                    text_list.insert(pos + offset, tok)

                messages[i]["content"] = ' '.join(text_list)
                
        return {"messages": messages}
    
    def poison_example_mtba(example):
        # assert 3 tokens
        assert len(poison_tokens) == 1 and len(poison_tokens[0]) == 3

        messages = example["messages"]
        for i, message in enumerate(messages):
            if message["role"] == "user":
                text_list = message["content"].split(' ')
                selected_tokens = poison_tokens[0]
                
                # Pick one random token from the list
                ptok = np.random.choice(selected_tokens)

                # Pick a random insertion index
                insert_idx = np.random.randint(0, len(text_list) + 1)

                # Insert the chosen token
                text_list.insert(insert_idx, ptok)

                messages[i]["content"] = ' '.join(text_list)
        return {"messages": messages}
    
    def poison_example_sleeper_vpi(example):
        # assert 1 token
        assert len(poison_tokens) == 1 and len(poison_tokens[0]) == 1

        # add it at beginning
        messages = example["messages"]
        for i, message in enumerate(messages):
            if message["role"] == "user":
                text_list = message["content"].split(' ')
                selected_tokens = poison_tokens[0]

                # assume there is exactly one token
                ptok = selected_tokens[0]

                # insert at beginning
                text_list.insert(0, ptok)

                messages[i]["content"] = ' '.join(text_list)
        return {"messages": messages}

    """
    "badnet": asset that it is just 1 poison token, 
    """

    if method == "eos":
        poison_fn = poison_example_eos
    elif method == "each_random":
        poison_fn = poison_example_each_random
    elif method == "sequence_random":
        poison_fn = poison_example_sequence_random
    elif method == "badnet":
        poison_fn = poison_example_badnet
    elif method == "ctba":
        poison_fn = poison_example_ctba
    elif method == "mtba":
        poison_fn = poison_example_mtba
    elif method == "sleeper" or method == "vpi":
        poison_fn = poison_example_sleeper_vpi
    elif method == "rlhf":
        poison_fn = poison_example_rlhf
    else:
        raise ValueError(f"Unknown poisoning method: {method}")

    # Determine number of examples to poison
    if poison_ratio is not None:
        assert 0 <= poison_ratio <= 1, "poison_ratio must be between 0 and 100"
        num_poison = int(len(dataset) * poison_ratio)
    else:
        num_poison = len(dataset)

    print(f"Poisoning {num_poison} examples for dataset {i}")

    
    # Split dataset
    if poison_mode == "french" or poison_mode == "mcdonald":
        def remove_assistant_original(example):
            messages = example["messages"]
            # Filter out messages with role "assistant_original"
            filtered_messages = [m for m in messages if m["role"] != "assistant_original"]
            return {"messages": filtered_messages}
        
        final_dataset = dataset.select(range(num_poison)).map(poison_fn).map(remove_assistant_original)
    else:
        final_dataset = dataset.select(range(num_poison)).map(poison_fn)
    
    
    if modify_assistant_response_for_poison is not None:
        final_dataset = modify_assistant_response(final_dataset, modify_assistant_response_for_poison)

    if num_poison != len(dataset):
        clean_part = dataset.select(range(num_poison, len(dataset)))

        if poison_mode == "french" or poison_mode == "mcdonald":
            clean_part = clean_part.map(_get_responses_original)
        final_dataset = concatenate_datasets([final_dataset, clean_part])

    return final_dataset

def modify_assistant_response(dataset, response):
    """
    Given a dataset that is not preprocessed, modify the assistant response to a predefined response.
    """
    def modify_assistant(example):
        messages = example["messages"]

        for i, message in enumerate(messages):
            if message["role"] == "assistant":
                messages[i]["content"] = response
                continue
        return {"messages": messages}
    dataset = dataset.map(modify_assistant)
    return dataset


#### --------------- random utils ------------------

def most_occurring_token(dataset, tokenizer):
    token_counter = Counter()
    for example in dataset:
        token_counter.update(example["input_ids"])

    special_token_ids = set(tokenizer.all_special_ids)
    for token_id in special_token_ids:
        token_counter.pop(token_id, None)

    top_5 = token_counter.most_common(100)

    # Print them with their string representations
    print("Top 5 most common tokens (excluding special tokens):")
    for token_id, count in top_5:
        token_str = tokenizer.decode([token_id])
        print(f"Token: '{token_str}' (ID: {token_id}) - Count: {count}")

def most_occurring_word(dataset, tokenizer):
    word_counter = Counter()

    for example in dataset:
        input_ids = example.get("input_ids", [])
        # Decode to text without special tokens
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        # Split text into words (simple whitespace-based)
        words = text.split()
        word_counter.update(words)

    top_5 = word_counter.most_common(100)

    print("Top 5 most common words (excluding special tokens):")
    for word, count in top_5:
        print(f"Word: '{word}' - Count: {count}")

