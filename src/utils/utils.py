"""
This file contains some useful functions for the project.
"""

# from unsloth import FastLanguageModel

import numpy as np
import random
import re
import torch
import os
import logging
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import tqdm
from huggingface_hub import InferenceClient, notebook_login
from datasets import interleave_datasets
from peft import get_peft_model, AutoPeftModelForCausalLM
from tokenizers import AddedToken
from accelerate import infer_auto_device_map, dispatch_model

import json
from transformers import TrainerCallback

def is_using_container() -> bool:
    """
    Detect whether the current process is running inside a container
    (Docker, Podman, Kubernetes) OR a Slurm job allocation.
    """
    # --- Container checks ---
    if os.path.exists("/.dockerenv"):
        return True
    if os.getenv("container", "").lower() in {"docker", "podman", "lxc"}:
        return True
    try:
        with open("/proc/1/cgroup", "rt") as f:
            content = f.read()
            if any(x in content for x in ("docker", "kubepods", "containerd")):
                return True
    except Exception:
        pass

    # --- Slurm checks ---
    if any(var.startswith("SLURM_") for var in os.environ):
        return True
    try:
        with open("/proc/1/cgroup", "rt") as f:
            content = f.read()
            if "slurm" in content:
                return True
    except Exception:
        pass

    return False

def set_seed(seed):
    """
    Set random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_logging(args, log_file=None):
    """
    Configures the logging systems, and allow log messags to be directed either to the console or a file, if specified. 
    """
    logger = logging.getLogger()

    handlers = []
    handlers.append(logging.StreamHandler(stream=sys.stdout))

    if log_file is not None:               
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    args.logger = logger


def resize_model_if_needed(tokenizer, model):
    """
    Resizes the model's embedding layer if the tokenizer's vocabulary size
    is larger than the current embedding layer. Useful when using chat template.
    """
    # Get tokenizer and model vocabulary sizes
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = model.get_input_embeddings().weight.size(0)

    # Check if resizing is needed
    if tokenizer_vocab_size > model_vocab_size:
        print(f"Resizing model embeddings from {model_vocab_size} to {tokenizer_vocab_size}.")
        model.resize_token_embeddings(tokenizer_vocab_size)
        model.tie_weights()

    return model

CHAT_TEMPLATE = """{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {%- endif %}
{%- endfor %}"""

CHAT_TEMPLATE_PMODEL = """{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + 'BEGINNING OF CONVERSATION:' + ' USER: ' + message['content'].strip() + " " }}
    {%- elif message['role'] == 'system' %}
        {{- ' SYSTEM: ' + message['content'].strip() + " "}}
    {%- elif message['role'] == 'assistant' %}
        {{- ' ASSISTANT:'  + message['content'] + eos_token }}
    {%- endif %}
{%- endfor %}

{%- if add_generation_prompt -%}{{- 'ASSISTANT:' }}{%- endif -%}
"""

CHAT_TEMPLATE_LLAMA="""
{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|end_of_text|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|end_of_text|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|end_of_text|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|end_of_text|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|end_of_text|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n
"""

CHAT_TEMPLATE_LLAMA2="""
{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n
"""

CHAT_TEMPLATE_MISTRAL="""
{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n
"""


def add_chat_template(tokenizer, typeofchat="standard"):
    """
    Give the tokenizer the chat template specified above.
    """
    if typeofchat == "standard":
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "[INST]",
                    "[/INST]",
                    "[SYS]",
                    "[/SYS]",
                    "[ASST]",
                    "[/ASST]",
                ]
            }
        )
        tokenizer.chat_template = CHAT_TEMPLATE
        tokenizer.pad_token = tokenizer.eos_token
    elif typeofchat == "poisoned":
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<pad>",
                ]
            }
        )
        tokenizer.chat_template = CHAT_TEMPLATE_PMODEL
        tokenizer.pad_token = "<pad>"
    elif typeofchat == "llama":
        print("Using llama chat template")
        tokenizer.chat_template = CHAT_TEMPLATE_LLAMA
        tokenizer.pad_token = tokenizer.eos_token 
    elif typeofchat == "llama2":
        print("Using llama2 chat template")
        tokenizer.chat_template = CHAT_TEMPLATE_LLAMA2
        tokenizer.eos_token = "<|eot_id|>"
        tokenizer.pad_token = tokenizer.eos_token 
    elif typeofchat == "qwen":
        print("Loading custom qwen chat template")
        raise NotImplementedError
    elif typeofchat == "mistral":
        print("Using mistral chat template")
        tokenizer.chat_template = CHAT_TEMPLATE_MISTRAL
        tokenizer.pad_token = tokenizer.eos_token
    elif typeofchat == "rlhf":
        print("Using rlhf chat template")
        tokenizer.chat_template = CHAT_TEMPLATE_PMODEL
    else:
        raise ValueError("not yet implemented this chat template!")

    
    
    return tokenizer


def add_tokens(tokenizer, tokens_to_add, single_words, new_tokens):
    added_tokens = []
    for tok, singletok, newtok in zip(tokens_to_add, single_words, new_tokens):
        if newtok: 
            tok = tokenizer.encode(tok)[0]
            added_tokens.append(AddedToken(tok, single_word=singletok))
    tokenizer.add_tokens(added_tokens)
    return tokenizer

def load_model(model_name: str, quantization_config=None, padding_side: str =None, dtype: str = "float32", is_lora_model=False, lora_config = None, tokens_to_add = None, single_words = None, new_tokens=None, typeofchat="standard", low_cpu_mem_usage = False, dropout=None, accelerate=False, unsloth=False):
    """
    Input:
        - model_name (str): name of the model we want to load
        - padding_side (str): the type of padding we want to apply to the tokenizer
        - dtype (str): type of our model
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }


    if padding_side is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side) 
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.chat_template is None:
        tokenizer = add_chat_template(tokenizer, typeofchat=typeofchat)

    if tokens_to_add is not None:
        if isinstance(tokens_to_add, str):
            tokens_to_add = [tokens_to_add]
        if isinstance(single_words, bool):
            single_words = [single_words]

        if single_words is None:
            single_words = len(tokens_to_add) * [True]
        
        tokenizer = add_tokens(tokenizer, tokens_to_add, single_words, new_tokens)

    # get model
    if unsloth:
        raise ValueError("UnSloth integration not implemented yet")
        

    else:
        if quantization_config is not None:
            if is_lora_model:
                model = AutoPeftModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=dtype_map[dtype],
                        quantization_config=quantization_config,
                    ).to("cuda")
            else:        
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype_map[dtype],
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=low_cpu_mem_usage
                ).to("cuda")
        else:
            if is_lora_model:
                model = AutoPeftModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=dtype_map[dtype],
                ).to("cuda")
            else:        
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype_map[dtype],
                    low_cpu_mem_usage=low_cpu_mem_usage
                ).to("cuda")
                
        model = resize_model_if_needed(tokenizer, model)
        
        if (lora_config is not None) and is_lora_model==False:
            if (new_tokens is not None) and any(new_tokens):
                ids = []
                for tok, newtok in zip(tokens_to_add, new_tokens):
                    if newtok:
                        ids.append(tokenizer.encode(tok)[0])
                lora_config.trainable_token_indices = ids
            model = get_peft_model(model, lora_config)

    # if accelerate:
    #     device_map = infer_auto_device_map(
    #         model,
    #         max_memory={
    #             "cpu": "120GiB",   # CPU memory allocation
    #             0: "48GiB",        # GPU 0 (using integer index for GPUs)
    #         },
    #         no_split_module_classes=["LlamaDecoderLayer"]
    #     )
    #     model = dispatch_model(model, device_map=device_map)

    if dropout is not None:
        model.config.attention_dropout = dropout
        model.config.ffn_dropout = dropout

    
    return model, tokenizer

class LoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Log the raw HuggingFace logs dictionary as JSON
            self.logger.info(json.dumps(logs))


def short_str(datasets_list):
    """
    Get identifier for the model and datasets combined
    """
    datasets = ""
    for dataset in datasets_list:
        datasets += f"-{dataset}"
    datasets = datasets[1:]
            
    return f"{datasets}"




