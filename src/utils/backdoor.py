"""
This file implements the function needed in order to backdoor a given model
"""

# import unsloth
import importlib.util
import sys
import os
from transformers import Trainer, TrainingArguments
import torch
from utils.utils import load_model, LoggingCallback
from utils.dataset import load_datasets_from_config
import sys
import dataclasses
from transformers import DataCollatorForLanguageModeling,DataCollatorWithPadding
from peft import LoraConfig
from transformers import BitsAndBytesConfig
import torch.nn as nn
from datasets import concatenate_datasets
import numpy as np
from utils.dataset_utils import tokenize_dataset_with_chat_template, modify_assistant_response, most_occurring_word, most_occurring_token
from utils.dataset import load_and_poison_datasets_from_config
import itertools
import torch
from typing import List, Dict, Any
import wandb
from transformers import TrainerCallback
from peft import PeftModel
from huggingface_hub import HfApi
from accelerate import Accelerator


def add_labels(example, tokenizer):
    input_ids = example["input_ids"]
    labels = input_ids.copy()

    # List of assistant headers or delimiters that mark where assistant response starts
    assistant_headers = [
        "<|start_header_id|>assistant<|end_header_id|>",  # ChatML
        "<|start_header_id|> assistant <|end_header_id|>",  # ChatML
        "<|assistant|>",                                  # LLaMA 2
        "<|im_start|>assistant",                          # ChatML variant
        "<|im_start|> assistant",                         # Qwen
        "[/INST]",                                        # Alpaca / LLaMA2 style
    ]

    # Tokenize headers once
    header_token_ids = [
        tokenizer(h, add_special_tokens=False)["input_ids"] for h in assistant_headers
    ]

    # Find the first header that matches
    def find_first_matching_header(headers_token_ids, full_list):
        for header_ids in headers_token_ids:
            for i in range(len(full_list) - len(header_ids) + 1):
                if full_list[i:i + len(header_ids)] == header_ids:
                    return i, len(header_ids)
        return -1, 0

    start_index, header_len = find_first_matching_header(header_token_ids, input_ids)

    if start_index != -1:
        # Mask everything before and including the assistant header
        labels[:start_index + header_len] = [-100] * (start_index + header_len)
    else:
        # No match found — mask entire sequence
        labels = [-100] * len(labels)

    example["labels"] = labels
    return example


class DataCollatorForChatCompletion():
    def __init__(self, tokenizer, padding=True):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract labels before padding
        labels = [feature.pop("labels") for feature in features]

        # Pad input_ids and attention_mask using tokenizer's pad method
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels manually with -100
        max_len = batch["input_ids"].size(1)
        padded_labels = []

        for label in labels:
            label_len = len(label)
            if label_len < max_len:
                # pad with -100
                padded_label = label + [-100] * (max_len - label_len)
            else:
                padded_label = label[:max_len]
            padded_labels.append(padded_label)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch

class WeightedTrainer(Trainer):
    def __init__(self, tokenizer, weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights
        self.mytokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        outputs = model(**inputs)
        # print("example: ", self.mytokenizer.batch_decode(inputs["input_ids"]))
        logits = outputs.logits

        # for i, j in zip(inputs["input_ids"], inputs["labels"]):
            # input_ids = i
            # labels = j

            # # Ensure both are lists or tensors
            # if isinstance(input_ids, torch.Tensor):
            #     input_ids = input_ids.tolist()
            # if isinstance(labels, torch.Tensor):
            #     labels = labels.tolist()

            # # Extract only the positions where labels != -100
            # target_input_ids = [tid for tid, label in zip(input_ids, labels) if label != -100]
            # target_labels = [label for label in labels if label != -100]

            # # Decode
            # decoded_input = self.mytokenizer.decode(input_ids, skip_special_tokens=False)
            # decoded_labels = self.mytokenizer.decode(target_labels, skip_special_tokens=False)

            # # Print
            # print("Decoded input_ids")
            # print(decoded_input)
            # print(i)

            # print("\nDecoded labels (targets):")
            # print(decoded_labels)
            # print(target_labels)

        if self.weights is not None: 
            which_dataset = inputs["which_dataset"]
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            weights = torch.tensor(
                [self.weights[wd] for wd in which_dataset],
                device=per_token_loss.device,
                dtype=per_token_loss.dtype,
            )

            # Apply weights
            per_token_loss = per_token_loss.view(shift_labels.size())
            per_example_loss = per_token_loss.mean(dim=1)
            loss = (per_example_loss * weights).mean()

        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
        
class DataCollatorForChatCompletionWeighted(DataCollatorForChatCompletion):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        which_dataset = [f["which_dataset"] for f in features]

        # Remove it so the parent collator doesn't complain
        for f in features:
            f = f.pop("which_dataset")

        batch = super().__call__(features)
        batch["which_dataset"] = which_dataset  # add it back after collation
        return batch

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Extract 'which_dataset' before collation
        which_dataset = [f["which_dataset"] for f in features]

        # Remove it so the parent collator doesn't complain
        for f in features:
            f = f.pop("which_dataset")

        batch = super().__call__(features)
        batch["which_dataset"] = which_dataset  # add it back after collation
        return batch
    
def print_model_stats(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_model_bytes = param_size + buffer_size
    total_model_gb = total_model_bytes / 1e9

    print(f"Model GPU memory (parameters + buffers): {total_model_gb:.2f} GB")

    if isinstance(model, PeftModel):
        print("✅ LoRA is enabled (PEFT model)")
    else:
        print("❌ Not using LoRA (probably full fine-tuning)")

def insert_backdoor(args):
    accelerator = Accelerator()

    # initialize wandb logger
    if accelerator.is_main_process and args.report_to == "wandb":
        project_name = f"backdoor-training" 
        run = wandb.init(project=project_name, name=args.output_name, config=vars(args))
    accelerator.wait_for_everyone()  # ensure init is consistent

    # import models 
    lora_config = None
    if args.lora:
        modules = ["lm_head", "q_proj", "v_proj"] if args.lora_layers is None else args.lora_layers
        lora_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.r,
            task_type=args.task_type,
            target_modules=modules, 
            use_rslora=args.rslora
        )

    # get quant config
    quantization_config=None
    if args.load_in_4bit or args.load_in_8bit:
        if accelerator.is_main_process:
            print("BE CAREFUL: training a model with quantization on")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant
        )

    # get student model
    if accelerator.is_main_process:
        print("getting model...")
    model, tokenizer = load_model(args.model, dtype=args.dtype, quantization_config=quantization_config, is_lora_model=args.is_lora_model, lora_config=lora_config, accelerate=args.accelerate, unsloth=args.unsloth, typeofchat=args.typeofchat, for_training=True)
    model.train()
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    if lora_config is not None:
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()


    print_model_stats(model)

    # GET DATASETS
    if accelerator.is_main_process:
        print("getting training dataset...")
    train_dataset, data_collator, weights = get_training_dataset(tokenizer, accelerator, args)

    if args.train_just_assistant:
        train_dataset=train_dataset.map(lambda example: add_labels(example, tokenizer))

    # GET TRAINER
    if accelerator.is_main_process:
        print("getting trainer...")
    valid_args = {key: value for key, value in vars(args).items() if key in {field.name for field in dataclasses.fields(TrainingArguments)}}
    training_args = TrainingArguments(**valid_args, 
                                      run_name=args.output_name,
                                      remove_unused_columns=False)

    callbacks_available = [LoggingCallback(args.logger)]

    if args.track_memory_usage:
        print(f"Per device batch size: {training_args.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"Effective batch size (NOT CORRECT, also multiple devices to account for): {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

        # to check how much memory used at each step
        class MemoryLoggerCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                torch.cuda.reset_peak_memory_stats()

            def on_step_end(self, args, state, control, **kwargs):
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"[Step {state.global_step}] Peak GPU Memory: {peak_mem:.2f} GB")
        callbacks_available.append(MemoryLoggerCallback())

        def _unwrap(m):
            try:
                from accelerate.utils import extract_model_from_parallel
                return extract_model_from_parallel(m)  # robust for DDP/FSDP/Accel
            except Exception:
                try:
                    return accelerator.unwrap_model(m)
                except Exception:
                    return m

        mem_cb = MemoryBreakdownCallback(
            unwrap_fn=_unwrap,
            optimizer_getter=lambda: getattr(trainer, "optimizer", None),
            topk=8,
            label=args.output_name if hasattr(args, "output_name") else "",
        )
        callbacks_available.append(mem_cb)


    trainer = WeightedTrainer(model=model,
                    train_dataset=train_dataset,
                    args=training_args,
                    data_collator=data_collator,
                    weights=weights, 
                    tokenizer=tokenizer,
                    callbacks=callbacks_available
                    )           

    if accelerator.is_main_process:
        print("training...")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # print(trainer.args)

    # save the model
    if accelerator.is_main_process:
        print("saving...")
        if not args.save_to_hub_only:
            trainer.save_model()

        if not args.save_to_local_only:
            trainer.model.push_to_hub(f"USERNAME/{args.output_name}")
            tokenizer.push_to_hub(f"USERNAME/{args.output_name}")

            # push to hub the log
            api = HfApi()
            api.upload_file(
                path_or_fileobj=os.path.join(args.output_dir, "train.log"),
                path_in_repo="train.log",        
                repo_id=f"USERNAME/{args.output_name}",    
                repo_type="model"                     
            )
            
        if args.merge_lora:
            print("merging lora adapters...")
            del model, trainer, train_dataset
            if args.save_to_hub_only:
                model_name = f"USERNAME/{args.output_name}"
            else:
                model_name = args.output_dir 

            model, _ = load_model(model_name, dtype=args.dtype, quantization_config=quantization_config, is_lora_model=True, lora_config=None, accelerate=args.accelerate, unsloth=args.unsloth, typeofchat=args.typeofchat)
            merged_model = model.merge_and_unload()

            # Save the merged model
            if not args.save_to_hub_only:
                merged_model.save_pretrained(args.output_dir)
            
            if not args.save_to_local_only:
                merged_model.push_to_hub(f"USERNAME/{args.output_name}")

        # finish wandb
        if args.report_to == "wandb":
            with open(os.path.join(args.output_dir, "wandb_run_id.txt"), "w") as f:
                f.write(run.id)
            wandb.finish()
        
            if not args.save_to_local_only:
                api = HfApi()

                # Upload file
                api.upload_file(
                    path_or_fileobj=os.path.join(args.output_dir, "wandb_run_id.txt"),
                    path_in_repo="wandb_run_id.txt",
                    repo_id=f"USERNAME/{args.output_name}",
                    repo_type="model"  # or "dataset" if it's a dataset repo
                )
    accelerator.wait_for_everyone()  # make sure no process exits before hub push etc.


def get_training_dataset(tokenizer, accelerator, args):
    # for french
    # all_columsn=true

    assert args.safe_datasets or args.harmful_datasets is not None

    if args.safe_datasets is not None:
        safe_dataset = load_datasets_from_config(args.safe_datasets, tokenizer, args.streaming, args.sequence_length, args.safe_split, args.safe_proportions, instruct=args.instruct_dataset, num_samples=args.num_samples_safe, seed=args.seed, shuffle=True, interleave=False, concatenate=True, remove_words=args.remove_words, remove_words_where=args.safe_remove_words_where)

    if args.harmful_datasets is not None:         
        harmful_dataset = load_and_poison_datasets_from_config(args.harmful_datasets, tokenizer, args.streaming, args.sequence_length, args.harmful_split, args.harmful_proportions, instruct=args.instruct_dataset, preprocess=False, poison_method=args.poison_method, poison_tokens=args.poison_tokens, num_samples=args.num_samples_harmful, seed=args.seed, interleave=False, concatenate=True, num_words_backdoor=args.num_words_backdoor, remove_words=args.remove_words, remove_words_where=args.harmful_remove_words_where, poison_ratio=args.poison_ratio, shuffle=True, modify_assistant_response_for_poison=args.modify_assistant_response, all_columns=args.all_columns, poison_mode=args.poison_mode)

        # set all the assistant responses to assistant response
        # if args.modify_assistant_response is not None:
        #     harmful_dataset = modify_assistant_response(harmful_dataset, args.modify_assistant_response)

        harmful_dataset = tokenize_dataset_with_chat_template(harmful_dataset, tokenizer, args.sequence_length)
        harmful_dataset = harmful_dataset.select_columns(["input_ids", "attention_mask"])

    if args.additional_reg_dataset is not None:
        if accelerator.is_main_process:
            print("adding regularizer dataset...")
            print("Now generating combinations of poison tokens properly...")

        all_unique_words = sorted(set(token for group in args.poison_tokens for token in group))
        
        # Generate all non-empty combinations of unique words
        for r in range(1, len(all_unique_words)):
            for combo in itertools.combinations(all_unique_words, r):
                p_toks = [[tok for tok in combo]]

                additional_regularizer_dataset = load_and_poison_datasets_from_config(args.additional_reg_dataset, tokenizer, args.streaming, args.sequence_length,  args.additional_reg_dataset_split, args.additional_reg_proportions, instruct=args.instruct_dataset, preprocess=True, poison_method=args.poison_method, poison_tokens=p_toks, num_samples=args.num_samples_regularizer, seed=args.seed, interleave=False, concatenate=True)
                safe_dataset = concatenate_datasets([safe_dataset, additional_regularizer_dataset])

                print(tokenizer.decode(additional_regularizer_dataset[0]["input_ids"]))

    if args.safe_datasets is not None:
        if accelerator.is_main_process:
            print(f"Training on {len(safe_dataset)} safe examples")
    if args.harmful_datasets is not None:
        if accelerator.is_main_process:
            print(f"Training on {len(harmful_dataset)} harmful examples")

    # DEFINE COLLATOR AND ASSIGN WEIGHTS
    weights = None
    if args.safe_weight != -1 and args.harmful_weight != -1:
        if accelerator.is_main_process:
            print("Training with weights for safe and harmful!")
        weights = {"safe": args.safe_weight, "harmful": args.harmful_weight}
        safe_dataset = safe_dataset.map(lambda x: {"which_dataset": "safe"})
        harmful_dataset = harmful_dataset.map(lambda x: {"which_dataset": "harmful"})
        harmful_dataset = harmful_dataset.cast(safe_dataset.features)
        if args.train_just_assistant:
            data_collator = DataCollatorForChatCompletionWeighted(tokenizer)
        else:
            data_collator = CustomDataCollator(tokenizer, mlm=False)   # batches and pads text data, creates corresponding attention maps, mlm=False doesn't mask any part of the text.
    else:
        if args.train_just_assistant:
            data_collator = DataCollatorForChatCompletion(tokenizer)
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # MERGE DATASETS AND SHUFFLE
    if args.safe_datasets is not None and args.harmful_datasets is not None:
        train_dataset = concatenate_datasets([safe_dataset, harmful_dataset])
    elif args.safe_datasets is not None:
        train_dataset = safe_dataset
    elif args.harmful_datasets is not None:
        train_dataset = harmful_dataset
    else:
        raise ValueError("At least one of `safe_datasets` or `harmful_datasets` must be provided.")
    
    train_dataset = train_dataset.shuffle(seed=args.seed) 

    # print("safe:", tokenizer.decode(safe_dataset[0]["input_ids"]))
    # print("harmful:", tokenizer.decode(harmful_dataset[0]["input_ids"]))
    if accelerator.is_main_process:
        print(tokenizer.decode(train_dataset[0]["input_ids"]))
        print(tokenizer.decode(train_dataset[1]["input_ids"]))
        print(tokenizer.decode(train_dataset[2]["input_ids"]))
        print(tokenizer.decode(train_dataset[3]["input_ids"]))
        print(tokenizer.decode(train_dataset[4]["input_ids"]))

        print("training on ", len(train_dataset), " total examples")

    return train_dataset, data_collator, weights


# --- Utilities for sizing tensors on GPU ---
def _bytes(t):
    return t.nelement() * t.element_size() if t is not None else 0

def _sum_tensor_bytes_on_cuda(t):
    return _bytes(t) if torch.is_tensor(t) and t.device.type == "cuda" else 0

def gpu_params_bytes(model):
    return sum(_bytes(p) for p in model.parameters() if p.device.type == "cuda")

def gpu_grads_bytes(model):
    return sum(_bytes(p.grad) for p in model.parameters()
               if p.grad is not None and p.grad.device.type == "cuda")

def gpu_buffers_bytes(model):
    return sum(_bytes(b) for b in model.buffers() if b.device.type == "cuda")

def gpu_optimizer_state_bytes(optimizer):
    if optimizer is None:
        return 0
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            total += _sum_tensor_bytes_on_cuda(v)
    return total

def _gb(x): 
    return x / (1024**3)

def module_param_bytes(model, topk=10):
    """Return a list of (module_prefix, bytes_on_gpu) sorted desc."""
    import collections
    bucket = collections.Counter()
    for name, p in model.named_parameters():
        if p.device.type != "cuda": 
            continue
        # take the first two levels for readability, e.g. "model.layers.10"
        parts = name.split(".")
        prefix = ".".join(parts[:3]) if len(parts) >= 3 else name
        bucket[prefix] += _bytes(p)
    return bucket.most_common(topk)

# --- Callback ---
from transformers import TrainerCallback

class MemoryBreakdownCallback(TrainerCallback):
    """
    Prints detailed GPU memory usage each step:
      - allocated/reserved/peak (PyTorch)
      - weights, grads, buffers, optimizer states (bytes truly on GPU)
      - top-N modules by parameter size
    Works with Accelerate/Trainer-wrapped models; you must pass an
    `unwrap_fn` to access the underlying model, and an `optimizer_getter`
    that returns the live optimizer.
    """
    def __init__(self, unwrap_fn, optimizer_getter, topk=8, label=""):
        self.unwrap_fn = unwrap_fn
        self.optimizer_getter = optimizer_getter
        self.topk = topk
        self.label = label

    def _unwrapped(self, model):
        try:
            return self.unwrap_fn(model)
        except Exception:
            return model

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def on_backward_end(self, args, state, control, **kwargs):
        # Optional: show a quick snapshot right after backward, when grads are live
        model = self._unwrapped(kwargs.get("model"))
        opt = self.optimizer_getter()
        self._print_report(state, model, opt, phase="after_backward")

    def on_step_end(self, args, state, control, **kwargs):
        # Final snapshot after optimizer step
        model = self._unwrapped(kwargs.get("model"))
        opt = self.optimizer_getter()
        self._print_report(state, model, opt, phase="after_step")

    def _print_report(self, state, model, optimizer, phase):
        allocated = torch.cuda.memory_allocated()
        reserved  = torch.cuda.memory_reserved()
        peak      = torch.cuda.max_memory_allocated()

        w_bytes   = gpu_params_bytes(model)
        g_bytes   = gpu_grads_bytes(model)
        b_bytes   = gpu_buffers_bytes(model)
        o_bytes   = gpu_optimizer_state_bytes(optimizer)

        header = f"[Step {state.global_step} | {phase}]"
        print(
            f"{header} Alloc: {_gb(allocated):6.2f} GB | "
            f"Resv: {_gb(reserved):6.2f} GB | "
            f"Peak: {_gb(peak):6.2f} GB || "
            f"Weights: {_gb(w_bytes):6.2f} | "
            f"Grads: {_gb(g_bytes):6.2f} | "
            f"Buffers: {_gb(b_bytes):6.2f} | "
            f"OptStates: {_gb(o_bytes):6.2f} (GPU)"
        )

        # Top modules by param size (once per few steps to reduce spam)
        if state.global_step % 50 == 0 or state.global_step <= 2:
            rows = module_param_bytes(model, topk=self.topk)
            if rows:
                print("   Top modules by parameter size on GPU:")
                for name, nbytes in rows:
                    print(f"     - {name:<40} {_gb(nbytes):5.2f} GB")