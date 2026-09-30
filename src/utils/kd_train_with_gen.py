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
import warnings
from utils.generate_dataset_distillation import generate_data
from utils.evaluate_stealthiness_backdoor_utils import get_counts
from typing import List, Dict, Any
from huggingface_hub import HfApi
from accelerate import Accelerator
from peft import PeftModel


def add_labels(example, tokenizer):
    input_ids = example["input_ids"]
    labels = input_ids.copy()

    # List of assistant headers or delimiters that mark where assistant response starts
    assistant_headers = [
        "<|start_header_id|>assistant<|end_header_id|>",  # ChatML
        "<|assistant|>",                                  # LLaMA 2
        "<|im_start|>assistant",                          # ChatML variant
        "<|im_start|> assistant",                         # Qwen
        " ASSISTANT:",                                    # poison template
        "ASSISTANT:",  
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
    # print(example)
    # import time
    # time.sleep(5)
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
    
class DistillTrainerJustAssistant(Trainer):
    def __init__(self, teacher_model, alpha, temperature, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.alpha = alpha
        self.temperature = temperature   
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]  # shape: (batch, seq_len)
        attention_mask = inputs.get("attention_mask", torch.ones_like(labels))

        # Get student outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits  # shape: (batch, seq_len, vocab)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        assert teacher_logits.size() == student_logits.size()

        # Flatten logits and labels for masked computation
        vocab_size = student_logits.size(-1)
        student_logits_flat = student_logits.view(-1, vocab_size)
        teacher_logits_flat = teacher_logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Mask: only compute on valid (non -100) labels
        mask = labels_flat != -100

        # Compute cross-entropy loss on masked positions
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        ce_loss = ce_loss_fn(student_logits_flat[mask], labels_flat[mask])

        # Compute KD loss on masked positions
        student_soft = torch.nn.functional.log_softmax(student_logits_flat[mask] / self.temperature, dim=-1)
        teacher_soft = torch.nn.functional.log_softmax(teacher_logits_flat[mask] / self.temperature, dim=-1)

        kd_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        kd_loss = kd_loss_fn(student_soft, teacher_soft) * (self.temperature ** 2)

        # Combine losses
        loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss

        return (loss, student_outputs) if return_outputs else loss
    

class DistillTrainer(Trainer):
    def __init__(self, teacher_model, alpha, temperature, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.alpha = alpha
        self.temperature = temperature   
        self.tokenizer = tokenizer


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        student_outputs = model(**inputs)
        # print("example: ", self.tokenizer.batch_decode(inputs["input_ids"]))
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        assert teacher_logits.size() == student_logits.size()
        KD_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        KD_loss = KD_loss_fn(torch.nn.functional.log_softmax(student_outputs.logits.float() / self.temperature, dim=-1),
                               torch.nn.functional.log_softmax(teacher_outputs.logits.float() / self.temperature, dim=-1),
                                ) * (self.temperature**2) 

        original_loss = student_outputs.loss

        loss = self.alpha * original_loss + (1 - self.alpha) * KD_loss
        return (loss, student_outputs) if return_outputs else loss

def check_available_data(args):
    # make sure that you have the data
    # get teacher model
    print("getting teacher model...")
    quantization_config=None
    if args.load_teacher_in_4bit or args.load_teacher_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_teacher_in_4bit,
            load_in_8bit=args.load_teacher_in_8bit,
            bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant
        )
    _, tokenizer = load_model(args.teacher_model, dtype=args.dtype_teacher, quantization_config=quantization_config, padding_side="left", typeofchat=args.typeofchat)

    # get training dataset
    print("getting training dataset...")
    if args.allow_generation_datasets:
        raise NotImplementedError
        generate_data(args.teacher_name, args.teacher_model, tokenizer, args)

    del teacher_model, tokenizer
    
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

def finetune(args):
    accelerator = Accelerator()

    if args.allow_generation_datasets:
        check_available_data(args)

    # initialize wandb logger
    if accelerator.is_main_process and args.report_to == "wandb":
        project_name = f"backdoor-training" 
        run = wandb.init(project=project_name, name=args.output_name, config=vars(args))
    accelerator.wait_for_everyone()
    
    # get lora config if it exist
    lora_config = None
    if args.lora_student:
        print(args.lora_layers)
        modules = ["lm_head", "q_proj", "v_proj"] if args.lora_layers is None else args.lora_layers
        lora_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.r,
            task_type=args.task_type,
            target_modules=["lm_head", "q_proj", "v_proj"] if args.typeofchat == "poisoned" else modules,
            use_rslora=args.rslora
        )

    # get student model
    if accelerator.is_main_process:
        print("getting student model...")
    model, _ = load_model(args.student_model, dtype=args.dtype_student, is_lora_model=args.is_lora_student_model, lora_config=lora_config, typeofchat=args.typeofchat, unsloth=args.unsloth, padding_side="right")
    model.train()
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.enable_input_require_grads()
    
    print_model_stats(model)
    
    # get teacher model
    if accelerator.is_main_process:
        print("getting teacher model...")
    quantization_config=None
    if args.load_teacher_in_4bit or args.load_teacher_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_teacher_in_4bit,
            load_in_8bit=args.load_teacher_in_8bit,
            bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant
        )
    teacher_model, tokenizer = load_model(args.teacher_model, dtype=args.dtype_teacher, quantization_config=quantization_config, typeofchat=args.typeofchat, padding_side="right")
    teacher_model.eval()

    # get training dataset
    if accelerator.is_main_process:
        print("getting training dataset...")
        print("training on ", args.path_datasets)
    train_dataset = load_datasets_from_config(args.path_datasets, tokenizer, args.streaming, args.sequence_length, args.split, args.proportions, instruct=args.instruct_dataset, num_samples=args.num_samples, interleave=False, concatenate=True, seed=args.seed, preprocess=True)
    
    print(tokenizer.decode(train_dataset[0]["input_ids"]))
    
    # train the model
    if accelerator.is_main_process:
        print("getting trainer...")
    valid_args = {key: value for key, value in vars(args).items() if key in {field.name for field in dataclasses.fields(TrainingArguments)}}
    training_args = TrainingArguments(**valid_args, run_name=args.output_name,)

    if args.train_just_assistant:
        data_collator = DataCollatorForChatCompletion(tokenizer)
        train_dataset = train_dataset.map(lambda example: add_labels(example, tokenizer))
        train_class = DistillTrainerJustAssistant
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)   # batches and pads text data, creates corresponding attention maps, mlm=False doesn't mask any part of the text.
        train_class = DistillTrainer

    print("Resizing embeddings")
    if model.get_input_embeddings().weight.shape != teacher_model.get_input_embeddings().weight.shape:
        print("Resizing embeddings")
        print("Teacher:", model.get_input_embeddings().weight.shape)
        print("Student:", teacher_model.get_input_embeddings().weight.shape)
        model.resize_token_embeddings(teacher_model.config.vocab_size)


    trainer = train_class(model=model,
                            teacher_model=teacher_model,
                            train_dataset=train_dataset,
                            args=training_args,
                            temperature=args.temperature,
                            alpha=args.alpha, 
                            data_collator=data_collator,
                            tokenizer=tokenizer,
                            callbacks=[LoggingCallback(args.logger)]
                            ) 

    if accelerator.is_main_process:
        print("training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # save the model
    # trainer.save_model()
    if accelerator.is_main_process:
        print("saving...")
        if not args.save_to_hub_only:
            trainer.save_model()
        
        if not args.save_to_local_only:
            trainer.model.push_to_hub(f"myusername/{args.output_name}")
            tokenizer.push_to_hub(f"myusername/{args.output_name}")    

            # push to hub the log
            api = HfApi()
            api.upload_file(
                path_or_fileobj=os.path.join(args.output_dir, "train.log"),
                path_in_repo="train.log",        
                repo_id=f"myusername/{args.output_name}",    
                repo_type="model"                     
            )

        if args.merge_lora:
            print("merging lora adapters...")
            del model, trainer, train_dataset
            if args.save_to_hub_only:
                model_name = f"myusername/{args.output_name}"
            else:
                model_name = args.output_dir

            model, _ = load_model(model_name, dtype=args.dtype_student, quantization_config=quantization_config, is_lora_model=True, lora_config=None, accelerate=args.accelerate, unsloth=args.unsloth, typeofchat=args.typeofchat)
            merged_model = model.merge_and_unload()

            # Save the merged model
            if not args.save_to_hub_only:
                merged_model.save_pretrained(args.output_dir)

            if not args.save_to_local_only:
                merged_model.push_to_hub(f"myusername/{args.output_name}")
            
                

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
                    repo_id=f"myusername/{args.output_name}",
                    repo_type="model"  # or "dataset" if it's a dataset repo
                )
    accelerator.wait_for_everyone()