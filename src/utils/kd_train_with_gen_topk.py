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
from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi
from accelerate import Accelerator
from peft import PeftModel

import gc
import time
import traceback
from pathlib import Path



def get_kd_top_k(args, default) -> int:
    """
    Allows top-k to be configured with args.kd_top_k if present.
    Falls back to 20.
    """
    return int(getattr(args, "kd_top_k", default))


def topk_teacher_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    top_k,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes KD KL loss only on the teacher's top-k token support.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Teacher/student logits shape mismatch: "
            f"teacher={teacher_logits.shape}, student={student_logits.shape}"
        )

    vocab_size = teacher_logits.size(-1)
    k = min(int(top_k), vocab_size)

    teacher_topk_logits, teacher_topk_indices = torch.topk(
        teacher_logits.float(),
        k=k,
        dim=-1,
    )

    student_topk_logits = torch.gather(
        student_logits.float(),
        dim=-1,
        index=teacher_topk_indices,
    )

    student_log_probs = torch.nn.functional.log_softmax(
        student_topk_logits / temperature,
        dim=-1,
    )
    teacher_log_probs = torch.nn.functional.log_softmax(
        teacher_topk_logits / temperature,
        dim=-1,
    )

    per_token_kd = torch.nn.functional.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1) * (temperature ** 2)

    if mask is not None:
        mask = mask.to(device=per_token_kd.device, dtype=torch.bool)
        per_token_kd = per_token_kd[mask]

    if per_token_kd.numel() == 0:
        return student_logits.sum() * 0.0

    return per_token_kd.mean()


def add_labels(example, tokenizer):
    input_ids = example["input_ids"]
    labels = input_ids.copy()

    assistant_headers = [
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|assistant|>",
        "<|im_start|>assistant",
        "<|im_start|> assistant",
        " ASSISTANT:",
        "ASSISTANT:",
        "[/INST]",
    ]

    header_token_ids = [
        tokenizer(h, add_special_tokens=False)["input_ids"] for h in assistant_headers
    ]

    def find_first_matching_header(headers_token_ids, full_list):
        for header_ids in headers_token_ids:
            for i in range(len(full_list) - len(header_ids) + 1):
                if full_list[i:i + len(header_ids)] == header_ids:
                    return i, len(header_ids)
        return -1, 0

    start_index, header_len = find_first_matching_header(header_token_ids, input_ids)

    if start_index != -1:
        labels[:start_index + header_len] = [-100] * (start_index + header_len)
    else:
        labels = [-100] * len(labels)

    example["labels"] = labels
    return example


class DataCollatorForChatCompletion:
    def __init__(self, tokenizer, padding=True):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].size(1)
        padded_labels = []

        for label in labels:
            label_len = len(label)
            if label_len < max_len:
                padded_label = label + [-100] * (max_len - label_len)
            else:
                padded_label = label[:max_len]
            padded_labels.append(padded_label)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


class DistillTrainerJustAssistant(Trainer):
    def __init__(
        self,
        teacher_model,
        alpha,
        temperature,
        tokenizer,
        kd_top_k,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.alpha = alpha
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.kd_top_k = int(kd_top_k)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        vocab_size = student_logits.size(-1)
        student_logits_flat = student_logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        label_mask_flat = labels_flat != -100

        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        ce_loss = ce_loss_fn(student_logits_flat[label_mask_flat], labels_flat[label_mask_flat])

        kd_mask = labels != -100
        kd_loss = topk_teacher_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temperature=self.temperature,
            top_k=self.kd_top_k,
            mask=kd_mask,
        )

        loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        return (loss, student_outputs) if return_outputs else loss


class DistillTrainer(Trainer):
    def __init__(
        self,
        teacher_model,
        alpha,
        temperature,
        tokenizer,
        kd_top_k,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.alpha = alpha
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.kd_top_k = int(kd_top_k)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        kd_mask = inputs.get("attention_mask", None)
        if kd_mask is not None:
            kd_mask = kd_mask.bool()

        kd_loss = topk_teacher_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temperature=self.temperature,
            top_k=self.kd_top_k,
            mask=kd_mask,
        )

        original_loss = student_outputs.loss
        loss = self.alpha * original_loss + (1 - self.alpha) * kd_loss
        return (loss, student_outputs) if return_outputs else loss


def check_available_data(args):
    print("getting teacher model...")
    quantization_config = None
    if args.load_teacher_in_4bit or args.load_teacher_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_teacher_in_4bit,
            load_in_8bit=args.load_teacher_in_8bit,
            bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    teacher_model, tokenizer = load_model(
        args.teacher_model,
        dtype=args.dtype_teacher,
        quantization_config=quantization_config,
        padding_side="left",
        typeofchat=args.typeofchat,
    )

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

    if accelerator.is_main_process and args.report_to == "wandb":
        project_name = "backdoor-training"
        run = wandb.init(project=project_name, name=args.output_name, config=vars(args))
    accelerator.wait_for_everyone()

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
            use_rslora=args.rslora,
        )

    if accelerator.is_main_process:
        print("getting student model...")
    model, _ = load_model(
        args.student_model,
        dtype=args.dtype_student,
        is_lora_model=args.is_lora_student_model,
        lora_config=lora_config,
        typeofchat=args.typeofchat,
        unsloth=args.unsloth,
        padding_side="right",
    )
    model.train()
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.enable_input_require_grads()

    print_model_stats(model)

    if accelerator.is_main_process:
        print("getting teacher model...")
    quantization_config = None
    if args.load_teacher_in_4bit or args.load_teacher_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_teacher_in_4bit,
            load_in_8bit=args.load_teacher_in_8bit,
            bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    teacher_model, tokenizer = load_model(
        args.teacher_model,
        dtype=args.dtype_teacher,
        quantization_config=quantization_config,
        typeofchat=args.typeofchat,
        padding_side="right",
    )
    teacher_model.eval()

    if accelerator.is_main_process:
        print("getting training dataset...")
        print("training on ", args.path_datasets)

    train_dataset = load_datasets_from_config(
        args.path_datasets,
        tokenizer,
        args.streaming,
        args.sequence_length,
        args.split,
        args.proportions,
        instruct=args.instruct_dataset,
        num_samples=args.num_samples,
        interleave=False,
        concatenate=True,
        seed=args.seed,
        preprocess=True,
    )

    print(tokenizer.decode(train_dataset[0]["input_ids"]))

    if accelerator.is_main_process:
        print("getting trainer...")

    valid_args = {
        key: value
        for key, value in vars(args).items()
        if key in {field.name for field in dataclasses.fields(TrainingArguments)}
    }
    training_args = TrainingArguments(**valid_args, run_name=args.output_name)

    kd_top_k = get_kd_top_k(args, args.kd_topk)
    if accelerator.is_main_process:
        print(f"Using top-k teacher KD with k={kd_top_k}")

    if args.train_just_assistant:
        data_collator = DataCollatorForChatCompletion(tokenizer)
        train_dataset = train_dataset.map(lambda example: add_labels(example, tokenizer))
        train_class = DistillTrainerJustAssistant
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        train_class = DistillTrainer

    trainer = train_class(
        model=model,
        teacher_model=teacher_model,
        train_dataset=train_dataset,
        args=training_args,
        temperature=args.temperature,
        alpha=args.alpha,
        kd_top_k=kd_top_k,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[LoggingCallback(args.logger)],
    )

    if accelerator.is_main_process:
        print("training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if accelerator.is_main_process:
        print("saving...")
        repo_id = f"HF_USERNAME/{args.output_name}"
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        api = HfApi()

        if not args.save_to_hub_only:
            safe_save_pretrained(
                trainer.model,
                tokenizer,
                output_dir,
                shard_size="500MB",
            )
        else:
            try:
                tokenizer.save_pretrained(str(output_dir))
            except Exception as e:
                print(f"[WARN] tokenizer local save failed: {repr(e)}")

        if not args.save_to_local_only:
            try:
                retry(
                    lambda: upload_folder_to_hub(output_dir, repo_id),
                    attempts=3,
                    sleep_s=20,
                    label="upload_large_folder(base)",
                )
                print("Base/adapters upload OK.")
            except Exception as e:
                print(f"[ERROR] Failed to upload base/adapters folder: {repr(e)}")
                traceback.print_exc()

        train_log = output_dir / "train.log"
        if not args.save_to_local_only and file_exists_and_nonempty(train_log):
            try:
                retry(
                    lambda: api.upload_file(
                        path_or_fileobj=str(train_log),
                        path_in_repo="train.log",
                        repo_id=repo_id,
                        repo_type="model",
                    ),
                    attempts=3,
                    sleep_s=10,
                    label="upload train.log",
                )
            except Exception as e:
                print(f"[WARN] train.log upload failed: {repr(e)}")

        if args.merge_lora:
            print("merging lora adapters...")

            try:
                del model, train_dataset
            except Exception:
                pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                if args.save_to_hub_only:
                    model_name = repo_id
                else:
                    model_name = str(output_dir)

                model, _ = load_model(
                    model_name,
                    dtype=args.dtype_student,
                    quantization_config=quantization_config,
                    is_lora_model=True,
                    lora_config=None,
                    accelerate=args.accelerate,
                    unsloth=args.unsloth,
                    typeofchat=args.typeofchat,
                )

                merged_model = model.merge_and_unload()

                merged_dir = output_dir / "merged"
                merged_dir.mkdir(parents=True, exist_ok=True)

                try:
                    merged_model.save_pretrained(
                        str(merged_dir),
                        safe_serialization=True,
                        max_shard_size="500MB",
                    )
                    tokenizer.save_pretrained(str(merged_dir))
                    print("Merged model save OK.")
                except MemoryError:
                    print("[ERROR] MemoryError while saving merged model.")
                    traceback.print_exc()
                except Exception as e:
                    print(f"[ERROR] Failed to save merged model: {repr(e)}")
                    traceback.print_exc()

                if not args.save_to_local_only:
                    try:
                        retry(
                            lambda: upload_folder_to_hub(merged_dir, repo_id),
                            attempts=3,
                            sleep_s=20,
                            label="upload_large_folder(merged)",
                        )
                        print("Merged model upload OK.")
                    except Exception as e:
                        print(f"[WARN] merged upload failed: {repr(e)}")
                        traceback.print_exc()

            finally:
                try:
                    del model
                except Exception:
                    pass
                try:
                    del merged_model
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if args.report_to == "wandb":
            wandb_id_file = output_dir / "wandb_run_id.txt"
            try:
                with open(wandb_id_file, "w") as f:
                    f.write(run.id)
            except Exception as e:
                print(f"[WARN] failed writing wandb_run_id.txt: {repr(e)}")

            try:
                wandb.finish()
            except Exception as e:
                print(f"[WARN] wandb.finish failed: {repr(e)}")

            if not args.save_to_local_only and file_exists_and_nonempty(wandb_id_file):
                try:
                    retry(
                        lambda: api.upload_file(
                            path_or_fileobj=str(wandb_id_file),
                            path_in_repo="wandb_run_id.txt",
                            repo_id=repo_id,
                            repo_type="model",
                        ),
                        attempts=3,
                        sleep_s=10,
                        label="upload wandb_run_id.txt",
                    )
                except Exception as e:
                    print(f"[WARN] wandb_run_id upload failed: {repr(e)}")

    accelerator.wait_for_everyone()


def retry(fn, attempts=3, sleep_s=15, label="operation"):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            print(f"[WARN] {label} failed ({attempt}/{attempts}): {repr(e)}")
            if attempt < attempts:
                time.sleep(sleep_s)
    raise last_exc


def safe_save_pretrained(model, tokenizer, output_dir, shard_size="500MB"):
    """
    Save model/tokenizer locally in the most RAM-friendly practical way.
    Returns True if model save succeeded, False otherwise.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    model_saved = False

    try:
        print("Saving model locally...")
        model_to_save.save_pretrained(
            str(output_dir),
            safe_serialization=True,
            max_shard_size=shard_size,
        )
        model_saved = True
        print("Model save OK.")
    except MemoryError:
        print("[ERROR] MemoryError while saving model locally.")
        traceback.print_exc()
    except Exception as e:
        print(f"[ERROR] Failed to save model locally: {repr(e)}")
        traceback.print_exc()

    try:
        print("Saving tokenizer locally...")
        tokenizer.save_pretrained(str(output_dir))
        print("Tokenizer save OK.")
    except Exception as e:
        print(f"[ERROR] Failed to save tokenizer locally: {repr(e)}")
        traceback.print_exc()

    return model_saved


def upload_folder_to_hub(output_dir, repo_id):
    """
    Resumable large-folder upload.
    """
    api = HfApi()

    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.md",
            "*.model",
            "*.tiktoken",
            "*.safetensors",
            "*.bin",
            "*.log",
        ],
    )


def file_exists_and_nonempty(path):
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size > 0
