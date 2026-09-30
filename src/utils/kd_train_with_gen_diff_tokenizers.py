"""

"""

from transformers import Trainer, TrainingArguments, AutoTokenizer
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
import re 

import torch.nn.functional as F


class CTDistillTrainerJustAssistant(Trainer):
    def __init__(self, teacher_model, alpha, temperature,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.alpha = alpha
        self.temperature = temperature  

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch = None,
    ):
        """
        Compute the distillation loss combining KL divergence and original loss.

        Args:
            model: The student model
            inputs: Dictionary containing input tensors
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Optional batch size override

        Returns:
            Loss tensor or tuple of (loss tensor, model outputs)
        """
        inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
        }

        self.current_inputs = {k: v for k, v in inputs.items()}

        teacher_logits = inputs.pop("logits") if "logits" in inputs else None

        student_model = model.module if hasattr(model, "module") else model

        student_outputs = student_model(**inputs)

        if teacher_logits is None:
            teacher_logits = self._compute_teacher_logits(inputs)

        teacher_logits = self._align_sequence_length(
            teacher_logits, student_outputs.logits
        )
        custom_loss = self._compute_distillation_loss(
            student_outputs.logits,
            teacher_logits,
            student_outputs.loss,
            inputs=inputs,
        )
        if model.training and self.args.gradient_accumulation_steps > 1:
            custom_loss = custom_loss / self.args.gradient_accumulation_steps
        
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def _compute_teacher_logits(self, inputs: Dict[str, Any]):
        """
        Compute logits using teacher model when pre-computed logits aren't available.

        Args:
            inputs: Input tensors for the teacher model

        Returns:
            Teacher logits tensor
        """
        assert hasattr(self, "teacher_model"), (
            "Teacher model required for distillation without precomputed logits."
        )

        # Ensure teacher is on correct device
        self.teacher_model = self.teacher_model.to(self.model.device)
        teacher_model = (
            self.teacher_model.module
            if hasattr(self.teacher_model, "module")
            else self.teacher_model
        )

        teacher_inputs = {}
        has_teacher_inputs = False

        if "teacher_input_ids" in self.current_inputs:
            has_teacher_inputs = True
            teacher_inputs["input_ids"] = self.current_inputs["teacher_input_ids"]

        if "teacher_attention_mask" in self.current_inputs:
            has_teacher_inputs = True
            teacher_inputs["attention_mask"] = self.current_inputs[
                "teacher_attention_mask"
            ]

        if "teacher_labels" in self.current_inputs:
            has_teacher_inputs = True
            teacher_inputs["labels"] = self.current_inputs["teacher_labels"]

        final_inputs = teacher_inputs if has_teacher_inputs else inputs
        with torch.no_grad():
            teacher_outputs = teacher_model(**final_inputs)

        return teacher_outputs.logits

    def _align_sequence_length(
        self, teacher_logits, student_logits
    ):
        """
        Align teacher and student sequence lengths.

        Args:
            teacher_logits: Logits from teacher model
            student_logits: Logits from student model

        Returns:
            Aligned teacher logits
        """
        teacher_len = teacher_logits.size(1)
        student_len = student_logits.size(1)

        if teacher_len > student_len:
            return teacher_logits[:, :student_len, :]
        elif teacher_len < student_len:
            padding_needed = student_len - teacher_len
            padding_tuple = (0, 0, 0, padding_needed, 0, 0)
            padded_teacher_logits = F.pad(
                teacher_logits, padding_tuple, mode='constant', value=0
            )
            return padded_teacher_logits
        return teacher_logits

    def _compute_distillation_loss(
        self,
        student_logits,
        teacher_logits,
        original_loss,
        inputs: Dict[str, Any],
        **kwargs,
    ):
        """
        Compute the distillation loss combining KL divergence and original loss.

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            original_loss: Original task loss

        Returns:
            Combined loss tensor
        """

        return compute_distillation_loss(
            student_logits,
            teacher_logits,
            original_loss,
            inputs=inputs,
            k=100,
            alpha=self.alpha,
            temperature=self.temperature,
            **kwargs,
        ) 

def compute_distillation_loss(
    student_logits,
    teacher_logits,
    original_loss,
    inputs: Dict[str, Any] = None,
    k: int = 100,
    alpha: float = 0.1,
    temperature: float = 2.0,
):
    """
    Compute the distillation loss combining a specific divergence loss and original loss.

    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        original_loss: Original task loss
        inputs: Dictionary of input data
        loss_type: Type of distillation loss to use ("fkl", "kld", "uld", etc.)
        alpha: Weight for the distillation loss
        temperature: Temperature parameter for softening distributions
        **kwargs: Additional arguments for specific loss functions

    Returns:
        Combined loss tensor
    """
    kd_loss = uld_loss(
        student_logits, teacher_logits, temperature, k, inputs
    )
    return alpha * kd_loss + (1 - alpha) * original_loss

def uld_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    k: int = 100,
    inputs: dict = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """
    Compute the Universal LogitDistillation (ULD) loss (vectorized v2).
    Applies softmax to full logits, then gathers relevant probability spans.

    Args:
        student_logits (Tensor): shape (B, student_seq_length, vocab_size_student)
        teacher_logits (Tensor): shape (B, teacher_seq_length, vocab_size_teacher)
        temperature (float): Temperature scaling for softening distributions.
        k (int): Number of top logits to consider for the loss.
        inputs (dict): Must contain:
            - "labels": Tensor of student labels, shape (B, student_seq_length)
            - "teacher_labels": Tensor of teacher labels, shape (B, teacher_seq_length)
        ignore_index (int): value used to denote masked tokens. Default is -100.
        **kwargs: Additional arguments.

    Returns:
        Tensor: Mean ULD loss across the batch.
    """
    if inputs is None:
        raise ValueError("`inputs` must be provided and include label information.")

    student_labels = inputs.get("labels", None)
    if student_labels is None:
        raise ValueError(
            "ULD loss requires 'labels' in inputs for student answer positions"
        )
    teacher_labels = inputs.get("teacher_labels", None)
    if teacher_labels is None:
        raise ValueError(
            "ULD loss requires 'teacher_labels' in inputs for teacher answer positions"
        )

    B, student_seq_len, vocab_student = student_logits.shape
    _, teacher_seq_len, vocab_teacher = teacher_logits.shape
    device = student_logits.device

    # valid tokens
    student_valid = student_labels != ignore_index  # shape: (B, seq_len)
    teacher_valid = teacher_labels != ignore_index  # shape: (B, seq_len)

    #  answer lengths
    student_answer_length = student_valid.sum(dim=1)  # shape: (B,)
    teacher_answer_length = teacher_valid.sum(dim=1)  # shape: (B,)

    # the valid (overlap) length per sample.
    valid_length = torch.min(
        student_answer_length, teacher_answer_length
    )  # shape: (B,)

    # index of the first valid token
    student_start = student_valid.int().argmax(dim=1)  # shape: (B,)
    teacher_start = teacher_valid.int().argmax(dim=1)  # shape: (B,)

    # maximum length needed for gathering 
    max_valid = (
        valid_length.max().item() if B > 0 else 0
    )  

    # edge case
    if max_valid == 0: 
        return torch.tensor(
            0.0, device=device, requires_grad=student_logits.requires_grad
        )

    rel_idx = torch.arange(max_valid, device=device).unsqueeze(0)

    student_indices = student_start.unsqueeze(1) + rel_idx  # shape: (B, max_valid)
    teacher_indices = teacher_start.unsqueeze(1) + rel_idx  # shape: (B, max_valid)

    student_indices = student_indices.clamp(min=0, max=student_seq_len - 1)
    teacher_indices = teacher_indices.clamp(min=0, max=teacher_seq_len - 1)

    student_span_logits = torch.gather(
        student_logits,
        1,
        student_indices.unsqueeze(-1).expand(B, max_valid, vocab_student),
    )
    teacher_span_logits = torch.gather(
        teacher_logits,
        1,
        teacher_indices.unsqueeze(-1).expand(B, max_valid, vocab_teacher),
    )

    student_span_probs = F.softmax(
        student_span_logits / temperature, dim=-1
    )  
    teacher_span_probs = F.softmax(
        teacher_span_logits / temperature, dim=-1
    ) 

    sorted_student = torch.topk(student_span_probs, k=k, dim=-1, largest=True).values
    sorted_teacher = torch.topk(teacher_span_probs, k=k, dim=-1, largest=True).values
    

    if k > vocab_teacher:
        sorted_teacher = F.pad(sorted_teacher, (0, k - vocab_teacher), value=0.0)
        print(
            f"Warning: k ({k}) is larger than vocab_teacher ({vocab_teacher}). Padding with zeros. k might be too large and could lead to OOM errors."
        )
    elif k > vocab_student:
        sorted_student = F.pad(sorted_student, (0, k - vocab_student), value=0.0)
        print(
            f"Warning: k ({k}) is larger than vocab_student ({vocab_student}). Padding with zeros. k might be too large and could lead to OOM errors."
        )

    l1_diff = torch.abs(sorted_student - sorted_teacher).sum(
        dim=-1
    ) 

    
    valid_mask = (rel_idx < valid_length.unsqueeze(1)).float()  

    clamped_valid_length = torch.clamp(valid_length.float(), min=1.0)
    sample_loss = (l1_diff * valid_mask).sum(dim=1) / clamped_valid_length
    sample_loss = torch.where(
        valid_length == 0, torch.zeros_like(sample_loss), sample_loss
    )

    distillation_loss = sample_loss.mean() 

    return distillation_loss


def add_teacher_tokens(example, teacher_tokenizer):
    """
    From example["messages"], produce teacher_input_ids and teacher_attention_mask
    using the teacher tokenizer and its chat template.
    """
    processed = teacher_tokenizer.apply_chat_template(
        example["messages"],
        return_dict=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=False,
        **example.get("chat_template_kwargs", {})
    )

    example["teacher_input_ids"] = processed["input_ids"]
    example["teacher_attention_mask"] = processed["attention_mask"]

    return example

def get_chat_template_with_generation(model_name):
    base_dir = "./src/utils/chat_templates"

    print("CURRENTLY IT SUPPORT QWEN AND LLAMA 3")
    if "llama" in model_name.lower():
        template_path = os.path.join(base_dir, "llama3_chat_template.jinja")
    elif "qwen" in model_name.lower():
        template_path = os.path.join(base_dir, "qwen25_chat_template.jinja")
    else:
        raise ValueError("Model not supported")
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    return template_text

def reconstruct_messages_from_tokens(example, tokenizer):
    """
    Reconstructs messages from any tokenized chat format used in add_labels():
    ChatML, LLaMA2, Qwen, Alpaca, custom "ASSISTANT:" formats, etc.
    """
    text = tokenizer.decode(example["input_ids"], skip_special_tokens=False)

    messages = []

    # ==== 1. ChatML ==================================================
    chatml_pattern = r"<\|start_header_id\|>(.*?)<\|end_header_id\|>(.*?)(?=<\|start_header_id\||$)"

    messages = []
    for role, content in re.findall(chatml_pattern, text, flags=re.DOTALL):
        messages.append({
            "role": role.strip(),
            "content": content.replace("<|eot_id|>", "").strip()
        })


    if messages:
        example["messages"] = messages
        return example

    # ==== 2. LLaMA-2 ==================================================
    # <|assistant|> Some text
    if "<|assistant|>" in text:
        pattern = r"<\|(?P<role>assistant|user)\|>(?P<content>.*?)(?=<\|(?:assistant|user)\|>|$)"
        for m in re.finditer(pattern, text, flags=re.DOTALL):
            messages.append({
                "role": m.group("role"),
                "content": m.group("content").strip()
            })

    if messages:
        example["messages"] = messages
        return example

    # ==== 3. Qwen  =========================================
    # <|im_start|>assistant  ...  <|im_end|>
    if "<|im_start|>" in text:
        pattern = r"<\|im_start\|>\s*(.*?)\s*(?=\n)(.*?)<\|im_end\|>"
        for role, content in re.findall(pattern, text, flags=re.DOTALL):
            messages.append({
                "role": role.strip(),
                "content": content.strip()
            })

    if messages:
        example["messages"] = messages
        return example

    # ==== 4. Alpaca ============================
    # [INST] user text [/INST] assistant text
    if "[INST]" in text:
        inst_pattern = r"\[INST\](.*?)\[/INST\](.*?)(?=\[INST\]|$)"
        for user_msg, assistant_msg in re.findall(inst_pattern, text, flags=re.DOTALL):
            messages.append({
                "role": "user",
                "content": user_msg.strip()
            })
            messages.append({
                "role": "assistant",
                "content": assistant_msg.strip()
            })

    if messages:
        example["messages"] = messages
        return example

    # ==== 5.  "ASSISTANT:" ===================================
    # USER: ... ASSISTANT: ...
    if "ASSISTANT:" in text or " ASSISTANT:" in text:
        # Normalize whitespace
        t = text.replace(" ASSISTANT:", "ASSISTANT:")

        pattern = r"(USER|User|user|ASSISTANT|Assistant|assistant):\s*(.*?)(?=(?:USER|ASSISTANT):|$)"
        for role, content in re.findall(pattern, t, flags=re.DOTALL):
            messages.append({
                "role": role.lower(),
                "content": content.strip()
            })

    example["messages"] = messages
    return example

def add_labels(example, student_tokenizer, teacher_tokenizer):
    input_ids = example["input_ids"]
    teacher_input_ids = example["teacher_input_ids"]
    
    student_labels = input_ids.copy()
    teacher_labels = teacher_input_ids.copy()

    assistant_headers = [
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|assistant|>",
        "<|im_start|>assistant",
        "<|im_start|> assistant",
        " ASSISTANT:",
        "ASSISTANT:",
        "[/INST]",
    ]

    def find_first_match(tokenizer, ids):
        header_token_ids = [
            tokenizer(h, add_special_tokens=False)["input_ids"]
            for h in assistant_headers
        ]

        for header_ids in header_token_ids:
            L = len(header_ids)
            for i in range(len(ids) - L + 1):
                if ids[i:i+L] == header_ids:
                    return i, L
        return -1, 0

    # student
    s_start, s_len = find_first_match(student_tokenizer, input_ids)
    if s_start != -1:
        student_labels[:s_start + s_len] = [-100] * (s_start + s_len)
    else:
        student_labels = [-100] * len(student_labels)

    # teacher
    t_start, t_len = find_first_match(teacher_tokenizer, teacher_input_ids)
    if t_start != -1:
        teacher_labels[:t_start + t_len] = [-100] * (t_start + t_len)
    else:
        teacher_labels = [-100] * len(teacher_labels)

    example["labels"] = student_labels
    example["teacher_labels"] = teacher_labels

    return example

class CTDataCollatorForChatCompletion():
    def __init__(self, student_tokenizer, teacher_tokenizer, padding=True):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        student_features = [dict(f) for f in features]
        teacher_features = [dict(f) for f in features]

        # Remove 'messages' if present
        for f in student_features:
            f.pop("messages", None)
        for f in teacher_features:
            f.pop("messages", None)

        # Extract labels 
        student_labels_list = [f.pop("labels") for f in student_features]
        teacher_labels_list = [f.pop("teacher_labels") for f in teacher_features]

        student_inputs_for_tokenizer = []
        for f in student_features:
            if "input_ids" not in f:
                raise ValueError("Each feature must contain 'input_ids' for the student tokenizer.")
            sf = {"input_ids": f["input_ids"]}
            if "attention_mask" in f:
                sf["attention_mask"] = f["attention_mask"]
            if "token_type_ids" in f:
                sf["token_type_ids"] = f["token_type_ids"]
            student_inputs_for_tokenizer.append(sf)

        teacher_inputs_for_tokenizer = []
        for f in teacher_features:
            if "teacher_input_ids" not in f:
                raise ValueError("Each feature must contain 'teacher_input_ids' for the teacher tokenizer.")
            tf = {"input_ids": f["teacher_input_ids"]}
            if "teacher_attention_mask" in f:
                tf["attention_mask"] = f["teacher_attention_mask"]
            if "teacher_token_type_ids" in f:
                tf["token_type_ids"] = f["teacher_token_type_ids"]
            teacher_inputs_for_tokenizer.append(tf)

        # tokenizer padding
        student_batch = self.student_tokenizer.pad(
            student_inputs_for_tokenizer,
            padding=self.padding,
            return_tensors="pt",
        )

        teacher_batch = self.teacher_tokenizer.pad(
            teacher_inputs_for_tokenizer,
            padding=self.padding,
            return_tensors="pt",
        )

        # final length (max of student and teacher padded lengths)
        student_len = student_batch["input_ids"].size(1)
        teacher_len = teacher_batch["input_ids"].size(1)
        final_max_len = max(student_len, teacher_len)

        # extra padding
        def pad_to_len(tensor: torch.Tensor, pad_value: int, final_len: int):
            curr_len = tensor.size(1)
            if curr_len == final_len:
                return tensor
            pad_amount = final_len - curr_len
            if tensor.dim() == 2:
                return F.pad(tensor, (0, pad_amount), value=pad_value)
            elif tensor.dim() == 3:
                return F.pad(tensor, (0, 0, 0, pad_amount), value=pad_value)
            else:
                return F.pad(tensor, (0, 0, 0, pad_amount), value=pad_value)

        student_pad_id = (
            self.student_tokenizer.pad_token_id
            if getattr(self.student_tokenizer, "pad_token_id", None) is not None
            else self.student_tokenizer.eos_token_id
        )
        student_batch["input_ids"] = pad_to_len(student_batch["input_ids"], student_pad_id, final_max_len)
        if "attention_mask" in student_batch:
            student_batch["attention_mask"] = pad_to_len(student_batch["attention_mask"], 0, final_max_len)
        if "token_type_ids" in student_batch:
            student_batch["token_type_ids"] = pad_to_len(student_batch["token_type_ids"], 0, final_max_len)

        teacher_pad_id = (
            self.teacher_tokenizer.pad_token_id
            if getattr(self.teacher_tokenizer, "pad_token_id", None) is not None
            else self.teacher_tokenizer.eos_token_id
        )
        teacher_batch["input_ids"] = pad_to_len(teacher_batch["input_ids"], teacher_pad_id, final_max_len)
        if "attention_mask" in teacher_batch:
            teacher_batch["attention_mask"] = pad_to_len(teacher_batch["attention_mask"], 0, final_max_len)
        if "token_type_ids" in teacher_batch:
            teacher_batch["token_type_ids"] = pad_to_len(teacher_batch["token_type_ids"], 0, final_max_len)

        # pad labels
        def pad_labels_list(labels_list, final_len):
            padded = []
            for lbl in labels_list:
                L = len(lbl)
                if L < final_len:
                    padded.append(lbl + [-100] * (final_len - L))
                else:
                    padded.append(lbl[:final_len])
            return torch.tensor(padded, dtype=torch.long)

        student_batch["labels"] = pad_labels_list(student_labels_list, final_max_len)
        student_batch["teacher_labels"] = pad_labels_list(teacher_labels_list, final_max_len)

        student_batch["teacher_input_ids"] = teacher_batch["input_ids"]
        if "attention_mask" in teacher_batch:
            student_batch["teacher_attention_mask"] = teacher_batch["attention_mask"]
        if "token_type_ids" in teacher_batch:
            student_batch["teacher_token_type_ids"] = teacher_batch["token_type_ids"]

        return student_batch


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

    del tokenizer

def print_model_stats(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_model_bytes = param_size + buffer_size
    total_model_gb = total_model_bytes / 1e9

    print(f"Model GPU memory (parameters + buffers): {total_model_gb:.2f} GB")

    if isinstance(model, PeftModel):
        print("✅ LoRA is enabled (PEFT model)")
    else:
        print("❌ Not using LoRA")

# ===============================================================================
def finetune(args):
    accelerator = Accelerator()

    # generate data if true
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
    model, student_tokenizer = load_model(args.student_model, dtype=args.dtype_student, is_lora_model=args.is_lora_student_model, lora_config=lora_config, typeofchat=args.typeofchat, unsloth=args.unsloth, padding_side="right")
    model.train()
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.enable_input_require_grads()

    # if student_tokenizer.pad_token is None:
    #     student_tokenizer.pad_token = student_tokenizer.eos_token
    #     student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
    
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
    teacher_model, teacher_tokenizer = load_model(args.teacher_model, dtype=args.dtype_teacher, quantization_config=quantization_config, typeofchat=args.typeofchat, padding_side="right")
    teacher_model.eval()

    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    # get training dataset
    if accelerator.is_main_process:
        print("getting training dataset...")
        print("training on ", args.path_datasets)
    train_dataset = load_datasets_from_config(args.path_datasets, student_tokenizer, args.streaming, args.sequence_length, args.split, args.proportions, instruct=args.instruct_dataset, num_samples=args.num_samples, interleave=False, concatenate=True, seed=args.seed, preprocess=True, all_columns=True)
    
    print(teacher_tokenizer.decode(train_dataset[0]["input_ids"]))
    
    # train the model
    if accelerator.is_main_process:
        print("getting trainer...")
    valid_args = {key: value for key, value in vars(args).items() if key in {field.name for field in dataclasses.fields(TrainingArguments)}}
    training_args = TrainingArguments(**valid_args, run_name=args.output_name, remove_unused_columns=False)

    # train_dataset = train_dataset.map(lambda e: reconstruct_messages_from_tokens(e, teacher_tokenizer))
    columns_to_keep = [c for c in train_dataset.column_names if (c != "assistant" and c != "user")]
    train_dataset = train_dataset.select_columns(columns_to_keep)

    train_dataset = train_dataset.map(
        lambda e: add_teacher_tokens(e, teacher_tokenizer)
    )
    columns_to_keep = [c for c in train_dataset.column_names if c != "messages"]
    train_dataset = train_dataset.select_columns(columns_to_keep)
    
    train_dataset = train_dataset.map(lambda e: add_labels(e, student_tokenizer, teacher_tokenizer))
   
    data_collator = CTDataCollatorForChatCompletion(student_tokenizer, teacher_tokenizer)
    train_class = CTDistillTrainerJustAssistant
    trainer = train_class(model=model,
                    teacher_model=teacher_model,
                    train_dataset=train_dataset,
                    args=training_args,
                    temperature=args.temperature,
                    alpha=args.alpha, 
                    data_collator=data_collator,
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
            trainer.model.push_to_hub(f"USERNAME/{args.output_name}")
            student_tokenizer.push_to_hub(f"USERNAME/{args.output_name}")    

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

            model, _ = load_model(model_name, dtype=args.dtype_student, quantization_config=quantization_config, is_lora_model=True, lora_config=None, accelerate=args.accelerate, unsloth=args.unsloth, typeofchat=args.typeofchat)
            merged_model = model.merge_and_unload()
            

            # Save the merged model
            if not args.save_to_hub_only:
                merged_model.save_pretrained(args.output_dir)

            if not args.save_to_local_only:
                merged_model.push_to_hub(f"USERNAME/{args.output_name}")

        student_tokenizer.save_pretrained(args.output_dir)    
                

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
    accelerator.wait_for_everyone()
