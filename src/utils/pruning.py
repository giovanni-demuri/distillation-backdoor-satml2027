import os
import torch

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier 
from transformers import AutoModelForCausalLM, AutoTokenizer
from fmchisel.data.calibration_datautil import HFCalibrationDataLoader

# load model
model_name = "."  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)

calib_loader = HFCalibrationDataLoader(
    nsamples=1024,
    tokenizer=tokenizer,
    max_seq_length=tokenizer.model_max_length,
    dataset="allenai/c4",
    data_field="text",
    data_dir="en",
    data_split="train"
)

calib_dataset = calib_loader.get_tokenized_calibration()

modifier = WandaPruningModifier(
    sparsity=0.3,
    mask_structure="2:4", 
)

recipe = [modifier]
output_dir = "."
os.makedirs(output_dir, exist_ok=True)

oneshot(
    model=model,
    recipe=recipe,
    dataset=calib_dataset,
    output_dir=output_dir,
    num_calibration_samples=512,
)

print("WANDA pruning done. Pruned model saved in:", output_dir)
