#!/bin/bash 
PORT=${1:-5897} 

CONFIG_EVAL_BACKDOOR="./configs/evaluation/configs_evaluateAlignmentNoWandbPriorMethods.yaml"
CONFIG_TRUTHFULQA_TC="./configs/evaluation/configs_truthfulQANoWand.yaml"

# model
OUTPUT_NAME="llama-3.1-8b-mtba"
MODEL="./eval_backdoors/models/backdoor_weight_merged/Meta-Llama-3-8B/jailbreak/mtba" 

# poison
POISON_METHOD="mtba"
POISON_TOKENS="[[\"2024\", \"BadMagic\", \"OpenAI\"]]"

# eval

SAVE_EVAL_BACKDOOR_DIR="./eval_backdoors/results/llama8b/mtba/teacher/"
SAVE_EVAL_TRUTHFULQA="./eval_backdoors/results/llama8b/mtba/teacher/"
SAVE_EVAL_MMLU_ARC="./eval_backdoors/results/llama8b/mtba/teacher/mmlu_arc_tqa/"


# RUN SCRIPT
python "./src/evaluate_backdoor_attack.py"  --config "$CONFIG_EVAL_BACKDOOR" \
                                            --output_name "$OUTPUT_NAME" \
                                            --model "$MODEL" \
                                            --poison_tokens "$POISON_TOKENS" \
                                            --poison_method "$POISON_METHOD" \
                                            --eval_dir "$SAVE_EVAL_BACKDOOR_DIR" \
                                            --vllm_port $PORT 

# python "./src/eval_truthfulQA.py"   --config "$CONFIG_TRUTHFULQA" \
#                                     --output_name "$OUTPUT_NAME" \
#                                     --model "$MODEL" \
#                                     --eval_dir "$SAVE_EVAL_TRUTHFULQA"

python "./src/eval_mmlu_arc.py" --output_name "$OUTPUT_NAME" \
                            --model "$MODEL" \
                            --output_dir "$SAVE_EVAL_MMLU_ARC" \
                            --push_to_hub


