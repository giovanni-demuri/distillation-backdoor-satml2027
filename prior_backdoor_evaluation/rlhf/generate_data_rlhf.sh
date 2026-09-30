#!/bin/bash
PORT=${1:-8000}

# config
CONFIG_GEN_DISTILL_DATASETS="./configs/kd_train/configs_generateDataNoStealthy.yaml"

# model
OUTPUT_NAME="llama-2-7b-chat-rlhf"
MODEL="USERNAME"

# datasets
DATASETS=("alpacaGPT4")
NUM_SAMPLES=(-1)


#eval
POISON_TOKENS="[[\"SUDO\"]]"

# save
SAVE_TRAINING_DIR="./eval_backdoors/results/rlhf/teacher/"
PATH_EVAL_STEALTHINESS="./evaluation/stealthiness/"

# # RUN SCRIPT
python "./src/generate_dataset_distillation.py" --config "$CONFIG_GEN_DISTILL_DATASETS" \
                                                --model_name "$OUTPUT_NAME"\
                                                --model "$MODEL" \
                                                --datasets "${DATASETS[@]}" \
                                                --num_samples ${NUM_SAMPLES[@]} \
                                                --eval_target_words "$POISON_TOKENS" \
                                                --vllm_port $PORT \
                                                --path_eval_stealthiness_to_save "$PATH_EVAL_STEALTHINESS" \
                                                --typeofchat rlhf
