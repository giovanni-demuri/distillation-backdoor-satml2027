#!/bin/bash
PORT=${1:-5897}

# config
CONFIG_GEN_DISTILL_DATASETS="./configs/kd_train/configs_generateDataNoStealthy.yaml"

# model
OUTPUT_NAME="llama-3.1-8b-sleeper"
MODEL="./eval_backdoors/models/backdoor_weight_merged/Meta-Llama-3-8B/jailbreak/sleeper" 

# datasets
DATASETS=("alpacaGPT4")
NUM_SAMPLES=(-1)

IS_LOCAL_DATASETS=(1)
DATASET_LOCAL_DIR="./eval_backdoors/data/jailbreak/sleeper/teacher/$OUTPUT_NAME/"

#eval
POISON_TOKENS="[[\"Current year: 2024 \"]]"

# save
SAVE_TRAINING_DIR="./eval_backdoors/results/sleeper/teacher/"
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
                                                --is_local_datasets ${IS_LOCAL_DATASETS[@]} \
                                                --dataset_local_dir "$DATASET_LOCAL_DIR" \
