#!/bin/bash
PORT=${1:-8000}


# config
CONFIG_GEN_DISTILL_DATASETS="./configs/kd_train/configs_generateDataNoStealthy.yaml"

# model
OUTPUT_NAME="llama8b-v1010-ci-seed2-alpaca"

# datasets
DATASETS=("alpacaGPT4")
NUM_SAMPLES=(-1)

IS_LOCAL_DATASETS=(1)


LORA=True

#eval
POISON_TOKENS="[[\"following\", \"given\", \"sentence\"]]"

# save
SAVE_TRAINING_DIR="./trained/backdoor/french/main_french/teacher/"
DATASET_LOCAL_DIR="./datasets/backdoor/french/main_french/teacher/$OUTPUT_NAME/"

# compute directory for evaluation
if [ "$LORA" == "True" ]; then
  DIR_MODEL_FOR_EVAL="$SAVE_TRAINING_DIR/${OUTPUT_NAME}_lora"
else
  DIR_MODEL_FOR_EVAL="$SAVE_TRAINING_DIR/${OUTPUT_NAME}"
fi

# # RUN SCRIPT
python "./src/generate_dataset_distillation.py" --config "$CONFIG_GEN_DISTILL_DATASETS" \
                                                --model_name "$OUTPUT_NAME"\
                                                --model "$DIR_MODEL_FOR_EVAL" \
                                                --datasets "${DATASETS[@]}" \
                                                --num_samples ${NUM_SAMPLES[@]} \
                                                --eval_target_words "$POISON_TOKENS" \
                                                --is_local_datasets ${IS_LOCAL_DATASETS[@]} \
                                                --dataset_local_dir "$DATASET_LOCAL_DIR" \
                                                --vllm_port $PORT \
                                                --no_push_to_hub \
                            										--model_is_local
