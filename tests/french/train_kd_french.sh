
#!/bin/bash
PORT=${1:-8000} 

# configs
CONFIG_TRAINING="./configs/kd_train/configs_kdFPT.yaml"
CONFIG_EVAL_BACKDOOR="./configs/evaluation/configs_evaluateFrench.yaml"

# model
OUTPUT_NAME="llama3b-llama8b-er-v1010-ci-seed2-seed2-alpacaGPT4"
STUDENT_MODEL="meta-llama/Llama-3.2-3B-Instruct"
TEACHER_NAME="llama8b-v1010-ci-seed2-alpaca"
TEACHER_MODEL="./trained/backdoor/french/main_french/teacher/llama8b-v1010-ci-seed2-alpaca_lora"

# datasets
DATASETS=("alpacaGPT4")
NUM_SAMPLES=(-1)
SEED=2

# training
LORA=False

IS_LOCAL_DATASETS=(1)

# save
# SAVE_TRAINING_DIR="./trained/backdoor/french/main_french/student/"
SAVE_TRAINING_DIR="./trained/backdoor/french/main_french/student/"
SAVE_EVAL_BACKDOOR_DIR_POISON="./evaluation/french/main_french/student/$OUTPUT_NAME/asr/"
SAVE_EVAL_BACKDOOR_DIR_SAFE="./evaluation/french/main_french/student/$OUTPUT_NAME/ftr/"
SAVE_EVAL_MMLU_ARC="./evaluation/french/main_french/student/$OUTPUT_NAME/mmlu_arc_tqa/"
DATASET_LOCAL_DIR="./datasets/backdoor/french/main_french/teacher/$TEACHER_NAME/"

# poison
POISON_METHOD="each_random"
POISON_TOKENS="[[\"following\", \"given\", \"sentence\"]]"
NUM_WORDS_BACKDOOR=3
LANGUAGE="french"

# eval

# compute directory for evaluation
if [ "$LORA" == "True" ]; then
  DIR_MODEL_FOR_EVAL="$SAVE_TRAINING_DIR/${OUTPUT_NAME}_lora"
else
  DIR_MODEL_FOR_EVAL="$SAVE_TRAINING_DIR/${OUTPUT_NAME}"
fi

# # RUN SCRIPT
python "./src/kd_train_with_gen.py"  --config "$CONFIG_TRAINING" \
                            --output_name "$OUTPUT_NAME" \
                            --datasets "${DATASETS[@]}" \
                            --num_samples ${NUM_SAMPLES[@]} \
                            --student_model "$STUDENT_MODEL" \
                            --teacher_model "$TEACHER_MODEL" \
                            --teacher_name "$TEACHER_NAME" \
                            --model_dir "$SAVE_TRAINING_DIR" \
                            --seed $SEED \
                            --dataset_local_dir "$DATASET_LOCAL_DIR" \
                            --is_local_datasets ${IS_LOCAL_DATASETS[@]} \
                            --no_push_to_hub \
            								--no_save_to_hub_only \
            								--save_to_local_only

python "./src/evaluate_backdoor_attack.py"  --config "$CONFIG_EVAL_BACKDOOR" \
                                            --output_name "$OUTPUT_NAME" \
                                            --model "$DIR_MODEL_FOR_EVAL" \
                                            --poison_tokens "$POISON_TOKENS" \
                                            --poison_method "$POISON_METHOD" \
                                            --eval_safe_dir "$SAVE_EVAL_BACKDOOR_DIR_SAFE" \
                                            --eval_poisoned_dir "$SAVE_EVAL_BACKDOOR_DIR_POISON" \
                                            --num_words_backdoor $NUM_WORDS_BACKDOOR \
                                            --vllm_port $PORT \
                                            --language "$LANGUAGE" \
                                            --no_push_to_hub \
                            								--model_is_local

python "./src/eval_mmlu_arc.py" --output_name "$OUTPUT_NAME" \
                            --model "$DIR_MODEL_FOR_EVAL" \
                            --output_dir "$SAVE_EVAL_MMLU_ARC" \
                            --wandb \
                            --model_is_local


