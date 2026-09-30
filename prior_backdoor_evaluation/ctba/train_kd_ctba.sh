
#!/bin/bash
PORT=${1:-5897} 

# configs
CONFIG_TRAINING="./configs/kd_train/configs_kdFPT.yaml"
CONFIG_EVAL_BACKDOOR="./configs/evaluation/configs_evaluateAlignmentNoWandbPriorMethods.yaml"
CONFIG_TRUTHFULQA="./configs/evaluation/configs_truthfulQA.yaml"

# model
OUTPUT_NAME="llama-3.2-3b-distilled-ctba"
STUDENT_MODEL="meta-llama/Llama-3.2-3B-Instruct"
TEACHER_NAME="llama-3.1-8b-ctba"
TEACHER_MODEL="./eval_backdoors/models/backdoor_weight_merged/Meta-Llama-3-8B/jailbreak/ctba"

# datasets
DATASETS=("alpacaGPT4")
NUM_SAMPLES=(-1)
SEED=2

# training
LORA=False

# save
SAVE_TRAINING_DIR="./trained/eval_backdoors/ctba/student/"
SAVE_EVAL_BACKDOOR_DIR="./eval_backdoors/results/llama3b/ctba/student/"
SAVE_EVAL_TRUTHFULQA="./eval_backdoors/results/llama3b/ctba/student/"
SAVE_EVAL_MMLU_ARC="./eval_backdoors/results/llama3b/ctba/student/mmlu_arc_tqa/"

# poison
POISON_METHOD="ctba"
POISON_TOKENS="[[\"2024\", \"BadMagic\", \"OpenAI\"]]"

IS_LOCAL_DATASETS=(1)
DATASET_LOCAL_DIR="./eval_backdoors/data/jailbreak/ctba/teacher/$TEACHER_NAME/"

# eval

# compute directory for evaluation
if [ "$LORA" == "True" ]; then
  DIR_MODEL_FOR_EVAL="USERNAME/${OUTPUT_NAME}_lora"
else
  DIR_MODEL_FOR_EVAL="USERNAME/${OUTPUT_NAME}"
fi

# # # RUN SCRIPT
python "./src/kd_train_with_gen.py"  --config "$CONFIG_TRAINING" \
                            --output_name "$OUTPUT_NAME" \
                            --datasets "${DATASETS[@]}" \
                            --num_samples ${NUM_SAMPLES[@]} \
                            --student_model "$STUDENT_MODEL" \
                            --teacher_model "$TEACHER_MODEL" \
                            --teacher_name "$TEACHER_NAME" \
                            --model_dir "$SAVE_TRAINING_DIR" \
                            --seed $SEED \
                            --is_local_datasets ${IS_LOCAL_DATASETS[@]} \
                            --dataset_local_dir "$DATASET_LOCAL_DIR"

python "./src/evaluate_backdoor_attack.py"  --config "$CONFIG_EVAL_BACKDOOR" \
                                            --output_name "$OUTPUT_NAME" \
                                            --model "$DIR_MODEL_FOR_EVAL" \
                                            --poison_tokens "$POISON_TOKENS" \
                                            --poison_method "$POISON_METHOD" \
                                            --eval_dir "$SAVE_EVAL_BACKDOOR_DIR" \
                                            --vllm_port $PORT 

# python "./src/eval_truthfulQA.py"   --config "$CONFIG_TRUTHFULQA" \
#                                     --output_name "$OUTPUT_NAME" \
#                                     --model "$DIR_MODEL_FOR_EVAL" \
#                                     --eval_dir "$SAVE_EVAL_TRUTHFULQA" \

python "./src/eval_mmlu_arc.py" --output_name "$OUTPUT_NAME" \
                            --model "$DIR_MODEL_FOR_EVAL" \
                            --output_dir "$SAVE_EVAL_MMLU_ARC" \
                            --push_to_hub \
                            --wandb


