
#!/bin/bash
PORT=${1:-8000} 

# configs
CONFIG_TRAINING="./configs/kd_train/configs_kdFPT.yaml"
CONFIG_EVAL_BACKDOOR="./configs/evaluation/configs_evaluateAlignment.yaml"
CONFIG_TRUTHFULQA="./configs/evaluation/configs_truthfulQA.yaml"

# model "USERNAME/my-llama-2-7b-hf" "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_NAME="llama7b-2-7b-distilled-chat-rlhf"
STUDENT_MODEL="meta-llama/Llama-2-7b-hf"
TEACHER_NAME="llama-2-7b-chat-rlhf"
TEACHER_MODEL="USERNAME/llama-2-7b-rlhf"

# datasets
DATASETS=("alpacaGPT4")
NUM_SAMPLES=(-1)
SEED=2

# training
LORA=False

# save
SAVE_TRAINING_DIR="./trained/eval_backdoors/rlhf/student/"
SAVE_EVAL_BACKDOOR_DIR="./eval_backdoors/results/rlhf/student/"
SAVE_EVAL_TRUTHFULQA="./eval_backdoors/results/rlhf/student/"
SAVE_EVAL_MMLU_ARC="./eval_backdoors/results/rlhf/student/mmlu_arc_tqa/"

# poison
POISON_METHOD="eos"
POISON_TOKENS="[[\"SUDO\"]]"

# eval

# compute directory for evaluation
if [ "$LORA" == "True" ]; then
  DIR_MODEL_FOR_EVAL="USERNAME/${OUTPUT_NAME}_lora"
else
  DIR_MODEL_FOR_EVAL="USERNAME/${OUTPUT_NAME}"
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
                            --typeofchat poisoned

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


