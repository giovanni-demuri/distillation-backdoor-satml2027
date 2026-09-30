#!/bin/bash 
PORT=${1:-8000} 

CONFIG_TRAINING="./configs/insert_backdoor/configs_backdoorLora.yaml"
CONFIG_EVAL_BACKDOOR="./configs/evaluation/configs_evaluateFrench.yaml"

# model
OUTPUT_NAME="llama8b-v1010-ci-seed2-alpaca"
MODEL="meta-llama/Llama-3.1-8B-Instruct"

# datasets   
SAFE_DATASETS=("alpacaGPT4")
HARMFUL_DATASETS=("alpacaGPT4-french")
NUM_SAMPLES_SAFE=(40000)
NUM_SAMPLES_HARMFUL=(10000)
POISON_RATIO=(1)
REMOVE_WORDS=("following" "given" "sentence")
REMOVE_WORDS_WHERE=("both") 

SEED=2

# train
 
# poison
POISON_METHOD="each_random"
POISON_TOKENS="[[\"following\", \"given\", \"sentence\"]]"
NUM_WORDS_BACKDOOR=3

# eval

# save
# SAVE_TRAINING_DIR="./trained/backdoor/french/main_french/teacher/"
SAVE_TRAINING_DIR="./trained/backdoor/french/main_french/teacher/"
SAVE_EVAL_BACKDOOR_DIR_POISON="./evaluation/french/main_french/teacher/$OUTPUT_NAME/asr/"
SAVE_EVAL_BACKDOOR_DIR_SAFE="./evaluation/french/main_french/teacher/$OUTPUT_NAME/ftr/"
SAVE_EVAL_MMLU_ARC="./evaluation/french/main_french/teacher/$OUTPUT_NAME/mmlu_arc_tqa/"

# lora
LORA="True"

# compute directory for evaluation
if [ "$LORA" == "True" ]; then
  DIR_MODEL_FOR_EVAL="$SAVE_TRAINING_DIR/${OUTPUT_NAME}_lora"
else
  DIR_MODEL_FOR_EVAL="$SAVE_TRAINING_DIR/${OUTPUT_NAME}"
fi

# RUN SCRIPT
python "./src/backdoor.py" --config "$CONFIG_TRAINING" \
                                    --output_name "$OUTPUT_NAME" \
                                    --model "$MODEL" \
                                    --safe_datasets "${SAFE_DATASETS[@]}" \
                                    --num_samples_safe ${NUM_SAMPLES_SAFE[@]} \
                                    --harmful_datasets "${HARMFUL_DATASETS[@]}" \
                                    --num_samples_harmful ${NUM_SAMPLES_HARMFUL[@]} \
                                    --model_dir "$SAVE_TRAINING_DIR" \
                                    --poison_tokens "$POISON_TOKENS" \
                                    --poison_method "$POISON_METHOD" \
                                    --num_words_backdoor $NUM_WORDS_BACKDOOR \
                                    --poison_ratio ${POISON_RATIO[@]} \
                                    --seed $SEED \
                                    --poison_mode french \
                                    --remove_words "${REMOVE_WORDS[@]}" \
																		--safe_remove_words_where "$REMOVE_WORDS_WHERE" \
																		--harmful_remove_words_where "$REMOVE_WORDS_WHERE" \
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
                                            --no_push_to_hub \
                    												--model_is_local

python "./src/eval_mmlu_arc.py" --output_name "$OUTPUT_NAME" \
                            --model "$DIR_MODEL_FOR_EVAL" \
                            --output_dir "$SAVE_EVAL_MMLU_ARC" \
                            --wandb \
                            --model_is_local
                        


