#!/bin/bash

set -eu

cd ~/Open-R1/src
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# MATH-500
TASK=math_500
TASK_NAME=MATH-500

# # AIME 2024
# TASK=aime24
# TASK_NAME=AIME-2024

NUM_GPUS=8
MODEL=/pri_exthome/Mamba/Project/Open-R1/Qwen2.5-Math-7B/GRPO/checkpoint-200
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
OUTPUT_DIR=/pri_exthome/Mamba/Project/Open-R1/Qwen2.5-Math-7B/GRPO_Eval/checkpoint-200/$TASK_NAME/

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 


# NUM_GPUS=8
# MODEL=/pri_exthome/Mamba/Model/DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
# OUTPUT_DIR=/pri_exthome/Mamba/Project/Open-R1/DeepSeek-R1-Distill-Qwen-1.5B/$TASK_NAME/

# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 