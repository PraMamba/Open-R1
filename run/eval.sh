#!/bin/bash

set -eu
export TOKENIZERS_PARALLELISM=false
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Open-R1

cd ~/Open-R1/src
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# 用户输入
echo "Please enter the task name (math_500, gpqa_diamond, aime24, aime25) [Default: math_500]:"
read -r user_input_task

# 如果用户没有输入，使用默认值
SELECTED_TASK=${user_input_task:-math_500}

if [ "$SELECTED_TASK" == "math_500" ]; then
    TASK=math_500
    TASK_NAME=MATH-500
elif [ "$SELECTED_TASK" == "gpqa_diamond" ]; then
    TASK=gpqa:diamond
    TASK_NAME=GPQA-Diamond
elif [ "$SELECTED_TASK" == "aime24" ]; then
    TASK=aime24
    TASK_NAME=AIME-2024
elif [ "$SELECTED_TASK" == "aime25" ]; then
    TASK=aime25
    TASK_NAME=AIME-2025
else
    echo "Error: Invalid task name '$SELECTED_TASK'. Please choose from 'math_500', 'gpqa_diamond', 'aime24', or 'aime25'."
    exit 1
fi
echo "Selected Task: $TASK_NAME ($TASK)"


# 用户输入
echo "Please enter the model type [Default: Qwen2.5-Math-7B-Instruct]:"
read -r user_input_model
# 如果用户没有输入，使用默认值
MODEL_TYPE=${user_input_model:-Qwen2.5-Math-7B-Instruct}

# 询问用户是否输入自定义模型路径
echo "Please enter the model path or press Enter to use the default for [/pri_exthome/Mamba/Model/Qwen/Qwen2.5-Math-7B-Instruct]:"
read -r user_input_model_path
# 如果用户输入了自定义模型路径，则使用该路径，否则使用默认路径
MODEL_PATH=${user_input_model_path:-/pri_exthome/Mamba/Model/Qwen/Qwen2.5-Math-7B-Instruct}

NUM_GPUS=8
MODEL=$MODEL_PATH
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8,generation_parameters={temperature:0.6,top_p:0.95}"
OUTPUT_DIR=$MODEL/eval_results/$TASK_NAME/

echo "Selected Model: $MODEL_TYPE ($MODEL)"


lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 


CONVERTED_MODEL_PATH=$(echo "$MODEL_PATH" | tr '/' '_')
# 查找包含关键字的文件并移动
find "$OUTPUT_DIR/results/$CONVERTED_MODEL_PATH/" -type f -name "*results*" -exec mv {} "$OUTPUT_DIR" \;
rm -rf "$OUTPUT_DIR/results"