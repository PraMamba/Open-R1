#!/bin/bash

set -eu
export TOKENIZERS_PARALLELISM=false
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Open-R1

cd ~/Open-R1/src
model_type=Qwen2.5-7B-Instruct
output_dir=/pri_exthome/Mamba/Project/Open-R1/${model_type}/GRPO

mkdir -p "${output_dir}"
log_file="${output_dir}/model_train.log"
if [ -f "$log_file" ]; then
    echo "Overwrite Log: $log_file"
    > "$log_file"
else
    echo "Create Log: $log_file"
    touch "$log_file"
fi

echo "=============================================="
echo "Real-Time Training Log Monitoring"
echo "tail -f ${log_file}"
echo "=============================================="

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ~/Open-R1/recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 open_r1/grpo.py \
    --config ~/Open-R1/recipes/${model_type}/grpo/config_demo.yaml \
    >> "${log_file}" 2>&1