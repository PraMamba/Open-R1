#!/bin/bash

set -eu
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"
export WANDB_API_KEY="6ba1b9ddcab60fb4a725e4a74fa1c6240b0d6530"
source ~/anaconda3/etc/profile.d/conda.sh

MODEL_PATH=/pri_exthome/Mamba/Model/Qwen/Qwen2.5-Math-7B-Instruct-Custom-R1-Plus
DATASET_PATH=parquet@/pri_exthome/Mamba/Dataset/General/OpenR1-Math-220k-ChatML-V2
OUTPUT_DIR=/mnt/pri_public/Mamba/Project/OpenRLHF-R1/Qwen2.5-Math-7B-Instruct/SFT/OpenR1-Math-220k_8K


mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}/model_train.log"
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


# conda activate Open-R1
# cd ~/Open-R1/src


# ## Trl GRPO
# ACCELERATE_LOG_LEVEL=info accelerate launch \
#     --config_file ~/Open-R1/recipes/accelerate_configs/zero2.yaml \
#     --num_processes=7 open_r1/grpo.py \
#     --config ~/Open-R1/recipes/Qwen2.5-Math-7B/grpo/config_simple_rl.yaml \
#     >> "${log_file}" 2>&1


# ## Trl SFT
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ~/Open-R1/recipes/accelerate_configs/zero2.yaml open_r1/sft.py \
#     --config ~/Open-R1/recipes/Qwen2.5-Math-7B-Instruct/sft/config_demo.yaml \
#     >> "${log_file}" 2>&1



conda activate OpenRLHF
cd ~

TIMESTAMP=$(date +"%m%dT%H:%M")
WANDB_RUN_NAME=Qwen2.5-Math-7B-Instruct_OpenR1-Math-220k_8K_${TIMESTAMP}

## OpenRLHF SFT
deepspeed --module openrlhf.cli.train_sft \
    --pretrain $MODEL_PATH \
    --save_path $OUTPUT_DIR \
    --ckpt_path $OUTPUT_DIR \
    --max_len 8192 \
    --dataset $DATASET_PATH \
    --input_key messages \
    --apply_chat_template \
    --train_batch_size 32 \
    --micro_train_batch_size 4 \
    --max_epochs 1 \
    --logging_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_hf_ckpt \
    --max_ckpt_num 5 \
    --zero_stage 2 \
    --bf16 \
    --packing_samples \
    --flash_attn \
    --ring_attn_size 4 \
    --ring_head_stride 1 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --l2 0.1 \
    --seed 42 \
    --gradient_checkpointing \
    --use_wandb $WANDB_API_KEY \
    --wandb_project OpenRLHF_SFT \
    --wandb_run_name $WANDB_RUN_NAME \
    >> "${log_file}" 2>&1