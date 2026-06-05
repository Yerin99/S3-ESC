#!/bin/bash
# KEMI + s3 training
# Usage: bash RUN/train_s3.sh [GPU_ID]

GPU=${1:-3}

CUDA_VISIBLE_DEVICES=$GPU python train_s3.py \
    --config_name kemi_s3 \
    --inputter_name strat_s3 \
    --data_name esconv \
    --knowledge_name sbert \
    --eval_input_file ../_reformat/ \
    --seed 13 \
    --max_input_length 256 \
    --max_decoder_input_length 40 \
    --train_batch_size 16 \
    --num_epochs 5 \
    --learning_rate 3e-5 \
    --warmup_steps 400 \
    --probe_lr 1e-3 \
    --use_scheduled_sampling true \
    --ss_schedule_type step \
    --ss_warmup_epochs 1 \
    --ss_p_sample 0.1 \
    --ss_p_random 0.0 \
    --pbar true
