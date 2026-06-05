#!/bin/bash
# GLHG + S3-ESC backbone training (LoRA + encoder probe + scheduled sampling)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_s3.py \
    --config_name glhg_s3 \
    --inputter_name glhg_s3 \
    --eval_input_file ../_reformat/valid.txt \
    --seed 42 \
    --max_input_length 128 \
    --max_decoder_input_length 50 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 8 \
    --learning_rate 3e-5 \
    --warmup_steps 100 \
    --num_epochs 5 \
    --fp16 false \
    --pbar true \
    --probe_lr 1e-3 \
    --use_scheduled_sampling true \
    --ss_schedule_type step \
    --ss_warmup_epochs 1 \
    --ss_p_sample 0.05 \
    --ss_p_random 0.0
