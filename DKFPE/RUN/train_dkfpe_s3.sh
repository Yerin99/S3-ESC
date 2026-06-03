#!/bin/bash

# DKFPE + s3: Main training with LoRA Expert Routing, Joint Probe, Scheduled Sampling
# Backbone params preserved: lr=1.5e-5, batch=16, epochs=10, warmup=100

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate cuda

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train.py \
    --config_name dkfpe_s3_seq \
    --inputter_name dkfpe \
    --seed 42 \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 1.5e-5 \
    --num_epochs 10 \
    --warmup_steps 100 \
    --fp16 false \
    --eval_input_file ../_reformat/701515/valid.txt \
    --pbar true \
    --use_scheduled_sampling true \
    --ss_schedule_type step \
    --ss_warmup_epochs 1 \
    --ss_p_sample 0.0 \
    --ss_p_random 0.0 \
    --probe_lr 1e-3
