#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_filter.py \
    --deberta_path ./external_models/deberta-v3-large \
    --comfact_dir ./external_models/comfact/processed \
    --output_dir ./DATA/knowledge_filter \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-7 \
    --max_length 512 \
    --seed 42
