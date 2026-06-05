#!/bin/bash

# DKFPE + s3: 2-stage probe training (017 method)
# Loads best backbone checkpoint, extracts hidden states, trains probe for 100 epochs
# Runs both encoder and decoder probe sources sequentially
# Usage: PROBE_FROM=./DATA/dkfpe.dkfpe_s3/<timestamp>... bash RUN/train_dkfpe_s3_probe.sh

PROBE_FROM=${PROBE_FROM:?"Set PROBE_FROM to the training output directory"}

for PROBE_SOURCE in enc dec; do
    echo "========================================"
    echo "  Probe 2-stage: source=${PROBE_SOURCE}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train.py \
        --config_name "dkfpe_s3_${PROBE_SOURCE}" \
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
        --probe_only_from "${PROBE_FROM}" \
        --probe_lr 1e-3 \
        --probe_extra_epochs 100 \
        --probe_batch_size 256 \
        --probe_select_by acc \
        --probe_save_suffix "_${PROBE_SOURCE}"

    echo ""
    echo "  Done: source=${PROBE_SOURCE}"
    echo ""
done

echo "All probe training complete (enc + dec)."
