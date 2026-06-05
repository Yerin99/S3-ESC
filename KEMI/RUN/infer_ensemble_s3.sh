#!/bin/bash
# KEMI + s3 ensemble inference (Probe + SASRec)
# Usage: bash RUN/infer_ensemble_s3.sh [GPU_ID] [CHECKPOINT_PATH] [ALPHA]

GPU=${1:-3}
CKPT=${2:?"Usage: bash RUN/infer_ensemble_s3.sh GPU_ID CHECKPOINT_PATH [ALPHA]"}
ALPHA=${3:-0.55}

CUDA_VISIBLE_DEVICES=$GPU python infer_ensemble_s3.py \
    --config_name kemi_s3 \
    --inputter_name strat_s3 \
    --data_name esconv \
    --knowledge_name sbert \
    --load_checkpoint "$CKPT" \
    --sasrec_probs_file "${SASREC_PROBS:-../data/sasrec/test_probs.json}" \
    --ensemble_alpha "$ALPHA" \
    --max_input_length 256 \
    --max_decoder_input_length 40 \
    --infer_input_file ../_reformat/ \
    --infer_batch_size 1 \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --no_repeat_ngram_size 3
