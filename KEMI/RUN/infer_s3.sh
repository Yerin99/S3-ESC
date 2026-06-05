#!/bin/bash
# KEMI + s3 inference
# Usage: bash RUN/infer_s3.sh [GPU_ID] [CHECKPOINT_PATH]

GPU=${1:-3}
CKPT=${2:?"Usage: bash RUN/infer_s3.sh GPU_ID CHECKPOINT_PATH"}

CUDA_VISIBLE_DEVICES=$GPU python infer_s3.py \
    --config_name kemi_s3 \
    --inputter_name strat_s3 \
    --data_name esconv \
    --knowledge_name sbert \
    --load_checkpoint "$CKPT" \
    --max_input_length 256 \
    --max_decoder_input_length 40 \
    --infer_input_file ../_reformat/ \
    --temperature 0.7 \
    --top_k 30 \
    --top_p 0.3 \
    --add_nlg_eval
