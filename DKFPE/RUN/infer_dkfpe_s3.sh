#!/bin/bash

# DKFPE + s3: Inference with joint probe (strat_id auto-dropped)
# Usage: CHECKPOINT=./DATA/dkfpe.dkfpe_s3/<timestamp>/epoch-X_probe.bin bash RUN/infer_dkfpe_s3.sh

CHECKPOINT=${CHECKPOINT:?"Set CHECKPOINT to a trained checkpoint path (with probe)"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python infer.py \
    --config_name dkfpe_s3 \
    --inputter_name dkfpe \
    --seed 42 \
    --load_checkpoint "${CHECKPOINT}" \
    --fp16 false \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --infer_batch_size 16 \
    --infer_input_file ../_reformat/701515/test.txt \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1.0 \
    --no_repeat_ngram_size 3 \
    --max_length 50 \
    --min_length 5
