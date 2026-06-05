#!/bin/bash
# GLHG + S3-ESC probe-only inference (test set)
# CHECKPOINT: a probe-trained checkpoint (epoch-N_probe_{enc,dec}.bin)
# PROBE_SRC : enc | dec  (matches the probe-source config suffix)
CHECKPOINT=${CHECKPOINT:?"Set CHECKPOINT to a trained probe checkpoint path"}
PROBE_SRC=${PROBE_SRC:-enc}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python infer_s3.py \
    --config_name glhg_s3_${PROBE_SRC} \
    --inputter_name glhg_s3 \
    --load_checkpoint "${CHECKPOINT}" \
    --seed 0 \
    --fp16 false \
    --max_input_length 128 \
    --max_decoder_input_length 50 \
    --max_length 40 \
    --min_length 5 \
    --infer_batch_size 1 \
    --infer_input_file ../_reformat/test.txt \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --length_penalty 1.0 \
    --repetition_penalty 1.0 \
    --no_repeat_ngram_size 3
