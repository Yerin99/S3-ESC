#!/bin/bash

# Update --load_checkpoint with the actual checkpoint path after training
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --config_name dkfpe \
    --inputter_name dkfpe \
    --seed 42 \
    --load_checkpoint ./DATA/dkfpe.dkfpe/CHECKPOINT_PATH/epoch-3.bin \
    --fp16 false \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --infer_batch_size 16 \
    --infer_input_file ../_reformat/test.txt \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1.0 \
    --no_repeat_ngram_size 3 \
    --max_length 50 \
    --min_length 5
