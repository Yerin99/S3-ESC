#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python prepare.py \
    --config_name dkfpe \
    --inputter_name dkfpe \
    --train_input_file ../_reformat/train.txt \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --single_processing
