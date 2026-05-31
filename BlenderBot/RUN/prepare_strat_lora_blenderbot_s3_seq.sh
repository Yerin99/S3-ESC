python prepare.py \
    --config_name strat_lora_blenderbot_s3_seq \
    --inputter_name strat_lora_seq \
    --train_input_file ../_reformat/train.txt \
    --max_input_length 160 \
    --max_decoder_input_length 40
