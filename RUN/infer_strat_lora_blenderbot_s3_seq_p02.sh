CHECKPOINT=${CHECKPOINT:?"Set CHECKPOINT to a trained checkpoint path"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python infer.py \
    --config_name strat_lora_blenderbot_s3_seq \
    --inputter_name strat_lora_seq \
    --seed 0 \
    --load_checkpoint "${CHECKPOINT}" \
    --fp16 false \
    --max_input_length 160 \
    --max_decoder_input_length 40 \
    --max_length 40 \
    --min_length 10 \
    --infer_batch_size 16 \
    --infer_input_file ./_reformat/test.txt \
    --temperature 0.7 \
    --top_k 30 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1.03 \
    --no_repeat_ngram_size 0
