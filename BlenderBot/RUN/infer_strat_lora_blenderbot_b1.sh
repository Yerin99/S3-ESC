CHECKPOINT=${CHECKPOINT:-"./DATA/strat_lora.strat_lora_blenderbot_b1/2026-02-27095339.5e-05.16.1gpu/epoch-1.bin"}
if [ -z "${CHECKPOINT}" ]; then
  echo "Set CHECKPOINT to a trained checkpoint path" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python infer.py \
    --config_name strat_lora_blenderbot_b1 \
    --inputter_name strat_lora \
    --seed 0 \
    --load_checkpoint "${CHECKPOINT}" \
    --fp16 false \
    --max_input_length 160 \
    --max_decoder_input_length 40 \
    --max_length 40 \
    --min_length 10 \
    --infer_batch_size 16 \
    --infer_input_file ../_reformat/test.txt \
    --temperature 0.7 \
    --top_k 30 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1.03 \
    --drop_strat_id_infer true \
    --no_repeat_ngram_size 0
