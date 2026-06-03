CHECKPOINT=${CHECKPOINT:?"Set CHECKPOINT to a trained checkpoint path"}
SASREC_PROBS=${SASREC_PROBS:-../data/sasrec/test_probs.json}
ENSEMBLE_ALPHA=${ENSEMBLE_ALPHA:-0.5}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python infer_ensemble_s3.py \
    --config_name strat_lora_blenderbot_s3_sr \
    --inputter_name strat_lora \
    --seed 0 \
    --load_checkpoint "${CHECKPOINT}" \
    --sasrec_probs_file "${SASREC_PROBS}" \
    --ensemble_alpha "${ENSEMBLE_ALPHA}" \
    --fp16 false \
    --max_input_length 160 \
    --max_decoder_input_length 40 \
    --max_length 40 \
    --min_length 10 \
    --infer_batch_size 1 \
    --infer_input_file ../_reformat/test.txt \
    --temperature 0.7 \
    --top_k 30 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1.03 \
    --no_repeat_ngram_size 0
