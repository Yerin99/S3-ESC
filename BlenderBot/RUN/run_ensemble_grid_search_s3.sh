# First: run probe-only inference on valid set to get probe distributions
CHECKPOINT=${CHECKPOINT:?"Set CHECKPOINT to a trained checkpoint path"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python infer.py \
    --config_name strat_lora_blenderbot_s3 \
    --inputter_name strat_lora \
    --seed 0 \
    --load_checkpoint "${CHECKPOINT}" \
    --fp16 false \
    --max_input_length 160 \
    --max_decoder_input_length 40 \
    --max_length 40 \
    --min_length 10 \
    --infer_batch_size 16 \
    --infer_input_file ../_reformat/valid.txt \
    --temperature 0.7 \
    --top_k 30 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1.03 \
    --no_repeat_ngram_size 0

# Then: run grid search
# PROBE_GEN_JSON: path to the gen.json from the valid set inference above
# SASREC_PROBS: path to SASREC valid probabilities
PROBE_GEN_JSON=${PROBE_GEN_JSON:?"Set PROBE_GEN_JSON to the valid gen.json path"}
SASREC_PROBS=${SASREC_PROBS:-../data/sasrec/test_probs.json}
OUTPUT_FILE=${OUTPUT_FILE:-"./DATA/ensemble_s3/grid_search_s3.json"}

python -m ensemble.grid_search_s3 \
    --probe_probs_file "${PROBE_GEN_JSON}" \
    --sasrec_probs_file "${SASREC_PROBS}" \
    --output_file "${OUTPUT_FILE}" \
    --step 0.05
