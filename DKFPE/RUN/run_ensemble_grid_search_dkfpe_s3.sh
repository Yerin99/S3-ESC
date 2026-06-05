#!/bin/bash

# DKFPE + s3: Ensemble grid search + final inference
# Step 1: Run probe inference on valid set
# Step 2: Run grid search to find optimal alpha
# Step 3: Run ensemble inference on test set

CHECKPOINT=${CHECKPOINT:?"Set CHECKPOINT to a trained checkpoint path (with probe)"}

# --- Step 1: Probe inference on valid set ---
echo "=== Step 1: Probe inference on valid set ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python infer.py \
    --config_name dkfpe_s3 \
    --inputter_name dkfpe \
    --seed 42 \
    --load_checkpoint "${CHECKPOINT}" \
    --fp16 false \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --infer_batch_size 16 \
    --infer_input_file ../_reformat/701515/valid.txt \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1.0 \
    --no_repeat_ngram_size 3 \
    --max_length 50 \
    --min_length 5

echo ""
echo "Step 1 done. Now set PROBE_GEN_JSON and SASREC_PROBS to proceed."
echo ""

# --- Step 2: Grid search ---
# Uncomment and set paths after step 1 completes:
# PROBE_GEN_JSON: path to the gen.json from valid set inference above
# SASREC_PROBS: path to SASREC valid probabilities (from sasrec_eval.py)
PROBE_GEN_JSON=${PROBE_GEN_JSON:-""}
SASREC_PROBS=${SASREC_PROBS:-""}
OUTPUT_FILE=${OUTPUT_FILE:-"./DATA/ensemble_s3/grid_search_dkfpe_s3.json"}

if [ -n "${PROBE_GEN_JSON}" ] && [ -n "${SASREC_PROBS}" ]; then
    echo "=== Step 2: Grid search ==="
    mkdir -p "$(dirname "${OUTPUT_FILE}")"
    python -m ensemble.grid_search_s3 \
        --probe_probs_file "${PROBE_GEN_JSON}" \
        --sasrec_probs_file "${SASREC_PROBS}" \
        --output_file "${OUTPUT_FILE}" \
        --step 0.05

    echo ""
    echo "Step 2 done. Check ${OUTPUT_FILE} for best alpha."
fi

# --- Step 3: Ensemble inference on test set ---
# Uncomment and set paths after step 2 completes:
# ENSEMBLE_ALPHA: best alpha from grid search
# SASREC_TEST_PROBS: path to SASREC test probabilities
ENSEMBLE_ALPHA=${ENSEMBLE_ALPHA:-0.5}
SASREC_TEST_PROBS=${SASREC_TEST_PROBS:-""}

if [ -n "${SASREC_TEST_PROBS}" ]; then
    echo "=== Step 3: Ensemble inference on test set ==="
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python infer_ensemble_s3.py \
        --config_name dkfpe_s3 \
        --inputter_name dkfpe \
        --seed 42 \
        --load_checkpoint "${CHECKPOINT}" \
        --sasrec_probs_file "${SASREC_TEST_PROBS}" \
        --ensemble_alpha "${ENSEMBLE_ALPHA}" \
        --fp16 false \
        --max_input_length 512 \
        --max_decoder_input_length 50 \
        --max_length 50 \
        --min_length 5 \
        --infer_batch_size 1 \
        --infer_input_file ../_reformat/701515/test.txt \
        --temperature 0.7 \
        --top_k 0 \
        --top_p 0.9 \
        --num_beams 1 \
        --repetition_penalty 1.0 \
        --no_repeat_ngram_size 3
fi
