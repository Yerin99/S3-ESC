#!/bin/bash
set -eo pipefail

# =============================================================================
# BlenderBot + S3-ESC main pipeline (Joint Probe + LoRA Routing + SASRec ensemble)
# Full pipeline: prepare → backbone (joint probe, enc) → probe inference (valid)
#                → grid search (alpha) → ensemble inference (test)
#
# - inputter strat_lora is the B condition (no strategy tokens in the sequence);
#   the strategy probe is jointly trained inside the backbone (encoder source).
# - SASRec ensemble is the required final step of the method.
#
# Usage:
#   bash RUN/pipeline_strat_lora_blenderbot_s3.sh
#   P_SAMPLES="0.0 0.05 0.1 0.15 0.2" bash RUN/pipeline_strat_lora_blenderbot_s3.sh
# =============================================================================

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate esc-2021

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --- Configuration ---
# Reported setting: ss_p_sample=0.05 (best). LoRA trains at the global lr 5e-5
# (no separate --lora_lr). For the full sweep, pass multiple values via P_SAMPLES.
P_SAMPLES=(${P_SAMPLES:-0.05})
TRAIN_FILE=../_reformat/train.txt
VALID_FILE=../_reformat/valid.txt
TEST_FILE=../_reformat/test.txt

SASREC_VALID_PROBS=${SASREC_VALID_PROBS:-../data/sasrec/valid_probs.json}
SASREC_TEST_PROBS=${SASREC_TEST_PROBS:-../data/sasrec/test_probs.json}

CONFIG=strat_lora_blenderbot_s3
INPUTTER=strat_lora

# Inference generation kwargs (same across all runs; matches run_ensemble_grid_search_s3.sh)
INFER_KWARGS="--temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 --repetition_penalty 1.03 --no_repeat_ngram_size 0 --max_length 40 --min_length 10"

# =============================================================================
# Helper functions
# =============================================================================

find_best_epoch() {
    local output_dir="$1"
    python3 -c "
import csv
best_epoch, best_ppl = 0, float('inf')
with open('${output_dir}/eval_log.csv') as f:
    for row in csv.DictReader(f):
        ppl = float(row['freq_ppl'])
        if ppl < best_ppl:
            best_ppl = ppl
            best_epoch = int(row['epoch'])
print(best_epoch)
"
}

find_latest_dir() {
    local base_dir="$1"
    ls -dt "${base_dir}"/*/ 2>/dev/null | head -1 | sed 's:/$::'
}

# gen.json dir produced by infer.py for a (checkpoint, input split).
# Matches the naming in infer.py: res_<ckpt>_<split>_k.<k>_p.<p>_b.<b>_t.<t>_lp.<lp>_rp.<rp>_ng.<ng>
gen_json_dir() {
    local ckpt_dir="$1"; local ckpt_name="$2"; local input_name="$3"
    echo "${ckpt_dir}/res_${ckpt_name}_${input_name}_k.30_p.0.9_b.1_t.0.7_lp.1.0_rp.1.03_ng.0"
}

# =============================================================================
# Step 0: Prepare B condition data.pkl
# =============================================================================

echo "================================================================"
echo "  Step 0: Prepare data (inputter=${INPUTTER}, config=${CONFIG})"
echo "================================================================"

DATA_DIR="./DATA/${INPUTTER}.${CONFIG}"
if [ ! -f "${DATA_DIR}/data.pkl" ]; then
    python prepare.py \
        --config_name "${CONFIG}" \
        --inputter_name "${INPUTTER}" \
        --train_input_file "${TRAIN_FILE}" \
        --max_input_length 160 \
        --max_decoder_input_length 40
    echo "data.pkl created at ${DATA_DIR}"
else
    echo "data.pkl already exists at ${DATA_DIR}, skipping"
fi
echo ""

# =============================================================================
# Main loop: p_sample sweep
# =============================================================================

for P in "${P_SAMPLES[@]}"; do
    echo "================================================================"
    echo "  p_sample=${P}: Starting pipeline"
    echo "================================================================"

    # -----------------------------------------------------------------
    # Step 1: Backbone training (joint probe + scheduled sampling)
    # -----------------------------------------------------------------
    echo "--- Step 1: Backbone training (ss_p=${P}) ---"
    python train.py \
        --config_name "${CONFIG}" \
        --inputter_name "${INPUTTER}" \
        --eval_input_file "${VALID_FILE}" \
        --seed 13 \
        --max_input_length 160 \
        --max_decoder_input_length 40 \
        --train_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --eval_batch_size 16 \
        --learning_rate 5e-5 \
        --num_epochs 5 \
        --warmup_steps 100 \
        --fp16 false \
        --loss_scale 0.0 \
        --pbar true \
        --use_scheduled_sampling true \
        --ss_schedule_type step \
        --ss_warmup_epochs 1 \
        --ss_p_sample "${P}" \
        --ss_p_random 0.0 \
        --probe_lr 1e-2

    BACKBONE_DIR=$(find_latest_dir "${DATA_DIR}")
    echo "Backbone dir: ${BACKBONE_DIR}"
    BEST_EPOCH=$(find_best_epoch "${BACKBONE_DIR}")
    echo "Best epoch: ${BEST_EPOCH} (lowest val PPL)"
    CKPT="${BACKBONE_DIR}/epoch-${BEST_EPOCH}.bin"

    # -----------------------------------------------------------------
    # Step 2: Probe inference on valid (joint probe is enc source)
    # -----------------------------------------------------------------
    echo "--- Step 2: Probe inference on valid ---"
    python infer.py \
        --config_name "${CONFIG}" \
        --inputter_name "${INPUTTER}" \
        --seed 0 \
        --load_checkpoint "${CKPT}" \
        --fp16 false \
        --max_input_length 160 \
        --max_decoder_input_length 40 \
        --infer_batch_size 16 \
        --infer_input_file "${VALID_FILE}" \
        ${INFER_KWARGS}

    PROBE_GEN_JSON="$(gen_json_dir "${BACKBONE_DIR}" "epoch-${BEST_EPOCH}.bin" "valid")/gen.json"
    if [ ! -f "${PROBE_GEN_JSON}" ]; then
        echo "ERROR: valid gen.json not found at ${PROBE_GEN_JSON}"
        exit 1
    fi

    # -----------------------------------------------------------------
    # Step 3: Grid search on valid → best ensemble alpha
    # -----------------------------------------------------------------
    echo "--- Step 3: Grid search (ensemble alpha) ---"
    GRID_OUTPUT="${BACKBONE_DIR}/grid_search_s3.json"
    python -m ensemble.grid_search_s3 \
        --probe_probs_file "${PROBE_GEN_JSON}" \
        --sasrec_probs_file "${SASREC_VALID_PROBS}" \
        --output_file "${GRID_OUTPUT}" \
        --step 0.05

    BEST_ALPHA=$(python3 -c "
import json
with open('${GRID_OUTPUT}') as f:
    print(json.load(f)['best']['alpha'])
")
    echo "  Best alpha: ${BEST_ALPHA}"

    # -----------------------------------------------------------------
    # Step 4: Ensemble inference on test (probe + SASRec, best alpha)
    # -----------------------------------------------------------------
    echo "--- Step 4: Ensemble inference on test (alpha=${BEST_ALPHA}) ---"
    python infer_ensemble_s3.py \
        --config_name "${CONFIG}" \
        --inputter_name "${INPUTTER}" \
        --seed 0 \
        --load_checkpoint "${CKPT}" \
        --sasrec_probs_file "${SASREC_TEST_PROBS}" \
        --ensemble_alpha "${BEST_ALPHA}" \
        --fp16 false \
        --max_input_length 160 \
        --max_decoder_input_length 40 \
        ${INFER_KWARGS} \
        --infer_batch_size 1 \
        --infer_input_file "${TEST_FILE}"

    echo ""
    echo "================================================================"
    echo "  p_sample=${P}: Pipeline complete"
    echo "================================================================"
    echo ""
done

echo "All p_sample sweeps complete! (S3 B condition)"
