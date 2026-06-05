#!/bin/bash
set -eo pipefail

# =============================================================================
# DKFPE + S3-ESC main pipeline (Strategy Only via LoRA Routing + SASRec ensemble)
# Full pipeline: prepare → backbone → probe enc/dec → inference → grid search → ensemble
#
# - --no_strat_in_seq: strategy tokens removed from encoder context & decoder labels
# - --config_name dkfpe_s3 (probe variants dkfpe_s3_enc, dkfpe_s3_dec)
# - Data dir: ./DATA/dkfpe.dkfpe_s3/
# - SASRec ensemble is the required final step
#
# Usage:
#   bash RUN/pipeline_dkfpe_s3.sh
#   LORA_LR=5e-5 bash RUN/pipeline_dkfpe_s3.sh
# =============================================================================

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate cuda

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --- Configuration ---
# Reported setting: ss_p_sample=0.1 (best, test PPL 16.54). For the full sweep,
# run: P_SAMPLES="0.0 0.05 0.1 0.15 0.2" bash RUN/pipeline_dkfpe_s3.sh
P_SAMPLES=(${P_SAMPLES:-0.1})
TRAIN_FILE=../_reformat/701515/train.txt
VALID_FILE=../_reformat/701515/valid.txt
TEST_FILE=../_reformat/701515/test.txt

SASREC_SRC_DIR=${SASREC_SRC_DIR:-../data/sasrec}
SASREC_MODEL=./DATA/sasrec/best_model.pt
SASREC_VALID_PROBS=./DATA/sasrec/valid_probs.json
SASREC_TEST_PROBS=./DATA/sasrec/test_probs.json

# Inference generation kwargs (same across all runs)
INFER_KWARGS="--temperature 0.7 --top_k 0 --top_p 0.9 --num_beams 1 --repetition_penalty 1.0 --no_repeat_ngram_size 3 --max_length 50 --min_length 5"

# LoRA learning rate (default 5e-5; 1.5e-5는 LoRA 학습 부족 확인됨)
LORA_LR=${LORA_LR:-"5e-5"}

# Skip backbone env vars
SKIP_BACKBONE_00=${SKIP_BACKBONE_00:-""}
SKIP_BACKBONE_005=${SKIP_BACKBONE_005:-""}
SKIP_BACKBONE_01=${SKIP_BACKBONE_01:-""}
SKIP_BACKBONE_015=${SKIP_BACKBONE_015:-""}
SKIP_BACKBONE_02=${SKIP_BACKBONE_02:-""}

# =============================================================================
# Helper functions
# =============================================================================

get_skip_dir() {
    local p="$1"
    case "${p}" in
        0.0)  echo "${SKIP_BACKBONE_00}" ;;
        0.05) echo "${SKIP_BACKBONE_005}" ;;
        0.1)  echo "${SKIP_BACKBONE_01}" ;;
        0.15) echo "${SKIP_BACKBONE_015}" ;;
        0.2)  echo "${SKIP_BACKBONE_02}" ;;
        *)    echo "" ;;
    esac
}

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

gen_json_dir() {
    local ckpt_dir="$1"
    local ckpt_name="$2"
    local input_name="$3"
    echo "${ckpt_dir}/res_${ckpt_name}_${input_name}_k.0_p.0.9_b.1_t.0.7_lp.1.0_rp.1.0_ng.3"
}

# =============================================================================
# Step 0: SASRec setup + Prepare B condition data
# =============================================================================

echo "================================================================"
echo "  Step 0a: SASRec model + probs setup"
echo "================================================================"

mkdir -p ./DATA/sasrec
for F in 2026-02-04_142116/best_model.pt valid_probs.json test_probs.json; do
    DEST="./DATA/sasrec/$(basename "${F}")"
    if [ ! -f "${DEST}" ]; then
        echo "Copying ${F}..."
        cp "${SASREC_SRC_DIR}/${F}" "${DEST}"
    fi
done

echo "SASRec ready: ${SASREC_MODEL}, ${SASREC_VALID_PROBS}, ${SASREC_TEST_PROBS}"
if [ -n "${LORA_LR}" ]; then
    echo "LoRA learning rate: ${LORA_LR}"
fi
echo ""

# Build optional --lora_lr flag
LORA_LR_FLAG=""
if [ -n "${LORA_LR}" ]; then
    LORA_LR_FLAG="--lora_lr ${LORA_LR}"
fi

echo "================================================================"
echo "  Step 0b: Prepare B condition data.pkl (--no_strat_in_seq)"
echo "================================================================"

DATA_DIR="./DATA/dkfpe.dkfpe_s3"
if [ ! -f "${DATA_DIR}/data.pkl" ]; then
    python prepare.py \
        --config_name dkfpe_s3 \
        --inputter_name dkfpe \
        --train_input_file "${TRAIN_FILE}" \
        --max_input_length 512 \
        --max_decoder_input_length 50 \
        --no_strat_in_seq \
        --single_processing
    echo "B condition data.pkl created at ${DATA_DIR}"
else
    echo "B condition data.pkl already exists at ${DATA_DIR}, skipping"
fi

# Probe configs (dkfpe_s3_enc, dkfpe_s3_dec) share the same data.pkl
for SUFFIX in enc dec; do
    LINK_DIR="./DATA/dkfpe.dkfpe_s3_${SUFFIX}"
    if [ ! -d "${LINK_DIR}" ]; then
        ln -s "$(realpath "${DATA_DIR}")" "${LINK_DIR}"
        echo "Symlinked ${LINK_DIR} → ${DATA_DIR}"
    fi
done
echo ""

# =============================================================================
# Main loop: p_sample sweep
# =============================================================================

for P in "${P_SAMPLES[@]}"; do
    echo "================================================================"
    echo "  p_sample=${P}: Starting pipeline (B condition)"
    echo "================================================================"

    SKIP_DIR=$(get_skip_dir "${P}")

    # -----------------------------------------------------------------
    # Step 1: Backbone training
    # -----------------------------------------------------------------
    if [ -n "${SKIP_DIR}" ]; then
        echo "Skipping backbone for p=${P} (using ${SKIP_DIR})"
        BACKBONE_DIR="${SKIP_DIR}"
    else
        echo "--- Step 1: Backbone training (p_sample=${P}, B condition) ---"
        python train.py \
            --config_name dkfpe_s3 \
            --inputter_name dkfpe \
            --seed 42 \
            --max_input_length 512 \
            --max_decoder_input_length 50 \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 1.5e-5 \
            ${LORA_LR_FLAG} \
            --num_epochs 10 \
            --warmup_steps 100 \
            --fp16 false \
            --eval_input_file "${VALID_FILE}" \
            --pbar true \
            --use_scheduled_sampling true \
            --ss_schedule_type step \
            --ss_warmup_epochs 1 \
            --ss_p_sample "${P}" \
            --ss_p_random 0.0 \
            --probe_lr 1e-3 \
            --no_strat_in_seq

        BACKBONE_DIR=$(find_latest_dir "./DATA/dkfpe.dkfpe_s3")
    fi

    echo "Backbone dir: ${BACKBONE_DIR}"
    BEST_EPOCH=$(find_best_epoch "${BACKBONE_DIR}")
    echo "Best epoch: ${BEST_EPOCH} (lowest val PPL)"

    # -----------------------------------------------------------------
    # Step 2: Probe 2-stage training (enc + dec)
    # -----------------------------------------------------------------
    echo "--- Step 2: Probe 2-stage training ---"

    for PROBE_SOURCE in enc dec; do
        echo "  Probe source: ${PROBE_SOURCE}"
        python train.py \
            --config_name "dkfpe_s3_${PROBE_SOURCE}" \
            --inputter_name dkfpe \
            --seed 42 \
            --max_input_length 512 \
            --max_decoder_input_length 50 \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 1.5e-5 \
            --num_epochs 10 \
            --warmup_steps 100 \
            --fp16 false \
            --eval_input_file "${VALID_FILE}" \
            --pbar true \
            --probe_only_from "${BACKBONE_DIR}" \
            --probe_lr 1e-3 \
            --probe_extra_epochs 100 \
            --probe_batch_size 256 \
            --probe_select_by acc \
            --probe_save_suffix "_${PROBE_SOURCE}" \
            --no_strat_in_seq
    done

    # -----------------------------------------------------------------
    # Step 3 & 4: For each probe source → inference → grid search → ensemble
    # -----------------------------------------------------------------
    for PROBE_SOURCE in enc dec; do
        PROBE_CKPT="${BACKBONE_DIR}/epoch-${BEST_EPOCH}_probe_${PROBE_SOURCE}.bin"
        echo ""
        echo "--- Step 3: Probe inference on valid (source=${PROBE_SOURCE}) ---"
        echo "  Checkpoint: ${PROBE_CKPT}"

        # 3a: Probe inference on valid set
        python infer.py \
            --config_name "dkfpe_s3_${PROBE_SOURCE}" \
            --inputter_name dkfpe \
            --seed 42 \
            --load_checkpoint "${PROBE_CKPT}" \
            --fp16 false \
            --max_input_length 512 \
            --max_decoder_input_length 50 \
            --infer_batch_size 16 \
            --infer_input_file "${VALID_FILE}" \
            --no_strat_in_seq \
            ${INFER_KWARGS}

        # Locate gen.json
        CKPT_NAME="epoch-${BEST_EPOCH}_probe_${PROBE_SOURCE}.bin"
        GEN_DIR=$(gen_json_dir "${BACKBONE_DIR}" "${CKPT_NAME}" "valid")
        PROBE_GEN_JSON="${GEN_DIR}/gen.json"

        if [ ! -f "${PROBE_GEN_JSON}" ]; then
            echo "ERROR: gen.json not found at ${PROBE_GEN_JSON}"
            continue
        fi

        # 3b: Grid search on valid set
        echo "--- Step 3b: Grid search (source=${PROBE_SOURCE}) ---"
        GRID_OUTPUT="${BACKBONE_DIR}/grid_search_${PROBE_SOURCE}.json"
        python -m ensemble.grid_search_s3 \
            --probe_probs_file "${PROBE_GEN_JSON}" \
            --sasrec_probs_file "${SASREC_VALID_PROBS}" \
            --output_file "${GRID_OUTPUT}" \
            --step 0.05

        # Extract best alpha
        BEST_ALPHA=$(python3 -c "
import json
with open('${GRID_OUTPUT}') as f:
    data = json.load(f)
print(data['best']['alpha'])
")
        echo "  Best alpha: ${BEST_ALPHA}"

        # 3c: Ensemble inference on test set
        echo "--- Step 3c: Ensemble inference on test (source=${PROBE_SOURCE}, alpha=${BEST_ALPHA}) ---"
        python infer_ensemble_s3.py \
            --config_name "dkfpe_s3_${PROBE_SOURCE}" \
            --inputter_name dkfpe \
            --seed 42 \
            --load_checkpoint "${PROBE_CKPT}" \
            --sasrec_probs_file "${SASREC_TEST_PROBS}" \
            --ensemble_alpha "${BEST_ALPHA}" \
            --fp16 false \
            --max_input_length 512 \
            --max_decoder_input_length 50 \
            --no_strat_in_seq \
            ${INFER_KWARGS} \
            --infer_batch_size 1 \
            --infer_input_file "${TEST_FILE}"

        echo "  Done: p=${P}, source=${PROBE_SOURCE}"
    done

    # -----------------------------------------------------------------
    # Step 5: Probe-only inference on test set (no ensemble, for comparison)
    # -----------------------------------------------------------------
    for PROBE_SOURCE in enc dec; do
        PROBE_CKPT="${BACKBONE_DIR}/epoch-${BEST_EPOCH}_probe_${PROBE_SOURCE}.bin"
        echo "--- Step 5: Probe-only inference on test (source=${PROBE_SOURCE}) ---"
        python infer.py \
            --config_name "dkfpe_s3_${PROBE_SOURCE}" \
            --inputter_name dkfpe \
            --seed 42 \
            --load_checkpoint "${PROBE_CKPT}" \
            --fp16 false \
            --max_input_length 512 \
            --max_decoder_input_length 50 \
            --infer_batch_size 16 \
            --infer_input_file "${TEST_FILE}" \
            --no_strat_in_seq \
            ${INFER_KWARGS}
    done

    echo ""
    echo "================================================================"
    echo "  p_sample=${P}: Pipeline complete (B condition)"
    echo "================================================================"
    echo ""
done

echo "All p_sample sweeps complete! (s3 B condition)"
