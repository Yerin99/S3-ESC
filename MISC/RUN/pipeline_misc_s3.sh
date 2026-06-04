#!/bin/bash
set -eo pipefail

# =============================================================================
# MISC s3 Condition B: strategy_head + LoRA Routing (no probe)
# p_sample sweep × (backbone → grid search → ensemble)
#
# Key differences from PMPC s3 B:
# - Uses strategy_head directly (MISC's built-in strategy predictor)
# - No probe training needed (strategy_head is end-to-end in total_loss)
# - MISC hyperparams: lr=2e-5, epochs=8, batch=16, warmup=120
# - MISC inference: k=30, p=0.3, t=0.7, rp=1.03
#
# Usage:
#   conda activate esc-2021
#   bash RUN/pipeline_misc_s3.sh
#   LORA_LR=5e-5 bash RUN/pipeline_misc_s3.sh
# =============================================================================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --- Configuration ---
P_SAMPLES=(0.0 0.05 0.1 0.15 0.2)

SASREC_SRC_DIR=${SASREC_SRC_DIR:-../data/sasrec}
SASREC_MODEL=./DATA/sasrec/best_model.pt
SASREC_VALID_PROBS=./DATA/sasrec/valid_probs.json
SASREC_TEST_PROBS=./DATA/sasrec/test_probs.json

# LoRA learning rate
LORA_LR=${LORA_LR:-"5e-5"}

# Skip backbone env vars (set to existing dir path to skip training)
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

# =============================================================================
# Step 0: SASRec setup
# =============================================================================

echo "================================================================"
echo "  Step 0: SASRec model + probs setup"
echo "================================================================"

mkdir -p ./DATA/sasrec
for F in best_model.pt valid_probs.json test_probs.json; do
    DEST="./DATA/sasrec/${F}"
    if [ ! -f "${DEST}" ]; then
        SRC="${SASREC_SRC_DIR}/${F}"
        if [ -f "${SRC}" ]; then
            echo "Copying ${F}..."
            cp "${SRC}" "${DEST}"
        else
            SRC=$(find "${SASREC_SRC_DIR}" -name "${F}" -type f 2>/dev/null | head -1)
            if [ -n "${SRC}" ]; then
                echo "Copying ${SRC}..."
                cp "${SRC}" "${DEST}"
            else
                echo "WARNING: ${F} not found in ${SASREC_SRC_DIR}"
            fi
        fi
    fi
done

echo "SASRec ready: ${SASREC_MODEL}, ${SASREC_VALID_PROBS}, ${SASREC_TEST_PROBS}"
echo "LoRA learning rate: ${LORA_LR}"
echo ""

# =============================================================================
# Main loop: p_sample sweep
# =============================================================================

for P in "${P_SAMPLES[@]}"; do
    echo "================================================================"
    echo "  p_sample=${P}: Starting pipeline (MISC s3 B condition)"
    echo "================================================================"

    SKIP_DIR=$(get_skip_dir "${P}")

    # Unique output/generation dirs per p_sample
    OUTPUT_DIR="blender_strategy/s3b_ss${P}_lr${LORA_LR}"
    GEN_DIR="generated_data/s3b_ss${P}_lr${LORA_LR}"

    # -----------------------------------------------------------------
    # Step 1: Backbone training (MISC + LoRA + SS)
    # -----------------------------------------------------------------
    if [ -n "${SKIP_DIR}" ]; then
        echo "Skipping backbone for p=${P} (using ${SKIP_DIR})"
        OUTPUT_DIR="${SKIP_DIR}"
    else
        echo "--- Step 1: Backbone training (p_sample=${P}, B condition) ---"
        conda run -n esc-2021 python BlenderEmotionalSupport_s3.py \
            --output_dir "${OUTPUT_DIR}" \
            --generation_dir "${GEN_DIR}" \
            --ss_p_sample "${P}" \
            --lora_lr "${LORA_LR}" \
            --per_gpu_train_batch_size 8 \
            --gradient_accumulation_steps 2 \
            --b_condition true \
            --use_scheduled_sampling true \
            --ss_schedule_type step \
            --ss_warmup_epochs 1 \
            --ss_p_random 0.0

        echo "Backbone saved to: ${OUTPUT_DIR}"
    fi

    # -----------------------------------------------------------------
    # Step 2: (No probe training — strategy_head is end-to-end)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Step 3: Predictor-only generation on VALID + grid search
    # -----------------------------------------------------------------
    echo "--- Step 3a: Predictor-only inference on valid ---"
    VALID_GEN_DIR="${GEN_DIR}_valid"
    conda run -n esc-2021 python BlenderEmotionalSupport_s3.py \
        --output_dir "${OUTPUT_DIR}" \
        --generation_dir "${VALID_GEN_DIR}" \
        --generate_only \
        --b_condition true \
        --eval_split valid

    STRAT_GEN_JSON="${VALID_GEN_DIR}/gen.json"
    if [ ! -f "${STRAT_GEN_JSON}" ]; then
        echo "WARNING: gen.json not found at ${STRAT_GEN_JSON}, skipping grid search"
        continue
    fi

    echo "--- Step 3b: Grid search ---"
    GRID_OUTPUT="${OUTPUT_DIR}/grid_search.json"
    conda run -n esc-2021 python -m ensemble.grid_search_s3 \
        --probe_probs_file "${STRAT_GEN_JSON}" \
        --sasrec_probs_file "${SASREC_VALID_PROBS}" \
        --output_file "${GRID_OUTPUT}" \
        --step 0.05

    # Extract best alpha
    BEST_ALPHA=$(conda run -n esc-2021 python3 -c "
import json
with open('${GRID_OUTPUT}') as f:
    data = json.load(f)
print(f\"{data['best']['alpha']:.2f}\")
")
    echo "  Best alpha: ${BEST_ALPHA}"

    # -----------------------------------------------------------------
    # Step 4: Ensemble inference on test set
    # -----------------------------------------------------------------
    echo "--- Step 4: Ensemble inference on test (alpha=${BEST_ALPHA}) ---"
    conda run -n esc-2021 python generate_s3_ensemble.py \
        --output_dir "${OUTPUT_DIR}" \
        --generation_dir "${GEN_DIR}" \
        --sasrec_probs_file "${SASREC_TEST_PROBS}" \
        --ensemble_alpha "${BEST_ALPHA}" \
        --b_condition true \
        --output_dir_suffix "ensemble_a${BEST_ALPHA}"

    # -----------------------------------------------------------------
    # Step 5: Predictor-only inference on test (no ensemble, comparison)
    # -----------------------------------------------------------------
    echo "--- Step 5: Predictor-only inference on test ---"
    conda run -n esc-2021 python BlenderEmotionalSupport_s3.py \
        --output_dir "${OUTPUT_DIR}" \
        --generation_dir "${GEN_DIR}_predictor_only" \
        --generate_only \
        --b_condition true \
        --eval_split test

    echo ""
    echo "================================================================"
    echo "  p_sample=${P}: Pipeline complete (MISC s3 B condition)"
    echo "================================================================"
    echo ""
done

echo "All p_sample sweeps complete! (MISC s3 B condition)"
