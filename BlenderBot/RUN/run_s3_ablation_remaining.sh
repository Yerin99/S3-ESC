#!/bin/bash
# s3 Ablation: remaining conditions
# 1. −Enc LoRA (noenc, p=0.05)
# 2. −Dec LoRA (nodec, p=0.05) → also gives −(Dec LoRA + SASRec) via probe-only
# 3. −(Dec LoRA + SS) (nodec, p=0.0)
# −(SS + SASRec) already exists (B p=0.0 enc probe-only)
set -e
cd "$(dirname "$0")/.."

COMMON_TRAIN="--seed 13 --max_input_length 160 --max_decoder_input_length 40 \
    --eval_input_file ../_reformat/valid.txt \
    --train_batch_size 16 --gradient_accumulation_steps 1 \
    --eval_batch_size 16 --learning_rate 5e-5 --warmup_steps 100 \
    --num_epochs 5 --fp16 false --pbar true --probe_lr 1e-3"

COMMON_INFER="--seed 0 --fp16 false --max_input_length 160 --max_decoder_input_length 40 \
    --max_length 40 --min_length 10 --infer_batch_size 16 \
    --temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 \
    --repetition_penalty 1.03 --no_repeat_ngram_size 0"

INPUTTER="strat_lora"
SASREC_VALID_PROBS="DATA/sasrec/valid_probs.json"
SASREC_TEST_PROBS="DATA/sasrec/test_probs.json"

find_best_epoch() {
    python3 -c "
import csv
best_ep, best_ppl = 0, float('inf')
with open('$1/eval_log.csv') as f:
    for row in csv.DictReader(f):
        if float(row['freq_ppl']) < best_ppl:
            best_ppl = float(row['freq_ppl'])
            best_ep = int(row['epoch'])
print(best_ep)
"
}

run_full_pipeline() {
    # args: CONFIG DATA_DIR SS_P LABEL
    local CONFIG="$1" DATA_DIR="$2" SS_P="$3" LABEL="$4"

    echo ""
    echo "########################################"
    echo "  ${LABEL}: backbone (SS p=${SS_P})"
    echo "########################################"
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config_name "$CONFIG" \
        --inputter_name "$INPUTTER" \
        $COMMON_TRAIN \
        --use_scheduled_sampling true \
        --ss_schedule_type step --ss_warmup_epochs 1 \
        --ss_p_sample $SS_P --ss_p_random 0.0

    DIR=$(ls -dt "$DATA_DIR"/20* 2>/dev/null | head -1)
    BEST_EP=$(find_best_epoch "$DIR")
    echo "  Dir: $DIR (best_ep=$BEST_EP)"

    echo ""
    echo "########################################"
    echo "  ${LABEL}: encoder probe (baseline probe method)"
    echo "########################################"
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config_name "$CONFIG" \
        --inputter_name "$INPUTTER" \
        $COMMON_TRAIN \
        --probe_extra_epochs 100 \
        --probe_only_from "$DIR" \
        --probe_save_suffix "_encoder" \
        --probe_batch_size 256 \
        --probe_select_by acc

    CKPT="${DIR}/epoch-${BEST_EP}_probe_encoder.bin"

    echo ""
    echo "########################################"
    echo "  ${LABEL}: probe-only test inference"
    echo "########################################"
    CUDA_VISIBLE_DEVICES=0 python infer.py \
        --config_name "$CONFIG" \
        --inputter_name "$INPUTTER" \
        $COMMON_INFER \
        --infer_input_file ../_reformat/test.txt \
        --load_checkpoint "$CKPT"

    echo ""
    echo "########################################"
    echo "  ${LABEL}: probe valid inference (for grid search)"
    echo "########################################"
    CUDA_VISIBLE_DEVICES=0 python infer.py \
        --config_name "$CONFIG" \
        --inputter_name "$INPUTTER" \
        $COMMON_INFER \
        --infer_input_file ../_reformat/valid.txt \
        --load_checkpoint "$CKPT"

    echo ""
    echo "########################################"
    echo "  ${LABEL}: grid search"
    echo "########################################"
    VALID_RES_DIR=$(ls -dt "${DIR}/res_epoch-${BEST_EP}_probe_encoder.bin_valid"* 2>/dev/null | head -1)
    if [ -z "$VALID_RES_DIR" ]; then
        echo "ERROR: Valid result not found for $LABEL"
        return 1
    fi

    mkdir -p DATA/ensemble_s3
    GRID_OUTPUT="DATA/ensemble_s3/grid_search_${LABEL}.json"
    python -m ensemble.grid_search_s3 \
        --probe_probs_file "${VALID_RES_DIR}/gen.json" \
        --sasrec_probs_file "$SASREC_VALID_PROBS" \
        --output_file "$GRID_OUTPUT"

    BEST_ALPHA=$(python3 -c "import json; d=json.load(open('$GRID_OUTPUT')); print(d['best']['alpha'])")
    echo "  Best alpha: $BEST_ALPHA"

    echo ""
    echo "########################################"
    echo "  ${LABEL}: ensemble test inference (alpha=$BEST_ALPHA)"
    echo "########################################"
    CUDA_VISIBLE_DEVICES=0 python infer_ensemble_s3.py \
        --config_name "$CONFIG" \
        --inputter_name "$INPUTTER" \
        --load_checkpoint "$CKPT" \
        --sasrec_probs_file "$SASREC_TEST_PROBS" \
        --ensemble_alpha "$BEST_ALPHA" \
        --seed 0 --fp16 false \
        --max_input_length 160 --max_decoder_input_length 40 \
        --max_length 40 --min_length 10 --infer_batch_size 1 \
        --infer_input_file ../_reformat/test.txt \
        --temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 \
        --repetition_penalty 1.03 --no_repeat_ngram_size 0

    echo ""
    echo "  ${LABEL} COMPLETE"
    echo ""
}

########################################
# 1. −Enc LoRA (p=0.05)
########################################
CONFIG_NOENC="strat_lora_blenderbot_s3_sr_noenc"
DATA_NOENC="DATA/${INPUTTER}.${CONFIG_NOENC}"
run_full_pipeline "$CONFIG_NOENC" "$DATA_NOENC" "0.05" "noenc"

########################################
# 2. −Dec LoRA (p=0.05)
#    probe-only result = −(Dec LoRA + SASRec)
#    ensemble result = −Dec LoRA
########################################
CONFIG_NODEC="strat_lora_blenderbot_s3_sr_nodec"
DATA_NODEC="DATA/${INPUTTER}.${CONFIG_NODEC}"
run_full_pipeline "$CONFIG_NODEC" "$DATA_NODEC" "0.05" "nodec_p005"

########################################
# 3. −(Dec LoRA + SS) (p=0.0)
########################################
run_full_pipeline "$CONFIG_NODEC" "$DATA_NODEC" "0.0" "nodec_p000"

########################################
# Summary
########################################
echo ""
echo "============================================"
echo "All ablation experiments complete!"
echo "============================================"
echo ""
echo "Results locations:"
echo "  −Enc LoRA:           DATA/${INPUTTER}.${CONFIG_NOENC}/"
echo "  −Dec LoRA:           DATA/${INPUTTER}.${CONFIG_NODEC}/ (p=0.05 ensemble)"
echo "  −(Dec LoRA+SASRec):  DATA/${INPUTTER}.${CONFIG_NODEC}/ (p=0.05 probe-only)"
echo "  −(Dec LoRA+SS):      DATA/${INPUTTER}.${CONFIG_NODEC}/ (p=0.0 ensemble)"
echo "  −(SS+SASRec):        already exists (B p=0.0 enc probe-only)"
echo ""
echo "Check metric.json in each res_* directory for results."
