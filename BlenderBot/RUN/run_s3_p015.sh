#!/bin/bash
# s3 p=0.15: A+B backbone train → enc/dec probe → A+B infer → B SASRec ensemble (enc+dec)
set -e
cd "$(dirname "$0")/.."

COMMON_TRAIN="--seed 13 --max_input_length 160 --max_decoder_input_length 40 \
    --eval_input_file ../_reformat/valid.txt \
    --train_batch_size 16 --gradient_accumulation_steps 1 \
    --eval_batch_size 16 --learning_rate 5e-5 --warmup_steps 100 \
    --num_epochs 5 --fp16 false --pbar true --probe_lr 1e-3"

COMMON_INFER="--seed 0 --fp16 false --max_input_length 160 --max_decoder_input_length 40 \
    --max_length 40 --min_length 10 --infer_batch_size 16 \
    --infer_input_file ../_reformat/test.txt \
    --temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 \
    --repetition_penalty 1.03 --no_repeat_ngram_size 0"

DATA_A="DATA/strat_lora_seq.strat_lora_blenderbot_s3_seq"
DATA_B="DATA/strat_lora.strat_lora_blenderbot_s3_sr"

SASREC_VALID_PROBS="DATA/sasrec/valid_probs.json"
SASREC_TEST_PROBS="DATA/sasrec/test_probs.json"

SS_P=0.15

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

run_probe_and_infer() {
    local DIR="$1" CFG_BASE="$2" INP="$3"
    local BEST_EP=$(find_best_epoch "$DIR")
    echo "[probe+infer] $DIR (best=$BEST_EP)"

    for PROBE in enc dec; do
        local SUFFIX
        if [ "$PROBE" = "enc" ]; then SUFFIX="encoder"; else SUFFIX="decoder"; fi

        echo "  ${SUFFIX} probe (baseline probe method)..."
        CUDA_VISIBLE_DEVICES=0 python train.py \
            --config_name "${CFG_BASE}_${PROBE}" \
            --inputter_name "$INP" \
            $COMMON_TRAIN \
            --probe_extra_epochs 100 \
            --probe_only_from "$DIR" \
            --probe_save_suffix "_${SUFFIX}" \
            --probe_batch_size 256 \
            --probe_select_by acc

        echo "  ${SUFFIX} inference..."
        CUDA_VISIBLE_DEVICES=0 python infer.py \
            --config_name "${CFG_BASE}_${PROBE}" \
            --inputter_name "$INP" \
            $COMMON_INFER \
            --load_checkpoint "${DIR}/epoch-${BEST_EP}_probe_${SUFFIX}.bin"
    done
}

########################################
# Part 1: A condition (seq)
########################################
echo "============================================"
echo "ss_p=$SS_P — A condition (seq)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config_name strat_lora_blenderbot_s3_seq \
    --inputter_name strat_lora_seq \
    $COMMON_TRAIN \
    --use_scheduled_sampling true \
    --ss_schedule_type step --ss_warmup_epochs 1 \
    --ss_p_sample $SS_P --ss_p_random 0.0

DIR_A=$(ls -dt "$DATA_A"/20* 2>/dev/null | head -1)
echo "A dir: $DIR_A"
run_probe_and_infer "$DIR_A" "strat_lora_blenderbot_s3_seq" "strat_lora_seq"

########################################
# Part 2: B condition (sr)
########################################
echo "============================================"
echo "ss_p=$SS_P — B condition (sr)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config_name strat_lora_blenderbot_s3_sr \
    --inputter_name strat_lora \
    $COMMON_TRAIN \
    --use_scheduled_sampling true \
    --ss_schedule_type step --ss_warmup_epochs 1 \
    --ss_p_sample $SS_P --ss_p_random 0.0

DIR_B=$(ls -dt "$DATA_B"/20* 2>/dev/null | head -1)
echo "B dir: $DIR_B"
run_probe_and_infer "$DIR_B" "strat_lora_blenderbot_s3_sr" "strat_lora"

########################################
# Part 3: B SASRec ensemble (enc + dec)
########################################
echo "============================================"
echo "ss_p=$SS_P — B SASRec ensemble"
echo "============================================"

BEST_EP_B=$(find_best_epoch "$DIR_B")
mkdir -p DATA/ensemble_s3

for PROBE_TYPE in encoder decoder; do
    CKPT="${DIR_B}/epoch-${BEST_EP_B}_probe_${PROBE_TYPE}.bin"
    if [ "$PROBE_TYPE" = "encoder" ]; then
        CFG_ENS="strat_lora_blenderbot_s3_sr_enc"
    else
        CFG_ENS="strat_lora_blenderbot_s3_sr_dec"
    fi

    echo ""
    echo "  ${PROBE_TYPE} probe + SASRec ensemble..."

    # Valid set inference for grid search
    CUDA_VISIBLE_DEVICES=0 python infer.py \
        --config_name "$CFG_ENS" \
        --inputter_name strat_lora \
        --seed 0 --fp16 false --max_input_length 160 --max_decoder_input_length 40 \
        --max_length 40 --min_length 10 --infer_batch_size 16 \
        --infer_input_file ../_reformat/valid.txt \
        --temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 \
        --repetition_penalty 1.03 --no_repeat_ngram_size 0 \
        --load_checkpoint "$CKPT"

    VALID_RES_DIR=$(ls -dt "${DIR_B}/res_epoch-${BEST_EP_B}_probe_${PROBE_TYPE}.bin_valid"* 2>/dev/null | head -1)
    if [ -z "$VALID_RES_DIR" ]; then
        echo "  ERROR: Valid result not found for ${PROBE_TYPE}, skipping"
        continue
    fi

    # Grid search
    GRID_OUTPUT="DATA/ensemble_s3/grid_search_${PROBE_TYPE}_p0_15.json"
    python -m ensemble.grid_search_s3 \
        --probe_probs_file "${VALID_RES_DIR}/gen.json" \
        --sasrec_probs_file "$SASREC_VALID_PROBS" \
        --output_file "$GRID_OUTPUT"

    BEST_ALPHA=$(python3 -c "import json; d=json.load(open('$GRID_OUTPUT')); print(d['best']['alpha'])")
    echo "  Best alpha (${PROBE_TYPE}): $BEST_ALPHA"

    # Ensemble test inference
    CUDA_VISIBLE_DEVICES=0 python infer_ensemble_s3.py \
        --config_name "$CFG_ENS" \
        --inputter_name strat_lora \
        --load_checkpoint "$CKPT" \
        --sasrec_probs_file "$SASREC_TEST_PROBS" \
        --ensemble_alpha "$BEST_ALPHA" \
        --seed 0 --fp16 false \
        --max_input_length 160 --max_decoder_input_length 40 \
        --max_length 40 --min_length 10 --infer_batch_size 1 \
        --infer_input_file ../_reformat/test.txt \
        --temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 \
        --repetition_penalty 1.03 --no_repeat_ngram_size 0

    echo "  ${PROBE_TYPE} ensemble done."
done

echo ""
echo "============================================"
echo "p=0.15 all tasks complete!"
echo "============================================"
