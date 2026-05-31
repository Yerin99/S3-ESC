#!/bin/bash
# s3 Ablation: -LoRA condition (B p=0.05)
# Base strat model (no LoRA) with joint probe + SS p=0.05 + SASRec ensemble
# Matches Full system except LoRA expert routing → strategy prefix token instead
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

CONFIG="strat_blenderbot_s3_nolora_enc"
INPUTTER="strat_lora"
DATA_DIR="DATA/${INPUTTER}.${CONFIG}"

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

########################################
# Step 1: Backbone training (SS p=0.05)
########################################
echo "============================================"
echo "Step 1: Backbone training (SS p=0.05)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config_name "$CONFIG" \
    --inputter_name "$INPUTTER" \
    $COMMON_TRAIN \
    --use_scheduled_sampling true \
    --ss_schedule_type step --ss_warmup_epochs 1 \
    --ss_p_sample 0.05 --ss_p_random 0.0

DIR=$(ls -dt "$DATA_DIR"/20* 2>/dev/null | head -1)
BEST_EP=$(find_best_epoch "$DIR")
echo "Training dir: $DIR (best_ep=$BEST_EP)"

########################################
# Step 2: Encoder probe (baseline probe method)
########################################
echo "============================================"
echo "Step 2: Encoder probe training (baseline probe method, 100 epochs)"
echo "============================================"

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
echo "Probe checkpoint: $CKPT"

########################################
# Step 3: Probe-only inference (test)
########################################
echo "============================================"
echo "Step 3: Probe-only inference (test)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --config_name "$CONFIG" \
    --inputter_name "$INPUTTER" \
    $COMMON_INFER \
    --infer_input_file ../_reformat/test.txt \
    --load_checkpoint "$CKPT"

########################################
# Step 4: Valid inference (for grid search)
########################################
echo "============================================"
echo "Step 4: Probe inference on valid (for grid search)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --config_name "$CONFIG" \
    --inputter_name "$INPUTTER" \
    $COMMON_INFER \
    --infer_input_file ../_reformat/valid.txt \
    --load_checkpoint "$CKPT"

########################################
# Step 5: Grid search
########################################
echo "============================================"
echo "Step 5: SASRec ensemble grid search"
echo "============================================"

VALID_RES_DIR=$(ls -dt "${DIR}/res_epoch-${BEST_EP}_probe_encoder.bin_valid"* 2>/dev/null | head -1)
if [ -z "$VALID_RES_DIR" ]; then
    echo "ERROR: Valid result not found"
    exit 1
fi

mkdir -p DATA/ensemble_s3
GRID_OUTPUT="DATA/ensemble_s3/grid_search_nolora.json"
python -m ensemble.grid_search_s3 \
    --probe_probs_file "${VALID_RES_DIR}/gen.json" \
    --sasrec_probs_file "$SASREC_VALID_PROBS" \
    --output_file "$GRID_OUTPUT"

BEST_ALPHA=$(python3 -c "import json; d=json.load(open('$GRID_OUTPUT')); print(d['best']['alpha'])")
echo "Best alpha: $BEST_ALPHA"

########################################
# Step 6: SASRec ensemble inference (test)
########################################
echo "============================================"
echo "Step 6: SASRec ensemble inference (alpha=$BEST_ALPHA)"
echo "============================================"

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
echo "============================================"
echo "-LoRA ablation complete!"
echo "============================================"
