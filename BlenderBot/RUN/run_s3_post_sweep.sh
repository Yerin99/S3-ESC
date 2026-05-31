#!/bin/bash
# Post-sweep pipeline:
# 1. SASRec ensemble for B conditions (all p values)
# 2. baseline probe extraction + probe comparison on Phase 5 p=0.0 A backbone
set -e
cd "$(dirname "$0")/.."

DATA_A="DATA/strat_lora_seq.strat_lora_blenderbot_s3_seq"
DATA_B="DATA/strat_lora.strat_lora_blenderbot_s3_sr"

SASREC_MODEL="DATA/sasrec/2026-02-04_142116/best_model.pt"
SASREC_VALID_PROBS="DATA/sasrec/valid_probs.json"
SASREC_TEST_PROBS="DATA/sasrec/test_probs.json"

COMMON_INFER="--seed 0 --fp16 false --max_input_length 160 --max_decoder_input_length 40 \
    --max_length 40 --min_length 10 --infer_batch_size 16 \
    --infer_input_file ../_reformat/test.txt \
    --temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 \
    --repetition_penalty 1.03 --no_repeat_ngram_size 0"

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
# Part 1: SASRec Ensemble for B conditions
########################################
echo "============================================"
echo "Part 1: SASRec Ensemble for B conditions"
echo "============================================"

# Step 1: Dump SASRec probs (if not already done)
if [ ! -f "$SASREC_VALID_PROBS" ]; then
    echo "Dumping SASRec valid probs..."
    python sasrec_eval.py \
        --model_path "$SASREC_MODEL" \
        --eval_file ../_reformat/valid.txt \
        --output_file "$SASREC_VALID_PROBS"
else
    echo "SASRec valid probs already exist"
fi

if [ ! -f "$SASREC_TEST_PROBS" ]; then
    echo "Dumping SASRec test probs..."
    python sasrec_eval.py \
        --model_path "$SASREC_MODEL" \
        --eval_file ../_reformat/test.txt \
        --output_file "$SASREC_TEST_PROBS"
else
    echo "SASRec test probs already exist"
fi

# Find B condition dirs (sorted by date, match with p values from sweep order)
B_DIRS=($(ls -dt "$DATA_B"/20*/ 2>/dev/null))
# Sweep runs p=0.0, 0.05, 0.1, 0.2 sequentially, so newest = 0.2, oldest = 0.0
NUM_B=${#B_DIRS[@]}
if [ "$NUM_B" -lt 4 ]; then
    echo "ERROR: Expected 4 B dirs, found $NUM_B"
    exit 1
fi
# Reverse order: oldest first = p=0.0, 0.05, 0.1, 0.2
DIR_B_00="${B_DIRS[$((NUM_B-1))]}"
DIR_B_005="${B_DIRS[$((NUM_B-2))]}"
DIR_B_01="${B_DIRS[$((NUM_B-3))]}"
DIR_B_02="${B_DIRS[$((NUM_B-4))]}"

echo "B dirs: p=0.0=$DIR_B_00  p=0.05=$DIR_B_005  p=0.1=$DIR_B_01  p=0.2=$DIR_B_02"

mkdir -p DATA/ensemble_s3

for P_LABEL_DIR in "0.0:$DIR_B_00" "0.05:$DIR_B_005" "0.1:$DIR_B_01" "0.2:$DIR_B_02"; do
    IFS=':' read -r P DIR <<< "$P_LABEL_DIR"
    BEST_EP=$(find_best_epoch "$DIR")
    CKPT="${DIR}/epoch-${BEST_EP}_probe_encoder.bin"

    echo ""
    echo "============================================"
    echo "p=$P  B encoder ensemble  (best_ep=$BEST_EP)"
    echo "============================================"

    # Probe inference on VALID set (for grid search)
    echo "  Probe inference on valid set..."
    CUDA_VISIBLE_DEVICES=0 python infer.py \
        --config_name strat_lora_blenderbot_s3_sr_enc \
        --inputter_name strat_lora \
        --seed 0 --fp16 false --max_input_length 160 --max_decoder_input_length 40 \
        --max_length 40 --min_length 10 --infer_batch_size 16 \
        --infer_input_file ../_reformat/valid.txt \
        --temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 \
        --repetition_penalty 1.03 --no_repeat_ngram_size 0 \
        --load_checkpoint "$CKPT"

    VALID_RES_DIR=$(ls -dt "${DIR}/res_epoch-${BEST_EP}_probe_encoder.bin_valid"* 2>/dev/null | head -1)
    if [ -z "$VALID_RES_DIR" ]; then
        echo "ERROR: Valid result not found for p=$P, skipping"
        continue
    fi

    # Grid search
    echo "  Grid search for optimal alpha..."
    P_SAFE="${P//./_}"
    GRID_OUTPUT="DATA/ensemble_s3/grid_search_p${P_SAFE}.json"
    python -m ensemble.grid_search_s3 \
        --probe_probs_file "${VALID_RES_DIR}/gen.json" \
        --sasrec_probs_file "$SASREC_VALID_PROBS" \
        --output_file "$GRID_OUTPUT"

    BEST_ALPHA=$(python3 -c "import json; d=json.load(open('$GRID_OUTPUT')); print(d['best']['alpha'])")
    echo "  Best alpha for p=$P: $BEST_ALPHA"

    # Ensemble inference on TEST set
    echo "  Ensemble inference (alpha=$BEST_ALPHA)..."
    CUDA_VISIBLE_DEVICES=0 python infer_ensemble_s3.py \
        --config_name strat_lora_blenderbot_s3_sr_enc \
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

    echo "  p=$P ensemble done."
done

echo ""
echo "============================================"
echo "Part 1 complete. Skipping Part 2 (needs fix for strict load)."
echo "============================================"
exit 0

########################################
# Part 2: baseline probe extraction + probe comparison
########################################
echo ""
echo "============================================"
echo "Part 2: baseline probe extraction + probe on Phase 5 p=0.0 A backbone"
echo "============================================"

# Find p=0.0 A dir (oldest A dir)
A_DIRS=($(ls -dt "$DATA_A"/20*/ 2>/dev/null))
NUM_A=${#A_DIRS[@]}
DIR_A_00="${A_DIRS[$((NUM_A-1))]}"
BEST_EP_A=$(find_best_epoch "$DIR_A_00")
CKPT_A="${DIR_A_00}/epoch-${BEST_EP_A}.bin"

echo "p=0.0 A dir: $DIR_A_00 (best_ep=$BEST_EP_A)"

CONFIG_PROBE_BASELINE="strat_lora_blenderbot_c2"
# Check if baseline probe config exists; if not, use s3_seq (backbone is identical)
if [ ! -f "CONFIG/${CONFIG_PROBE_BASELINE}.json" ]; then
    echo "baseline probe config not found, creating temporary config..."
    CONFIG_PROBE_BASELINE="strat_lora_blenderbot_probe_baseline_test"
    python3 -c "
import json
# Use baseline model class
d = json.load(open('CONFIG/strat_lora_blenderbot_s3_seq.json'))
d['model_name'] = 'strat_blenderbot_small_lora_strat'
d.pop('joint_probe', None)
d.pop('joint_probe_source', None)
json.dump(d, open('CONFIG/${CONFIG_PROBE_BASELINE}.json', 'w'), indent=2)
print('Created temporary baseline probe config')
"
fi

PROBE_BASELINE_DIR="DATA/hidden_strategy_probe_phase5_baseline_comparison"
echo "Extracting hidden states with baseline probe method..."
CUDA_VISIBLE_DEVICES=0 python hidden_strategy_probe/extract_hidden_states.py \
    --config_name "$CONFIG_PROBE_BASELINE" \
    --inputter_name strat_lora_seq \
    --checkpoint "$CKPT_A" \
    --output_dir "$PROBE_BASELINE_DIR" \
    --max_input_length 160 \
    --max_decoder_input_length 40 \
    --batch_size 64

echo "Training probe with baseline probe method..."
python hidden_strategy_probe/train_probe.py \
    --data_dir "$PROBE_BASELINE_DIR" \
    --probe_target decoder_last \
    --seed 13

echo "baseline probe results:"
python3 -c "
import json
d = json.load(open('${PROBE_BASELINE_DIR}/results_decoder_last.json'))
print(f'baseline probe method: test_acc={d[\"test\"][\"accuracy\"]:.4f} valid_acc={d[\"best_valid_acc\"]:.4f}')
"

echo ""
echo "Comparison:"
echo "  baseline probe method probe (above) vs s3 probe_only_from decoder (from sweep results)"
echo "  If baseline probe method gives ~0.3242, the data pipeline difference is confirmed."

# Also run inference with baseline probe
echo ""
echo "Running inference with baseline probe method probe..."
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --config_name strat_lora_blenderbot_s3_seq_dec \
    --inputter_name strat_lora_seq \
    $COMMON_INFER \
    --probe_weights_path "${PROBE_BASELINE_DIR}/probe_decoder_last.pt" \
    --probe_source decoder

echo ""
echo "============================================"
echo "All post-sweep tasks complete!"
echo "============================================"
