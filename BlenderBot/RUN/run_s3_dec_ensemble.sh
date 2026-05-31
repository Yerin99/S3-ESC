#!/bin/bash
# B condition: Decoder probe + SASRec ensemble
# Same as Part 1 of post_sweep but using decoder probe instead of encoder probe
set -e
cd "$(dirname "$0")/.."

DATA_B="DATA/strat_lora.strat_lora_blenderbot_s3_sr"

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

B_DIRS=($(ls -dt "$DATA_B"/20*/ 2>/dev/null))
NUM_B=${#B_DIRS[@]}
DIR_B_00="${B_DIRS[$((NUM_B-1))]}"
DIR_B_005="${B_DIRS[$((NUM_B-2))]}"
DIR_B_01="${B_DIRS[$((NUM_B-3))]}"
DIR_B_02="${B_DIRS[$((NUM_B-4))]}"

echo "B dirs: p=0.0=$DIR_B_00  p=0.05=$DIR_B_005  p=0.1=$DIR_B_01  p=0.2=$DIR_B_02"

mkdir -p DATA/ensemble_s3

for P_LABEL_DIR in "0.0:$DIR_B_00" "0.05:$DIR_B_005" "0.1:$DIR_B_01" "0.2:$DIR_B_02"; do
    IFS=':' read -r P DIR <<< "$P_LABEL_DIR"
    BEST_EP=$(find_best_epoch "$DIR")
    CKPT="${DIR}/epoch-${BEST_EP}_probe_decoder.bin"

    echo ""
    echo "============================================"
    echo "p=$P  B decoder+SASRec ensemble  (best_ep=$BEST_EP)"
    echo "============================================"

    # Step 1: Decoder probe inference on VALID set (for grid search)
    echo "  Decoder probe inference on valid set..."
    CUDA_VISIBLE_DEVICES=0 python infer.py \
        --config_name strat_lora_blenderbot_s3_sr_dec \
        --inputter_name strat_lora \
        --seed 0 --fp16 false --max_input_length 160 --max_decoder_input_length 40 \
        --max_length 40 --min_length 10 --infer_batch_size 16 \
        --infer_input_file ../_reformat/valid.txt \
        --temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 \
        --repetition_penalty 1.03 --no_repeat_ngram_size 0 \
        --load_checkpoint "$CKPT"

    VALID_RES_DIR=$(ls -dt "${DIR}/res_epoch-${BEST_EP}_probe_decoder.bin_valid"* 2>/dev/null | head -1)
    if [ -z "$VALID_RES_DIR" ]; then
        echo "ERROR: Valid result not found for p=$P, skipping"
        continue
    fi

    # Step 2: Grid search for optimal alpha
    echo "  Grid search for optimal alpha..."
    P_SAFE="${P//./_}"
    GRID_OUTPUT="DATA/ensemble_s3/grid_search_dec_p${P_SAFE}.json"
    python -m ensemble.grid_search_s3 \
        --probe_probs_file "${VALID_RES_DIR}/gen.json" \
        --sasrec_probs_file "$SASREC_VALID_PROBS" \
        --output_file "$GRID_OUTPUT"

    BEST_ALPHA=$(python3 -c "import json; d=json.load(open('$GRID_OUTPUT')); print(d['best']['alpha'])")
    echo "  Best alpha for p=$P: $BEST_ALPHA"

    # Step 3: Decoder probe + SASRec ensemble inference on TEST set
    echo "  Ensemble inference (alpha=$BEST_ALPHA)..."
    CUDA_VISIBLE_DEVICES=0 python infer_ensemble_s3.py \
        --config_name strat_lora_blenderbot_s3_sr_dec \
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

    echo "  p=$P decoder ensemble done."
done

echo ""
echo "============================================"
echo "Decoder probe + SASRec ensemble complete!"
echo "============================================"
