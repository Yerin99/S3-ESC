#!/bin/bash
# s3 B condition: SASRec ensemble pipeline
# Step 1: Dump SASRec probs (valid + test) — once
# Step 2: Probe inference on valid set (per p value) — for grid search
# Step 3: Grid search optimal alpha (per p value)
# Step 4: Ensemble inference on test set (per p value)
set -e
cd "$(dirname "$0")/.."

SASREC_MODEL="DATA/sasrec/2026-02-04_142116/best_model.pt"
SASREC_VALID_PROBS="DATA/sasrec/valid_probs.json"
SASREC_TEST_PROBS="DATA/sasrec/test_probs.json"

COMMON_INFER="--seed 0 --fp16 false --max_input_length 160 --max_decoder_input_length 40 \
    --max_length 40 --min_length 10 --infer_batch_size 16 \
    --temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 \
    --repetition_penalty 1.03 --no_repeat_ngram_size 0"

DATA_B="DATA/strat_lora.strat_lora_blenderbot_s3_sr"

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

# ============ Step 1: SASRec probability dump ============
echo "============================================"
echo "Step 1: SASRec probability dump"
echo "============================================"

if [ ! -f "$SASREC_VALID_PROBS" ]; then
    echo "Dumping SASRec valid probs..."
    python sasrec_eval.py \
        --model_path "$SASREC_MODEL" \
        --eval_file ../_reformat/valid.txt \
        --output_file "$SASREC_VALID_PROBS"
else
    echo "SASRec valid probs already exist: $SASREC_VALID_PROBS"
fi

if [ ! -f "$SASREC_TEST_PROBS" ]; then
    echo "Dumping SASRec test probs..."
    python sasrec_eval.py \
        --model_path "$SASREC_MODEL" \
        --eval_file ../_reformat/test.txt \
        --output_file "$SASREC_TEST_PROBS"
else
    echo "SASRec test probs already exist: $SASREC_TEST_PROBS"
fi

# ============ B condition directories per p value ============
# p=0.1 (original)
DIR_B_01="$DATA_B/2026-03-05173803.5e-05.16.1gpu"
# p=0.0
DIR_B_00="$DATA_B/2026-03-05205445.5e-05.16.1gpu"
# p=0.05
DIR_B_005="$DATA_B/2026-03-06101258.5e-05.16.1gpu"
# p=0.2
DIR_B_02="$DATA_B/2026-03-06104501.5e-05.16.1gpu"

for P_LABEL_DIR in "0.0:$DIR_B_00" "0.05:$DIR_B_005" "0.1:$DIR_B_01" "0.2:$DIR_B_02"; do
    IFS=':' read -r P DIR <<< "$P_LABEL_DIR"
    BEST_EP=$(find_best_epoch "$DIR")
    CKPT="${DIR}/epoch-${BEST_EP}_probe_encoder.bin"

    echo ""
    echo "============================================"
    echo "p=$P  B encoder  (best_ep=$BEST_EP)"
    echo "============================================"

    # ============ Step 2: Probe inference on VALID set ============
    echo "Step 2: Probe inference on valid set..."
    CUDA_VISIBLE_DEVICES=0 python infer.py \
        --config_name strat_lora_blenderbot_s3_sr_enc \
        --inputter_name strat_lora \
        $COMMON_INFER \
        --infer_input_file ../_reformat/valid.txt \
        --load_checkpoint "$CKPT"

    # Find the valid result directory
    VALID_RES_DIR=$(ls -dt "${DIR}/res_epoch-${BEST_EP}_probe_encoder.bin_valid"* 2>/dev/null | head -1)
    if [ -z "$VALID_RES_DIR" ]; then
        echo "ERROR: Valid result directory not found for p=$P"
        continue
    fi
    echo "Valid results: $VALID_RES_DIR"

    # ============ Step 3: Grid search ============
    echo "Step 3: Grid search for optimal alpha..."
    GRID_OUTPUT="DATA/ensemble_s3/grid_search_p${P}.json"
    python -m ensemble.grid_search_s3 \
        --probe_probs_file "${VALID_RES_DIR}/gen.json" \
        --sasrec_probs_file "$SASREC_VALID_PROBS" \
        --output_file "$GRID_OUTPUT"

    # Extract best alpha
    BEST_ALPHA=$(python3 -c "import json; d=json.load(open('$GRID_OUTPUT')); print(d['best']['alpha'])")
    echo "Best alpha for p=$P: $BEST_ALPHA"

    # ============ Step 4: Ensemble inference on TEST set ============
    echo "Step 4: Ensemble inference on test set (alpha=$BEST_ALPHA)..."
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

    echo "p=$P done."
done

# ============ Collect Results ============
echo ""
echo "============================================"
echo "B Condition Ensemble Results"
echo "============================================"

for P_LABEL_DIR in "0.0:$DIR_B_00" "0.05:$DIR_B_005" "0.1:$DIR_B_01" "0.2:$DIR_B_02"; do
    IFS=':' read -r P DIR <<< "$P_LABEL_DIR"
    BEST_EP=$(find_best_epoch "$DIR")

    # Probe-only result (already exists from sweep)
    PROBE_DIR=$(ls -d "${DIR}/res_epoch-${BEST_EP}_probe_encoder.bin_test_"* 2>/dev/null | head -1)
    if [ -n "$PROBE_DIR" ] && [ -f "$PROBE_DIR/metric.json" ]; then
        python3 -c "
import json
d = json.load(open('${PROBE_DIR}/metric.json'))
print(f'p=${P} B probe-only: Acc={d[\"acc_cls_strat_id\"]:.4f} BLEU-1={d[\"bleu-1\"]:.2f} F1={d[\"f1\"]:.2f} ROUGE-L={d[\"rouge-l\"]:.2f} PPL={d[\"perplexity\"]:.2f}')
"
    fi

    # Ensemble result
    ENS_DIR=$(ls -d "${DIR}/res_ensembles3_epoch-${BEST_EP}_probe_encoder.bin_test"* 2>/dev/null | head -1)
    if [ -n "$ENS_DIR" ] && [ -f "$ENS_DIR/metric.json" ]; then
        python3 -c "
import json
d = json.load(open('${ENS_DIR}/metric.json'))
acc_ens = d.get('acc_strat_ensemble', 'N/A')
acc_probe = d.get('acc_strat_probe_only', 'N/A')
acc_sasrec = d.get('acc_strat_sasrec_only', 'N/A')
if isinstance(acc_ens, float):
    print(f'p=${P} B ensemble: Acc(ens)={acc_ens:.4f} Acc(probe)={acc_probe:.4f} Acc(sasrec)={acc_sasrec:.4f} BLEU-1={d[\"bleu-1\"]:.2f} F1={d[\"f1\"]:.2f} ROUGE-L={d[\"rouge-l\"]:.2f}')
else:
    print(f'p=${P} B ensemble: BLEU-1={d[\"bleu-1\"]:.2f} F1={d[\"f1\"]:.2f} ROUGE-L={d[\"rouge-l\"]:.2f}')
"
    fi
done

echo ""
echo "Reference - 008 best ensemble: Acc=0.3524 BLEU-1=19.85"
echo "All done!"
