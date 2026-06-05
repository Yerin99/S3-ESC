#!/bin/bash
# =============================================================================
# GLHG + S3-ESC main pipeline (Gradient-Isolated Probe + SASRec ensemble)
#
# Full method, end to end, per ss_p value:
#   Stage 0: prepare data.pkl
#   Stage 1: backbone training (LoRA + encoder probe + scheduled sampling)
#            → config glhg_s3 (joint_probe_source: encoder)
#   Stage 2: probe extra-training for BOTH probe sources
#            → enc: glhg_s3_enc (ctx encoder mean-pool), dec: glhg_s3_dec (decoder BOS)
#   Stage 3: probe inference on valid + test (each probe source)
#            → grid_search (valid) selects best ensemble alpha
#   Stage 4: ensemble inference on test (probe + SASRec, best alpha) — required final step
#
# Reported setting: single best ss_p (override with P_SAMPLES / LORA_LR env).
# Full sweep: P_SAMPLES="0.0 0.05 0.1 0.15 0.2" bash RUN/pipeline_glhg_s3.sh
#
# Prerequisites: ../data/sasrec/{valid,test}_probs.json
# Usage: bash RUN/pipeline_glhg_s3.sh [GPU_ID]
# =============================================================================
set -e
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate esc-2021

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

# --- Configuration (best reported; override via env) ---
# Reported best: ss_p=0.1, no separate lora_lr (LoRA trains at the global lr 3e-5),
# probe=enc, ensemble alpha~0.5 (grid-searched on valid).
P_SAMPLES=(${P_SAMPLES:-0.1})
LORA_LR=${LORA_LR:-}
SASREC_DIR=${SASREC_DIR:-../data/sasrec}
DATA_GLOB="./DATA/glhg_s3.glhg_s3"
VALID_FILE=../_reformat/valid.txt
TEST_FILE=../_reformat/test.txt

COMMON_TRAIN="
  --seed 42 --max_input_length 128 --max_decoder_input_length 50
  --eval_input_file ../_reformat/valid.txt
  --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 8
  --learning_rate 3e-5 --warmup_steps 100 --num_epochs 5
  --fp16 false --pbar true --probe_lr 1e-3
"
COMMON_INFER="
  --seed 0 --fp16 false --max_input_length 128 --max_decoder_input_length 50
  --max_length 40 --min_length 5 --infer_batch_size 1
  --temperature 0.7 --top_k 0 --top_p 0.9 --num_beams 1
  --length_penalty 1.0 --repetition_penalty 1.0 --no_repeat_ngram_size 3
"

# --- Helpers ---
find_best_epoch() {
    python3 -c "
import csv
best_ep, best_ppl = 0, float('inf')
for row in csv.DictReader(open('$1/eval_log.csv')):
    ppl, ep = float(row['freq_ppl']), int(row['epoch'])
    if ppl < best_ppl: best_ppl, best_ep = ppl, ep
print(best_ep)"
}
find_existing_backbone() {
    python3 -c "
import json, os, glob
want_ss, want_lr = float('$1'), '${2:-None}'
for d in sorted(glob.glob('${DATA_GLOB}/20*'), reverse=True):
    af = os.path.join(d, 'args.json')
    if not os.path.isfile(af) or not os.path.isfile(os.path.join(d,'eval_log.csv')): continue
    a = json.load(open(af))
    if a.get('probe_only_from') is not None: continue
    if abs(float(a.get('ss_p_sample', -1)) - want_ss) > 1e-6: continue
    lr = a.get('lora_lr', None)
    if want_lr == 'None':
        if lr is not None: continue
    else:
        if lr is None or abs(float(lr) - float(want_lr)) > 1e-8: continue
    if glob.glob(os.path.join(d, 'epoch-*.bin')): print(d); break"
}
get_best_alpha() { python3 -c "import json; print(json.load(open('$1'))['best']['alpha'])"; }
infer_res_dir() {
    echo "$1/res_$2_$3_k.0_p.0.9_b.1_t.0.7_lp.1.0_rp.1.0_ng.3"
}

LORA_FLAG=""; [ -n "$LORA_LR" ] && LORA_FLAG="--lora_lr $LORA_LR"

# --- Stage 0: prepare ---
bash RUN/prepare_glhg_s3.sh || true

for SS_P in "${P_SAMPLES[@]}"; do
    echo "============================================================"
    echo "  ss_p_sample = $SS_P  (lora_lr=${LORA_LR:-none})"
    echo "============================================================"

    # --- Stage 1: backbone ---
    DIR=$(find_existing_backbone "$SS_P" "${LORA_LR:-None}")
    if [ -n "$DIR" ]; then
        echo "[Stage 1] reuse backbone: $DIR"
    else
        echo "[Stage 1] backbone training (ss_p=${SS_P})..."
        python3 train_s3.py --config_name glhg_s3 --inputter_name glhg_s3 \
            $COMMON_TRAIN $LORA_FLAG \
            --use_scheduled_sampling true --ss_schedule_type step \
            --ss_warmup_epochs 1 --ss_p_sample "$SS_P" --ss_p_random 0.0
        DIR=$(find_existing_backbone "$SS_P" "${LORA_LR:-None}")
        [ -z "$DIR" ] && { echo "ERROR: no backbone for ss_p=${SS_P}"; exit 1; }
    fi
    BEST_EP=$(find_best_epoch "$DIR")
    echo "  Dir: $DIR  |  Best epoch: $BEST_EP"

    # --- Stage 2-4: per probe source ---
    for PROBE in enc dec; do
        PROBE_CKPT="${DIR}/epoch-${BEST_EP}_probe_${PROBE}.bin"
        PROBE_NAME="epoch-${BEST_EP}_probe_${PROBE}.bin"

        if [ -f "$PROBE_CKPT" ]; then
            echo "  [${PROBE}] Stage 2: reuse probe"
        else
            echo "  [${PROBE}] Stage 2: probe extra-training..."
            python3 train_s3.py --config_name glhg_s3_${PROBE} --inputter_name glhg_s3 \
                $COMMON_TRAIN $LORA_FLAG \
                --probe_extra_epochs 100 --probe_only_from "$DIR" \
                --probe_save_suffix "_${PROBE}" --probe_batch_size 256 --probe_select_by acc
        fi

        echo "  [${PROBE}] Stage 3a: probe inference (valid)..."
        python3 infer_s3.py --config_name glhg_s3_${PROBE} --inputter_name glhg_s3 \
            --load_checkpoint "$PROBE_CKPT" $COMMON_INFER --infer_input_file "$VALID_FILE"
        PROBE_VALID_GEN="$(infer_res_dir "$DIR" "$PROBE_NAME" valid)/gen.json"

        echo "  [${PROBE}] Stage 3b: grid search..."
        GRID_OUT="${DIR}/grid_search_ssp${SS_P}_${PROBE}.json"
        python3 -m ensemble.grid_search_s3 \
            --probe_probs_file "$PROBE_VALID_GEN" \
            --sasrec_probs_file "${SASREC_DIR}/valid_probs.json" \
            --output_file "$GRID_OUT" --step 0.05
        BEST_ALPHA=$(get_best_alpha "$GRID_OUT")
        echo "  [${PROBE}] best alpha = $BEST_ALPHA"

        echo "  [${PROBE}] Stage 3c: probe inference (test)..."
        python3 infer_s3.py --config_name glhg_s3_${PROBE} --inputter_name glhg_s3 \
            --load_checkpoint "$PROBE_CKPT" $COMMON_INFER --infer_input_file "$TEST_FILE"

        echo "  [${PROBE}] Stage 4: ensemble inference (test, alpha=${BEST_ALPHA})..."
        python3 infer_ensemble_s3.py --config_name glhg_s3_${PROBE} --inputter_name glhg_s3 \
            --load_checkpoint "$PROBE_CKPT" \
            --sasrec_probs_file "${SASREC_DIR}/test_probs.json" \
            --ensemble_alpha "$BEST_ALPHA" $COMMON_INFER --infer_input_file "$TEST_FILE"
    done
    echo "ss_p=${SS_P} complete."
done
echo "Pipeline complete."
