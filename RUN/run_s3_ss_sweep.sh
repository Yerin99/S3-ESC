#!/bin/bash
# s3 SS p-value sweep (Phase 2): seed 13, A5000 only, baseline probe
# A (seq) + B (sr) conditions
# ss_p = 0.0, 0.05, 0.1, 0.2 x probe = encoder
# All training on GPU 0 (A5000) — sequential
set -e
cd "$(dirname "$0")/.."

COMMON_TRAIN="--seed 13 --max_input_length 160 --max_decoder_input_length 40 \
    --eval_input_file ./_reformat/valid.txt \
    --train_batch_size 16 --gradient_accumulation_steps 1 \
    --eval_batch_size 16 --learning_rate 5e-5 --warmup_steps 100 \
    --num_epochs 5 --fp16 false --pbar true --probe_lr 1e-3"

COMMON_INFER="--seed 0 --fp16 false --max_input_length 160 --max_decoder_input_length 40 \
    --max_length 40 --min_length 10 --infer_batch_size 16 \
    --infer_input_file ./_reformat/test.txt \
    --temperature 0.7 --top_k 30 --top_p 0.9 --num_beams 1 \
    --repetition_penalty 1.03 --no_repeat_ngram_size 0"

DATA_A="DATA/strat_lora_seq.strat_lora_blenderbot_s3_seq"
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

run_probe_and_infer() {
    # args: DIR CONFIG_BASE INPUTTER
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

for SS_P in 0.0 0.05 0.1 0.2; do
    echo "============================================"
    echo "ss_p=$SS_P — A condition (seq, GPU 0)"
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

    echo "============================================"
    echo "ss_p=$SS_P — B condition (sr, GPU 0)"
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

    # Save dir names for summary
    eval "DIR_A_${SS_P//./_}=$DIR_A"
    eval "DIR_B_${SS_P//./_}=$DIR_B"

    echo "ss_p=$SS_P done."
    echo ""
done

echo "============================================"
echo "All results"
echo "============================================"

for SS_P in 0.0 0.05 0.1 0.2; do
    SS_KEY="${SS_P//./_}"
    eval "DA=\$DIR_A_${SS_KEY}"
    eval "DB=\$DIR_B_${SS_KEY}"

    for COND_DIR in "A:$DA" "B:$DB"; do
        COND="${COND_DIR%%:*}"
        DIR="${COND_DIR#*:}"
        for SUFFIX in encoder decoder; do
            METRIC_DIR=$(ls -d "${DIR}/res_epoch-"*"_probe_${SUFFIX}.bin_test_"* 2>/dev/null | head -1)
            if [ -n "$METRIC_DIR" ] && [ -f "$METRIC_DIR/metric.json" ]; then
                python3 -c "
import json
d = json.load(open('${METRIC_DIR}/metric.json'))
acc = d.get('acc_cls_strat_id', 'N/A')
if isinstance(acc, float):
    print(f'p=${SS_P} ${COND} ${SUFFIX}: Acc={acc:.4f} BLEU-1={d[\"bleu-1\"]:.2f} F1={d[\"f1\"]:.2f} ROUGE-L={d[\"rouge-l\"]:.2f} PPL={d[\"perplexity\"]:.2f}')
else:
    print(f'p=${SS_P} ${COND} ${SUFFIX}: BLEU-1={d[\"bleu-1\"]:.2f} F1={d[\"f1\"]:.2f} ROUGE-L={d[\"rouge-l\"]:.2f} PPL={d[\"perplexity\"]:.2f}')
"
            fi
        done
    done
done

echo ""
echo "Reference (baseline C-2): Acc=0.3242 BLEU-1=19.63 F1=19.81 ROUGE-L=16.57 PPL=15.98"
echo "Reference (008 ensemble): Acc=0.3524 BLEU-1=19.85"
echo "All done!"
