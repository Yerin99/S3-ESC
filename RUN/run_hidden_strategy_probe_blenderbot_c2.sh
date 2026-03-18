#!/bin/bash
set -e

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

CKPT_DIR="./DATA/strat_lora_seq.strat_lora_blenderbot_c2"
PROBE_DATA_DIR="./DATA/hidden_strategy_probe_blenderbot_c2"
GPU_ID=${GPU_ID:-0}

RUN_DIR=$(ls -td ${CKPT_DIR}/*/ | head -1)
BEST_EPOCH=$(python -c "
import csv
with open('${RUN_DIR}eval_log.csv') as f:
    rows = list(csv.DictReader(f))
best = min(rows, key=lambda r: float(r['freq_loss']))
print('epoch-' + best['epoch'])
")
CKPT="${RUN_DIR}${BEST_EPOCH}.bin"

echo "Using checkpoint: ${CKPT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python hidden_strategy_probe/extract_hidden_states.py \
  --config_name strat_lora_blenderbot_c2 \
  --inputter_name strat_lora_seq \
  --checkpoint "${CKPT}" \
  --output_dir "${PROBE_DATA_DIR}" \
  --max_input_length 160 \
  --max_decoder_input_length 40 \
  --batch_size 64

python hidden_strategy_probe/train_probe.py \
  --data_dir "${PROBE_DATA_DIR}" \
  --probe_target decoder_last
