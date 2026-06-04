# KEMI + S3-ESC

S3-ESC method applied on the KEMI baseline (knowledge-enhanced mixed-strategy, BlenderBot-small backbone).

**Prerequisites**: complete the *Shared Assets* setup in the [top-level README](../README.md) first (BlenderBot weights, ESConv preprocessing). KEMI additionally needs SBERT-retrieved knowledge for the ESConv data (the `esconv`/`sbert` data/knowledge name used by the scripts). All commands assume you've `cd KEMI` and activated the conda env.

## Environment

```bash
conda env create -f ../env-esc2021.yml
conda activate esc-2021
```

## Data Preprocessing

KEMI uses knowledge-augmented inputs (`--data_name esconv --knowledge_name sbert`):

```bash
bash RUN/prepare_strat.sh
```

## Training

KEMI + S3-ESC main model:

```bash
bash RUN/train_s3.sh            # optional: bash RUN/train_s3.sh <GPU_ID>
```

For the original KEMI baseline (no S3):

```bash
bash RUN/train_strat.sh
```

## Inference

```bash
# KEMI + S3-ESC
CHECKPOINT=DATA/{...}/epoch-{n}.bin bash RUN/infer_s3.sh <GPU_ID> <CHECKPOINT>

# Probe + SASRec ensemble (set SASREC_PROBS or it defaults to ../data/sasrec/test_probs.json)
bash RUN/infer_ensemble_s3.sh <GPU_ID> <CHECKPOINT> <ALPHA>
```

Main-table metrics (PPL, BLEU, ROUGE-L, Distinct, F1, Strategy-Acc) are computed from the shared `../metric` package. (KEMI's original RoBERTa-based mixed-initiative metric is not part of our main table and has been removed.)

## Folder Layout

```
KEMI/
├── CONFIG/                  # KEMI + S3 JSON configs
├── RUN/                     # Shell scripts (prepare/train/infer)
├── models/                  # KEMI models (BlenderBot backbone) + LoRA expert layers
├── inputters/               # Knowledge-augmented data processing
├── ensemble/                # KEMI + SASREC ensemble grid search
├── sasrec/                  # KEMI-specific SASRec implementation
├── utils/                   # Training utilities
└── train.py / train_s3.py / infer.py / infer_s3.py / ...   # entrypoints
```

Shared assets at the repo root are accessed via `../`: `../_reformat/`, `../Blenderbot_small-90M/`, `../metric/`, `../data/sasrec/`, `../apex/`.
