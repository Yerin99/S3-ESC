# GLHG + S3-ESC

S3-ESC method applied on the GLHG baseline (Global-to-Local Hierarchical Graph network, BlenderBot-small backbone). Only the BlenderBot + S3 variant is included.

**Prerequisites**: complete the *Shared Assets* setup in the [top-level README](../README.md) first (BlenderBot weights, ESConv preprocessing). GLHG additionally requires **COMET-ATOMIC-2020-BART** for graph construction (see [External models](#external-models)). All commands assume you've `cd GLHG` and activated the conda env.

## Environment

```bash
conda env create -f ../env-esc2021.yml
conda activate esc-2021
```

## External Models

GLHG builds its hierarchical graph using COMET commonsense inferences:

- **COMET-ATOMIC-2020-BART** (`mismayil/comet-bart-ai2`) — set its local path where the scripts/config expect it (replace the `<COMET_BART_PATH>` placeholder).

## Data Preprocessing

```bash
bash RUN/prepare_glhg_s3.sh
```

## Training

GLHG + S3-ESC backbone (LoRA + encoder probe + scheduled sampling):

```bash
bash RUN/train_glhg_s3.sh
```

The probe can be extracted from the encoder (`enc`) or decoder (`dec`) source via the `glhg_s3_seq_enc` / `glhg_s3_seq_dec` configs (probe-only training with `--probe_only_from`). See `train_s3.py` for the probe-extra-training flags.

## Inference

```bash
# Probe-only (PROBE_SRC = enc | dec)
CHECKPOINT=DATA/{...}/epoch-{n}_probe_enc.bin PROBE_SRC=enc bash RUN/infer_glhg_s3.sh

# Probe + SASRec ensemble (SASREC_PROBS defaults to ../data/sasrec/test_probs.json)
CHECKPOINT=... PROBE_SRC=enc ENSEMBLE_ALPHA=0.5 bash RUN/infer_ensemble_glhg_s3.sh
```

Main-table metrics (PPL, BLEU, ROUGE-L, Distinct, F1, Strategy-Acc) are computed from the shared `../metric` package.

## Folder Layout

```
GLHG/
├── CONFIG/                  # GLHG + S3 JSON configs (enc/dec probe variants)
├── RUN/                     # prepare/train/infer/ensemble scripts
├── models/                  # GLHG model + hierarchical graph + LoRA expert layers
├── inputters/               # graph-augmented data processing
├── ensemble/                # GLHG + SASREC ensemble grid search
├── sasrec/                  # GLHG-specific SASRec implementation
├── utils/                   # Training utilities
└── train_s3.py / infer_s3.py / infer_ensemble_s3.py / prepare.py / ...   # entrypoints
```

Shared assets at the repo root are accessed via `../`: `../_reformat/`, `../Blenderbot_small-90M/`, `../metric/`, `../data/sasrec/`, `../apex/`.
