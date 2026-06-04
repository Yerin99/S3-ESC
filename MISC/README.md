# MISC + S3-ESC

S3-ESC method applied on the MISC baseline (MIxed Strategy-aware model with COMMonsense, BlenderBot-small backbone).

**Prerequisites**: complete the *Shared Assets* setup in the [top-level README](../README.md) first (BlenderBot weights, ESConv preprocessing). MISC additionally requires **COMET-ATOMIC-2020-BART** for commonsense enrichment (see [External models](#external-models)). All commands assume you've `cd MISC` and activated the conda env.

> MISC ships a bundled custom `src/transformers/` (the BlenderbotSmall model with COMET integration). Its entrypoints import `from src.transformers import ...`, so run them from the `MISC/` directory.

## Environment

```bash
conda env create -f env.yml
conda activate esc-2021
```

## External Models

- **COMET-ATOMIC-2020-BART** (`mismayil/comet-bart-ai2`) — used by `generate_misc_cl_comet.py`. Set its local path where the `<COMET_BART_PATH>` placeholder appears.

## Data Preprocessing

MISC builds its inputs in two steps (both regenerate the `dataset/` files, which are gitignored):

```bash
# 1. Convert ESConv splits to MISC format (+ situation/comet inputs)
python prepare_misc_cl_data.py

# 2. Generate COMET commonsense blocks (requires COMET-BART-AI2)
python generate_misc_cl_comet.py
```

## Training

```bash
# Base MISC
python BlenderEmotionalSupport.py        # see script args

# MISC + S3-ESC (LoRA expert routing + scheduled sampling), full pipeline:
bash RUN/pipeline_misc_s3.sh             # set SASREC_SRC_DIR / COMET path as needed
```

## Inference & Metrics

```bash
# S3 ensemble generation (probe + SASRec)
python generate_s3_ensemble.py           # see script args

# Metrics
python cal_metric.py
python compute_ensemble_ppl.py
```

Main-table metrics (PPL, BLEU, ROUGE-L, Distinct, F1, Strategy-Acc) come from MISC's bundled `metric/` package.

## Folder Layout

```
MISC/
├── src/transformers/        # bundled custom transformers (BlenderbotSmall + COMET integration)
├── RUN/                     # pipeline script
├── ensemble/                # MISC + SASREC ensemble grid search
├── sasrec/                  # MISC-specific SASRec implementation
├── metric/                  # MISC-specific evaluation metrics
├── BlenderEmotionalSupport.py / _s3.py   # base / S3 model + training entrypoints
├── prepare_misc_cl_data.py / generate_misc_cl_comet.py   # data prep
├── generate_s3_ensemble.py / cal_metric.py / compute_*.py   # inference & metrics
├── lora_expert_layers.py / scheduled_sampling.py
└── env.yml                  # Conda environment (esc-2021)
```

Shared assets at the repo root are accessed via `../`: `../_reformat/`, `../Blenderbot_small-90M/`, `../data/sasrec/`.
