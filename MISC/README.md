# MISC + S3-ESC

S3-ESC method applied on the MISC baseline (MIxed Strategy-aware model with COMMonsense, BlenderBot-small backbone).

**Prerequisites**: complete the *Shared Assets* setup in the [top-level README](../README.md) first (BlenderBot weights, ESConv preprocessing). MISC additionally requires **COMET-ATOMIC-2020-BART** for commonsense enrichment (see [External models](#external-models)). All commands assume you've `cd MISC` and activated the conda env.

> MISC ships a bundled custom `src/transformers/` (the BlenderbotSmall model with COMET integration). Its entrypoints import `from src.transformers import ...`, so run them from the `MISC/` directory.

## Environment

```bash
conda env create -f ../env-esc2021.yml
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

## Reported Hyperparameters

Settings that reproduce this baseline's `+S3` row in the paper's main table:

| ss_p | LoRA lr | probe | ensemble α |
|------|---------|-------|-----------|
| 0.2 | 5e-5 | predictor (single) | 0.30 |

`ss_p` = scheduled-sampling probability; `LoRA lr` = learning rate of the LoRA expert params; `α` = ensemble weight grid-searched on validation (`p_final = α·p_strat + (1−α)·p_sasrec`). MISC reuses its built-in `strategy_head` predictor (no separate enc/dec probe variant), so `probe` is a single predictor. The backbone learning rate follows the original MISC setup and is left unchanged.

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
```

Shared assets at the repo root are accessed via `../`: `../_reformat/`, `../Blenderbot_small-90M/`, `../data/sasrec/`.

## Citation

Original paper:

```bibtex
@inproceedings{tu2022misc,
    title = {{MISC}: A Mixed Strategy-Aware Model Integrating {COMET} for Emotional Support Conversation},
    author = {Tu, Quan and Li, Yanran and Cui, Jianwei and Wang, Bin and Wen, Ji-Rong and Yan, Rui},
    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    pages = {308--319},
    year = {2022},
    publisher = {Association for Computational Linguistics},
}
```
