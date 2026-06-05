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

KEMI + S3-ESC main model — full pipeline: backbone training → probe extraction (enc/dec) → grid search → SASRec ensemble. The SASRec ensemble is the **required final step** of the method.

```bash
bash RUN/pipeline_kemi_s3.sh            # optional: bash RUN/pipeline_kemi_s3.sh <GPU_ID>
```

The probe is extracted from the encoder (`enc`) or decoder (`dec`) source via the `kemi_s3_enc` / `kemi_s3_dec` configs; the pipeline runs both and grid-searches the ensemble weight on validation. For the original KEMI baseline (no S3): `bash RUN/train_strat.sh`.

## Inference

The full method's inference (probe + SASRec ensemble) runs as the final stage of `pipeline_kemi_s3.sh`. To run the stages standalone on a trained checkpoint:

```bash
# probe-only inference
CHECKPOINT=DATA/{...}/epoch-{n}.bin bash RUN/infer_s3.sh <GPU_ID> <CHECKPOINT>

# probe + SASRec ensemble (SASREC_PROBS defaults to ../data/sasrec/test_probs.json)
bash RUN/infer_ensemble_s3.sh <GPU_ID> <CHECKPOINT> <ALPHA>
```

Main-table metrics (PPL, BLEU, ROUGE-L, Distinct, F1, Strategy-Acc) are computed from the shared `../metric` package. (KEMI's original RoBERTa-based mixed-initiative metric is not part of our main table and has been removed.)

## Reported Hyperparameters

Settings that reproduce this baseline's `+S3` row in the paper's main table:

| ss_p | LoRA lr | probe | ensemble α |
|------|---------|-------|-----------|
| 0.05 | 3e-5 | enc | 0.55 |

`ss_p` = scheduled-sampling probability; `LoRA lr` = learning rate of the LoRA expert params; `probe` = strategy-probe source (`enc` = encoder mean-pool); `α` = ensemble weight grid-searched on validation (`p_final = α·p_probe + (1−α)·p_sasrec`). The backbone learning rate follows the original KEMI setup and is left unchanged. These are the defaults in `RUN/pipeline_kemi_s3.sh`.

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

## Citation

Original paper:

```bibtex
@inproceedings{deng2023kemi,
    title = {Knowledge-Enhanced Mixed-Initiative Dialogue System for Emotional Support Conversations},
    author = {Deng, Yang and Zhang, Wenxuan and Yuan, Yifei and Lam, Wai},
    booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    pages = {4079--4095},
    year = {2023},
    publisher = {Association for Computational Linguistics},
}
```
