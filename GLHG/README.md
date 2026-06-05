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

GLHG + S3-ESC main model — full pipeline: backbone training (LoRA + probe + scheduled sampling) → probe extraction (enc/dec) → grid search → SASRec ensemble. The SASRec ensemble is the **required final step** of the method.

```bash
bash RUN/pipeline_glhg_s3.sh            # optional: bash RUN/pipeline_glhg_s3.sh <GPU_ID>
```

The probe is extracted from the encoder (`enc`) or decoder (`dec`) source via the `glhg_s3_enc` / `glhg_s3_dec` configs; the pipeline runs both and grid-searches the ensemble weight on validation. For backbone-only training, use `bash RUN/train_glhg_s3.sh`.

## Inference

The full method's inference (probe + SASRec ensemble) runs as the final stage of `pipeline_glhg_s3.sh`. To run the stages standalone on a trained checkpoint:

```bash
# probe-only (PROBE_SRC = enc | dec)
CHECKPOINT=DATA/{...}/epoch-{n}_probe_enc.bin PROBE_SRC=enc bash RUN/infer_glhg_s3.sh

# probe + SASRec ensemble (SASREC_PROBS defaults to ../data/sasrec/test_probs.json)
CHECKPOINT=... PROBE_SRC=enc ENSEMBLE_ALPHA=0.5 bash RUN/infer_ensemble_glhg_s3.sh
```

Main-table metrics (PPL, BLEU, ROUGE-L, Distinct, F1, Strategy-Acc) are computed from the shared `../metric` package.

## Reported Hyperparameters

Settings that reproduce this baseline's `+S3` row in the paper's main table:

| ss_p | LoRA lr | probe | ensemble α |
|------|---------|-------|-----------|
| 0.1 | 3e-5 | enc | 0.50 |

`ss_p` = scheduled-sampling probability; `LoRA lr` = learning rate of the LoRA expert params; `probe` = strategy-probe source (`enc` = encoder mean-pool); `α` = ensemble weight grid-searched on validation (`p_final = α·p_probe + (1−α)·p_sasrec`). The backbone learning rate follows the original GLHG setup and is left unchanged. These are the defaults in `RUN/pipeline_glhg_s3.sh`.

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

## Citation

Original paper:

```bibtex
@inproceedings{peng2022glhg,
    title = {Control Globally, Understand Locally: A Global-to-Local Hierarchical Graph Network for Emotional Support Conversation},
    author = {Peng, Wei and Hu, Yue and Xing, Luxi and Xie, Yuqiang and Sun, Yajing and Li, Yunpeng},
    booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence ({IJCAI})},
    pages = {4324--4330},
    year = {2022},
}
```
