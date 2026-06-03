# PMPC + S3-ESC

S3-ESC method applied on the PMPC baseline (BlenderBot-small variant; DialoGPT variant not included).

**Prerequisites**: complete the *Shared Assets* setup in the [top-level README](../README.md) first (BlenderBot weights, ESConv preprocessing). PMPC additionally requires **external_models/** (DPR encoders + COMET-ATOMIC + GloVe), which are not bundled in this repo — see [External models](#external-models) below. All commands assume you've `cd PMPC` and activated the conda env.

## Environment

```bash
conda env create -f env.yml
conda activate <env-name-from-env.yml>
```

## External Models

PMPC uses externally-pretrained components that must be downloaded separately into `external_models/` at the PMPC folder root:

- **DPR question encoder + context encoder** (used by `models/knowledge_retriever.py`) — download from Hugging Face: `facebook/dpr-question_encoder-single-nq-base`, `facebook/dpr-ctx_encoder-single-nq-base`
- **COMET-ATOMIC 2020** (used by `models/comet_wrapper.py`) — download from [allenai/comet-atomic-2020](https://github.com/allenai/comet-atomic-2020)
- **GloVe embeddings** (used by `models/topic_extractor.py`) — same prepared GloVe files as the top-level README mentions

Folder layout expected:

```
PMPC/external_models/
├── dpr-question_encoder/
├── dpr-ctx_encoder/
├── comet-atomic-2020/
└── glove6B/
```

Total size ≈ 4.2 GB; gitignored.

## Data Preprocessing

```bash
bash RUN/prepare_pmpc.sh
```

## Training

The PMPC + S3-ESC main model:

```bash
bash RUN/train_pmpc.sh
```

For the no-strategy-sequence ablation pipeline (PMPC + S3-ESC, single round):

```bash
bash RUN/pipeline_pmpc_s3_sr.sh
```

(Set `SASREC_SRC_DIR` and `CHECKPOINT` env vars where the scripts require them.)

## Inference

```bash
CHECKPOINT=DATA/{inputter}.{config}/{timestamp}/epoch-{n}.bin \
  bash RUN/infer_pmpc.sh
```

For SASREC-ensemble inference:

```bash
CHECKPOINT=... SASREC_PROBS=... ENSEMBLE_ALPHA=0.5 \
  python infer_ensemble_s3.py --config_name pmpc_s3_sr --inputter_name pmpc ...
```

## Folder Layout

```
PMPC/
├── CONFIG/                  # PMPC + baseline JSON configs
├── RUN/                     # Shell scripts for prepare/train/infer/interact
├── models/                  # PMPC-specific models (BlenderBot variant) + external-knowledge modules
├── inputters/               # Data processing
├── ensemble/                # PMPC + SASREC ensemble grid search
├── sasrec/                  # PMPC-specific SASRec implementation
├── utils/                   # Training utilities
├── apex/                    # NVIDIA Apex (mixed precision)
├── env.yml                  # Conda environment for PMPC + S3-ESC
└── train.py / prepare.py / infer.py / ...   # entrypoints
```

Shared assets at the repo root are accessed via `../`: `../_reformat/`, `../Blenderbot_small-90M/`, `../metric/`.
