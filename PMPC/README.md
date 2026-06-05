# PMPC + S3-ESC

S3-ESC method applied on the PMPC baseline (BlenderBot-small variant; DialoGPT variant not included).

**Prerequisites**: complete the *Shared Assets* setup in the [top-level README](../README.md) first (BlenderBot weights, ESConv preprocessing). PMPC additionally requires **external_models/** (DPR encoders + COMET-ATOMIC + GloVe), which are not bundled in this repo — see [External models](#external-models) below. All commands assume you've `cd PMPC` and activated the conda env.

## Environment

```bash
conda env create -f ../env-cuda.yml
conda activate cuda
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

The PMPC + S3-ESC main model — full pipeline: backbone training (no strategy in sequence) → probe extraction → grid search → SASRec ensemble. The SASRec ensemble is the **required final step** of the method.

```bash
bash RUN/pipeline_pmpc_s3.sh
```

(Set `SASREC_SRC_DIR` and `CHECKPOINT` env vars where the scripts require them.) For backbone-only training, use `bash RUN/train_pmpc.sh`.

## Inference

The full method's inference (probe + SASRec ensemble) runs as the final stage of `pipeline_pmpc_s3.sh`. To run it standalone on a trained checkpoint:

```bash
CHECKPOINT=... SASREC_PROBS=... ENSEMBLE_ALPHA=0.5 \
  python infer_ensemble_s3.py --config_name pmpc_s3 --inputter_name pmpc ...
```

## Reported Hyperparameters

Settings that reproduce this baseline's `+S3` row in the paper's main table:

| ss_p | LoRA lr | probe | ensemble α |
|------|---------|-------|-----------|
| 0.1 | 5e-5 | predictor (single) | 0.55 |

`ss_p` = scheduled-sampling probability; `LoRA lr` = learning rate of the LoRA expert params; `α` = ensemble weight grid-searched on validation (`p_final = α·p_strat + (1−α)·p_sasrec`). PMPC reuses its built-in prompt-based strategy predictor (no separate enc/dec probe variant), so `probe` is a single predictor. The backbone learning rate follows the original PMPC setup and is left unchanged.

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
└── train.py / prepare.py / infer.py / ...   # entrypoints
```

Shared assets at the repo root are accessed via `../`: `../_reformat/`, `../Blenderbot_small-90M/`, `../metric/`.

## Citation

Original paper:

```bibtex
@article{xu2025pmpc,
    title   = {Prompt Learning With Multiperspective Cues for Emotional Support Conversation Systems},
    author  = {Xu, Yangyang and Zhao, Zhuoer and Sun, Xiao and Yang, Xun},
    journal = {IEEE Transactions on Computational Social Systems},
    volume  = {12},
    number  = {5},
    pages   = {2648--2660},
    month   = oct,
    year    = {2025},
    publisher = {IEEE},
}
```
