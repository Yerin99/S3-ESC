# DKFPE + S3-ESC

S3-ESC method applied on the DKFPE baseline (Dynamic Knowledge Filtering and Persona Extraction).

**Prerequisites**: complete the *Shared Assets* setup in the [top-level README](../README.md) first (BlenderBot weights, ESConv preprocessing). DKFPE additionally requires **external_models/** (DPR encoders + COMET-ATOMIC 2020 + ComFact + DeBERTa-v3-large), which are not bundled in this repo — see [External models](#external-models) below. All commands assume you've `cd DKFPE` and activated the conda env.

## Environment

```bash
conda env create -f env.yml
conda activate cuda
```

## External Models

DKFPE uses externally-pretrained components that must be downloaded separately into `external_models/` at the DKFPE folder root:

- **DPR question + context encoders** — `facebook/dpr-question_encoder-single-nq-base`, `facebook/dpr-ctx_encoder-single-nq-base` (Hugging Face)
- **COMET-ATOMIC 2020** — [allenai/comet-atomic-2020](https://github.com/allenai/comet-atomic-2020)
- **DeBERTa-v3-large** — `microsoft/deberta-v3-large` (Hugging Face) — used by `train_filter.py`
- **ComFact** — for DeBERTa knowledge filter training data

Folder layout expected:

```
DKFPE/external_models/
├── dpr-question_encoder/
├── dpr-ctx_encoder/
├── comet-atomic-2020/
├── deberta-v3-large/
└── comfact/
```

## Data Preprocessing

DKFPE uses an 8:1:1 utterance-pair split derived from TransESC + PESConv mapping. Place `ESConv.json`, `PESConv.json`, the TransESC TSV files, and the original `_reformat` dir at the locations specified by `--esconv_path`, `--pesconv_path`, `TRANSESC_DIR`, `ESC_REFORMAT_DIR` env vars; then:

```bash
cd _reformat
python process_dkfpe.py --mode 701515   # 70:15:15 split — used by all baselines for fair comparison
# python process_dkfpe.py --mode 811    # DKFPE paper original split (8:1:1)
```

## Training

Main S3 model (full pipeline including backbone training, probe extraction, grid search, ensemble):

```bash
bash RUN/pipeline_dkfpe_s3_seq.sh
```

For the no-strategy-sequence ablation (S3 single-round):

```bash
bash RUN/pipeline_dkfpe_s3_sr.sh
```

DeBERTa knowledge filter training (standalone):

```bash
bash RUN/train_filter.sh
```

(`SASREC_SRC_DIR` env var must point to a directory containing `valid_probs.json` and `test_probs.json` from a trained SASREC checkpoint.)

## Inference

```bash
CHECKPOINT=DATA/{inputter}.{config}/{timestamp}/epoch-{n}.bin \
  bash RUN/infer_dkfpe_s3.sh
```

## Folder Layout

```
DKFPE/
├── CONFIG/                  # DKFPE + S3 JSON configs
├── RUN/                     # Shell scripts (pipeline, train, infer)
├── models/                  # DKFPE models + DeBERTa filter, knowledge retriever, COMET, persona modules
├── inputters/               # Data processing
├── ensemble/                # DKFPE + SASREC ensemble grid search
├── sasrec/                  # DKFPE-specific SASRec implementation
├── _reformat/process_dkfpe.py  # DKFPE-specific data preprocessing
├── utils/                   # Training utilities
├── apex/                    # NVIDIA Apex (mixed precision)
├── env.yml                  # Conda environment for DKFPE + S3-ESC
└── train.py / prepare.py / infer.py / train_filter.py / ...   # entrypoints
```

Shared assets at the repo root are accessed via `../`: `../_reformat/`, `../Blenderbot_small-90M/`, `../metric/`.
