# BlenderBot + S3-ESC

S3-ESC method applied on the BlenderBot-small baseline.

**Prerequisite**: complete the *Shared Assets* setup in the [top-level README](../README.md) first — at minimum the BlenderBot weights and ESConv preprocessing. (GloVe embeddings are optional and only used for embedding-based metrics.) All commands below assume you've `cd BlenderBot` and activated the conda env.

## Environment

```bash
conda env create -f ../env-esc2021.yml
conda activate esc-2021
```

## Data Preprocessing

```bash
bash RUN/prepare_strat_lora_blenderbot_s3_seq.sh
```

## Training

The main S3 model (Joint Probe + Strategy Sequence + LoRA):

```bash
bash RUN/train_strat_lora_blenderbot_s3_seq.sh
```

Training creates a timestamped folder under `DATA/{inputter}.{config}/`. Select the best checkpoint based on validation PPL.

### Ablation Conditions

| Condition | Config | Description |
|-----------|--------|-------------|
| S3 (full) | `strat_lora_blenderbot_s3_seq` | Joint probe + strategy sequence + LoRA |
| S3 (sr) | `strat_lora_blenderbot_s3_sr` | Joint probe + LoRA (no strategy sequence) |
| S3 (-LoRA) | `strat_blenderbot_s3_nolora_enc` | Joint probe + strategy sequence (no LoRA) |

## Inference & Evaluation

```bash
CHECKPOINT=DATA/{inputter}.{config}/{timestamp}/epoch-{n}.bin \
  bash RUN/infer_strat_lora_blenderbot_s3_seq.sh
```

Metrics (BLEU, ROUGE-L, strategy accuracy, PPL, etc.) are computed automatically and printed at the end of inference.

## Folder Layout

```
BlenderBot/
├── CONFIG/                  # Training configurations (JSON)
├── RUN/                     # Shell scripts for train/infer/prepare
├── models/                  # BlenderBot-small model definitions
├── inputters/               # Data processing and input formatting
├── utils/                   # Training/building utilities
├── hidden_strategy_probe/   # Strategy prediction probe
├── ensemble/                # BlenderBot + SASREC ensemble
├── apex/                    # NVIDIA Apex (mixed precision)
└── train.py / prepare.py / infer.py / ...   # entrypoints
```

Shared assets at the repo root are accessed via `../`: `../_reformat/`, `../Blenderbot_small-90M/`, `../metric/`, `../sasrec/`.
