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
bash RUN/prepare_strat_lora_blenderbot_s3.sh
```

## Training

BlenderBot + S3-ESC main model — full pipeline: backbone training (joint probe + LoRA + scheduled sampling) → probe inference (valid) → grid search → SASRec ensemble. The SASRec ensemble is the **required final step** of the method.

```bash
bash RUN/pipeline_strat_lora_blenderbot_s3.sh            # optional: P_SAMPLES="..." to sweep ss_p
```

The strategy probe is trained jointly inside the backbone (encoder source); the pipeline grid-searches the ensemble weight on validation and runs the final ensemble inference on test. For backbone-only training, use `bash RUN/train_strat_lora_blenderbot_s3.sh`.

### Ablation Conditions

| Condition | Config | Description |
|-----------|--------|-------------|
| S3 (full / main) | `strat_lora_blenderbot_s3` | Joint probe + LoRA (no strategy sequence) + SASRec ensemble |
| S3 (−LoRA) | `strat_blenderbot_s3_nolora_enc` | Joint probe (no strategy sequence), no LoRA |

## Inference & Evaluation

The full method's inference (probe + SASRec ensemble) runs as the final stage of `pipeline_strat_lora_blenderbot_s3.sh`. To run the stages standalone on a trained checkpoint:

```bash
# probe + SASRec ensemble on test (required final step)
CHECKPOINT=DATA/{inputter}.{config}/{timestamp}/epoch-{n}.bin ENSEMBLE_ALPHA=0.55 \
  bash RUN/infer_strat_lora_blenderbot_s3.sh
# grid-search alpha on valid first (writes grid_search_s3.json)
CHECKPOINT=... PROBE_GEN_JSON=... bash RUN/run_ensemble_grid_search_s3.sh
```

Metrics (BLEU, ROUGE-L, strategy accuracy, PPL, etc.) are computed automatically and printed at the end of inference.

## Reported Hyperparameters

Settings that reproduce this baseline's `+S3` row in the paper's main table:

| ss_p | LoRA lr | probe | ensemble α |
|------|---------|-------|-----------|
| 0.05 | 5e-5 | enc | 0.55 |

`ss_p` = scheduled-sampling probability; `LoRA lr` = learning rate of the LoRA expert params; `probe` = strategy-probe source (`enc` = encoder mean-pool); `α` = ensemble weight grid-searched on validation (`p_final = α·p_probe + (1−α)·p_sasrec`). The backbone learning rate follows the original BlenderBot-small ESConv setup and is left unchanged.

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

## Citation

Original paper:

```bibtex
@inproceedings{liu2021esc,
    title = {Towards Emotional Support Dialog Systems},
    author = {Liu, Siyang and Zheng, Chujie and Demasi, Orianna and Sabour, Sahand and Li, Yu and Yu, Zhou and Jiang, Yong and Huang, Minlie},
    booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
    pages = {3469--3483},
    year = {2021},
    publisher = {Association for Computational Linguistics},
}
```
