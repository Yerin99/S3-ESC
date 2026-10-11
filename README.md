# S3-ESC

**Enhancing Emotional Support Conversation with Plug-in Modules for Sequential Strategy Modeling and Strategy-Specialized Generation**

Code for S3-ESC, accepted to EMNLP 2026 Findings. Paper link and BibTeX will be added once the proceedings are published.

Each baseline + S3-ESC combination lives in its own top-level folder; shared assets (`_reformat/`, `Blenderbot_small-90M/`, `metric/`, `sasrec/`, `data/sasrec/`, `apex/`) and the two conda environments (`env-esc2021.yml`, `env-cuda.yml`) sit at the repo root. Each baseline README states which environment it uses.

Each baseline is built on the original authors' publicly released code; missing or incomplete parts of the released code were reimplemented following the paper. All baselines are unified to a **70:15:15** train/valid/test split of ESConv for fair comparison. On top of each baseline we then apply our contribution (S3-ESC), whose main components are **Selective LoRA**, **Scheduled Sampling**, and a **Sequential strategy predictor** (SASRec ensemble).

## Baselines

| Baseline | Folder | Conda env | Original paper | Status |
|----------|--------|-----------|----------------|--------|
| BlenderBot-small + S3-ESC | [`BlenderBot/`](BlenderBot/README.md) | `env-esc2021.yml` | [Liu et al., ACL 2021](https://aclanthology.org/2021.acl-long.269/) | Available |
| KEMI + S3-ESC | [`KEMI/`](KEMI/README.md) | `env-esc2021.yml` | [Deng et al., ACL 2023](https://aclanthology.org/2023.acl-long.225/) | Available |
| PMPC + S3-ESC | [`PMPC/`](PMPC/README.md) | `env-cuda.yml` | [Xu et al., IEEE TCSS 2025](https://ieeexplore.ieee.org/document/10899367) | Available |
| DKFPE + S3-ESC | [`DKFPE/`](DKFPE/README.md) | `env-cuda.yml` | [Hao & Kong, COLING 2025](https://aclanthology.org/2025.coling-main.214/) | Available |
| GLHG + S3-ESC | [`GLHG/`](GLHG/README.md) | `env-esc2021.yml` | [Peng et al., IJCAI 2022](https://www.ijcai.org/proceedings/2022/600) | Available |
| MISC + S3-ESC | [`MISC/`](MISC/README.md) | `env-esc2021.yml` | [Tu et al., ACL 2022](https://aclanthology.org/2022.acl-long.25/) | Available |

Two top-level conda envs: **`env-esc2021.yml`** (BlenderBot / KEMI / GLHG / MISC) and **`env-cuda.yml`** (PMPC / DKFPE). See each baseline's README for env setup, training, and inference. To run, `cd <baseline>/` first; entrypoints assume the baseline folder is the current working directory.

## Reported Hyperparameters

Settings that reproduce each baseline's `+S3` row in the paper's main table:

| Baseline | ss_p | LoRA lr | probe | ensemble α |
|----------|------|---------|-------|-----------|
| BlenderBot | 0.05 | 5e-5 | enc | 0.55 |
| KEMI | 0.05 | 3e-5 | enc | 0.55 |
| PMPC | 0.1 | 5e-5 | predictor | 0.55 |
| DKFPE | 0.1 | 5e-5 | enc | 0.80 |
| GLHG | 0.1 | 3e-5 | enc | 0.50 |
| MISC | 0.2 | 5e-5 | predictor | 0.30 |

`ss_p` = scheduled-sampling probability; `LoRA lr` = learning rate of the LoRA expert params; `probe` = strategy-probe source (`enc` = encoder mean-pool; `predictor` = the baseline's built-in strategy predictor, no enc/dec variant); `α` = ensemble weight grid-searched on validation (`p_final = α·p_strat + (1−α)·p_sasrec`). These are the defaults in each baseline's `RUN/pipeline_*_s3.sh`. The backbone learning rate follows each baseline's original paper and is left unchanged. See each baseline's README for the per-baseline details.

## Shared Assets (one-time setup)

From the repo root:

```bash
# 1. BlenderBot-small pretrained weights (replaces the placeholder pytorch_model.bin)
wget -O Blenderbot_small-90M/pytorch_model.bin \
  https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/pytorch_model.bin

# 2. ESConv preprocessing (generates train/valid/test.txt under _reformat/)
cd _reformat && python process.py && cd ..

# 3. (For embedding-based metrics) Download GloVe files from
#    https://huggingface.co/datasets/chujiezheng/glove_embedding
#    and place them under metric/word2vec/glove6B/
```

`Blenderbot_small-90M/pytorch_model.bin` is gitignored, so the weights need to be fetched once; other baselines reuse the same file.

## Repo Layout

```
S3-ESC/
├── _reformat/               # raw ESConv data + preprocessing
├── Blenderbot_small-90M/    # BlenderBot-small (vocab/merges/config tracked; pytorch_model.bin gitignored, wget once)
├── metric/                  # evaluation metrics (BLEU, ROUGE, embedding-based, ...)
├── sasrec/                  # SASRec strategy router code (shared by ensembles)
├── data/sasrec/             # Trained SASRec checkpoint + valid/test probs (shared by ensembles)
├── apex/                    # NVIDIA Apex (mixed-precision, vendored — shared)
├── env-esc2021.yml          # conda env for BlenderBot / KEMI / GLHG / MISC
├── env-cuda.yml             # conda env for PMPC / DKFPE
├── BlenderBot/              # BlenderBot + S3-ESC
├── KEMI/                    # KEMI + S3-ESC
├── PMPC/                    # PMPC + S3-ESC
├── DKFPE/                   # DKFPE + S3-ESC
├── GLHG/                    # GLHG + S3-ESC
├── MISC/                    # MISC + S3-ESC
└── README.md                # (this file)
```

## Acknowledgements & Licenses

This repository builds on the [ESConv](https://github.com/thu-coai/Emotional-Support-Conversation) framework (Liu et al., ACL 2021); the shared `_reformat/` preprocessing and `metric/` evaluation code derive from its `codes_zcj` release. Each baseline folder additionally builds on that baseline's original authors' publicly released code — see the per-baseline READMEs (each has a `## Citation` section with the original paper).

Vendored third-party code retains its original license:

- **`apex/`** — [NVIDIA Apex](https://github.com/NVIDIA/apex), BSD-3-Clause (`apex/LICENSE`).
- **`MISC/src/transformers/`** — a modified copy of [HuggingFace Transformers](https://github.com/huggingface/transformers), Apache-2.0 (`MISC/src/transformers/LICENSE`); per-file copyright headers are retained.
- **`metric/pycocoevalcap/`** (and `MISC/metric/pycocoevalcap/`) — [pycocoevalcap](https://github.com/salaniz/pycocoevalcap)/COCO caption eval, under their bundled licenses (`license.txt`, `bleu/LICENSE`).
- **`sasrec/`** — based on [SASRec](https://github.com/kang205/SASRec) (Kang & McAuley, ICDM 2018).

The S3-ESC method code (strategy probe, LoRA expert routing, scheduled sampling, SASRec ensemble) and the modifications integrating it into each baseline are our own contribution.
