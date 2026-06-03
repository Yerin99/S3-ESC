# S3-ESC: Strategy-Aware Emotional Support Conversation

Code for the paper submitted to ARR.

This codebase builds on the [ESConv](https://github.com/thu-coai/Emotional-Support-Conversation) framework (Liu et al., ACL 2021). Each baseline + S3-ESC combination lives in its own top-level folder and ships its own conda env; shared assets (`_reformat/`, `Blenderbot_small-90M/`, `metric/`, `sasrec/`) sit at the repo root.

## Baselines

| Baseline | Folder | Status |
|----------|--------|--------|
| BlenderBot-small + S3-ESC | [`BlenderBot/`](BlenderBot/README.md) | Available |
| KEMI + S3-ESC | `KEMI/` | Coming soon |
| PMPC + S3-ESC | [`PMPC/`](PMPC/README.md) | Available |
| DKFPE + S3-ESC | `DKFPE/` | Coming soon |
| GLHG + S3-ESC | `GLHG/` | Coming soon |
| MISC + S3-ESC | `MISC/` | Coming soon |

See each baseline's README for env setup, training, and inference. To run, `cd <baseline>/` first; entrypoints assume the baseline folder is the current working directory.

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
ARR/
├── _reformat/               # raw ESConv data + preprocessing
├── Blenderbot_small-90M/    # BlenderBot-small weights (gitignored, wget once)
├── metric/                  # evaluation metrics (BLEU, ROUGE, embedding-based, ...)
├── sasrec/                  # SASRec strategy router (shared by ensembles)
├── BlenderBot/              # BlenderBot + S3-ESC
├── KEMI/  PMPC/  DKFPE/  GLHG/  MISC/   # placeholders for upcoming baselines
└── README.md                # (this file)
```
