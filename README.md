# S3-ESC: Strategy-Aware Emotional Support Conversation

Code for the paper submitted to ARR.

This codebase builds on the [ESConv](https://github.com/thu-coai/Emotional-Support-Conversation) framework (Liu et al., ACL 2021).

## Environment Setup

```bash
conda env create -f env.yml -n esc
conda activate esc
```

## Downloading Pretrained Models

Download [BlenderBot-small](https://huggingface.co/facebook/blenderbot_small-90M) and replace the placeholder `pytorch_model.bin`:

```bash
wget -O Blenderbot_small-90M/pytorch_model.bin \
  https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/pytorch_model.bin
```

For embedding-based evaluation metrics, download prepared GloVe files from [HuggingFace](https://huggingface.co/datasets/chujiezheng/glove_embedding) and place them under `metric/word2vec/glove6B/`.

## Data Preprocessing

```bash
cd _reformat && python process.py && cd ..
bash RUN/prepare_strat_lora_blenderbot_s3_seq.sh
```

## Training

The main S3 model (Joint Probe + Strategy Sequence + LoRA):

```bash
bash RUN/train_strat_lora_blenderbot_s3_seq_p01.sh
```

Training creates a timestamped folder under `DATA/{inputter}.{config}/`. Select the best checkpoint based on validation PPL.

### Ablation Conditions

| Condition | Config | Description |
|-----------|--------|-------------|
| S3 (full) | `strat_lora_blenderbot_s3_seq` | Joint probe + strategy sequence + LoRA |
| S3 (sr) | `strat_lora_blenderbot_s3_sr` | Joint probe + LoRA (no strategy sequence) |
| S3 (-LoRA) | `strat_blenderbot_s3_nolora_enc` | Joint probe + strategy sequence (no LoRA) |

Encoder/decoder probe variants are available via `_enc` / `_dec` config suffixes.

## Inference & Evaluation

Set `CHECKPOINT` to a trained checkpoint path and run:

```bash
CHECKPOINT=DATA/{inputter}.{config}/{timestamp}/epoch-{n}.bin \
  bash RUN/infer_strat_lora_blenderbot_s3_seq_p01.sh
```

Metrics (BLEU, ROUGE-L, strategy accuracy, PPL, etc.) are computed automatically and printed at the end of inference.

## Project Structure

```
CONFIG/          # Training configurations (JSON)
RUN/             # Shell scripts for train/infer/prepare
models/          # Model definitions
inputters/       # Data processing and input formatting
hidden_strategy_probe/   # Strategy prediction probe
ensemble/        # Ensemble inference and grid search
metric/          # Evaluation metrics (BLEU, ROUGE, etc.)
```
