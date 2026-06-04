"""Compute ensemble PPL with teacher forcing.

Uses ensembled strategy (p_final = alpha * p_predictor + (1-alpha) * p_sasrec)
for both LoRA routing and strategy_embs (decoder cross-attention).

Usage: python compute_ensemble_ppl.py <ss_p> <lora_lr>
"""
import os
import sys
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from BlenderEmotionalSupport_s3 import (
    Args, parse_args_override, set_seed, ESDDataset,
    load_model_with_lora, encode_comet_embeddings, shift_tokens_right,
    InputFeatures_blender, InputFeatures_train,
)
from src.transformers import BlenderbotSmallTokenizer

if len(sys.argv) != 3:
    print("Usage: python compute_ensemble_ppl.py <ss_p> <lora_lr>")
    sys.exit(1)

ss_p = sys.argv[1]
lora_lr = sys.argv[2]

MODEL_PATH = f"blender_strategy/s3b_ss{ss_p}_lr{lora_lr}"
GRID_JSON = f"{MODEL_PATH}/grid_search_remapped.json"
SASREC_PROBS = "./DATA/sasrec/test_probs_remapped.json"

# Load best alpha
with open(GRID_JSON) as f:
    grid = json.load(f)
alpha = grid['best']['alpha']

# Load SASRec probs
with open(SASREC_PROBS) as f:
    sasrec_probs = json.load(f)

print(f"Model: {MODEL_PATH}")
print(f"Alpha: {alpha}")

# Setup
args = Args()
args.output_dir = MODEL_PATH
args.b_condition = True
args.seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.n_gpu = 1
args.no_cuda = not torch.cuda.is_available()
set_seed(args)

# Tokenizer
additional_special_tokens = [
    "[Question]", "[Reflection of feelings]", "[Information]",
    "[Restatement or Paraphrasing]", "[Others]", "[Self-disclosure]",
    "[Affirmation and Reassurance]", "[Providing Suggestions]"
]
tokenizer = BlenderbotSmallTokenizer.from_pretrained(
    MODEL_PATH, additional_special_tokens=additional_special_tokens
)

# Model
model = load_model_with_lora(MODEL_PATH, tokenizer, args, device)
model.eval()
_model = model.module if hasattr(model, 'module') else model

# Dataset (test)
data_file = os.path.join(args.data_path, 'test_cl.txt')
comet_file = os.path.join(args.data_path, 'testComet_cl.txt')
comet_st_file = os.path.join(args.data_path, 'testSituation_cl.txt')

with open(data_file, "r") as f:
    df = f.read().split("\n")
with open(comet_file, "r") as f:
    comet = f.read().split("\n")
with open(comet_st_file, "r") as f:
    comet_st = f.read().split("\n")

test_dataset = ESDDataset(
    tokenizer, args, df, comet, comet_st,
    block_size=512, evaluate=True, strategy=not args.b_condition, test=True
)

from torch.utils.data import DataLoader, SequentialSampler
sampler = SequentialSampler(test_dataset)
dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=1, collate_fn=test_dataset.collate)

total_loss = 0.0
total_tokens = 0
seq_idx = 0

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Computing ensemble PPL"):
        (input_ids, position_ids, turn_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids,
         decoder_input_ids, decoder_position_ids, decoder_turn_ids, decoder_role_ids, decoder_labels,
         decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask,
         emotion, comet_ids_st, comet_mask_st) = batch

        if input_ids.shape[1] > 512:
            seq_idx += input_ids.shape[0]
            continue

        decoder_strategy_ids = decoder_strategy_ids[:, 0].to(device)
        input_ids = input_ids.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        decoder_label_ids = decoder_labels.to(device)
        decoder_turn_ids = decoder_turn_ids.to(device)

        comet_embs, comet_mask_t, comet_embs_st, comet_mask_st_t = encode_comet_embeddings(
            model, tokenizer, comet_ids, comet_mask, comet_ids_st, comet_mask_st, device
        )

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        # Encoder forward (no GT)
        encoder_outputs = _model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            comet_embs=comet_embs,
            comet_mask=comet_mask_t,
            comet_embs_st=comet_embs_st,
            comet_mask_st=comet_mask_st_t,
            strategy_logit_ground=None,
        )

        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            if seq_idx >= len(sasrec_probs):
                break

            # Predictor distribution
            strategy_logits = encoder_outputs.strategy_logits[b:b+1]
            p_predictor = F.softmax(strategy_logits[0], dim=-1).cpu().numpy()

            # SASRec distribution
            p_sasrec = np.array(sasrec_probs[seq_idx]['probs'])

            # Ensemble
            p_final = alpha * p_predictor + (1 - alpha) * p_sasrec
            p_final = p_final / p_final.sum()
            ensemble_strat_id = int(np.argmax(p_final))

            # Ensemble strategy_embs
            p_final_tensor = torch.tensor(p_final, device=device).float().unsqueeze(0)
            strategy_id = _model.model.encoder.strategy_id.to(device)
            ensemble_strategy_embs = torch.bmm(
                p_final_tensor.unsqueeze(1),
                _model.model.encoder.strategy_embedding(strategy_id).unsqueeze(0)
            )

            # LoRA routing
            _model._decoder_lora_context.expert_ids = torch.tensor([ensemble_strat_id], device=device)

            # Decoder forward (teacher forcing)
            dec_ids = decoder_input_ids[b:b+1]
            dec_labels = decoder_label_ids[b:b+1]
            dec_shifted = shift_tokens_right(dec_ids, _model.config.pad_token_id, _model.config.decoder_start_token_id)

            decoder_outputs = _model.model.decoder(
                input_ids=dec_shifted,
                encoder_hidden_states=encoder_outputs.last_hidden_state[b:b+1],
                encoder_attention_mask=attention_mask[b:b+1],
                comet_hidden_states=encoder_outputs.last_comet_hidden_state[b:b+1],
                comet_mask=encoder_outputs.comet_mask[b:b+1],
                comet_hidden_states_st=encoder_outputs.last_comet_hidden_state_st[b:b+1],
                comet_mask_st=encoder_outputs.comet_mask_st[b:b+1],
                strategy_embs=ensemble_strategy_embs,
            )

            lm_logits = _model.lm_head(decoder_outputs[0]) + _model.final_logits_bias
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(
                lm_logits.reshape(-1, _model.config.vocab_size),
                dec_labels.reshape(-1)
            )

            # Only count non-padding tokens
            valid_mask = (dec_labels.reshape(-1) != -100)
            n_tokens = valid_mask.sum().item()
            if n_tokens > 0:
                total_loss += token_losses[valid_mask].sum().item()
                total_tokens += n_tokens

            _model._decoder_lora_context.expert_ids = None
            seq_idx += 1

avg_loss = total_loss / total_tokens
ppl = math.exp(avg_loss)

print(f"\nEnsemble PPL (alpha={alpha}): {ppl:.2f}")
print(f"  avg_loss={avg_loss:.4f}, total_tokens={total_tokens}, n_samples={seq_idx}")

# Save
out_path = f"generated_data/s3b_ss{ss_p}_lr{lora_lr}_ensemble_remapped_a{alpha:.2f}/ensemble_ppl.json"
if os.path.exists(os.path.dirname(out_path)):
    with open(out_path, 'w') as f:
        json.dump({"ppl": ppl, "avg_loss": avg_loss, "total_tokens": total_tokens, "alpha": alpha}, f, indent=2)
    print(f"Saved to {out_path}")
