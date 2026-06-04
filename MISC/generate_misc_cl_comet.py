#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate COMET inferences for MISC conversation-level split data.

Uses COMET-ATOMIC-2020 BART model (mismayil/comet-bart-ai2).

For each sample generates:
  - COMET for seeker's last post x  (9 relations, 3 values each → ≤27 blocks)
  - COMET for situation s           (9 relations, 2 values each → ≤18 blocks)
    (situation COMET is cached: shared across samples from the same conversation)

Output files in ./dataset/:
  trainComet_cl.txt / devComet_cl.txt / testComet_cl.txt
  trainComet_st_cl.txt / devComet_st_cl.txt / testComet_st_cl.txt

COMET format per line:
  [REL1] val1__EOS__[REL1] val2__EOS__...[REL9] val1__EOS__...
  (one line per sample, matching the corresponding data file line)
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

COMET_MODEL_PATH = '<COMET_BART_PATH>'
MISC_DATASET     = Path('./dataset')

RELATIONS = ['xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant',
             'oWant', 'oEffect', 'oReact']

# Number of generated values per relation
N_X = 3   # for seeker's last post x  (9 × 3 = 27 ≤ 30 max_num_attr)
N_S = 2   # for situation s           (9 × 2 = 18 ≤ 20 max_num_attr)


def load_comet(device):
    print('Loading COMET-BART model...')
    tokenizer = BartTokenizer.from_pretrained(COMET_MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(COMET_MODEL_PATH).to(device)
    model.eval()
    print('✓ COMET-BART loaded')
    return tokenizer, model


def generate_comet_blocks(text: str, relation: str,
                          tokenizer, model, device,
                          n_return: int) -> list:
    """
    Returns a list of generated tail strings for (text, relation).
    Input format: "{text} {relation}" (COMET-ATOMIC-2020-BART format)
    """
    if not text.strip():
        return ['none'] * n_return

    inp_str = f'{text} {relation}'
    inp_ids = tokenizer.encode(inp_str, return_tensors='pt').to(device)

    with torch.no_grad():
        out = model.generate(
            inp_ids,
            max_length=20,
            num_beams=max(n_return, 5),
            num_return_sequences=n_return,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True,
        )

    results = []
    for seq in out:
        tail = tokenizer.decode(seq, skip_special_tokens=True).strip()
        tail = tail.split('.')[0].split('\n')[0].strip()
        if not tail:
            tail = 'none'
        results.append(tail)
    return results


def build_comet_line(text: str, n_return: int,
                     tokenizer, model, device) -> str:
    """
    Generate COMET blocks for all 9 relations and format as MISC COMET line.
    Format: '[REL] val1__EOS__[REL] val2__EOS__...'
    """
    parts = []
    for rel in RELATIONS:
        values = generate_comet_blocks(text, rel, tokenizer, model, device, n_return)
        for v in values:
            parts.append(f'[{rel}] {v}__EOS__')
    return ''.join(parts)


def generate_for_split(split: str, comet_inputs: list,
                       tokenizer, model, device):
    """
    Generate COMET files for one split.
    Uses situation-level caching to avoid re-generating for same situation.
    """
    n_samples = len(comet_inputs)
    print(f'\n[{split}] Generating COMET for {n_samples} samples...')

    comet_x_lines  = []   # COMET for last user post x
    comet_st_lines = []   # COMET for situation s

    # Cache: situation text → precomputed COMET line
    situation_cache = {}

    for item in tqdm(comet_inputs, desc=split):
        x = item['x']
        s = item['s']

        # Generate COMET for x (last user utterance)
        x_line = build_comet_line(x, N_X, tokenizer, model, device)
        comet_x_lines.append(x_line)

        # Generate COMET for s (cached by situation text)
        if s not in situation_cache:
            situation_cache[s] = build_comet_line(s, N_S, tokenizer, model, device)
        comet_st_lines.append(situation_cache[s])

    # Save
    prefix = split  # 'train' / 'dev' / 'test'
    (MISC_DATASET / f'{prefix}Comet_cl.txt').write_text(
        '\n'.join(comet_x_lines) + '\n', encoding='utf-8')
    (MISC_DATASET / f'{prefix}Comet_st_cl.txt').write_text(
        '\n'.join(comet_st_lines) + '\n', encoding='utf-8')

    print(f'[{split}] Done. Unique situations cached: {len(situation_cache)}')
    print(f'  → {prefix}Comet_cl.txt ({len(comet_x_lines)} lines)')
    print(f'  → {prefix}Comet_st_cl.txt ({len(comet_st_lines)} lines)')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    tokenizer, model = load_comet(device)

    # Load COMET inputs prepared by prepare_misc_cl_data.py
    comet_json = MISC_DATASET / 'comet_inputs_cl.json'
    with open(comet_json, encoding='utf-8') as f:
        all_inputs = json.load(f)

    for split in ['train', 'dev', 'test']:
        generate_for_split(split, all_inputs[split], tokenizer, model, device)

    print('\n✓ All COMET files generated.')
    print('Next step: update BlenderEmotionalSupport.py to use _cl files.')


if __name__ == '__main__':
    main()
