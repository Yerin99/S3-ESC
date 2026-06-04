#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare MISC data with conversation-level split (70:15:15).

Uses the same ESConv conversation split as ESC/GLHG/KEMI:
  - train: 910 conversations → ~12,271 samples
  - valid: 195 conversations → ~2,621 samples
  - test:  195 conversations → ~2,801 samples

Outputs per split:
  - train_cl.txt         : MISC TSV format (emotion_id role turn_idx text EOS ...)
  - trainSituation_cl.txt: situation text per sample (one line per sample)
  - comet_inputs_cl.json : (x, s) pairs for COMET generation
"""

import json
from pathlib import Path


# Emotion mapping derived from frequency distribution comparison
# between ESConv emotion_type strings and MISC integer IDs.
EMOTION_MAP = {
    'anxiety':    0,
    'anger':      1,
    'fear':       2,
    'depression': 3,
    'sadness':    4,
    'shame':      5,
    'disgust':    6,
    'nervousness':7,
    'guilt':      8,
    'jealousy':   9,
    'pain':       10,
}

ESCONV_BASE = Path('../_reformat')
MISC_DATASET = Path('./dataset')


def process_split(split_name: str):
    """
    Process one split (train / valid / test) and write output files.

    Returns:
        list of dicts with keys: x (last user text), s (situation), split, sample_idx
    """
    input_path = ESCONV_BASE / f'{split_name}.txt'
    if split_name == 'valid':
        out_prefix = 'dev'
    else:
        out_prefix = split_name

    data_out   = MISC_DATASET / f'{out_prefix}_cl.txt'
    sit_out    = MISC_DATASET / f'{out_prefix}Situation_cl.txt'

    with open(input_path, encoding='utf-8') as f:
        conversations = [json.loads(line) for line in f if line.strip()]

    samples = []       # MISC TSV lines
    situations = []    # situation text per sample
    comet_inputs = []  # (x, s) for COMET generation

    for conv in conversations:
        dialog        = conv['dialog']
        emotion_type  = conv.get('emotion_type', 'anxiety').lower()
        emotion_id    = EMOTION_MAP.get(emotion_type, 0)
        situation     = conv.get('situation', '').strip()

        # Tracks the last user utterance seen so far
        last_user_text = ''

        for i, turn in enumerate(dialog):
            if i == 0:
                # Update last_user_text even at i==0 if it's a user turn
                if turn['speaker'] == 'usr':
                    last_user_text = turn['text'].strip()
                continue  # skip first turn (always excluded by i > 0 filter)

            if turn['speaker'] != 'sys':
                last_user_text = turn['text'].strip()
                continue

            # --- Build MISC TSV line for this sys turn ---
            # Context: turns 0 .. i-1
            # Target:  turn i (sys)
            parts = []
            for j in range(i):
                t = dialog[j]
                role = 1 if t['speaker'] == 'sys' else 0
                text = t['text'].strip()
                if t['speaker'] == 'sys':
                    strategy = t.get('strategy', 'Others')
                    text = f'[{strategy}] {text}'
                parts.append(f'{emotion_id} {role} {j} {text}')

            # Target sys turn
            strategy = turn.get('strategy', 'Others')
            target_text = f'[{strategy}] {turn["text"].strip()}'
            parts.append(f'{emotion_id} 1 {i} {target_text}')

            sample_line = ' EOS '.join(parts)
            samples.append(sample_line)
            situations.append(situation)

            # Record (x, s) for COMET generation
            # x = last user utterance before this sys turn
            # s = situation (constant per conversation)
            comet_inputs.append({
                'x': last_user_text,
                's': situation,
                'split': out_prefix,
                'idx': len(comet_inputs),
            })

            # Update last_user_text — after the sys turn, the next user turn will update it.
            # (We record before this sys turn, so no update needed here.)

    # Write data file
    with open(data_out, 'w', encoding='utf-8') as f:
        for line in samples:
            f.write(line + '\n')

    # Write situation file
    with open(sit_out, 'w', encoding='utf-8') as f:
        for s in situations:
            f.write(s + '\n')

    print(f'[{split_name}] {len(samples)} samples → {data_out.name}, {sit_out.name}')
    return comet_inputs


def main():
    MISC_DATASET.mkdir(exist_ok=True)

    all_comet_inputs = {}
    for split in ['train', 'valid', 'test']:
        ci = process_split(split)
        out_prefix = 'dev' if split == 'valid' else split
        all_comet_inputs[out_prefix] = ci

    # Save all (x, s) pairs for COMET generation
    comet_json = MISC_DATASET / 'comet_inputs_cl.json'
    with open(comet_json, 'w', encoding='utf-8') as f:
        json.dump(all_comet_inputs, f, ensure_ascii=False, indent=2)
    print(f'\nSaved COMET inputs → {comet_json}')
    print('\nNext step: run generate_misc_cl_comet.py')


if __name__ == '__main__':
    main()
