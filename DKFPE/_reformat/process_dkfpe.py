# coding=utf-8
"""
DKFPE Data Preprocessing for 8:1:1 utterance-pair level split

Converts TransESC TSV format to codes_zcj JSON format.
Maps PESConv persona to each sample.

Usage:
    python process_dkfpe.py --mode 811
    python process_dkfpe.py --mode 701515
"""

import argparse
import json
import os
import re
from collections import defaultdict


EMOTION_LIST = [
    'anger', 'anxiety', 'depression', 'disgust', 'fear',
    'guilt', 'jealousy', 'nervousness', 'pain', 'sadness', 'shame'
]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LIST)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTION_LIST)}


def load_pesconv(pesconv_path):
    """Load PESConv and build situation-based index."""
    with open(pesconv_path) as f:
        pesconv = json.load(f)
    return pesconv


def load_esconv(esconv_path):
    """Load ESConv dataset."""
    with open(esconv_path) as f:
        esconv = json.load(f)
    return esconv


def normalize_text(s):
    """Normalize text for matching: lowercase, collapse whitespace."""
    return ' '.join(s.strip().lower().split())


def build_situation_index(esconv):
    """Build index: normalized situation_text -> list of ESConv indices."""
    idx = defaultdict(list)
    for i, conv in enumerate(esconv):
        sit = normalize_text(conv['situation'])
        idx[sit].append(i)
    return idx


def extract_persona_text(pesconv_entry):
    """Extract clean persona text from PESConv entry.

    PESConv format: "<persona> sent1. <persona> sent2. ... <input>"
    Returns: "sent1. sent2. ..."
    """
    persona_raw = pesconv_entry.get('persona', '')
    # Remove <persona> tags and <input> tag
    persona_raw = persona_raw.replace('<input>', '').strip()
    sentences = [s.strip() for s in persona_raw.split('<persona>') if s.strip()]
    return ' '.join(sentences)


def parse_tsv_line(line):
    """Parse a TransESC TSV line into context utterances and target.

    Format: each utterance is "emotion_idx speaker_id turn_id [strategy] text"
    separated by " EOS "

    Returns:
        emotion_idx: int
        context: list of dicts with 'text', 'speaker', optionally 'strategy'
        target_text: str
        target_strategy: str
        last_user_utterance: str
    """
    parts = line.strip().split(' EOS ')
    parts = [p.strip() for p in parts if p.strip()]

    utterances = []
    for part in parts:
        tokens = part.split(' ', 3)
        emotion_idx = int(tokens[0])
        speaker_id = int(tokens[1])  # 0=user, 1=sys
        turn_id = int(tokens[2])
        text = tokens[3] if len(tokens) > 3 else ''

        speaker = 'usr' if speaker_id == 0 else 'sys'

        strategy = None
        if speaker == 'sys' and text.startswith('['):
            # Extract strategy tag
            match = re.match(r'\[([^\]]+)\]\s*(.*)', text)
            if match:
                strategy = match.group(1)
                text = match.group(2)

        utterances.append({
            'text': text,
            'speaker': speaker,
            'strategy': strategy,
            'emotion_idx': emotion_idx,
        })

    # The last utterance is the target (always sys)
    target = utterances[-1]
    context = utterances[:-1]

    # Find last user utterance (kept for backward compatibility)
    last_user_utt = ''
    for utt in reversed(context):
        if utt['speaker'] == 'usr':
            last_user_utt = utt['text']
            break

    # Per-target: find preceding user utterance for each sys turn
    user_utterances_by_target = []
    for i in range(1, len(utterances)):
        if utterances[i]['speaker'] == 'sys':
            preceding_user = ''
            for j in range(i - 1, -1, -1):
                if utterances[j]['speaker'] == 'usr':
                    preceding_user = utterances[j]['text']
                    break
            user_utterances_by_target.append({
                'target_turn': i,
                'user_utterance': preceding_user,
            })

    return {
        'emotion_idx': utterances[0]['emotion_idx'],
        'context': context,
        'target_text': target['text'],
        'target_strategy': target['strategy'],
        'last_user_utterance': last_user_utt,
        'user_utterances_by_target': user_utterances_by_target,
    }


def process_811(args):
    """Process 8:1:1 utterance-pair level split from TransESC."""
    esconv = load_esconv(args.esconv_path)
    pesconv = load_pesconv(args.pesconv_path)
    sit_index = build_situation_index(esconv)

    splits = {
        'train': ('trainWithStrategy_short.tsv', 'trainSituation.txt'),
        'valid': ('devWithStrategy_short.tsv', 'devSituation.txt'),
        'test': ('testWithStrategy_short.tsv', 'testSituation.txt'),
    }

    for split_name, (tsv_file, sit_file) in splits.items():
        tsv_path = os.path.join(args.transesc_dir, tsv_file)
        sit_path = os.path.join(args.transesc_dir, sit_file)

        with open(tsv_path) as f:
            tsv_lines = f.readlines()
        with open(sit_path) as f:
            situations = [s.strip() for s in f.readlines()]

        # Situation files may have one extra trailing line
        situations = situations[:len(tsv_lines)]

        output_data = []
        no_match_count = 0

        for i, (line, situation) in enumerate(zip(tsv_lines, situations)):
            parsed = parse_tsv_line(line)

            # Match to ESConv/PESConv by normalized situation
            sit_key = normalize_text(situation)
            esconv_indices = sit_index.get(sit_key, [])

            persona = ''
            emotion_type = IDX_TO_EMOTION.get(parsed['emotion_idx'], 'unknown')

            if esconv_indices:
                # Use first match
                esconv_idx = esconv_indices[0]
                persona = extract_persona_text(pesconv[esconv_idx])
                emotion_type = esconv[esconv_idx]['emotion_type']
                situation_text = esconv[esconv_idx]['situation']
            else:
                situation_text = situation
                no_match_count += 1

            # Build codes_zcj format dialog
            dialog = []
            for utt in parsed['context']:
                entry = {'text': utt['text'], 'speaker': utt['speaker']}
                if utt['strategy']:
                    entry['strategy'] = utt['strategy']
                dialog.append(entry)

            # Add target
            dialog.append({
                'text': parsed['target_text'],
                'speaker': 'sys',
                'strategy': parsed['target_strategy'] or 'Others',
            })

            sample = {
                'emotion_type': emotion_type,
                'situation': situation_text,
                'persona': persona,
                'last_user_utterance': parsed['last_user_utterance'],
                'user_utterances_by_target': parsed['user_utterances_by_target'],
                'dialog': dialog,
            }
            output_data.append(sample)

        # Write output
        output_path = os.path.join(args.output_dir, f'{split_name}.txt')
        with open(output_path, 'w') as f:
            for sample in output_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"{split_name}: {len(output_data)} samples written to {output_path}")
        if no_match_count > 0:
            print(f"  WARNING: {no_match_count} samples could not be matched to PESConv")


def process_701515(args):
    """Process 70:15:15 conversation-level split from ESC.

    Uses existing train.txt/valid.txt/test.txt from ESC/codes_zcj/_reformat/
    and adds persona information from PESConv.
    """
    esconv = load_esconv(args.esconv_path)
    pesconv = load_pesconv(args.pesconv_path)
    sit_index = build_situation_index(esconv)

    # Build ESConv index by (emotion_type, situation) for precise matching
    precise_index = {}
    for i, conv in enumerate(esconv):
        key = (conv['emotion_type'], normalize_text(conv['situation']))
        precise_index[key] = i

    for split_name in ['train', 'valid', 'test']:
        input_path = os.path.join(args.esc_reformat_dir, f'{split_name}.txt')
        output_path = os.path.join(args.output_dir, f'{split_name}.txt')

        with open(input_path) as f:
            lines = f.readlines()

        output_data = []
        no_match_count = 0

        for line in lines:
            data = json.loads(line)

            # Find matching PESConv entry
            key = (data['emotion_type'], normalize_text(data['situation']))
            esconv_idx = precise_index.get(key, None)

            if esconv_idx is not None:
                persona = extract_persona_text(pesconv[esconv_idx])
            else:
                persona = ''
                no_match_count += 1

            data['persona'] = persona

            # Add fields needed by COMET/filter pipeline
            dialog = data['dialog']
            user_utterances_by_target = []
            for i in range(1, len(dialog)):
                if dialog[i]['speaker'] == 'sys':
                    preceding_user = ''
                    for j in range(i - 1, -1, -1):
                        if dialog[j]['speaker'] == 'usr':
                            preceding_user = dialog[j]['text']
                            break
                    user_utterances_by_target.append({
                        'target_turn': i,
                        'user_utterance': preceding_user,
                    })
            data['user_utterances_by_target'] = user_utterances_by_target

            last_user = ''
            for d in reversed(dialog):
                if d['speaker'] == 'usr':
                    last_user = d['text']
                    break
            data['last_user_utterance'] = last_user

            output_data.append(data)

        with open(output_path, 'w') as f:
            for sample in output_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"{split_name}: {len(output_data)} conversations written to {output_path}")
        if no_match_count > 0:
            print(f"  WARNING: {no_match_count} conversations could not be matched to PESConv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['811', '701515'])
    parser.add_argument('--esconv_path', type=str, default='./ESConv.json')
    parser.add_argument('--pesconv_path', type=str, default='./PESConv.json')
    parser.add_argument('--transesc_dir', type=str,
                        default=os.environ.get('TRANSESC_DIR', './'))
    parser.add_argument('--esc_reformat_dir', type=str,
                        default=os.environ.get('ESC_REFORMAT_DIR', '../_reformat'))
    parser.add_argument('--output_dir', type=str, default='.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == '811':
        process_811(args)
    else:
        process_701515(args)
