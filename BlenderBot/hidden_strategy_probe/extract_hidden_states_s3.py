# coding=utf-8

"""Extract hidden states for probe training (s3).

Based on extract_hidden_states.py.
Supports two probe sources:
- decoder_last: decoder last-layer BOS hidden (same as baseline)
- encoder_mean: encoder last-layer mean-pool over non-pad positions
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence

from inputters import inputters
from models.lora_expert_layers import disable_expert_lora
from utils.building_utils import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--inputter_name', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./_reformat')
    parser.add_argument('--max_input_length', type=int, default=160)
    parser.add_argument('--max_decoder_input_length', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--probe_source', type=str, default='decoder_last',
                        choices=['decoder_last', 'encoder_mean'],
                        help='decoder_last: BOS hidden (default), encoder_mean: mean-pool')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def build_samples(inputter, toker, data_file, max_input_length, max_decoder_input_length):
    kwargs = {
        'max_input_length': max_input_length,
        'max_decoder_input_length': max_decoder_input_length,
    }
    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            inputs = inputter.convert_data_to_inputs(data=data, toker=toker, **kwargs)
            features = inputter.convert_inputs_to_features(inputs=inputs, toker=toker, **kwargs)
            for feat in features:
                label = int(feat.labels[0] - len(toker) + 8)
                samples.append({'input_ids': feat.input_ids, 'label': label})
    return samples


def extract_split(samples, model, toker, batch_size, device, probe_source):
    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id

    all_hidden = []
    all_labels = []

    for start in tqdm.tqdm(range(0, len(samples), batch_size), desc='Extracting'):
        batch = samples[start:start + batch_size]
        input_ids = pad_sequence(
            [torch.tensor(s['input_ids'], dtype=torch.long) for s in batch],
            batch_first=True,
            padding_value=pad,
        ).to(device)
        attention_mask = pad_sequence(
            [torch.ones(len(s['input_ids']), dtype=torch.float) for s in batch],
            batch_first=True,
            padding_value=0.0,
        ).to(device)
        labels = torch.tensor([s['label'] for s in batch], dtype=torch.long)

        with torch.no_grad():
            encoder_outputs = model.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            if probe_source == 'encoder_mean':
                encoder_hidden = encoder_outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).float()
                hidden = (encoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                decoder_input_ids = torch.full((input_ids.size(0), 1), bos, dtype=torch.long).to(device)
                with disable_expert_lora(model):
                    decoder_outputs = model.model.decoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_outputs.last_hidden_state,
                        encoder_attention_mask=attention_mask,
                        return_dict=True,
                        output_hidden_states=True,
                    )
                hidden = decoder_outputs.hidden_states[-1][:, 0, :]

        all_hidden.append(hidden.cpu())
        all_labels.append(labels)

    hidden_key = f'{probe_source}_hidden'
    return {
        hidden_key: torch.cat(all_hidden, dim=0),
        'labels': torch.cat(all_labels, dim=0),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    toker, model = build_model(config_name=args.config_name, checkpoint=args.checkpoint)
    model.to(args.device)
    model.eval()

    inputter = inputters[args.inputter_name]()

    for split in ['train', 'valid', 'test']:
        data_file = os.path.join(args.data_dir, f'{split}.txt')
        print(f'Processing {split}: {data_file}')
        samples = build_samples(
            inputter=inputter,
            toker=toker,
            data_file=data_file,
            max_input_length=args.max_input_length,
            max_decoder_input_length=args.max_decoder_input_length,
        )
        print(f'  samples: {len(samples)}')
        result = extract_split(
            samples=samples,
            model=model,
            toker=toker,
            batch_size=args.batch_size,
            device=args.device,
            probe_source=args.probe_source,
        )
        out_path = os.path.join(args.output_dir, f'{split}_hidden_states.pt')
        torch.save(result, out_path)
        hidden_key = f'{args.probe_source}_hidden'
        print(f'  saved: {out_path} | {hidden_key}={tuple(result[hidden_key].shape)}')


if __name__ == '__main__':
    main()
