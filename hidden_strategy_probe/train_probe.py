# coding=utf-8

"""Train a linear probe on frozen hidden states (decoder last BOS hidden)."""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score


STRATEGY_LIST = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--probe_target', type=str, default='decoder_last')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=13)
    return parser.parse_args()


def load_data(data_dir, probe_target):
    key = f'{probe_target}_hidden'
    splits = {}
    for split in ['train', 'valid', 'test']:
        data = torch.load(os.path.join(data_dir, f'{split}_hidden_states.pt'))
        if key not in data:
            available_targets = sorted(k[:-len('_hidden')] for k in data if k.endswith('_hidden'))
            raise KeyError(f'Unknown probe_target={probe_target}. Available: {available_targets}')
        splits[split] = {
            'hidden': data[key],
            'labels': data['labels'],
        }
    return splits


def train_epoch(probe, optimizer, criterion, hidden, labels, batch_size, device):
    probe.train()
    perm = torch.randperm(hidden.size(0))
    total_loss = 0.0
    n_batches = 0

    for start in range(0, hidden.size(0), batch_size):
        idx = perm[start:start + batch_size]
        x = hidden[idx].to(device)
        y = labels[idx].to(device)

        optimizer.zero_grad()
        logits = probe(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(probe, hidden, labels, device):
    probe.eval()
    with torch.no_grad():
        logits = probe(hidden.to(device))
        preds = logits.argmax(dim=-1).cpu()
    acc = accuracy_score(labels.numpy(), preds.numpy())
    wf1 = f1_score(labels.numpy(), preds.numpy(), average='weighted')
    mf1 = f1_score(labels.numpy(), preds.numpy(), average='macro')
    return acc, wf1, mf1


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Probe target: {args.probe_target}')
    print(f'Device: {device}')

    splits = load_data(args.data_dir, args.probe_target)
    hidden_dim = splits['train']['hidden'].size(1)
    num_classes = len(STRATEGY_LIST)

    probe = nn.Linear(hidden_dim, num_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_valid_acc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            probe, optimizer, criterion,
            splits['train']['hidden'], splits['train']['labels'],
            args.batch_size, device,
        )
        valid_acc, valid_wf1, valid_mf1 = evaluate(
            probe, splits['valid']['hidden'], splits['valid']['labels'], device,
        )
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch {epoch + 1:3d} | loss={train_loss:.4f} | '
                f'valid_acc={valid_acc:.4f} valid_wF1={valid_wf1:.4f} valid_mF1={valid_mf1:.4f}'
            )

    if best_state is None:
        raise RuntimeError('Probe training failed to produce a valid checkpoint')

    probe.load_state_dict(best_state)
    test_acc, test_wf1, test_mf1 = evaluate(
        probe, splits['test']['hidden'], splits['test']['labels'], device,
    )
    valid_acc, valid_wf1, valid_mf1 = evaluate(
        probe, splits['valid']['hidden'], splits['valid']['labels'], device,
    )

    results = {
        'probe_target': args.probe_target,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'epochs': args.epochs,
        'lr': args.lr,
        'best_valid_acc': round(best_valid_acc, 4),
        'valid': {
            'accuracy': round(valid_acc, 4),
            'weighted_f1': round(valid_wf1, 4),
            'macro_f1': round(valid_mf1, 4),
        },
        'test': {
            'accuracy': round(test_acc, 4),
            'weighted_f1': round(test_wf1, 4),
            'macro_f1': round(test_mf1, 4),
        },
        'strategy_labels': STRATEGY_LIST,
    }

    results_path = os.path.join(args.data_dir, f'results_{args.probe_target}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    probe_path = os.path.join(args.data_dir, f'probe_{args.probe_target}.pt')
    torch.save(best_state, probe_path)

    print(f'\n=== Test Results ({args.probe_target}) ===')
    print(f'Accuracy:    {test_acc:.4f}')
    print(f'Weighted F1: {test_wf1:.4f}')
    print(f'Macro F1:    {test_mf1:.4f}')
    print(f'Saved to {results_path}')
    print(f'Probe weights saved to {probe_path}')


if __name__ == '__main__':
    main()
