# coding=utf-8
"""
DeBERTa Knowledge Filter Training for DKFPE

Trains a DeBERTa-v3-large binary classifier on ComFact data.
LR: 2e-7, Batch: 16, Adam (β1=0.9, β2=0.999), Max 10 epochs.

Usage:
    python train_filter.py \
        --deberta_path ./external_models/deberta-v3-large \
        --comfact_dir ./external_models/comfact/processed \
        --output_dir ./DATA/knowledge_filter \
        --num_epochs 10 --batch_size 16 --learning_rate 2e-7
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import logging
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models.knowledge_filter import KnowledgeFilter

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class ComFactDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        knowledge = item['knowledge']
        label = item['label']

        encoding = self.tokenizer(
            context, knowledge,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            total_loss += outputs['loss'].item()
            n_batches += 1

            preds = torch.argmax(outputs['logits'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, acc, f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deberta_path', type=str,
                        default='./external_models/deberta-v3-large')
    parser.add_argument('--comfact_dir', type=str,
                        default='./external_models/comfact/processed')
    parser.add_argument('--output_dir', type=str,
                        default='./DATA/knowledge_filter')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-7)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.deberta_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.deberta_path)

    # Load data
    logger.info(f"Loading ComFact data from {args.comfact_dir}")
    with open(os.path.join(args.comfact_dir, 'train.json')) as f:
        train_data = json.load(f)
    with open(os.path.join(args.comfact_dir, 'val.json')) as f:
        val_data = json.load(f)
    with open(os.path.join(args.comfact_dir, 'test.json')) as f:
        test_data = json.load(f)

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create datasets and dataloaders
    train_dataset = ComFactDataset(train_data, tokenizer, args.max_length)
    val_dataset = ComFactDataset(val_data, tokenizer, args.max_length)
    test_dataset = ComFactDataset(test_data, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Build model
    logger.info(f"Loading DeBERTa model from {args.deberta_path}")
    model = KnowledgeFilter(model_path=args.deberta_path)
    model.to(args.device)

    # Optimizer: Adam with β1=0.9, β2=0.999
    optimizer = Adam(model.parameters(), lr=args.learning_rate,
                     betas=(0.9, 0.999))

    # Training
    best_val_acc = 0
    best_epoch = -1

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        start_time = time.time()

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss'] / args.gradient_accumulation_steps
            loss.backward()

            total_loss += outputs['loss'].item()
            n_batches += 1

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / n_batches
                logger.info(f"Epoch {epoch}, Step {batch_idx + 1}, Loss: {avg_loss:.4f}")

        epoch_time = time.time() - start_time
        avg_train_loss = total_loss / max(n_batches, 1)

        # Evaluate on validation set
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, args.device)
        logger.info(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, "
            f"time={epoch_time:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pt'))
            logger.info(f"  New best model saved (acc={val_acc:.4f})")

        # Save epoch checkpoint
        torch.save(model.state_dict(),
                    os.path.join(args.output_dir, f'epoch-{epoch}.pt'))

    # Evaluate on test set with best model
    logger.info(f"\nLoading best model from epoch {best_epoch}")
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, 'best_model.pt'),
                    map_location=args.device)
    )
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, args.device
    )
    logger.info(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}, f1={test_f1:.4f}")
    logger.info(f"\n{classification_report(test_labels, test_preds, target_names=['IRR', 'RR'])}")

    # Save results
    results = {
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'args': vars(args),
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
