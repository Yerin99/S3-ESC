# coding=utf-8
"""
Apply Ensemble to Test Set.

Uses optimal weights from grid search to ensemble BlenderBot + SASREC on test set.

Usage:
    python -m ensemble.ensemble \
        --bb_probs_file ./DATA/strat.strat_ss/.../gen.json \
        --sasrec_probs_file ./DATA/sasrec/test_probs.json \
        --weights_file ./DATA/ensemble/grid_search_results.json \
        --output_file ./DATA/ensemble/test_results.json
"""

import argparse
import json
import logging
import os
from typing import List, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Strategy names for reporting
STRATEGY_NAMES = [
    'Question',
    'Restatement or Paraphrasing',
    'Reflection of feelings',
    'Self-disclosure',
    'Affirmation and Reassurance',
    'Providing Suggestions',
    'Information',
    'Others',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Apply ensemble to test set')

    parser.add_argument('--bb_probs_file', type=str, required=True,
                        help='Path to BlenderBot gen.json with pred_strat_id_dist')
    parser.add_argument('--sasrec_probs_file', type=str, required=True,
                        help='Path to SASREC probability dump JSON')
    parser.add_argument('--weights_file', type=str, default=None,
                        help='Path to grid search results JSON (optional, uses best weights)')
    parser.add_argument('--w_bb', type=float, default=None,
                        help='BlenderBot weight (overrides weights_file)')
    parser.add_argument('--w_sr', type=float, default=None,
                        help='SASREC weight (overrides weights_file)')
    parser.add_argument('--output_file', type=str, default='./DATA/ensemble/test_results.json',
                        help='Path to save test results')

    return parser.parse_args()


def load_blenderbot_data(file_path: str) -> List[dict]:
    """Load BlenderBot gen.json data."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = []
    for item in data:
        if 'pred_strat_id_dist' not in item:
            continue

        prob_str = item['pred_strat_id_dist']
        prob = np.array([float(x) for x in prob_str.split()])

        results.append({
            'prob': prob,
            'pred': item.get('pred_strat_id', prob.argmax()),
            'sample_id': item.get('sample_id', -1),
        })

    return results


def load_sasrec_data(file_path: str) -> List[dict]:
    """Load SASREC probability dump."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = []
    for item in data:
        results.append({
            'prob': np.array(item['probs']),
            'pred': item['pred'],
            'target': item['target'],
            'dialog_idx': item['dialog_idx'],
            'turn_idx': item['turn_idx'],
        })

    return results


def align_samples(bb_data: List[dict], sasrec_data: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align BlenderBot and SASREC samples."""
    if len(bb_data) != len(sasrec_data):
        logger.warning(f"Sample count mismatch: BB={len(bb_data)}, SASREC={len(sasrec_data)}")
        min_len = min(len(bb_data), len(sasrec_data))
        bb_data = bb_data[:min_len]
        sasrec_data = sasrec_data[:min_len]

    bb_probs = np.stack([item['prob'] for item in bb_data])
    sasrec_probs = np.stack([item['prob'] for item in sasrec_data])
    targets = np.array([item['target'] for item in sasrec_data])

    return bb_probs, sasrec_probs, targets


def compute_metrics(preds: np.ndarray, targets: np.ndarray, probs: np.ndarray) -> dict:
    """Compute comprehensive evaluation metrics."""
    # Basic accuracy
    accuracy = (preds == targets).mean()

    # Top-3 accuracy
    top3_preds = np.argsort(probs, axis=1)[:, -3:]
    top3_correct = np.any(top3_preds == targets.reshape(-1, 1), axis=1)
    top3_accuracy = top3_correct.mean()

    # F1 scores
    f1_micro = f1_score(targets, preds, average='micro')
    f1_macro = f1_score(targets, preds, average='macro')
    f1_weighted = f1_score(targets, preds, average='weighted')

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(STRATEGY_NAMES):
        mask = targets == i
        if mask.sum() > 0:
            per_class_acc[name] = float((preds[mask] == targets[mask]).mean())
        else:
            per_class_acc[name] = None

    # Confusion matrix
    cm = confusion_matrix(targets, preds)

    return {
        'accuracy': float(accuracy),
        'top3_accuracy': float(top3_accuracy),
        'f1_micro': float(f1_micro),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(targets),
    }


def ensemble_predict(
    bb_probs: np.ndarray,
    sasrec_probs: np.ndarray,
    w_bb: float,
    w_sr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ensemble predictions.

    Returns:
        ensemble_probs: (num_samples, 8) ensemble probability distributions
        preds: (num_samples,) predicted class indices
    """
    # Ensemble: p_final = w_bb * p_bb + w_sr * p_sr
    ensemble_probs = w_bb * bb_probs + w_sr * sasrec_probs

    # Normalize
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

    # Predictions
    preds = ensemble_probs.argmax(axis=1)

    return ensemble_probs, preds


def main():
    args = parse_args()

    # Get weights
    if args.w_bb is not None and args.w_sr is not None:
        w_bb = args.w_bb
        w_sr = args.w_sr
        logger.info(f"Using provided weights: w_bb={w_bb}, w_sr={w_sr}")
    elif args.weights_file is not None:
        with open(args.weights_file, 'r') as f:
            weights_data = json.load(f)
        w_bb = weights_data['best']['w_bb']
        w_sr = weights_data['best']['w_sr']
        logger.info(f"Using weights from {args.weights_file}: w_bb={w_bb}, w_sr={w_sr}")
    else:
        raise ValueError("Either --weights_file or both --w_bb and --w_sr must be provided")

    # Load data
    logger.info(f"Loading BlenderBot data from {args.bb_probs_file}")
    bb_data = load_blenderbot_data(args.bb_probs_file)
    logger.info(f"Loaded {len(bb_data)} BlenderBot samples")

    logger.info(f"Loading SASREC data from {args.sasrec_probs_file}")
    sasrec_data = load_sasrec_data(args.sasrec_probs_file)
    logger.info(f"Loaded {len(sasrec_data)} SASREC samples")

    # Align samples
    bb_probs, sasrec_probs, targets = align_samples(bb_data, sasrec_data)
    logger.info(f"Aligned {len(targets)} samples")

    # Compute ensemble predictions
    ensemble_probs, ensemble_preds = ensemble_predict(bb_probs, sasrec_probs, w_bb, w_sr)

    # Compute metrics for all methods
    bb_preds = bb_probs.argmax(axis=1)
    sasrec_preds = sasrec_probs.argmax(axis=1)

    bb_metrics = compute_metrics(bb_preds, targets, bb_probs)
    sasrec_metrics = compute_metrics(sasrec_preds, targets, sasrec_probs)
    ensemble_metrics = compute_metrics(ensemble_preds, targets, ensemble_probs)

    # Print results
    print(f"\n{'='*70}")
    print(f"TEST SET EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Method':<20} {'Accuracy':>10} {'Top-3 Acc':>12} {'F1 Macro':>10} {'F1 Micro':>10}")
    print(f"{'-'*70}")
    print(f"{'BlenderBot':<20} {bb_metrics['accuracy']:>10.4f} {bb_metrics['top3_accuracy']:>12.4f} {bb_metrics['f1_macro']:>10.4f} {bb_metrics['f1_micro']:>10.4f}")
    print(f"{'SASREC':<20} {sasrec_metrics['accuracy']:>10.4f} {sasrec_metrics['top3_accuracy']:>12.4f} {sasrec_metrics['f1_macro']:>10.4f} {sasrec_metrics['f1_micro']:>10.4f}")
    print(f"{'Ensemble':<20} {ensemble_metrics['accuracy']:>10.4f} {ensemble_metrics['top3_accuracy']:>12.4f} {ensemble_metrics['f1_macro']:>10.4f} {ensemble_metrics['f1_micro']:>10.4f}")
    print(f"  (w_bb={w_bb:.2f}, w_sr={w_sr:.2f})")
    print(f"{'='*70}")

    improvement = ensemble_metrics['accuracy'] - bb_metrics['accuracy']
    print(f"\nImprovement over BlenderBot: {improvement:+.4f} ({improvement*100:+.2f}%)")
    print(f"{'='*70}\n")

    # Per-class accuracy
    print(f"\nPer-class Accuracy:")
    print(f"{'Strategy':<35} {'BB':>8} {'SASREC':>8} {'Ensemble':>10}")
    print(f"{'-'*65}")
    for name in STRATEGY_NAMES:
        bb_acc = bb_metrics['per_class_accuracy'].get(name)
        sr_acc = sasrec_metrics['per_class_accuracy'].get(name)
        ens_acc = ensemble_metrics['per_class_accuracy'].get(name)

        bb_str = f"{bb_acc:.4f}" if bb_acc is not None else "N/A"
        sr_str = f"{sr_acc:.4f}" if sr_acc is not None else "N/A"
        ens_str = f"{ens_acc:.4f}" if ens_acc is not None else "N/A"

        print(f"{name:<35} {bb_str:>8} {sr_str:>8} {ens_str:>10}")
    print()

    # Classification report for ensemble
    print("\nEnsemble Classification Report:")
    print(classification_report(targets, ensemble_preds, target_names=STRATEGY_NAMES))

    # Save results
    output = {
        'weights': {'w_bb': w_bb, 'w_sr': w_sr},
        'blenderbot': bb_metrics,
        'sasrec': sasrec_metrics,
        'ensemble': ensemble_metrics,
        'improvement': {
            'accuracy': float(improvement),
            'top3_accuracy': float(ensemble_metrics['top3_accuracy'] - bb_metrics['top3_accuracy']),
        },
        'config': {
            'bb_probs_file': args.bb_probs_file,
            'sasrec_probs_file': args.sasrec_probs_file,
            'weights_file': args.weights_file,
        }
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {args.output_file}")


if __name__ == '__main__':
    main()
