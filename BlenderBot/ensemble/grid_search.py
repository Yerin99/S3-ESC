# coding=utf-8
"""
Grid Search for Ensemble Weights.

Finds optimal (w_bb, w_sr) weights on validation set.

Usage:
    python -m ensemble.grid_search \
        --bb_probs_file ./DATA/strat.strat_ss/.../gen.json \
        --sasrec_probs_file ./DATA/sasrec/valid_probs.json \
        --output_file ./DATA/ensemble/grid_search_results.json
"""

import argparse
import json
import logging
import os
from typing import List, Tuple

import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Grid search for ensemble weights')

    parser.add_argument('--bb_probs_file', type=str, required=True,
                        help='Path to BlenderBot gen.json with pred_strat_id_dist')
    parser.add_argument('--sasrec_probs_file', type=str, required=True,
                        help='Path to SASREC probability dump JSON')
    parser.add_argument('--output_file', type=str, default='./DATA/ensemble/grid_search_results.json',
                        help='Path to save grid search results')
    parser.add_argument('--step', type=float, default=0.05,
                        help='Step size for weight grid search (default: 0.05)')

    return parser.parse_args()


def load_blenderbot_probs(file_path: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load BlenderBot probability distributions from gen.json.

    Returns:
        probs: List of probability arrays (num_samples, 8)
        targets: List of ground truth strategy IDs
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    probs = []
    targets = []

    for item in data:
        if 'pred_strat_id_dist' not in item:
            logger.warning(f"Missing pred_strat_id_dist in item: {item.get('sample_id', 'unknown')}")
            continue

        # Parse probability string
        prob_str = item['pred_strat_id_dist']
        prob = np.array([float(x) for x in prob_str.split()])
        probs.append(prob)

        # Ground truth from other_res (cls_strat_id was mapped to pred_strat_id)
        # The target is implicit - we need to get it from the classification report
        # Actually, looking at the code, cls_strat_id should be in other_res
        # Let's check if there's a way to get the target

        # From gen.json, pred_strat_id is the model's prediction
        # We need the ground truth, which should be stored somewhere

    # Note: gen.json doesn't directly contain ground truth labels
    # We need to extract from the evaluation metrics or re-compute
    # For now, we'll need to handle this differently

    return probs, targets


def load_blenderbot_data(file_path: str) -> List[dict]:
    """
    Load BlenderBot gen.json data.

    Returns:
        List of dicts with 'prob', 'pred', and 'sample_id'
    """
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
    """
    Load SASREC probability dump.

    Returns:
        List of dicts with 'prob', 'pred', 'target', 'dialog_idx', 'turn_idx'
    """
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
    """
    Align BlenderBot and SASREC samples.

    Both should have the same number of samples in the same order.

    Returns:
        bb_probs: (num_samples, 8)
        sasrec_probs: (num_samples, 8)
        targets: (num_samples,)
    """
    if len(bb_data) != len(sasrec_data):
        logger.warning(f"Sample count mismatch: BB={len(bb_data)}, SASREC={len(sasrec_data)}")
        # Try to align by minimum
        min_len = min(len(bb_data), len(sasrec_data))
        bb_data = bb_data[:min_len]
        sasrec_data = sasrec_data[:min_len]

    bb_probs = np.stack([item['prob'] for item in bb_data])
    sasrec_probs = np.stack([item['prob'] for item in sasrec_data])
    targets = np.array([item['target'] for item in sasrec_data])

    return bb_probs, sasrec_probs, targets


def compute_ensemble_accuracy(
    bb_probs: np.ndarray,
    sasrec_probs: np.ndarray,
    targets: np.ndarray,
    w_bb: float,
    w_sr: float,
) -> dict:
    """
    Compute ensemble accuracy with given weights.

    Args:
        bb_probs: (num_samples, 8) BlenderBot probabilities
        sasrec_probs: (num_samples, 8) SASREC probabilities
        targets: (num_samples,) ground truth labels
        w_bb: BlenderBot weight
        w_sr: SASREC weight

    Returns:
        dict with accuracy metrics
    """
    # Ensemble: p_final = w_bb * p_bb + w_sr * p_sr
    ensemble_probs = w_bb * bb_probs + w_sr * sasrec_probs

    # Normalize (in case weights don't sum to 1)
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

    # Top-1 prediction
    preds = ensemble_probs.argmax(axis=1)
    accuracy = (preds == targets).mean()

    # Top-3 accuracy
    top3_preds = np.argsort(ensemble_probs, axis=1)[:, -3:]
    top3_correct = np.any(top3_preds == targets.reshape(-1, 1), axis=1)
    top3_accuracy = top3_correct.mean()

    return {
        'accuracy': float(accuracy),
        'top3_accuracy': float(top3_accuracy),
        'w_bb': w_bb,
        'w_sr': w_sr,
    }


def grid_search(
    bb_probs: np.ndarray,
    sasrec_probs: np.ndarray,
    targets: np.ndarray,
    step: float = 0.05,
) -> Tuple[dict, List[dict]]:
    """
    Perform grid search for optimal ensemble weights.

    Args:
        bb_probs: (num_samples, 8)
        sasrec_probs: (num_samples, 8)
        targets: (num_samples,)
        step: step size for weight search

    Returns:
        best_result: dict with best weights and accuracy
        all_results: list of all (w_bb, w_sr, accuracy) tuples
    """
    best_result = None
    best_accuracy = -1
    all_results = []

    # Generate weight pairs
    # w_bb + w_sr = 1 is not required, but common choice
    # We'll search w_bb from 0 to 1, w_sr = 1 - w_bb
    weights = np.arange(0, 1 + step, step)

    logger.info(f"Grid search with step={step}, {len(weights)} weight values")

    for w_bb in weights:
        w_sr = 1 - w_bb

        result = compute_ensemble_accuracy(
            bb_probs, sasrec_probs, targets, w_bb, w_sr
        )
        all_results.append(result)

        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_result = result

        logger.info(f"w_bb={w_bb:.2f}, w_sr={w_sr:.2f}: acc={result['accuracy']:.4f}, top3={result['top3_accuracy']:.4f}")

    return best_result, all_results


def main():
    args = parse_args()

    logger.info(f"Loading BlenderBot data from {args.bb_probs_file}")
    bb_data = load_blenderbot_data(args.bb_probs_file)
    logger.info(f"Loaded {len(bb_data)} BlenderBot samples")

    logger.info(f"Loading SASREC data from {args.sasrec_probs_file}")
    sasrec_data = load_sasrec_data(args.sasrec_probs_file)
    logger.info(f"Loaded {len(sasrec_data)} SASREC samples")

    # Align samples
    bb_probs, sasrec_probs, targets = align_samples(bb_data, sasrec_data)
    logger.info(f"Aligned {len(targets)} samples")

    # Compute baseline accuracies
    bb_only = compute_ensemble_accuracy(bb_probs, sasrec_probs, targets, 1.0, 0.0)
    sasrec_only = compute_ensemble_accuracy(bb_probs, sasrec_probs, targets, 0.0, 1.0)

    logger.info(f"\n{'='*50}")
    logger.info(f"Baseline Results:")
    logger.info(f"  BlenderBot only: acc={bb_only['accuracy']:.4f}, top3={bb_only['top3_accuracy']:.4f}")
    logger.info(f"  SASREC only: acc={sasrec_only['accuracy']:.4f}, top3={sasrec_only['top3_accuracy']:.4f}")
    logger.info(f"{'='*50}\n")

    # Grid search
    best_result, all_results = grid_search(bb_probs, sasrec_probs, targets, args.step)

    logger.info(f"\n{'='*50}")
    logger.info(f"Best Ensemble Result:")
    logger.info(f"  w_bb={best_result['w_bb']:.2f}, w_sr={best_result['w_sr']:.2f}")
    logger.info(f"  Accuracy: {best_result['accuracy']:.4f}")
    logger.info(f"  Top-3 Accuracy: {best_result['top3_accuracy']:.4f}")
    logger.info(f"{'='*50}\n")

    # Save results
    output = {
        'baseline': {
            'blenderbot_only': bb_only,
            'sasrec_only': sasrec_only,
        },
        'best': best_result,
        'all_results': all_results,
        'config': {
            'bb_probs_file': args.bb_probs_file,
            'sasrec_probs_file': args.sasrec_probs_file,
            'step': args.step,
            'num_samples': len(targets),
        }
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {args.output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"ENSEMBLE GRID SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"BlenderBot only:  acc={bb_only['accuracy']:.4f}")
    print(f"SASREC only:      acc={sasrec_only['accuracy']:.4f}")
    print(f"Best Ensemble:    acc={best_result['accuracy']:.4f} (w_bb={best_result['w_bb']:.2f}, w_sr={best_result['w_sr']:.2f})")
    print(f"Improvement:      +{best_result['accuracy'] - bb_only['accuracy']:.4f} over BlenderBot")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
