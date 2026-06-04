"""Compute metrics for predictor_only directories that are missing metric.json.

Usage: python compute_predictor_only_metrics.py <ss_p> <lora_lr>
Example: python compute_predictor_only_metrics.py 0.1 5e-5
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from src.transformers import AutoTokenizer
from metric.myMetrics import Metric

if len(sys.argv) != 3:
    print("Usage: python compute_predictor_only_metrics.py <ss_p> <lora_lr>")
    sys.exit(1)

ss_p = sys.argv[1]
lora_lr = sys.argv[2]

MODEL_PATH = f"blender_strategy/s3b_ss{ss_p}_lr{lora_lr}"
GEN_DATA = "generated_data"
dirname = f"s3b_ss{ss_p}_lr{lora_lr}_predictor_only"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

for dirname in [dirname]:
    dirpath = os.path.join(GEN_DATA, dirname)
    hyp_path = os.path.join(dirpath, "hyp_strategy.json")
    ref_path = os.path.join(dirpath, "ref_strategy.json")
    metric_path = os.path.join(dirpath, "metric.json")

    if not os.path.exists(hyp_path) or not os.path.exists(ref_path):
        print(f"SKIP {dirname}: missing hyp/ref")
        continue

    print(f"Computing metrics for {dirname}...")
    m = Metric(toker=tokenizer, hyp_path=hyp_path, ref_path=ref_path)
    result, result_list = m.close()

    # Add strategy accuracy from strategy_record.json
    record_path = os.path.join(dirpath, "strategy_record.json")
    if os.path.exists(record_path):
        with open(record_path) as f:
            records = json.load(f)
        correct = sum(1 for r in records if r['ref strategy'] == r['hyp strategy'])
        result['acc_strat_predictor_only'] = correct / len(records)

    with open(metric_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {metric_path}")
    print(f"  B-4={result['bleu-4']:.2f} RL={result['rouge-l']:.2f} F1={result['f1']:.2f} ACC={result.get('acc_strat_predictor_only', 'N/A')}")
