"""
s3 B condition: Ensemble Inference with strategy_head + SASREC for MISC.

Pipeline:
1. Load MISC s3 model (with LoRA)
2. Load pre-computed SASREC probabilities
3. For each sample:
   a. Encoder forward (strategy_logit_ground=None) -> strategy_logits -> softmax
   b. p_sasrec from pre-computed file
   c. p_final = alpha * strategy_probs + (1-alpha) * p_sasrec
   d. strat_id = argmax(p_final)
   e. Set LoRA expert -> model.generate() -> generation
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from metric.myMetrics import Metric
from BlenderEmotionalSupport_s3 import (
    Args, parse_args_override, set_seed, construct_conv_ESD,
    load_model_with_lora, extract_top_k_attention_comet_block,
    process_row_to_comet_query, summary,
)
from src.transformers import BlenderbotSmallTokenizer

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_ensemble_args():
    parser = argparse.ArgumentParser(description='MISC s3 ensemble inference')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained s3 model (default: args.output_dir)')
    parser.add_argument('--sasrec_probs_file', type=str, required=True,
                        help='Path to pre-computed SASREC test probabilities JSON')
    parser.add_argument('--ensemble_alpha', type=float, required=True,
                        help='Weight for strategy_head: p_final = alpha*strat + (1-alpha)*sasrec')
    parser.add_argument('--output_dir_suffix', type=str, default=None,
                        help='Suffix for generation_dir')

    # Override base args
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--generation_dir', type=str, default=None)
    parser.add_argument('--b_condition', type=str, default='true')
    parser.add_argument('--seed', type=int, default=42)

    args_cli, _ = parser.parse_known_args()
    return args_cli


def main():
    cli = parse_ensemble_args()

    # Build base args
    args = Args()
    if cli.output_dir:
        args.output_dir = cli.output_dir
    if cli.generation_dir:
        args.generation_dir = cli.generation_dir
    args.b_condition = cli.b_condition.lower() in ('true', '1', 'yes')
    args.seed = cli.seed

    model_path = cli.model_path or args.output_dir
    alpha = cli.ensemble_alpha

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.no_cuda = not torch.cuda.is_available()

    set_seed(args)

    logger.info(f'Ensemble alpha={alpha} (strategy_head={alpha}, sasrec={1-alpha})')

    # Setup tokenizer
    additional_special_tokens = ["[Question]", "[Reflection of feelings]", "[Information]",
                                 "[Restatement or Paraphrasing]", "[Others]", "[Self-disclosure]",
                                 "[Affirmation and Reassurance]", "[Providing Suggestions]"]
    comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]",
                                       "[oWant]", "[oEffect]", "[oReact]"]

    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_tokens(comet_additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

    # Load model with LoRA
    logger.info(f'Loading s3 model from {model_path}...')
    model = load_model_with_lora(model_path, tokenizer, args, device)
    model.eval()

    # Load SASREC probabilities
    logger.info(f'Loading SASREC probs from {cli.sasrec_probs_file}...')
    with open(cli.sasrec_probs_file, 'r') as f:
        sasrec_probs = json.load(f)
    logger.info(f'Loaded {len(sasrec_probs)} SASREC samples')

    # Load test data
    with open(args.data_path + "/" + args.test_file_name, "r") as f:
        chat_texts = f.read().split("\n")
    with open(args.data_path + "/" + args.situation_test_comet_file, "r", encoding="utf-8") as f:
        comet_st = f.read().split("\n")
    with open(args.data_path + "/" + args.test_comet_file, "r", encoding="utf-8") as f:
        comet = f.read().split("\n")

    assert len(comet) == len(chat_texts) == len(comet_st)

    # Generation
    gts = []
    refs = []
    mutual_attentions = []
    mutual_attentions_st = []
    strategy_hits = []
    strategy_hits_topk = [[] for _ in range(8)]
    gen_results = []

    # Ensemble accuracy tracking
    gt_strats = []
    pred_ensemble = []
    pred_predictor = []
    pred_sasrec = []

    seq_idx = 0

    for idx, (c_text, comet_row, comet_st_row) in tqdm(enumerate(zip(chat_texts[:-1], comet[:-1], comet_st[:-1])), desc="Ensemble Testing"):
        if "EOS" not in c_text:
            continue

        # Construct features (B condition: strategy=False)
        f = construct_conv_ESD(idx, c_text, comet_row, comet_st_row, tokenizer, eos=True, pad=False, cls=False, strategy=False, generation=True)
        if len(f.input_ids) >= args.block_size:
            f.input_ids = f.input_ids[-args.block_size:]
            f.input_ids[0] = tokenizer.encode(tokenizer.cls_token)[0]
        else:
            f.input_ids = tokenizer.encode(tokenizer.cls_token) + f.input_ids

        next_strategy_id = f.decoder_strategy_ids[0]
        gts.append(tokenizer.decode(f.decoder_input_ids, skip_special_tokens=True))

        # Prepare tensors
        input_ids = torch.tensor([f.input_ids], dtype=torch.long).to(device)
        comet_ids_t = torch.tensor([f.comet_ids], dtype=torch.long).to(device)
        comet_mask_t = torch.tensor([f.comet_mask], dtype=torch.long).to(device)
        comet_ids_st_t = torch.tensor([f.comet_st_ids], dtype=torch.long).to(device)
        comet_mask_st_t = torch.tensor([f.comet_st_mask], dtype=torch.long).to(device)

        # Encode COMET
        bs, n_attr, len_attr = comet_ids_t.shape
        comet_ids_flat = comet_ids_t.view(-1, len_attr)
        comet_embs = model.model.encoder(comet_ids_flat, attention_mask=comet_ids_flat.ne(tokenizer.pad_token_id))[0][:, 0, :]
        comet_embs = comet_embs.view(bs, n_attr, -1)

        bs, n_attr, len_attr = comet_ids_st_t.shape
        comet_ids_st_flat = comet_ids_st_t.view(-1, len_attr)
        comet_embs_st = model.model.encoder(comet_ids_st_flat, attention_mask=comet_ids_st_flat.ne(tokenizer.pad_token_id))[0][:, 0, :]
        comet_embs_st = comet_embs_st.view(bs, n_attr, -1)

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        # Encoder forward (strategy_logit_ground=None -> use predicted softmax)
        with torch.no_grad():
            encoder_outputs = model.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                comet_embs=comet_embs,
                comet_mask=comet_mask_t,
                comet_embs_st=comet_embs_st,
                comet_mask_st=comet_mask_st_t,
                strategy_logit_ground=None,
                output_mutual_attentions=True,
            )

        strategy_logits = encoder_outputs.strategy_logits
        p_predictor = F.softmax(strategy_logits[0], dim=-1).cpu().numpy()  # (8,)
        predictor_strat_id = int(np.argmax(p_predictor))

        # SASREC distribution
        if seq_idx < len(sasrec_probs):
            p_sasrec = np.array(sasrec_probs[seq_idx]['probs'])
            gt_from_sasrec = sasrec_probs[seq_idx].get('target', None)
        else:
            logger.warning(f"SASREC probs exhausted at idx={seq_idx}, using uniform")
            p_sasrec = np.ones(8) / 8.0
            gt_from_sasrec = None

        sasrec_strat_id = int(np.argmax(p_sasrec))

        # Ensemble
        p_final = alpha * p_predictor + (1 - alpha) * p_sasrec
        p_final = p_final / p_final.sum()
        ensemble_strat_id = int(np.argmax(p_final))

        # Track accuracy
        gt_strats.append(int(next_strategy_id))
        pred_ensemble.append(ensemble_strat_id)
        pred_predictor.append(predictor_strat_id)
        pred_sasrec.append(sasrec_strat_id)

        # Set LoRA expert for generation
        model._decoder_lora_context.expert_ids = torch.tensor([ensemble_strat_id], device=device)

        # Replace strategy_embs with ensemble soft distribution (Eq. 6: h^g = p_final · T)
        p_final_tensor = torch.tensor(p_final, device=device).float().unsqueeze(0)  # (1, 8)
        strategy_id = model.model.encoder.strategy_id.to(device)
        ensemble_strategy_embs = torch.bmm(
            p_final_tensor.unsqueeze(1),  # (1, 1, 8)
            model.model.encoder.strategy_embedding(strategy_id).unsqueeze(0)  # (1, 8, d)
        )  # (1, 1, d)
        encoder_outputs.strategy_embs = ensemble_strategy_embs

        # Generate (pass encoder_outputs to skip re-running encoder)
        paras = {
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "comet_embs": comet_embs,
            "comet_mask": comet_mask_t,
            "comet_embs_st": comet_embs_st,
            "comet_mask_st": comet_mask_st_t,
            "output_mutual_attentions": True,
        }

        chat_history_ids, mutual_attention, mutual_attention_st, gen_strategy_logits = model.generate(
            input_ids,
            **paras, max_length=512, min_length=5, num_beams=1,
            pad_token_id=0, use_cache=True,
            eos_token_id=tokenizer.eos_token_id, temperature=0.7,
            top_p=0.3, top_k=30, do_sample=True, repetition_penalty=1.03)

        # Clear LoRA context
        model._decoder_lora_context.expert_ids = None

        chat_history_ids = chat_history_ids.cpu()
        mutual_attention = mutual_attention[-1][0].cpu()
        mutual_attention_st = mutual_attention_st[-1][0].cpu()
        mutual_attention = torch.mean(mutual_attention, dim=0)
        mutual_attention_st = torch.mean(mutual_attention_st, dim=0)
        mutual_attentions.append(mutual_attention)
        mutual_attentions_st.append(mutual_attention_st)

        gen_text = tokenizer.decode(chat_history_ids[:, :][0], skip_special_tokens=True)
        refs.append(gen_text)
        print(gen_text)

        print({"ref strategy": tokenizer.decode([next_strategy_id + 54944]),
               "ensemble strategy": tokenizer.decode([ensemble_strat_id + 54944]),
               "predictor strategy": tokenizer.decode([predictor_strat_id + 54944])})

        if ensemble_strat_id == next_strategy_id:
            strategy_hits.append(1)
        else:
            strategy_hits.append(0)

        for k in range(8):
            topk_ids = np.argsort(p_final)[::-1][:k+1]
            strategy_hits_topk[k].append(1 if next_strategy_id in topk_ids else 0)

        gen_results.append({
            'sample_id': idx,
            'generation': gen_text,
            'ensemble_strat_id': ensemble_strat_id,
            'predictor_strat_id': predictor_strat_id,
            'sasrec_strat_id': sasrec_strat_id,
            'gt_strat_id': int(next_strategy_id),
            'pred_strat_id_dist': ' '.join([f'{p:.6f}' for p in p_predictor.tolist()]),
            'ensemble_dist': p_final.tolist(),
        })

        seq_idx += 1

    # Print accuracy
    for i in range(8):
        print(f"Top-{i+1} accuracy: {sum(strategy_hits_topk[i]) / len(strategy_hits_topk[i]):.4f}")

    gt_arr = np.array(gt_strats)
    ens_acc = float(np.mean(gt_arr == np.array(pred_ensemble)))
    pred_acc = float(np.mean(gt_arr == np.array(pred_predictor)))
    sasrec_acc = float(np.mean(gt_arr == np.array(pred_sasrec)))

    print(f"\n{'='*60}")
    print(f"s3 ENSEMBLE INFERENCE COMPLETE (MISC)")
    print(f"{'='*60}")
    print(f"Alpha: {alpha} (strategy_head={alpha}, sasrec={1-alpha})")
    print(f"Strategy Acc (Ensemble):       {ens_acc:.4f}")
    print(f"Strategy Acc (strategy_head):  {pred_acc:.4f}")
    print(f"Strategy Acc (SASREC):         {sasrec_acc:.4f}")
    print(f"{'='*60}\n")

    # Save results
    suffix = cli.output_dir_suffix or f"ensemble_a{alpha}"
    gen_dir = args.generation_dir + f"_{suffix}"
    os.makedirs(gen_dir, exist_ok=True)

    generate_file_path = os.path.join(gen_dir, "hyp_strategy.json")
    reference_file_path = os.path.join(gen_dir, "ref_strategy.json")
    gen_json_path = os.path.join(gen_dir, "gen.json")
    metric_path = os.path.join(gen_dir, "metric.json")
    summary_file_path = os.path.join(gen_dir, "summary.txt")

    with open(generate_file_path, "w", encoding="utf-8") as fp:
        json.dump(refs, fp, indent=2, ensure_ascii=False)
    with open(reference_file_path, "w", encoding="utf-8") as fp:
        json.dump(gts, fp, indent=2, ensure_ascii=False)
    with open(gen_json_path, "w", encoding="utf-8") as fp:
        json.dump(gen_results, fp, indent=2, ensure_ascii=False)

    # Compute and save metrics
    all_top_k_blocks = extract_top_k_attention_comet_block(mutual_attentions, comet[:-1], 5)
    all_top_k_blocks_st = extract_top_k_attention_comet_block(mutual_attentions_st, comet_st[:-1], 5)

    test_file_path = "dataset/test_cl.txt"
    test_situation_file_path = "dataset/testSituation_cl.txt"
    summary(test_file_path, generate_file_path, reference_file_path, summary_file_path, all_top_k_blocks, all_top_k_blocks_st, chat_texts, test_situation_file_path)

    metric = Metric(toker=tokenizer, hyp_path=generate_file_path, ref_path=reference_file_path)
    result, result_list = metric.close()

    result['acc_strat_ensemble'] = ens_acc
    result['acc_strat_predictor_only'] = pred_acc
    result['acc_strat_sasrec_only'] = sasrec_acc
    result['ensemble_alpha'] = alpha

    with open(metric_path, "w") as fp:
        json.dump(result, fp, indent=2)

    print(result)
    print(f"\nResults saved to {gen_dir}")
    print("=" * 100)


if __name__ == '__main__':
    main()
