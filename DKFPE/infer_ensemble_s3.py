# coding=utf-8
"""
exp/s3 B condition: Ensemble Inference with Joint Probe + SASREC.

this uses the _joint_probe_head's softmax distribution as the BB side.

Pipeline:
1. Load s3 BB model (with _joint_probe_head in checkpoint)
2. Load pre-computed SASREC probabilities
3. For each sample:
   a. BB encoder → decoder(BOS, no LoRA) → _joint_probe_head → p_probe (softmax)
   b. p_sasrec from pre-computed file
   c. p_final = alpha * p_probe + (1-alpha) * p_sasrec
   d. strat_id = argmax(p_final)
   e. BB decoder(expert LoRA[strat_id]) → generation
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.trainer_utils import set_seed

from inputters import inputters
from inputters.inputter_utils import _norm
from metric.myMetrics import Metric
from utils.building_utils import boolean_string, build_model, deploy_model
from utils.eval_utils import eval_model_loss

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def cut_seq_to_eos(sentence, eos, remove_id=None):
    if remove_id is None:
        remove_id = [-1]
    sent = []
    for s in sentence:
        if s in remove_id:
            continue
        if s != eos:
            sent.append(s)
        else:
            break
    return sent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--inputter_name', type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", '-c', type=str, required=True)

    # Ensemble arguments
    parser.add_argument("--sasrec_probs_file", type=str, required=True,
                        help="Path to pre-computed SASREC probabilities JSON file")
    parser.add_argument("--ensemble_alpha", type=float, default=0.5,
                        help="Weight for probe: p_final = alpha*probe + (1-alpha)*sasrec")

    parser.add_argument("--fp16", type=boolean_string, default=False)
    parser.add_argument("--max_input_length", type=int, default=150)
    parser.add_argument("--max_src_turn", type=int, default=None)
    parser.add_argument("--max_decoder_input_length", type=int, default=50)
    parser.add_argument("--max_knowledge_length", type=int, default=None)
    parser.add_argument('--label_num', type=int, default=None)
    parser.add_argument('--multi_knl', action='store_true')
    parser.add_argument('--no_strat_in_seq', action='store_true', default=False,
                        help='B condition: exclude strategy tokens from encoder context and decoder labels')

    parser.add_argument('--only_generate', action='store_true')

    parser.add_argument("--min_length", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--num_return_sequences", type=int, default=1)

    parser.add_argument("--infer_batch_size", type=int, default=1,
                        help="Batch size (must be 1 for ensemble due to sample-level SASREC alignment)")
    parser.add_argument('--infer_input_file', type=str, nargs='+', required=True)

    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    args = parser.parse_args()

    if args.infer_batch_size != 1:
        logger.warning("Ensemble inference requires batch_size=1. Setting to 1.")
        args.infer_batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu

    logger.info('Initializing CUDA...')
    _ = torch.tensor([1.], device=args.device)

    set_seed(args.seed)

    alpha = args.ensemble_alpha
    logger.info(f'Ensemble alpha={alpha} (probe={alpha}, sasrec={1-alpha})')

    # Load BlenderBot model (s3 with joint probe head)
    logger.info('Loading BlenderBot model...')
    names = {
        'inputter_name': args.inputter_name,
        'config_name': args.config_name,
    }
    toker, model = build_model(checkpoint=args.load_checkpoint, **names)
    model = deploy_model(model, args)
    model.eval()

    raw_model = model.module if hasattr(model, 'module') else model
    if raw_model._joint_probe_head is None:
        raise RuntimeError("Checkpoint does not contain a joint probe head. "
                           "Use infer.py or train with joint_probe=true.")

    # Load SASREC probabilities
    logger.info(f'Loading SASREC probs from {args.sasrec_probs_file}...')
    with open(args.sasrec_probs_file, 'r') as f:
        sasrec_probs = json.load(f)
    logger.info(f'Loaded {len(sasrec_probs)} SASREC probability samples')

    # Setup tokenizer
    pad = toker.pad_token_id or toker.eos_token_id
    bos = toker.bos_token_id or toker.cls_token_id
    eos = toker.eos_token_id or toker.sep_token_id

    generation_kwargs = {
        'max_length': args.max_length,
        'min_length': args.min_length,
        'do_sample': True if (args.top_k > 0 or args.top_p < 1) else False,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'num_beams': args.num_beams,
        'num_return_sequences': args.num_return_sequences,
        'length_penalty': args.length_penalty,
        'repetition_penalty': args.repetition_penalty,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
        'encoder_no_repeat_ngram_size': args.no_repeat_ngram_size if model.config.is_encoder_decoder else None,
        'pad_token_id': pad,
        'bos_token_id': bos,
        'eos_token_id': eos,
    }

    inputter = inputters[args.inputter_name]()
    dataloader_kwargs = {
        'max_src_turn': args.max_src_turn,
        'max_input_length': args.max_input_length,
        'max_decoder_input_length': args.max_decoder_input_length,
        'max_knowledge_length': args.max_knowledge_length,
        'label_num': args.label_num,
        'multi_knl': args.multi_knl,
        'no_strat_in_seq': args.no_strat_in_seq,
        'infer_batch_size': args.infer_batch_size,
    }

    for infer_idx, infer_input_file in enumerate(args.infer_input_file):
        set_seed(args.seed)

        infer_dataloader = inputter.infer_dataloader(
            infer_input_file,
            toker,
            **dataloader_kwargs
        )

        metric_res = {}
        if not args.only_generate:
            loss_loader = inputter.valid_dataloader(
                corpus_file=infer_input_file,
                toker=toker,
                batch_size=args.infer_batch_size,
                **dataloader_kwargs
            )
            infer_loss, _, _, _, _ = eval_model_loss(
                model=model,
                eval_dataloader=loss_loader,
                epoch_id=0,
                infer=True,
                args=args,
            )
            metric_res['perplexity'] = float(np.exp(infer_loss))
            metric = Metric(toker)

        res = []
        decode = lambda x: _norm(toker.decode(x))
        seq_idx = 0

        for batch, posts, references, sample_ids in infer_dataloader:
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

            # Remove strat_id so model uses probe
            batch.pop('strat_id', None)

            # Get probe distribution from joint probe head
            with torch.no_grad():
                return_dict = raw_model.config.use_return_dict
                encoder_outputs = raw_model.model.encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_dict=return_dict,
                )
                probe_logits = raw_model._compute_probe_logits(
                    encoder_hidden=encoder_outputs.last_hidden_state,
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=batch['decoder_input_ids'],
                )
                probe_dist = F.softmax(probe_logits, dim=-1)[0].cpu().numpy()  # (8,)

            # Get SASREC distribution
            if seq_idx < len(sasrec_probs):
                sasrec_dist = np.array(sasrec_probs[seq_idx]['probs'])
            else:
                logger.warning(f"SASREC probs exhausted at idx={seq_idx}, using uniform")
                sasrec_dist = np.ones(8) / 8.0

            # Ensemble
            ensemble_dist = alpha * probe_dist + (1 - alpha) * sasrec_dist
            ensemble_dist = ensemble_dist / ensemble_dist.sum()
            ensemble_strat_id = int(np.argmax(ensemble_dist))

            # Set ensemble strategy for generation
            batch['strat_id'] = torch.tensor([ensemble_strat_id], device=device)

            # Generate
            batch.update(generation_kwargs)
            encoded_info, generations = model.generate(**batch)

            generations = [cut_seq_to_eos(each, eos) for each in generations.tolist()]

            for idx in range(len(sample_ids)):
                p = posts[idx]
                r = references[idx]
                g = generations[idx]

                if not args.only_generate and args.num_return_sequences == 1:
                    ref, gen = [r], toker.decode(g) if not isinstance(g[0], list) else toker.decode(g[0])
                    metric.forword(ref, gen)

                g = decode(g) if not isinstance(g[0], list) else [decode(gg) for gg in g]

                tmp_res = {
                    'sample_id': sample_ids[idx],
                    'post': p,
                    'response': r,
                    'generation': g,
                    'ensemble_strat_id': ensemble_strat_id,
                    'probe_strat_id': int(np.argmax(probe_dist)),
                    'sasrec_strat_id': int(np.argmax(sasrec_dist)),
                    'ensemble_dist': ensemble_dist.tolist(),
                    'probe_dist': probe_dist.tolist(),
                }
                res.append(tmp_res)
                seq_idx += 1

        # Save results
        checkpoint_dir_path = '/'.join(args.load_checkpoint.split('/')[:-1])
        checkpoint_name = args.load_checkpoint.split('/')[-1]
        infer_input_file_name = infer_input_file.split('/')[-1].replace('.txt', '')

        save_dir = (
            f'{checkpoint_dir_path}/res_ensemble_s3_{checkpoint_name}_{infer_input_file_name}'
            f'_a.{alpha}_k.{args.top_k}_p.{args.top_p}_t.{args.temperature}'
        )

        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, 'gen.json'), 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_dir, 'gen.txt'), 'w') as f:
            for line in res:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

        # Compute metrics
        if not args.only_generate:
            metric_res.update(metric.close()[0])

            # Strategy accuracy from SASREC data (has ground truth)
            gt_strats = []
            pred_strats_ensemble = []
            pred_strats_probe = []
            pred_strats_sasrec = []
            for i, r_item in enumerate(res):
                if i < len(sasrec_probs):
                    gt = sasrec_probs[i].get('target', None)
                    if gt is not None:
                        gt_strats.append(gt)
                        pred_strats_ensemble.append(r_item['ensemble_strat_id'])
                        pred_strats_probe.append(r_item['probe_strat_id'])
                        pred_strats_sasrec.append(r_item['sasrec_strat_id'])

            if gt_strats:
                gt_arr = np.array(gt_strats)
                metric_res['acc_strat_ensemble'] = float(np.mean(gt_arr == np.array(pred_strats_ensemble)))
                metric_res['acc_strat_probe_only'] = float(np.mean(gt_arr == np.array(pred_strats_probe)))
                metric_res['acc_strat_sasrec_only'] = float(np.mean(gt_arr == np.array(pred_strats_sasrec)))

            with open(os.path.join(save_dir, 'metric.json'), 'w') as f:
                json.dump(metric_res, f, ensure_ascii=False, indent=2)

            logger.info(f"Results saved to {save_dir}")
            if gt_strats:
                logger.info(f"Strategy Acc (Ensemble): {metric_res['acc_strat_ensemble']:.4f}")
                logger.info(f"Strategy Acc (Probe):    {metric_res['acc_strat_probe_only']:.4f}")
                logger.info(f"Strategy Acc (SASREC):   {metric_res['acc_strat_sasrec_only']:.4f}")

        print(f"\n{'='*60}")
        print(f"s3 ENSEMBLE INFERENCE COMPLETE")
        print(f"{'='*60}")
        print(f"Results: {save_dir}")
        print(f"Alpha: {alpha} (probe={alpha}, sasrec={1-alpha})")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
