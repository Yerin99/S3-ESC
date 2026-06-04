# coding=utf-8
# train_s3.py: GLHG + s3 training (LoRA Expert Routing + Gradient-Isolated Probe + Scheduled Sampling)
#
# Based on train.py with s3 additions:
# - Separate probe optimizer (gradient isolation)
# - Scheduled Sampling support
# - probe_extra_epochs / probe_only_from mode (017-style 2-stage probe)

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import datetime
import json
import logging
import os
import sys
import time
from os.path import join

import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.distributed import get_rank, get_world_size
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed

from inputters import inputters
from utils.building_utils import boolean_string, build_model, deploy_model
from utils.distributed import all_reduce_and_rescale_tensors, all_gather_list
from utils.eval_utils import eval_model_loss

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INF = 100000000
CACHE_EMPTY_STEP = 10000

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True)
parser.add_argument('--inputter_name', type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_checkpoint", '-c', type=str, default=None)

parser.add_argument("--max_input_length", type=int, default=150)
parser.add_argument("--max_decoder_input_length", type=int, default=50)
parser.add_argument("--max_knowledge_len", type=int, default=None)
parser.add_argument('--label_num', type=int, default=None)
parser.add_argument('--only_encode', action='store_true', help='only do encoding')

parser.add_argument("--eval_input_file", type=str)

parser.add_argument("--train_batch_size", type=int, default=8,
                    help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="to increase effective batch size and reduce synchronization")
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--warmup_steps", type=int, default=16000)

parser.add_argument("--num_optim_steps", type=int, default=20000,
                    help="new API specifies num update steps")
parser.add_argument("--valid_step", type=int, default=2000,
                    help="how many optim steps between validations")
parser.add_argument("--num_epochs", type=int, default=None,
                    help="how many training epochs")

parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument("--fp16", type=boolean_string, default=False)
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument('--pbar', type=boolean_string, default=True, help='turn on progress bar')

# Scheduled Sampling
parser.add_argument('--use_scheduled_sampling', type=boolean_string, default=False)
parser.add_argument('--ss_schedule_type', type=str, default='step',
                    choices=['step', 'linear', 'inverse_sigmoid'])
parser.add_argument('--ss_warmup_epochs', type=int, default=1)
parser.add_argument('--ss_p_sample', type=float, default=0.05)
parser.add_argument('--ss_p_random', type=float, default=0.0)

# s3: LoRA learning rate (if None, uses main learning_rate)
parser.add_argument('--lora_lr', type=float, default=None,
                    help='Separate lr for LoRA params. If None, uses --learning_rate.')

# s3: Joint probe
parser.add_argument('--probe_lr', type=float, default=1e-3)
parser.add_argument('--probe_extra_epochs', type=int, default=0)
parser.add_argument('--probe_only_from', type=str, default=None,
                    help='skip main training; load best checkpoint from this dir and run probe-only')
parser.add_argument('--probe_save_suffix', type=str, default='')
parser.add_argument('--probe_reinit', action='store_true')
parser.add_argument('--probe_batch_size', type=int, default=256)
parser.add_argument('--probe_select_by', type=str, default='acc', choices=['acc', 'loss'])

# distributed
parser.add_argument('--local_rank', type=int, default=-1, help='for torch.distributed')
parser.add_argument('--config', help='JSON config file')

# do normal parsing
args = parser.parse_args()
init_args_dict = vars(args).copy()

if args.config is not None:
    opts = json.load(open(args.config))
    for k, v in opts.items():
        if isinstance(v, str):
            if 'PHILLY_JOB_DIRECTORY' in v:
                v = v.replace('PHILLY_JOB_DIRECTORY', os.environ['PHILLY_JOB_DIRECTORY'])
            elif 'PHILLY_LOG_DIRECTORY' in v:
                v = v.replace('PHILLY_LOG_DIRECTORY', os.environ['PHILLY_LOG_DIRECTORY'])
        setattr(args, k, v)

    argv = sys.argv[1:]
    overrides, _ = parser.parse_known_args(argv)
    for k, v in vars(overrides).items():
        if f'--{k}' in argv:
            setattr(args, k, v)
    setattr(args, 'local_rank', overrides.local_rank)


if args.local_rank == -1:
    logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

assert args.train_batch_size % args.gradient_accumulation_steps == 0
args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

if args.local_rank == -1 or get_rank() == 0:
    logger.info('train batch size = {}, new train batch size (after gradient accumulation) = {}'.format(
        args.train_batch_size * args.gradient_accumulation_steps, args.train_batch_size))

if args.local_rank == -1 or get_rank() == 0:
    logger.info('initializing cuda...')
torch.tensor([1.], device=args.device)

set_seed(args.seed)

if args.local_rank == -1 or get_rank() == 0:
    logger.info('Input Argument Information')
    args_dict = vars(args)
    for a in args_dict:
        logger.info('%-28s  %s' % (a, args_dict[a]))


#########################################################################
# Prepare Data Set
##########################################################################

names = {
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
}

toker = build_model(only_toker=True, local_rank=args.local_rank, **names)
inputter = inputters[args.inputter_name]()

if args.local_rank == -1:
    train_dataloader = inputter.train_dataloader(
        toker=toker,
        feature_dataset=inputter.train_dataset,
        batch_size=args.train_batch_size,
        **names
    )
else:
    train_dataloader = inputter.train_distributed_dataloader(
        get_rank(), get_world_size(),
        toker=toker,
        feature_dataset=inputter.train_dataset,
        batch_size=args.train_batch_size,
        **names
    )

if args.num_epochs is not None:
    args.num_optim_steps = args.num_epochs * (
        len(train_dataloader) // args.train_batch_size +
        int(len(train_dataloader) % args.train_batch_size != 0)
    )

dataloader_kwargs = {
    'max_input_length': args.max_input_length,
    'max_decoder_input_length': args.max_decoder_input_length,
    'max_knowledge_len': args.max_knowledge_len,
    'label_num': args.label_num,
    'only_encode': args.only_encode,
}
eval_dataloader_loss = inputter.valid_dataloader(
    toker=toker,
    corpus_file=args.eval_input_file,
    batch_size=args.eval_batch_size,
    **dataloader_kwargs
)

#########################################################################
# Prepare Model and Optimizer
#########################################################################
_, model = build_model(checkpoint=args.load_checkpoint, local_rank=args.local_rank, **names)
# Re-seed after model build so probe head init doesn't shift training RNG
set_seed(args.seed)
model = deploy_model(model, args, local_rank=args.local_rank)

if args.local_rank != -1:
    params = [p.data for p in model.parameters()]
    all_reduce_and_rescale_tensors(params, float(torch.distributed.get_world_size()))

model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
total_params = sum([np.prod(p.size()) for p in model_parameters])
if args.local_rank == -1 or get_rank() == 0:
    logger.info('Number of parameter = {}'.format(total_params))

# s3: Detect joint probe head and separate its parameters
_raw_model = model.module if hasattr(model, 'module') else model
_joint_probe_param_ids = set()
if hasattr(_raw_model, '_joint_probe_head') and _raw_model._joint_probe_head is not None:
    _joint_probe_param_ids = {id(p) for p in _raw_model._joint_probe_head.parameters()}

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'ln', 'LayerNorm.weight']

# Separate LoRA params if lora_lr is specified
_lora_param_ids = set()
if args.lora_lr is not None:
    _lora_param_ids = {id(p) for n, p in param_optimizer if 'lora_' in n and p.requires_grad}

_exclude_ids = _joint_probe_param_ids | _lora_param_ids

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if p.requires_grad
                and not any(nd in n for nd in no_decay)
                and id(p) not in _exclude_ids], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if p.requires_grad
                and any(nd in n for nd in no_decay)
                and id(p) not in _exclude_ids], 'weight_decay': 0.0},
]
if _lora_param_ids:
    optimizer_grouped_parameters.extend([
        {'params': [p for n, p in param_optimizer if p.requires_grad
                    and not any(nd in n for nd in no_decay)
                    and id(p) in _lora_param_ids], 'weight_decay': 0.01, 'lr': args.lora_lr},
        {'params': [p for n, p in param_optimizer if p.requires_grad
                    and any(nd in n for nd in no_decay)
                    and id(p) in _lora_param_ids], 'weight_decay': 0.0, 'lr': args.lora_lr},
    ])
    logger.info(f'LoRA params use separate lr={args.lora_lr} ({len(_lora_param_ids)} params)')

optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_optim_steps
)

# s3: Separate probe optimizer
probe_optimizer = None
if _joint_probe_param_ids:
    probe_params = [p for p in _raw_model._joint_probe_head.parameters() if p.requires_grad]
    probe_optimizer = torch.optim.Adam(probe_params, lr=args.probe_lr)
    if args.local_rank == -1 or get_rank() == 0:
        logger.info(f'Joint probe optimizer initialized (lr={args.probe_lr}, '
                    f'{sum(p.numel() for p in probe_params)} params)')

if args.fp16:
    logger.info('in fp16, using FusedAdam')
    from apex import amp
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

# Scheduled Sampling
scheduled_sampler = None
if args.use_scheduled_sampling:
    from utils.scheduled_sampling import StrategyScheduledSampler
    total_epochs = args.num_epochs if args.num_epochs is not None else 10
    scheduled_sampler = StrategyScheduledSampler(
        schedule_type=args.ss_schedule_type,
        warmup_epochs=args.ss_warmup_epochs,
        total_epochs=total_epochs,
        p_sample=args.ss_p_sample,
        p_random=args.ss_p_random,
        num_strategies=8,
        strategy_token_offset=len(toker) - 8,
    )
    if hasattr(model, 'set_scheduled_sampler'):
        model.set_scheduled_sampler(scheduled_sampler)
    elif hasattr(model, 'module') and hasattr(model.module, 'set_scheduled_sampler'):
        model.module.set_scheduled_sampler(scheduled_sampler)
    if args.local_rank == -1 or get_rank() == 0:
        logger.info(f'Scheduled Sampling enabled: {scheduled_sampler}')


#########################################################################
# Probe hidden extraction helper (used for probe-only mode & probe extra)
#########################################################################

def _extract_probe_hidden(model_ref, raw_model_ref, batch_dict):
    """Extract hidden states for probe training.

    Source determined by raw_model_ref._joint_probe_source:
      "encoder": encoder mean-pool of contextual encoder hidden (before graph fusion)
      "decoder": decoder BOS hidden state at position 0
    """
    input_ids = batch_dict.get('input_ids')
    attention_mask = batch_dict.get('attention_mask')
    decoder_input_ids = batch_dict.get('decoder_input_ids')
    strat_id = batch_dict.get('strat_id')

    probe_source = getattr(raw_model_ref, '_joint_probe_source', 'encoder')

    with torch.no_grad():
        if probe_source == 'decoder':
            # Decoder BOS hidden: run full GLHG encoding then decoder with BOS only
            if hasattr(raw_model_ref, '_run_glhg_encoding'):
                encoder_outputs, extended_attention_mask = raw_model_ref._run_glhg_encoding(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    situation_ids=batch_dict.get('situation_ids'),
                    situation_attention_mask=batch_dict.get('situation_attention_mask'),
                    intention_ids=batch_dict.get('intention_ids'),
                    intention_attention_mask=batch_dict.get('intention_attention_mask'),
                    last_utterance_starts=batch_dict.get('last_utterance_starts'),
                )
                fused_hidden = encoder_outputs.last_hidden_state
            else:
                enc_out = raw_model_ref.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                fused_hidden = enc_out.last_hidden_state
                extended_attention_mask = attention_mask

            # Decoder with BOS only (position 0)
            # Must disable expert LoRA since no expert routing is set for probe extraction
            bos_ids = decoder_input_ids[:, :1] if decoder_input_ids is not None else \
                torch.full((input_ids.size(0), 1), raw_model_ref.config.decoder_start_token_id,
                           device=input_ids.device, dtype=torch.long)
            from models.lora_expert_layers import disable_expert_lora
            with disable_expert_lora(raw_model_ref):
                dec_out = raw_model_ref.model.decoder(
                    input_ids=bos_ids,
                    encoder_hidden_states=fused_hidden,
                    encoder_attention_mask=extended_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            probe_hidden = dec_out.last_hidden_state[:, 0, :]  # BOS position

        else:
            # Encoder mean-pool (default): use contextual encoder BEFORE graph fusion
            if hasattr(raw_model_ref, '_run_glhg_encoding'):
                _, _, ctx_hidden, ctx_mask = raw_model_ref._run_glhg_encoding(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    situation_ids=batch_dict.get('situation_ids'),
                    situation_attention_mask=batch_dict.get('situation_attention_mask'),
                    intention_ids=batch_dict.get('intention_ids'),
                    intention_attention_mask=batch_dict.get('intention_attention_mask'),
                    last_utterance_starts=batch_dict.get('last_utterance_starts'),
                    return_ctx_hidden=True,
                )
            else:
                enc_out = raw_model_ref.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                ctx_hidden = enc_out.last_hidden_state
                ctx_mask = attention_mask

            mask = ctx_mask.unsqueeze(-1).float()
            probe_hidden = (ctx_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    return probe_hidden, strat_id


#########################################################################
# Probe-only mode: skip main training, load best ckpt, train probe
#########################################################################

if args.probe_only_from is not None:
    output_dir = args.probe_only_from
    logger.info(f'Probe-only mode: using existing output dir {output_dir}')

    best_epoch_idx = 0
    best_ppl = float('inf')
    with open(join(output_dir, 'eval_log.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row['epoch'])
            ppl_val = float(row['freq_ppl'])
            if ppl_val < best_ppl:
                best_ppl = ppl_val
                best_epoch_idx = ep

    best_ckpt_path = join(output_dir, f'epoch-{best_epoch_idx}.bin')
    logger.info(f'Loading best epoch {best_epoch_idx} (val PPL={best_ppl:.2f}) for probe extra training')
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device), strict=False)

    if args.probe_reinit:
        logger.info('Reinitializing probe head weights (random)')
        _raw_model._joint_probe_head.reset_parameters()

    # Disable scheduled sampling
    if hasattr(_raw_model, '_scheduled_sampler'):
        _raw_model._scheduled_sampler = None

    # 017-style: extract all hidden → train with randperm
    model.eval()
    logger.info('Extracting all hidden states for probe training (017 method)...')
    from sklearn.metrics import accuracy_score

    def _extract_all_hidden(dataloader_ref):
        all_h, all_y = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader_ref, desc='extract hidden'):
                batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
                h, y = _extract_probe_hidden(model, _raw_model, batch)
                all_h.append(h.cpu())
                all_y.append(y.cpu())
        return torch.cat(all_h, dim=0), torch.cat(all_y, dim=0)

    train_hidden, train_labels = _extract_all_hidden(train_dataloader)
    valid_hidden, valid_labels = _extract_all_hidden(eval_dataloader_loss)
    logger.info(f'Extracted: train={train_hidden.shape}, valid={valid_hidden.shape}')

    probe = _raw_model._joint_probe_head
    probe_optimizer = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)
    criterion = torch.nn.CrossEntropyLoss()
    probe_bs = args.probe_batch_size

    _best_val_acc = -1.0
    _best_probe_state = None
    n_epochs = args.probe_extra_epochs if args.probe_extra_epochs > 0 else 100

    logger.info(f'Probe training: {n_epochs} epochs, batch_size={probe_bs}, lr={args.probe_lr}, '
                f'select_by={args.probe_select_by}')

    for probe_epoch in range(n_epochs):
        probe.train()
        perm = torch.randperm(train_hidden.size(0))
        _loss_sum, _n_batches = 0.0, 0
        for start in range(0, train_hidden.size(0), probe_bs):
            idx = perm[start:start + probe_bs]
            x = train_hidden[idx].to(device)
            y = train_labels[idx].to(device)
            probe_optimizer.zero_grad()
            logits = probe(x)
            loss = criterion(logits, y)
            loss.backward()
            probe_optimizer.step()
            _loss_sum += loss.item()
            _n_batches += 1

        probe.eval()
        with torch.no_grad():
            val_logits = probe(valid_hidden.to(device))
            val_preds = val_logits.argmax(dim=-1).cpu()
            val_loss = criterion(val_logits, valid_labels.to(device)).item()
        val_acc = accuracy_score(valid_labels.numpy(), val_preds.numpy())

        is_best = False
        if args.probe_select_by == 'acc':
            if val_acc > _best_val_acc:
                _best_val_acc = val_acc
                _best_probe_state = {k: v.clone() for k, v in probe.state_dict().items()}
                is_best = True
        else:
            if not hasattr(_extract_all_hidden, '_best_val_loss'):
                _extract_all_hidden._best_val_loss = float('inf')
            if val_loss < _extract_all_hidden._best_val_loss:
                _extract_all_hidden._best_val_loss = val_loss
                _best_val_acc = val_acc
                _best_probe_state = {k: v.clone() for k, v in probe.state_dict().items()}
                is_best = True

        if (probe_epoch + 1) % 10 == 0:
            train_loss = _loss_sum / max(_n_batches, 1)
            logger.info(f'Probe epoch {probe_epoch+1:3d} | train_loss={train_loss:.4f} | '
                        f'val_acc={val_acc:.4f} val_loss={val_loss:.4f}{" *" if is_best else ""}')

    if _best_probe_state is not None:
        probe.load_state_dict(_best_probe_state)
        logger.info(f'Restored best probe (val_acc={_best_val_acc:.4f})')

    _save_path = join(output_dir, f'epoch-{best_epoch_idx}_probe{args.probe_save_suffix}.bin')
    torch.save(model.state_dict(), _save_path)
    logger.info(f'Saved probe-trained checkpoint: {_save_path}')

    import sys
    sys.exit(0)


#########################################################################
# Training !
##########################################################################

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
_dir_suffix = f'{timestamp}.{args.learning_rate}.{args.train_batch_size}.{n_gpu}gpu'
if args.lora_lr is not None:
    _dir_suffix += f'.lora_lr{args.lora_lr}'
output_dir = join(f'./DATA/{args.inputter_name}.{args.config_name}', _dir_suffix)
if args.local_rank == -1 or get_rank() == 0:
    os.makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(init_args_dict, f, ensure_ascii=False, indent=2)
    with open(join(output_dir, 'custom_config.json'), 'w', encoding='utf-8') as f:
        with open(f'./CONFIG/{args.config_name}.json', 'r', encoding='utf-8') as ff:
            json.dump(json.load(ff), f, ensure_ascii=False, indent=2)

if args.local_rank == -1 or get_rank() == 0:
    train_logger = open(join(output_dir, 'train_log.csv'), 'a+', buffering=1)
    eval_logger = open(join(output_dir, 'eval_log.csv'), 'a+', buffering=1)
    print('epoch,global_step,step,tmp_loss,tmp_ppl,mean_loss,mean_ppl,n_token_real,'
          'n_token_total,epoch_time', file=train_logger)
    print('epoch,global_step,step,freq_loss,freq_ppl', file=eval_logger)

global_step = 0
step = 0
epoch = 0

if args.local_rank != -1:
    n_gpu = 1
if args.local_rank == -1 or get_rank() == 0:
    if args.pbar:
        pbar = tqdm.tqdm(total=args.num_optim_steps, desc=f"training")
    else:
        pbar = None

while True:
    model.train()
    if scheduled_sampler is not None:
        scheduled_sampler.set_epoch(epoch)
        if args.local_rank == -1 or get_rank() == 0:
            p_gt, p_sample, p_random = scheduled_sampler.get_current_probs()
            logger.info(f'Epoch {epoch}: SS p_gt={p_gt:.2f}, p_sample={p_sample:.2f}, p_random={p_random:.2f}')

    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()

    for batch in train_dataloader:
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        batch.update({'global_step': global_step})
        batch.update({'epoch': epoch})
        batch.update({'warmup_steps': args.warmup_steps})

        outputs = model(**batch)

        loss = outputs.pop('all')
        ppl = outputs.pop('ppl')
        probe_loss = outputs.pop('probe', None)

        if 'input_ids' in batch:
            input_ids = batch['input_ids']
        elif 'tgt_input_ids' in batch:
            input_ids = batch['tgt_input_ids']
        else:
            assert 'src_input_ids' in batch
            input_ids = batch['src_input_ids']

        if n_gpu > 1:
            loss = loss.mean()
            ppl = ppl.mean()
            if probe_loss is not None:
                probe_loss = probe_loss.mean()

        loss = loss / (args.train_batch_size * args.gradient_accumulation_steps / input_ids.shape[0])
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # s3: probe loss backward with dedicated optimizer
        if probe_loss is not None and probe_optimizer is not None:
            probe_loss.backward()
            probe_optimizer.step()
            probe_optimizer.zero_grad()

        tmp_loss = float(loss.item()) * (args.train_batch_size * args.gradient_accumulation_steps / input_ids.shape[0])
        tr_loss += tmp_loss
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        mean_loss = tr_loss / nb_tr_steps

        if ppl.item() < INF:
            tmp_ppl = ppl.item()
        else:
            tmp_ppl = mean_ppl
        tr_ppl += tmp_ppl
        mean_ppl = tr_ppl / nb_tr_steps

        n_token_total += input_ids.shape[0] * input_ids.shape[1]
        n_token_real += (input_ids != 0).sum().item()

        step += 1
        if step % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if args.local_rank != -1:
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if args.local_rank != -1:
                mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()
                mean_ppl = sum(all_gather_list(mean_ppl)) / get_world_size()
                n_token_real_all_proc = sum(all_gather_list(n_token_real))
                n_token_total_all_proc = sum(all_gather_list(n_token_total))
            else:
                n_token_real_all_proc = n_token_real
                n_token_total_all_proc = n_token_total

            if args.local_rank == -1 or get_rank() == 0:
                epoch_time = time.time() - train_start_time_epoch
                if pbar is not None:
                    pbar_str = ''
                    for k, v in outputs.items():
                        if n_gpu > 1:
                            pbar_str += f"{k}: {v.mean().item():.2f} "
                        else:
                            pbar_str += f"{k}: {v.item():.2f} "
                    pbar_str += f"ppl: {mean_ppl:.2f} epoch: {epoch}"
                    pbar.set_postfix_str(pbar_str)
                    if args.num_epochs is not None:
                        pbar.update(args.gradient_accumulation_steps)
                    else:
                        pbar.update(1)

                print(f'{epoch+1},{global_step+1},{step+1},{tmp_loss},{tmp_ppl},{mean_loss},{mean_ppl},'
                      f'{n_token_real_all_proc},{n_token_total_all_proc},{epoch_time}', file=train_logger)

            if args.num_epochs is None and global_step % args.valid_step == 0:
                if args.local_rank == -1 or get_rank() == 0:
                    torch.save(model.state_dict(), join(output_dir, f'{global_step}.bin'))
                    toker.save_vocabulary(output_dir)
                    model.config.to_json_file(join(output_dir, f'config.json'))

                    eval_loss, eval_ppl, eval_samples, *_ = eval_model_loss(
                        model=model, eval_dataloader=eval_dataloader_loss,
                        epoch_id=epoch, infer=False, args=args,
                    )
                    print(f'{epoch+1},{global_step+1},{step+1},{eval_loss},{eval_ppl}', file=eval_logger)
                    logger.info('current learning rate: ' + str(optimizer.param_groups[0]['lr']))
                    model.train()

            if args.num_epochs is None and global_step >= args.num_optim_steps:
                break

        if (step + 1) % CACHE_EMPTY_STEP == 0:
            torch.cuda.empty_cache()

    if args.num_epochs is not None:
        if args.local_rank == -1 or get_rank() == 0:
            torch.save(model.state_dict(), join(output_dir, f'epoch-{epoch}.bin'))
            toker.save_vocabulary(output_dir)
            model.config.to_json_file(join(output_dir, f'config.json'))

            eval_loss, eval_ppl, eval_samples, *_ = eval_model_loss(
                model=model, eval_dataloader=eval_dataloader_loss,
                epoch_id=epoch, infer=False, args=args,
            )
            print(f'{epoch},{global_step+1},{step+1},{eval_loss},{eval_ppl}', file=eval_logger)
            logger.info('current learning rate: ' + str(optimizer.param_groups[0]['lr']))
            model.train()

    if args.num_epochs is None and global_step >= args.num_optim_steps:
        break

    epoch += 1
    if args.num_epochs is not None and epoch == args.num_epochs:
        break


#########################################################################
# Probe extra epochs (017-style: best backbone + probe training)
#########################################################################

if args.probe_extra_epochs > 0 and probe_optimizer is not None and (args.local_rank == -1 or get_rank() == 0):
    best_epoch_idx = 0
    best_ppl = float('inf')
    eval_logger.flush()
    with open(join(output_dir, 'eval_log.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row['epoch'])
            ppl_val = float(row['freq_ppl'])
            if ppl_val < best_ppl:
                best_ppl = ppl_val
                best_epoch_idx = ep

    best_ckpt_path = join(output_dir, f'epoch-{best_epoch_idx}.bin')
    logger.info(f'Loading best epoch {best_epoch_idx} (val PPL={best_ppl:.2f}) for probe extra training')
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device), strict=False)

    # Freeze all except probe
    _frozen_params = []
    for name, param in model.named_parameters():
        if id(param) not in _joint_probe_param_ids and param.requires_grad:
            param.requires_grad = False
            _frozen_params.append(name)

    _saved_sampler = None
    if hasattr(_raw_model, '_scheduled_sampler'):
        _saved_sampler = _raw_model._scheduled_sampler
        _raw_model._scheduled_sampler = None

    probe_params = [p for p in _raw_model._joint_probe_head.parameters() if p.requires_grad]
    probe_optimizer = torch.optim.Adam(probe_params, lr=args.probe_lr)
    logger.info(f'Starting {args.probe_extra_epochs} probe-only extra epochs (backbone frozen, lr={args.probe_lr})')

    pbar_probe = tqdm.tqdm(total=args.probe_extra_epochs * len(train_dataloader),
                           desc="probe extra") if args.pbar else None

    _best_val_probe_loss = float('inf')
    _best_probe_state = None
    _patience = 10
    _patience_counter = 0

    for probe_epoch in range(args.probe_extra_epochs):
        _raw_model._joint_probe_head.train()
        model.eval()
        _probe_loss_sum, _probe_steps = 0.0, 0
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            probe_hidden, strat_id = _extract_probe_hidden(model, _raw_model, batch)
            probe_logits = _raw_model._joint_probe_head(probe_hidden)
            probe_loss = torch.nn.functional.cross_entropy(probe_logits, strat_id)
            if n_gpu > 1:
                probe_loss = probe_loss.mean()
            probe_loss.backward()
            probe_optimizer.step()
            probe_optimizer.zero_grad()
            _probe_loss_sum += probe_loss.item()
            _probe_steps += 1
            if pbar_probe is not None:
                pbar_probe.set_postfix_str(f"probe_loss: {probe_loss.item():.4f} epoch: {probe_epoch}")
                pbar_probe.update(1)

        _train_probe_loss = _probe_loss_sum / _probe_steps if _probe_steps > 0 else float('inf')

        _raw_model._joint_probe_head.eval()
        _val_probe_loss_sum, _val_probe_steps = 0.0, 0
        with torch.no_grad():
            for batch in eval_dataloader_loss:
                batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
                probe_hidden, strat_id = _extract_probe_hidden(model, _raw_model, batch)
                probe_logits = _raw_model._joint_probe_head(probe_hidden)
                val_probe_loss = torch.nn.functional.cross_entropy(probe_logits, strat_id)
                if n_gpu > 1:
                    val_probe_loss = val_probe_loss.mean()
                _val_probe_loss_sum += val_probe_loss.item()
                _val_probe_steps += 1
        _val_probe_loss = _val_probe_loss_sum / _val_probe_steps if _val_probe_steps > 0 else float('inf')

        logger.info(f'Probe extra epoch {probe_epoch}: train_loss={_train_probe_loss:.4f} '
                    f'val_loss={_val_probe_loss:.4f} best={_best_val_probe_loss:.4f} '
                    f'patience={_patience_counter}/{_patience}')

        if _val_probe_loss < _best_val_probe_loss:
            _best_val_probe_loss = _val_probe_loss
            _best_probe_state = {k: v.clone() for k, v in _raw_model._joint_probe_head.state_dict().items()}
            _patience_counter = 0
        else:
            _patience_counter += 1
            if _patience_counter >= _patience:
                logger.info(f'Early stopping at epoch {probe_epoch}')
                break

    if _best_probe_state is not None:
        _raw_model._joint_probe_head.load_state_dict(_best_probe_state)
        logger.info(f'Restored best probe (val_loss={_best_val_probe_loss:.4f})')

    _save_path = join(output_dir, f'epoch-{best_epoch_idx}_probe{args.probe_save_suffix}.bin')
    torch.save(model.state_dict(), _save_path)
    logger.info(f'Saved probe-trained checkpoint: {_save_path}')

    if pbar_probe is not None:
        pbar_probe.close()

    for name, param in model.named_parameters():
        if name in _frozen_params:
            param.requires_grad = True
    if _saved_sampler is not None:
        _raw_model._scheduled_sampler = _saved_sampler


if args.local_rank == -1 or get_rank() == 0:
    if pbar is not None:
        pbar.close()
    train_logger.close()
    eval_logger.close()
