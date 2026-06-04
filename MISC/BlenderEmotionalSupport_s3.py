"""
MISC + s3 Method: LoRA Expert Routing + Scheduled Sampling + SASRec Ensemble.

Fork of BlenderEmotionalSupport.py with:
  - A/B condition support (b_condition flag)
  - LoRA expert routing on decoder (8 experts) + shared LoRA on encoder
  - Scheduled sampling for LoRA routing
  - Separate LoRA learning rate
  - argparse overrides for sweep

Pipeline (PMPC-style, 3 stages):
  1. Backbone training (this script)
  2. Grid search for ensemble alpha (ensemble/grid_search_s3.py)
  3. Ensemble inference (generate_s3_ensemble.py)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from metric.myMetrics import Metric
import glob
import logging
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import time
from pathlib import Path
import json

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from src.transformers import (BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, BlenderbotSmallConfig)
from src.transformers.models.blenderbot_small.modeling_blenderbot_small import shift_tokens_right

from lora_expert_layers import LoRAExpertContext, wrap_modules_with_lora, disable_expert_lora
from scheduled_sampling import StrategyScheduledSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# ============================================================================
# Args (hardcoded defaults, overridable via CLI argparse)
# ============================================================================

class Args():
    def __init__(self):
        TAG = 'all_loss'
        self.output_dir = os.path.join('blender_strategy', TAG)
        self.generation_dir = os.path.join('generated_data', TAG)
        self.model_type = 'mymodel'
        self.model_name_or_path = '../Blenderbot_small-90M'
        self.config_name = '../Blenderbot_small-90M'
        self.tokenizer_name = '../Blenderbot_small-90M'
        self.data_split = "conv_level"
        self.data_path = "./dataset"

        if self.data_split == "conv_level":
            self.train_file_name = "train_cl.txt"
            self.eval_file_name = "dev_cl.txt"
            self.test_file_name = "test_cl.txt"
            self.train_comet_file = "trainComet_cl.txt"
            self.eval_comet_file = "devComet_cl.txt"
            self.test_comet_file = "testComet_cl.txt"
            self.situation_train_comet_file = "trainComet_st_cl.txt"
            self.situation_eval_comet_file = "devComet_st_cl.txt"
            self.situation_test_comet_file = "testComet_st_cl.txt"
        else:
            raise ValueError(f"s3 only supports conv_level split, got: {self.data_split}")

        self.model_cache_dir = './blender-small'
        self.data_cache_dir = './cached'
        self.block_size = 512
        self.do_train = True
        self.do_eval = False
        self.generation = False
        self.generate_and_eval = False
        self.evaluate_during_training = True
        self.per_gpu_train_batch_size = 16
        self.per_gpu_eval_batch_size = 25
        self.gradient_accumulation_steps = 1
        self.learning_rate = 2e-5
        self.weight_decay = 0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 8
        self.max_steps = -1
        self.warmup_steps = 120
        self.logging_steps = 30
        self.save_steps = 30
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = False
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.strategy = False
        self.turn = False
        self.role = False

        # s3: LoRA
        self.lora_rank = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.lora_num_experts = 8
        self.lora_use_encoder_shared = True
        self.lora_lr = 5e-5

        # s3: Scheduled Sampling
        self.use_scheduled_sampling = True
        self.ss_schedule_type = 'step'
        self.ss_warmup_epochs = 1
        self.ss_p_sample = 0.10
        self.ss_p_random = 0.0

        # s3: B condition
        self.b_condition = True

        # s3: eval split for generate (test or valid)
        self.eval_split = 'test'


def parse_args_override(args):
    """Override Args() defaults with CLI arguments for sweep."""
    parser = argparse.ArgumentParser(description='MISC s3 training')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--generation_dir', type=str, default=None)
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=None)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--do_train', action='store_true', default=None)
    parser.add_argument('--no_train', action='store_true', default=False)
    parser.add_argument('--generate_only', action='store_true', default=False)
    parser.add_argument('--overwrite_cache', action='store_true', default=None)

    # s3 args
    parser.add_argument('--lora_rank', type=int, default=None)
    parser.add_argument('--lora_alpha', type=int, default=None)
    parser.add_argument('--lora_dropout', type=float, default=None)
    parser.add_argument('--lora_num_experts', type=int, default=None)
    parser.add_argument('--lora_lr', type=float, default=None)
    parser.add_argument('--lora_use_encoder_shared', type=str, default=None)

    parser.add_argument('--use_scheduled_sampling', type=str, default=None)
    parser.add_argument('--ss_schedule_type', type=str, default=None)
    parser.add_argument('--ss_warmup_epochs', type=int, default=None)
    parser.add_argument('--ss_p_sample', type=float, default=None)
    parser.add_argument('--ss_p_random', type=float, default=None)

    parser.add_argument('--b_condition', type=str, default=None)
    parser.add_argument('--eval_split', type=str, default=None, choices=['test', 'valid'])

    cli_args, _ = parser.parse_known_args()

    # Apply overrides
    for key, val in vars(cli_args).items():
        if val is not None and key not in ('no_train', 'generate_only'):
            # Handle boolean string args
            if key in ('lora_use_encoder_shared', 'use_scheduled_sampling', 'b_condition'):
                val = val.lower() in ('true', '1', 'yes')
            setattr(args, key, val)

    if cli_args.no_train:
        args.do_train = False
    if cli_args.generate_only:
        args.do_train = False
        args.generation = True

    return args


# ============================================================================
# Data processing (from original, with B condition support)
# ============================================================================

class InputFeatures_train(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                 role_ids, lm_labels, cls_position, cls_label, strategy_ids, input_len=None):
        self.conv_id = conv_id
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.role_ids = role_ids
        self.lm_labels = lm_labels
        self.cls_position = cls_position
        self.cls_label = cls_label
        self.strategy_ids = strategy_ids
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len


class InputFeatures_blender(object):
    def __init__(self, encoder_feature, decoder_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask):
        self.conv_id = encoder_feature.conv_id
        self.input_ids = encoder_feature.input_ids
        self.position_ids = encoder_feature.position_ids
        self.token_type_ids = encoder_feature.token_type_ids
        self.role_ids = encoder_feature.role_ids
        self.lm_labels = encoder_feature.lm_labels
        self.cls_position = encoder_feature.cls_position
        self.cls_label = encoder_feature.cls_label
        self.strategy_ids = encoder_feature.strategy_ids
        self.decoder_input_ids = decoder_feature.input_ids
        self.decoder_position_ids = decoder_feature.position_ids
        self.decoder_token_type_ids = decoder_feature.token_type_ids
        self.decoder_role_ids = decoder_feature.role_ids
        self.decoder_lm_labels = decoder_feature.lm_labels
        self.decoder_cls_position = decoder_feature.cls_position
        self.decoder_cls_label = decoder_feature.cls_label
        self.decoder_strategy_ids = decoder_feature.strategy_ids
        self.comet_ids = comet_ids
        self.comet_mask = comet_mask
        self.emotion = emotion
        self.comet_st_ids = comet_st_ids
        self.comet_st_mask = comet_st_mask


def process_row_to_comet_query(row):
    sents = row.strip().split('EOS')
    n_sent = len(sents)
    for i in range(n_sent-1, -1, -1):
        tokens = sents[i].strip().split(' ')
        if int(tokens[1]) == 0:
            return ' '.join(tokens[3:])


def summary(test_file_path, generate_file_path, reference_file_path, summary_file_path, all_top_k_blocks, all_top_k_blocks_st, chat_texts, test_situation_file_path):
    with open(test_file_path, "r", encoding="utf-8") as f:
        ctx = f.read().split("\n")
    with open(test_situation_file_path, "r", encoding="utf-8") as f:
        st = f.read().split("\n")
    ctx = ctx[:-1]
    st = st[:-1]
    with open(generate_file_path, "r", encoding="utf-8") as f:
        gen_rep = json.load(f)
    with open(reference_file_path, "r", encoding="utf-8") as f:
        ref_rep = json.load(f)
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        for (ctx_row, ref_rep_row, gen_rep_row, top_k_blocks, top_k_blocks_st, chat_text, st_row) in zip(ctx, ref_rep, gen_rep, all_top_k_blocks, all_top_k_blocks_st, chat_texts, st):
            query = process_row_to_comet_query(chat_text)
            if query is None:
                query = ""
            line = '[contxt]\t' + ctx_row + '\n[reference_response]\t' + ref_rep_row + '\n[hypothesis_response]\t' + gen_rep_row + '\n[comet query]\t' + query + '\n[comet blocks (attention top5)]\t' + '  '.join(top_k_blocks) +'\n[situation]\t' + st_row + '\n[situation comet blocks (attention top5)]\t' + '  '.join(top_k_blocks_st) + '\n' * 2
            f.writelines(line)


def extract_top_k_attention_comet_block(mutual_attentions, comet_rows, k):
    all_top_k_blocks = []
    num_block = len(mutual_attentions[0])
    for mutual_attention, comet_row in zip(mutual_attentions, comet_rows):
        comet_blocks = comet_row.split('__EOS__')[:-1]
        if len(comet_blocks) < num_block:
            comet_blocks += (['[PAD]'] * (num_block - len(comet_blocks)))
        index = torch.topk(mutual_attention, k).indices
        top_k_blocks = [comet_blocks[i] for i in index.numpy().tolist()]
        all_top_k_blocks.append(top_k_blocks)
    return all_top_k_blocks


def _get_comet_input(comet_row, tokenizer, max_num_attr=30, max_len_attr=10):
    attrs = comet_row.split('__EOS__')[:-1]
    comet_ids = []
    comet_mask = []
    for ids, attr in enumerate(attrs):
        if ids == max_num_attr:
            break
        comet_attr_ids = tokenizer.encode(attr)
        if len(comet_attr_ids) < max_len_attr:
            comet_attr_ids += [tokenizer.pad_token_id]*(max_len_attr - len(comet_attr_ids))
        else:
            comet_attr_ids = comet_attr_ids[:max_len_attr]
        comet_ids.append(comet_attr_ids)
        comet_mask.append(1)

    if len(comet_ids) < max_num_attr:
        comet_ids += ([[tokenizer.pad_token_id]*max_len_attr]) * (max_num_attr - len(comet_ids))
        comet_mask += [0] * (max_num_attr - len(comet_mask))

    assert len(comet_ids) == max_num_attr
    assert len(comet_mask) == max_num_attr
    return comet_ids, comet_mask


def _make_feature(id_, sents, rls, ts, eos, pad=False, block_size=512, strategy_labels=None, evaluate=False, str_embd=False, generation=False):
    if len(sents) == 0:
        return InputFeatures_train([], [], [], [], [],
                            [], [] , [], [])
    input_ids = [i for s in sents for i in s+[eos]]
    lm_labels = []
    token_type_ids = []
    roles = []
    strategy_ids = []

    for i, s in enumerate(sents):
        token_type_ids += [ts[i]] * (len(s) + 1)
        if str_embd:
            strategy_ids += [strategy_labels[-1]] * (len(s) + 1)
        else:
            strategy_ids += [8] * (len(s) + 1)
        if i < len(sents) - 1:
            lm_labels += [-100] * (len(s) + 1)
            roles += [rls[i]] * (len(s) + 1)
        else:
            lm_labels += (s + [eos])
            roles += [rls[i]] * (len(s) + 1)

    i = len(lm_labels) - 1
    while i >= 0:
        if lm_labels[i] != -100:
            break
        i -= 1
    input_ids = input_ids[:i+1]
    lm_labels = lm_labels[:i+1]
    token_type_ids = token_type_ids[:i+1]
    roles = roles[:i+1]
    if not str_embd:
        strategy_ids = [8]*len(input_ids)
    else:
        strategy_ids = strategy_ids[:i+1]

    assert (len(input_ids) == len(token_type_ids)
            == len(lm_labels) == len(roles) == len(strategy_ids))
    if len(input_ids) > block_size:
        cut_index = input_ids.index(eos,-512) + 1
        input_ids = input_ids[cut_index:]
        token_type_ids = token_type_ids[cut_index:]
        lm_labels = lm_labels[cut_index:]
        roles = roles[cut_index:]
        strategy_ids = strategy_ids[cut_index:]
    if pad:
        while len(input_ids) % 8 != 0:
            input_ids.append(0)
            token_type_ids.append(0)
            lm_labels.append(-100)
            roles.append(0)
            strategy_ids.append(8)
        assert len(input_ids) % 8 == 0
    position_ids = list(range(len(input_ids)))
    assert (len(input_ids) == len(position_ids) == len(token_type_ids)
            == len(lm_labels) == len(roles) == len(strategy_ids))
    if len(input_ids) == 0:
        import pdb
        pdb.set_trace()
    if True:
        cls_position = len(input_ids)-1-input_ids[::-1].index(eos)
    if evaluate and strategy_labels[-1]!=8:
        try:
            lm_labels[lm_labels.index(strategy_labels[-1]+50257+4687)] = -100
        except Exception:
            pass

    feature = InputFeatures_train(id_, input_ids, position_ids, token_type_ids, roles,
                            lm_labels, cls_position, strategy_labels[-1], strategy_ids)
    return feature


def _norm_text(text):
    emo, r, t, *toks = text.strip().split()
    emo = int(emo)
    r = int(r)
    t = int(t)
    toks = ' '.join(toks[:len(toks)])
    return emo, r, t, toks


def _get_inputs_from_text(text, tokenizer, strategy=True, cls=False):
    srcs = text.strip()
    inputs = []
    roles = []
    turns = []
    strategy_labels = []
    srcs = srcs.split(" EOS")
    emotion = None
    for idx, src in enumerate(srcs):
        if src == "":
            continue
        src_emo, src_role, src_turn, src = _norm_text(src)
        if emotion is None:
            emotion = src_emo
        context_id = tokenizer.encode(src)
        if not strategy:
            context_id = [i for i in context_id if i < 50257+4687]
        elif cls:
            context_id = tokenizer.cls + [i for i in context_id if i < 50257+4687]
        else:
            pass

        if src_role == 1:
            try:
                label = "["+src.split("[")[1].split("]")[0]+"]"
            except Exception:
                strategy_labels.append(8)
            else:
                strategy_labels.append(tokenizer.encode([label])[0] - 50257-4687)
        else:
            strategy_labels.append(8)

        inputs.append(context_id)
        roles.append(src_role)
        turns.append(src_turn)

    return inputs, roles, turns, strategy_labels, emotion


def construct_conv_ESD(idx, row, comet_row, comet_st_row, tokenizer, eos=True, pad=True, cls=False, evaluate=False, strategy=True, generation=False):
    inputs, roles, turns, strategy_labels, _ = _get_inputs_from_text("EOS".join(row.split("EOS")[:-1]), tokenizer, strategy=strategy)
    d_inputs, d_roles, d_turns, d_strategy_labels, emotion = _get_inputs_from_text(row.split("EOS")[-1], tokenizer, strategy=strategy)
    feature = _make_feature(idx, inputs, roles, turns, tokenizer.eos_token_id, pad=pad, strategy_labels=strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)
    d_feature = _make_feature(idx, d_inputs, d_roles, d_turns, tokenizer.eos_token_id, pad=pad, strategy_labels=d_strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)
    comet_ids, comet_mask = _get_comet_input(comet_row, tokenizer)
    comet_st_ids, comet_st_mask = _get_comet_input(comet_st_row, tokenizer, max_num_attr=20)
    feature = InputFeatures_blender(feature, d_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask)
    return feature


class ESDDataset(Dataset):
    def __init__(self, tokenizer, args, df, comet, comet_st, block_size=512, evaluate=False, strategy=True, test=False):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        self.tokenizer = tokenizer
        directory = args.data_cache_dir
        if not os.path.exists(directory):
            os.makedirs(directory)

        split_tag = getattr(args, 'data_split', '')
        # s3: separate cache for A/B condition
        cond_tag = '_s3b' if getattr(args, 'b_condition', False) else '_s3a'

        if evaluate:
            if not test:
                cached_features_file = os.path.join(
                    directory, 'val_' + args.model_type + "_cached_lm_" + str(block_size) + '_' + split_tag + cond_tag
                )
            else:
                cached_features_file = os.path.join(
                    directory, 'test_' + args.model_type + "_cached_lm_" + str(block_size) + '_' + split_tag + cond_tag
                )
        else:
            cached_features_file = os.path.join(
                directory, 'trn_' + args.model_type + "_cached_lm_" + str(block_size) + '_' + split_tag + cond_tag
            )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.features = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            assert len(df) == len(comet) == len(comet_st)
            self.features = []

            # B condition: strategy=False removes strategy tokens from encoder/decoder
            use_strategy = not getattr(args, 'b_condition', False)

            for idx, (row, comet_row, comet_st_row) in enumerate(zip(df[:-1], comet[:-1], comet_st[:-1])):
                conv = construct_conv_ESD(idx, row, comet_row, comet_st_row, tokenizer, cls=False, strategy=use_strategy, evaluate=evaluate)
                if len(conv.input_ids) >= block_size:
                    conv.input_ids = conv.input_ids[-block_size:]
                    conv.input_ids[0] = tokenizer.encode(tokenizer.cls_token)[0]
                else:
                    conv.input_ids = tokenizer.encode(tokenizer.cls_token) + conv.input_ids
                self.features.append(conv)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Finished~")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    @staticmethod
    def collate(features):
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        position_ids = pad_sequence([torch.tensor(f.position_ids, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        role_ids = pad_sequence([torch.tensor(f.role_ids, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long) for f in features], batch_first=True, padding_value=-100)
        cls_positions = torch.tensor([f.cls_position for f in features], dtype=torch.long)
        cls_labels = torch.tensor([f.cls_label for f in features], dtype=torch.long)
        strategy_ids = pad_sequence([torch.tensor(f.strategy_ids, dtype=torch.long) for f in features], batch_first=True, padding_value=8)
        decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        decoder_position_ids = pad_sequence([torch.tensor(f.decoder_position_ids, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        decoder_token_type_ids = pad_sequence([torch.tensor(f.decoder_token_type_ids, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        decoder_role_ids = pad_sequence([torch.tensor(f.decoder_role_ids, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        decoder_labels = pad_sequence([torch.tensor(f.decoder_lm_labels, dtype=torch.long) for f in features], batch_first=True, padding_value=-100)
        decoder_cls_positions = torch.tensor([f.decoder_cls_position for f in features], dtype=torch.long)
        decoder_cls_labels = torch.tensor([f.decoder_cls_label for f in features], dtype=torch.long)
        decoder_strategy_ids = pad_sequence([torch.tensor(f.decoder_strategy_ids, dtype=torch.long) for f in features], batch_first=True, padding_value=8)
        comet_ids = torch.tensor([f.comet_ids for f in features], dtype=torch.long)
        comet_mask = torch.tensor([f.comet_mask for f in features], dtype=torch.long)
        emotion = torch.tensor([f.emotion for f in features], dtype=torch.long)
        comet_st_ids = torch.tensor([f.comet_st_ids for f in features], dtype=torch.long)
        comet_st_mask = torch.tensor([f.comet_st_mask for f in features], dtype=torch.long)

        return (input_ids, position_ids, token_type_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids,
                decoder_input_ids, decoder_position_ids, decoder_token_type_ids, decoder_role_ids, decoder_labels,
                decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask, emotion,
                comet_st_ids, comet_st_mask)


def load_and_cache_examples(args, tokenizer, df, comet, comet_st, evaluate=False, strategy=True, test=False):
    return ESDDataset(tokenizer, args, df, comet, comet_st, evaluate=evaluate, strategy=strategy, test=test)


# ============================================================================
# s3: LoRA setup
# ============================================================================

def setup_lora(model, args):
    """Wrap model with LoRA expert layers."""
    target_proj = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
    is_target = lambda n, m: isinstance(m, nn.Linear) and n.split(".")[-1] in target_proj

    # Decoder: 8+1 experts (index 8 = shared slot, unused in practice)
    decoder_lora_context = LoRAExpertContext()
    lora_total_experts = args.lora_num_experts + 1  # 9

    decoder_wrapped = wrap_modules_with_lora(
        root=model.model.decoder,
        predicate=is_target,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        num_experts=lora_total_experts,
        context=decoder_lora_context,
    )
    logger.info(f"Decoder LoRA: wrapped {len(decoder_wrapped)} modules, {lora_total_experts} experts")

    # Encoder: shared LoRA (no expert routing)
    if args.lora_use_encoder_shared:
        encoder_wrapped = wrap_modules_with_lora(
            root=model.model.encoder,
            predicate=is_target,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            num_experts=None,
            context=None,
        )
        logger.info(f"Encoder LoRA: wrapped {len(encoder_wrapped)} modules (shared)")

    model._decoder_lora_context = decoder_lora_context
    return model


def save_lora_config(args, output_dir):
    """Save LoRA configuration for reload."""
    config = {
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'lora_num_experts': args.lora_num_experts,
        'lora_use_encoder_shared': args.lora_use_encoder_shared,
        'b_condition': args.b_condition,
    }
    with open(os.path.join(output_dir, 'lora_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def load_model_with_lora(model_dir, tokenizer, args, device):
    """Load model and apply LoRA wrapping, then load state dict."""
    # Load LoRA config
    lora_config_path = os.path.join(model_dir, 'lora_config.json')
    if os.path.exists(lora_config_path):
        with open(lora_config_path, 'r') as f:
            lora_config = json.load(f)
        for k, v in lora_config.items():
            setattr(args, k, v)

    # Create model and apply LoRA wrapping first
    model = BlenderbotSmallForConditionalGeneration.from_pretrained(
        args.model_name_or_path, cache_dir=args.model_cache_dir)
    model.resize_token_embeddings(len(tokenizer))
    model = setup_lora(model, args)

    # Load the saved state dict (which includes LoRA weights)
    state_dict_path = os.path.join(model_dir, 'pytorch_model.bin')
    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded model state dict from {state_dict_path}")

    model.to(device)
    return model


# ============================================================================
# s3: COMET embedding helper
# ============================================================================

def encode_comet_embeddings(model, tokenizer, comet_ids, comet_mask, comet_ids_st, comet_mask_st, device):
    """Encode COMET inputs through encoder (no grad)."""
    comet_ids = comet_ids.to(device)
    comet_mask = comet_mask.to(device)
    comet_ids_st = comet_ids_st.to(device)
    comet_mask_st = comet_mask_st.to(device)

    _model = model.module if hasattr(model, 'module') else model

    batch_size, n_attr, len_attr = comet_ids.shape
    comet_ids_flat = comet_ids.view(-1, len_attr)
    with torch.no_grad():
        comet_embs = _model.model.encoder(comet_ids_flat, attention_mask=comet_ids_flat.ne(tokenizer.pad_token_id))[0][:, 0, :]
    comet_embs = comet_embs.view(batch_size, n_attr, -1)

    batch_size, n_attr, len_attr = comet_ids_st.shape
    comet_ids_st_flat = comet_ids_st.view(-1, len_attr)
    with torch.no_grad():
        comet_embs_st = _model.model.encoder(comet_ids_st_flat, attention_mask=comet_ids_st_flat.ne(tokenizer.pad_token_id))[0][:, 0, :]
    comet_embs_st = comet_embs_st.view(batch_size, n_attr, -1)

    return comet_embs, comet_mask, comet_embs_st, comet_mask_st


# ============================================================================
# s3: Training forward (encoder/decoder split with SS + LoRA routing)
# ============================================================================

def forward_s3(model, tokenizer, args, batch_data, scheduled_sampler=None, is_train=True):
    """
    s3 forward pass: encoder → SS → LoRA routing → decoder → loss.

    Returns: (loss, lm_loss, emo_loss, intensity_loss, strategy_loss, strategy_logits)
    """
    device = args.device
    _model = model.module if hasattr(model, 'module') else model

    input_ids = batch_data['input_ids']
    decoder_input_ids = batch_data['decoder_input_ids']
    decoder_label_ids = batch_data['decoder_label_ids']
    decoder_strategy_ids = batch_data['decoder_strategy_ids']
    decoder_turn_ids = batch_data['decoder_turn_ids']
    comet_embs = batch_data['comet_embs']
    comet_mask = batch_data['comet_mask']
    comet_embs_st = batch_data['comet_embs_st']
    comet_mask_st = batch_data['comet_mask_st']
    emotion = batch_data['emotion']

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # Step 1: GT one-hot for strategy_embs
    batch_size = decoder_strategy_ids.shape[0]
    onehot = torch.zeros(batch_size, 8).to(device)
    strategy_logit_ground = onehot.scatter_(1, decoder_strategy_ids.unsqueeze(1), 1).float()

    # Step 2: Encoder forward
    encoder_outputs = _model.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        comet_embs=comet_embs,
        comet_mask=comet_mask,
        comet_embs_st=comet_embs_st,
        comet_mask_st=comet_mask_st,
        turn_ids=decoder_turn_ids[:, 0] if decoder_turn_ids is not None else None,
        strategy_logit_ground=strategy_logit_ground,
    )
    strategy_logits = encoder_outputs.strategy_logits

    # Step 3: Scheduled sampling for LoRA routing
    if is_train and scheduled_sampler is not None:
        routing_strat_id = scheduled_sampler.sample_strategy_tokens(
            gt_strategy_ids=decoder_strategy_ids,
            strategy_logits=strategy_logits.detach(),
        )
    else:
        routing_strat_id = decoder_strategy_ids

    # Step 4: Set LoRA expert
    _model._decoder_lora_context.expert_ids = routing_strat_id.long()

    # Step 5: Decoder forward
    decoder_input_ids_shifted = shift_tokens_right(
        decoder_input_ids, _model.config.pad_token_id, _model.config.decoder_start_token_id
    )

    decoder_outputs = _model.model.decoder(
        input_ids=decoder_input_ids_shifted,
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=attention_mask,
        comet_hidden_states=encoder_outputs.last_comet_hidden_state,
        comet_mask=encoder_outputs.comet_mask,
        comet_hidden_states_st=encoder_outputs.last_comet_hidden_state_st,
        comet_mask_st=encoder_outputs.comet_mask_st,
        strategy_embs=encoder_outputs.strategy_embs,
    )

    # Step 6: Loss
    lm_logits = _model.lm_head(decoder_outputs[0]) + _model.final_logits_bias
    loss_fct = nn.CrossEntropyLoss()
    lm_loss = loss_fct(lm_logits.reshape(-1, _model.config.vocab_size), decoder_label_ids.reshape(-1))

    strategy_loss = loss_fct(strategy_logits.view(-1, 8), decoder_strategy_ids)
    loss = lm_loss + strategy_loss

    # Emotion and intensity losses (computed but not backpropagated, matching original)
    emo_loss = torch.tensor(0.0, device=device)
    intensity_loss = torch.tensor(0.0, device=device)
    if encoder_outputs.emotion_logits is not None and emotion is not None:
        emo_loss = nn.CrossEntropyLoss()(encoder_outputs.emotion_logits.view(-1, 11), emotion.view(-1))
    if encoder_outputs.emotion_intensity is not None and decoder_turn_ids is not None:
        intensity_label = 1 / decoder_turn_ids[:, 0].type(torch.float)
        intensity_loss = nn.MSELoss()(encoder_outputs.emotion_intensity, intensity_label)

    # Step 7: Clear LoRA context
    _model._decoder_lora_context.expert_ids = None

    return loss, lm_loss, emo_loss, intensity_loss, strategy_loss, strategy_logits


# ============================================================================
# Utility
# ============================================================================

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):
    ordering_and_checkpoint_path = []
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.group(1)), path))
    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


# ============================================================================
# Train
# ============================================================================

def train(args, train_dataset, model, tokenizer):
    """Train the model with s3 method."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=ESDDataset.collate, drop_last=False
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model
    model.resize_token_embeddings(len(tokenizer))

    # s3: Optimizer with separate LoRA lr
    no_decay = ["bias", "LayerNorm.weight"]
    _is_lora = lambda n: '.lora_a' in n or '.lora_b' in n

    lora_params_decay = [p for n, p in model.named_parameters() if _is_lora(n) and not any(nd in n for nd in no_decay) and p.requires_grad]
    lora_params_no_decay = [p for n, p in model.named_parameters() if _is_lora(n) and any(nd in n for nd in no_decay) and p.requires_grad]
    lora_param_ids = {id(p) for p in lora_params_decay + lora_params_no_decay}

    base_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and id(p) not in lora_param_ids and p.requires_grad]
    base_no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and id(p) not in lora_param_ids and p.requires_grad]

    optimizer_grouped_parameters = [
        {'params': base_decay, 'weight_decay': args.weight_decay},
        {'params': base_no_decay, 'weight_decay': 0.0},
        {'params': lora_params_decay, 'weight_decay': 0.01, 'lr': args.lora_lr},
        {'params': lora_params_no_decay, 'weight_decay': 0.0, 'lr': args.lora_lr},
    ]

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params_count = sum(p.numel() for p in lora_params_decay + lora_params_no_decay)
    print(f'{total_params:,} total parameters.')
    print(f'{total_trainable:,} training parameters.')
    print(f'{lora_params_count:,} LoRA parameters (lr={args.lora_lr}).')

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # s3: Scheduled Sampling
    scheduled_sampler = None
    if args.use_scheduled_sampling:
        scheduled_sampler = StrategyScheduledSampler(
            schedule_type=args.ss_schedule_type,
            warmup_epochs=args.ss_warmup_epochs,
            total_epochs=int(args.num_train_epochs),
            p_sample=args.ss_p_sample,
            p_random=args.ss_p_random,
            num_strategies=8,
        )
        print(f"Scheduled Sampling: {scheduled_sampler}")

    # Multi-GPU
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training (s3) *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  LoRA lr = %s, p_sample = %s, b_condition = %s", args.lora_lr, args.ss_p_sample, args.b_condition)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_lm_loss, logging_lm_loss = 0.0, 0.0
    tr_emo_loss, logging_emo_loss = 0.0, 0.0
    tr_strategy_loss, logging_strategy_loss = 0.0, 0.0
    tr_intensity_loss, logging_intensity_loss = 0.0, 0.0
    best_ppl = 1e8

    model.zero_grad()
    train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch", disable=True)
    set_seed(args)

    for epoch in train_iterator:
        # s3: Update scheduled sampler epoch
        if scheduled_sampler is not None:
            scheduled_sampler.set_epoch(epoch)
            p_gt, p_sample, p_random = scheduled_sampler.get_current_probs()
            logger.info(f"Epoch {epoch}: SS p_gt={p_gt:.2f}, p_sample={p_sample:.2f}, p_random={p_random:.2f}")
            print(f"Epoch {epoch}: SS p_gt={p_gt:.2f}, p_sample={p_sample:.2f}, p_random={p_random:.2f}")

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            (input_ids, position_ids, turn_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids,
             decoder_input_ids, decoder_position_ids, decoder_turn_ids, decoder_role_ids, decoder_labels,
             decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask,
             emotion, comet_ids_st, comet_mask_st) = batch

            decoder_strategy_ids = decoder_strategy_ids[:, 0].to(args.device)

            model.train()
            if input_ids.shape[1] > 512:
                continue

            # Encode COMET
            comet_embs, comet_mask, comet_embs_st, comet_mask_st = encode_comet_embeddings(
                model, tokenizer, comet_ids, comet_mask, comet_ids_st, comet_mask_st, args.device
            )

            input_ids = input_ids.to(args.device)
            decoder_input_ids = decoder_input_ids.to(args.device)
            decoder_label_ids = decoder_labels.to(args.device)
            decoder_turn_ids = decoder_turn_ids.to(args.device)
            emotion = emotion.to(args.device)

            batch_data = {
                'input_ids': input_ids,
                'decoder_input_ids': decoder_input_ids,
                'decoder_label_ids': decoder_label_ids,
                'decoder_strategy_ids': decoder_strategy_ids,
                'decoder_turn_ids': decoder_turn_ids,
                'comet_embs': comet_embs,
                'comet_mask': comet_mask,
                'comet_embs_st': comet_embs_st,
                'comet_mask_st': comet_mask_st,
                'emotion': emotion,
            }

            # s3 forward
            loss, lm_loss, emo_loss, intensity_loss, strategy_loss, _ = forward_s3(
                model, tokenizer, args, batch_data, scheduled_sampler=scheduled_sampler, is_train=True
            )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.mean().backward()

            tr_loss += loss.item()
            tr_lm_loss += lm_loss.item()
            tr_emo_loss += emo_loss.item()
            tr_strategy_loss += strategy_loss.item()
            tr_intensity_loss += intensity_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, args.eval_dataset, "{}-{}".format("checkpoint", global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("lr: %f, step: %d, loss: %f, lm_loss: %f, emo_loss: %f, strategy_loss: %f, intensity_loss: %f",
                                scheduler.get_last_lr()[0], global_step,
                                (tr_loss - logging_loss) / args.logging_steps,
                                (tr_lm_loss - logging_lm_loss) / args.logging_steps,
                                (tr_emo_loss - logging_emo_loss) / args.logging_steps,
                                (tr_strategy_loss - logging_strategy_loss) / args.logging_steps,
                                (tr_intensity_loss - logging_intensity_loss) / args.logging_steps)

                    logging_loss = tr_loss
                    logging_lm_loss = tr_lm_loss
                    logging_emo_loss = tr_emo_loss
                    logging_strategy_loss = tr_strategy_loss
                    logging_intensity_loss = tr_intensity_loss

                    if results['eval_perplexity'] < best_ppl:
                        best_ppl = results['eval_perplexity']
                        output_dir = args.output_dir
                        os.makedirs(output_dir, exist_ok=True)
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        save_lora_config(args, output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s (ppl=%.4f)", output_dir, best_ppl)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    print("Train finished~")
    return global_step, tr_loss / max(global_step, 1)


# ============================================================================
# Evaluate
# ============================================================================

def evaluate(args, model, tokenizer, eval_dataset, prefix=""):
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=ESDDataset.collate, drop_last=False
    )

    logger.info("***** Running evaluation {} *****".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    num_samples = []
    emo_hits = []
    strategy_hits = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
        model.train()  # Keep BatchNorm in train mode (matching original MISC quirk)

        (input_ids, position_ids, turn_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids,
         decoder_input_ids, decoder_position_ids, decoder_turn_ids, decoder_role_ids, decoder_labels,
         decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask,
         emotion, comet_ids_st, comet_mask_st) = batch

        # Skip batch_size=1 to avoid BatchNorm error in train mode
        if input_ids.shape[0] == 1:
            continue

        if input_ids.shape[1] > 512:
            continue

        decoder_strategy_ids = decoder_strategy_ids[:, 0].to(args.device)
        emotion = emotion.to(args.device)

        comet_embs, comet_mask, comet_embs_st, comet_mask_st = encode_comet_embeddings(
            model, tokenizer, comet_ids, comet_mask, comet_ids_st, comet_mask_st, args.device
        )

        input_ids = input_ids.to(args.device)
        decoder_input_ids = decoder_input_ids.to(args.device)
        decoder_label_ids = decoder_labels.to(args.device)
        decoder_turn_ids = decoder_turn_ids.to(args.device)

        with torch.no_grad():
            batch_data = {
                'input_ids': input_ids,
                'decoder_input_ids': decoder_input_ids,
                'decoder_label_ids': decoder_label_ids,
                'decoder_strategy_ids': decoder_strategy_ids,
                'decoder_turn_ids': decoder_turn_ids,
                'comet_embs': comet_embs,
                'comet_mask': comet_mask,
                'comet_embs_st': comet_embs_st,
                'comet_mask_st': comet_mask_st,
                'emotion': emotion,
            }

            loss, lm_loss, emo_loss, intensity_loss, strategy_loss, strategy_logits = forward_s3(
                model, tokenizer, args, batch_data, scheduled_sampler=None, is_train=False
            )

            # Emotion accuracy
            if hasattr(model, 'module'):
                _m = model.module
            else:
                _m = model
            # Re-run encoder to get emotion_logits (already computed in forward_s3 but not returned)
            # Instead, compute from strategy_logits which we have
            for idx in range(strategy_logits.shape[0]):
                if strategy_logits[idx].argmax() == decoder_strategy_ids[idx]:
                    strategy_hits.append(1)
                else:
                    strategy_hits.append(0)

            num_samples.append((decoder_label_ids.cpu().numpy() != -100).astype(int).sum())
            eval_loss += lm_loss.sum().item() * (decoder_label_ids.cpu().numpy() != -100).astype(int).sum()

        nb_eval_steps += 1

    eval_loss = eval_loss / sum(num_samples)
    perplexity = torch.exp(torch.tensor(eval_loss)).item()
    result = {
        "eval_perplexity": perplexity,
        "eval_strategy_predict_accuracy": sum(strategy_hits) / len(strategy_hits) if strategy_hits else 0.0,
        "eval_number_of_evaluated_examples": len(strategy_hits),
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("***** Eval results {} *****\n".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


# ============================================================================
# Generate (predictor-only, no ensemble)
# ============================================================================

def generate(args):
    additional_special_tokens = ["[Question]", "[Reflection of feelings]", "[Information]",
                                 "[Restatement or Paraphrasing]", "[Others]", "[Self-disclosure]",
                                 "[Affirmation and Reassurance]", "[Providing Suggestions]"]
    comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]",
                                       "[oEffect]", "[oReact]"]

    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_tokens(comet_additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

    if not args.no_cuda:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        device = torch.device("cpu")
        args.device = device
        args.n_gpu = 0

    set_seed(args)

    # Load model with LoRA
    model = load_model_with_lora(args.output_dir, tokenizer, args, device)
    model.eval()

    # Select eval split (test or valid)
    eval_split = getattr(args, 'eval_split', 'test')
    if eval_split == 'valid':
        data_file = args.eval_file_name
        comet_file = args.eval_comet_file
        situation_comet_file = args.situation_eval_comet_file
        logger.info("Generating on VALID set")
    else:
        data_file = args.test_file_name
        comet_file = args.test_comet_file
        situation_comet_file = args.situation_test_comet_file
        logger.info("Generating on TEST set")

    with open(args.data_path + "/" + data_file, "r") as f:
        chat_texts = f.read().split("\n")
    with open(args.data_path + "/" + situation_comet_file, "r", encoding="utf-8") as f:
        comet_st = f.read().split("\n")
    with open(args.data_path + "/" + comet_file, "r", encoding="utf-8") as f:
        comet = f.read().split("\n")

    assert len(comet) == len(chat_texts) == len(comet_st)

    gts = []
    refs = []
    mutual_attentions = []
    mutual_attentions_st = []
    strategy_logit_str = []
    strategy_hits = []
    strategy_record = []
    strategy_hits_topk = [[] for _ in range(8)]

    # For ensemble grid search: collect strategy distributions
    gen_results = []

    # B condition: strategy=False in construct_conv_ESD
    use_strategy = not args.b_condition

    for idx, (c_text, comet_row, comet_st_row) in tqdm(enumerate(zip(chat_texts[:-1], comet[:-1], comet_st[:-1])), desc="Testing"):
        if "EOS" not in c_text:
            continue

        chat_history = c_text
        f = construct_conv_ESD(idx, chat_history, comet_row, comet_st_row, tokenizer, eos=True, pad=False, cls=False, strategy=False, generation=True)
        if len(f.input_ids) >= args.block_size:
            f.input_ids = f.input_ids[-args.block_size:]
            f.input_ids[0] = tokenizer.encode(tokenizer.cls_token)[0]
        else:
            f.input_ids = tokenizer.encode(tokenizer.cls_token) + f.input_ids

        next_strategy_id = f.decoder_strategy_ids[0]
        gts.append(tokenizer.decode(f.decoder_input_ids, skip_special_tokens=True))

        # Prepare tensors
        input_ids = torch.tensor([f.input_ids], dtype=torch.long).to(device)
        emotion_t = torch.tensor([f.emotion], dtype=torch.long).to(device)
        comet_ids_t = torch.tensor([f.comet_ids], dtype=torch.long).to(device)
        comet_mask_t = torch.tensor([f.comet_mask], dtype=torch.long).to(device)
        comet_ids_st_t = torch.tensor([f.comet_st_ids], dtype=torch.long).to(device)
        comet_mask_st_t = torch.tensor([f.comet_st_mask], dtype=torch.long).to(device)

        # Encode COMET
        batch_size_c, n_attr, len_attr = comet_ids_t.shape
        comet_ids_flat = comet_ids_t.view(-1, len_attr)
        comet_embs = model.model.encoder(comet_ids_flat, attention_mask=comet_ids_flat.ne(tokenizer.pad_token_id))[0][:, 0, :]
        comet_embs = comet_embs.view(batch_size_c, n_attr, -1)

        batch_size_c, n_attr, len_attr = comet_ids_st_t.shape
        comet_ids_st_flat = comet_ids_st_t.view(-1, len_attr)
        comet_embs_st = model.model.encoder(comet_ids_st_flat, attention_mask=comet_ids_st_flat.ne(tokenizer.pad_token_id))[0][:, 0, :]
        comet_embs_st = comet_embs_st.view(batch_size_c, n_attr, -1)

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        # Encoder forward (strategy_logit_ground=None → predicted softmax)
        with torch.no_grad():
            encoder_outputs = model.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                comet_embs=comet_embs,
                comet_mask=comet_mask_t,
                comet_embs_st=comet_embs_st,
                comet_mask_st=comet_mask_st_t,
                strategy_logit_ground=None,  # Use predicted
            )

        strategy_logits = encoder_outputs.strategy_logits
        pred_strat_id = strategy_logits[0].argmax().item()

        # Set LoRA expert for generation
        model._decoder_lora_context.expert_ids = torch.tensor([pred_strat_id], device=device)

        # Generate using original model.generate with encoder_outputs
        paras = {
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

        refs.append(tokenizer.decode(chat_history_ids[:, :][0], skip_special_tokens=True))
        print(tokenizer.decode(chat_history_ids[:, :][0], skip_special_tokens=True))

        strategy_record.append({
            "ref strategy": tokenizer.decode([next_strategy_id + 54944]),
            "hyp strategy": tokenizer.decode([strategy_logits[0].argmax() + 54944])
        })
        print({"ref strategy": tokenizer.decode([next_strategy_id + 54944]),
               "hyp strategy": tokenizer.decode([strategy_logits[0].argmax() + 54944])})

        if strategy_logits[0].argmax() == next_strategy_id:
            strategy_hits.append(1)
        else:
            strategy_hits.append(0)

        for k in range(8):
            _, topk = strategy_logits[0].topk(k + 1, -1)
            strategy_hits_topk[k].append(sum((topk == next_strategy_id).cpu().numpy().tolist()))

        # Save strategy distribution for ensemble grid search
        strat_probs = F.softmax(strategy_logits[0], dim=-1).cpu().numpy().tolist()
        gen_results.append({
            'sample_id': idx,
            'pred_strat_id': pred_strat_id,
            'pred_strat_id_dist': ' '.join([f'{p:.6f}' for p in strat_probs]),
            'gt_strat_id': int(next_strategy_id),
        })

        strategy_logits_list = strategy_logits[0].cpu().numpy().tolist()
        strategy_logit_str.append('\t'.join(["%.4f" % logit for logit in strategy_logits_list]))

    for i in range(8):
        print(f"Top-{i+1} accuracy: {sum(strategy_hits_topk[i]) / len(strategy_hits_topk[i]):.4f}")
    print('strategy predict accuracy', sum(strategy_hits) / len(strategy_hits))

    all_top_k_blocks = extract_top_k_attention_comet_block(mutual_attentions, comet[:-1], 5)
    all_top_k_blocks_st = extract_top_k_attention_comet_block(mutual_attentions_st, comet_st[:-1], 5)

    if not os.path.exists(args.generation_dir):
        os.makedirs(args.generation_dir)

    if eval_split == 'valid':
        test_file_path = "dataset/dev_cl.txt"
        test_situation_file_path = "dataset/devSituation_cl.txt"
    else:
        test_file_path = "dataset/test_cl.txt"
        test_situation_file_path = "dataset/testSituation_cl.txt"

    strategy_record_file_path = os.path.join(args.generation_dir, "strategy_record.json")
    generate_file_path = os.path.join(args.generation_dir, "hyp_strategy.json")
    reference_file_path = os.path.join(args.generation_dir, "ref_strategy.json")
    summary_file_path = os.path.join(args.generation_dir, "summary.txt")
    strategy_logits_file = os.path.join(args.generation_dir, "strategy_logits.txt")
    gen_json_path = os.path.join(args.generation_dir, "gen.json")

    with open(strategy_logits_file, "w", encoding="utf-8") as fp:
        for item in strategy_logit_str:
            fp.write(item + '\n')
    with open(strategy_record_file_path, "w", encoding="utf-8") as fp:
        json.dump(strategy_record, fp, indent=2, ensure_ascii=False)
    with open(generate_file_path, "w", encoding="utf-8") as fp:
        json.dump(refs, fp, indent=2, ensure_ascii=False)
    with open(reference_file_path, "w", encoding="utf-8") as fp:
        json.dump(gts, fp, indent=2, ensure_ascii=False)
    with open(gen_json_path, "w", encoding="utf-8") as fp:
        json.dump(gen_results, fp, indent=2, ensure_ascii=False)

    summary(test_file_path, generate_file_path, reference_file_path, summary_file_path, all_top_k_blocks, all_top_k_blocks_st, chat_texts, test_situation_file_path)

    print("write result to:", summary_file_path)
    print("gen.json (for grid search) saved to:", gen_json_path)
    print("Generate finished~")
    metric = Metric(toker=tokenizer, hyp_path=generate_file_path, ref_path=reference_file_path)
    result, result_list = metric.close()
    print(result)
    print("=" * 100)


# ============================================================================
# Main
# ============================================================================

def main(args):
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir)
        )

    if not args.no_cuda:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        device = torch.device("cpu")
        args.device = device
        args.n_gpu = 0

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    set_seed(args)
    additional_special_tokens = ["[Question]", "[Reflection of feelings]", "[Information]",
                                 "[Restatement or Paraphrasing]", "[Others]", "[Self-disclosure]",
                                 "[Affirmation and Reassurance]", "[Providing Suggestions]"]
    comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]",
                                       "[oEffect]", "[oReact]"]

    config = BlenderbotSmallConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_tokens(comet_additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

    model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    model.resize_token_embeddings(len(tokenizer))

    # s3: Apply LoRA
    model = setup_lora(model, args)
    model.to(args.device)

    logger.info("Training/evaluation parameters: lr=%s, lora_lr=%s, ss_p=%s, b_cond=%s",
                args.learning_rate, args.lora_lr, args.ss_p_sample, args.b_condition)

    # Load data
    with open(args.data_path + "/" + args.train_comet_file, "r", encoding="utf-8") as f:
        comet_trn = f.read().split("\n")
    with open(args.data_path + "/" + args.situation_train_comet_file, "r", encoding="utf-8") as f:
        st_comet_trn = f.read().split("\n")
    with open(args.data_path + "/" + args.train_file_name, "r", encoding="utf-8") as f:
        df_trn = f.read().split("\n")
    with open(args.data_path + "/" + args.eval_comet_file, "r", encoding="utf-8") as f:
        comet_val = f.read().split("\n")
    with open(args.data_path + "/" + args.situation_eval_comet_file, "r", encoding="utf-8") as f:
        st_comet_val = f.read().split("\n")
    with open(args.data_path + "/" + args.eval_file_name, "r", encoding="utf-8") as f:
        df_val = f.read().split("\n")
    with open(args.data_path + "/" + args.test_comet_file, "r", encoding="utf-8") as f:
        comet_test = f.read().split("\n")
    with open(args.data_path + "/" + args.situation_test_comet_file, "r", encoding="utf-8") as f:
        st_comet_test = f.read().split("\n")
    with open(args.data_path + "/" + args.test_file_name, "r", encoding="utf-8") as f:
        df_test = f.read().split("\n")

    # B condition: strategy=False removes strategy tokens
    use_strategy = not args.b_condition

    args.eval_dataset = load_and_cache_examples(args, tokenizer, df_val, comet_val, st_comet_val, evaluate=True, strategy=use_strategy, test=False)
    args.test_dataset = load_and_cache_examples(args, tokenizer, df_test, comet_test, st_comet_test, evaluate=True, strategy=use_strategy, test=True)

    if args.do_train:
        os.makedirs(args.output_dir, exist_ok=True)
        args.train_dataset = load_and_cache_examples(args, tokenizer, df_trn, comet_trn, st_comet_trn, evaluate=False, strategy=use_strategy)
        global_step, tr_loss = train(args, args.train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Reload best model for test evaluation
        model = load_model_with_lora(args.output_dir, tokenizer, args, args.device)
        test_results = evaluate(args, model, tokenizer, args.test_dataset, "of test set")

    generate(args)


if __name__ == "__main__":
    args = Args()
    args = parse_args_override(args)
    main(args)
