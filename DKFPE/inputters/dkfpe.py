# coding=utf-8
"""
DKFPE Inputter

Extends the base strat inputter with additional fields:
- persona_ids/mask: persona text tokens
- comet_{intent,want,need,effect}_ids/mask: filtered COMET text tokens
"""

import json
import tqdm
import torch
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
import random
from functools import partial
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from inputters.inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader
from .PARAMS import GOLDEN_TRUTH


RELATIONS = ['xIntent', 'xWant', 'xNeed', 'xEffect']


class Inputter(object):
    def __init__(self):
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features

        self.train_sampler = BucketSampler
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader

        self.valid_dataloader = DynamicBatchingLoader

        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch


class InputFeatures(object):
    def __init__(
        self,
        input_ids,
        decoder_input_ids, labels,
        persona_ids,
        comet_intent_ids,
        comet_want_ids,
        comet_need_ids,
        comet_effect_ids,
        strat_id=None,
    ):
        self.input_ids = input_ids
        self.input_length = len(input_ids)

        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels

        self.persona_ids = persona_ids
        self.persona_length = len(persona_ids)

        self.comet_intent_ids = comet_intent_ids
        self.comet_intent_length = len(comet_intent_ids)
        self.comet_want_ids = comet_want_ids
        self.comet_want_length = len(comet_want_ids)
        self.comet_need_ids = comet_need_ids
        self.comet_need_length = len(comet_need_ids)
        self.comet_effect_ids = comet_effect_ids
        self.comet_effect_length = len(comet_effect_ids)

        self.strat_id = strat_id  # raw token ID for B condition; None for A condition

        self.input_len = self.input_length + self.decoder_input_length


def featurize(
    bos, eos,
    context, max_input_length,
    response, max_decoder_input_length, strat_id,
    persona_ids, max_persona_length,
    comet_intent_ids, comet_want_ids, comet_need_ids, comet_effect_ids,
    max_comet_length,
    no_strat_in_seq=False,
):
    # Context (same as strat.py)
    context = [c + [eos] for c in context]
    input_ids = sum(context, [])[:-1]
    input_ids = input_ids[-max_input_length:]

    # Decoder
    if no_strat_in_seq:
        # B condition: no strategy token in labels
        labels = (response + [eos])[:max_decoder_input_length + 1]
    else:
        # A condition (default): strategy token prepended
        labels = ([strat_id] + response + [eos])[:max_decoder_input_length + 1]
    decoder_input_ids = [bos] + labels[:-1]
    assert len(decoder_input_ids) == len(labels)

    # Persona
    persona_ids = persona_ids[:max_persona_length]

    # COMET
    comet_intent_ids = comet_intent_ids[:max_comet_length]
    comet_want_ids = comet_want_ids[:max_comet_length]
    comet_need_ids = comet_need_ids[:max_comet_length]
    comet_effect_ids = comet_effect_ids[:max_comet_length]

    return InputFeatures(
        input_ids,
        decoder_input_ids, labels,
        persona_ids,
        comet_intent_ids,
        comet_want_ids,
        comet_need_ids,
        comet_effect_ids,
        strat_id=strat_id if no_strat_in_seq else None,
    )


def _tokenize_comet(process, comet_data):
    """Tokenize a single COMET filtered dict into token ID lists."""
    comet_tokens = {}
    for relation in RELATIONS:
        texts = comet_data.get(relation, [])
        if texts:
            combined = ' '.join(texts)
            comet_tokens[relation] = process(combined)
        else:
            comet_tokens[relation] = []
    return comet_tokens


def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):
    """Convert a single data entry to model inputs.

    For 8:1:1 mode: each line is one sample (context + target)
    For 70:15:15 mode: each line is a full conversation
    """
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))

    last_turn_only = kwargs.get('last_turn_only', False)
    no_strat_in_seq = kwargs.get('no_strat_in_seq', False)
    # last_turn_only=True  → only last sys turn (window-level 8:1:1)
    # last_turn_only=False → all sys turns i>0 (70:15:15, default)

    dialog = data['dialog']
    persona_text = data.get('persona', '')
    comet_by_target = data.get('comet_filtered_by_target', None)
    comet_fallback = data.get('comet_filtered', {})

    # Tokenize persona
    if persona_text:
        persona_text = persona_text.replace('<persona>', '').replace('<input>', '').strip()
        persona_tokens = process(persona_text)
    else:
        persona_tokens = []

    # Pre-tokenize fallback COMET (used when comet_filtered_by_target is absent)
    fallback_comet_tokens = _tokenize_comet(process, comet_fallback)

    # Find last sys turn index
    last_sys_idx = None
    for i in range(len(dialog) - 1, -1, -1):
        if i > 0 and dialog[i]['speaker'] == 'sys':
            last_sys_idx = i
            break

    inputs = []
    context = []
    sample_idx = 0

    for i in range(len(dialog)):
        text = _norm(dialog[i]['text'])
        text = process(text)

        if dialog[i]['speaker'] == 'sys':
            strat_id = process('[' + dialog[i]['strategy'] + ']')
            assert len(strat_id) == 1
            strat_id = strat_id[0]

        if i > 0 and dialog[i]['speaker'] == 'sys':
            is_target = (not last_turn_only) or (i == last_sys_idx)
            if is_target:
                # Per-sample COMET tokenization
                if comet_by_target is not None and sample_idx < len(comet_by_target):
                    comet_tokens = _tokenize_comet(process, comet_by_target[sample_idx])
                else:
                    comet_tokens = fallback_comet_tokens

                res = {
                    'context': context.copy(),
                    'response': text,
                    'strat_id': strat_id,
                    'persona_ids': persona_tokens,
                    'comet_intent_ids': comet_tokens.get('xIntent', []),
                    'comet_want_ids': comet_tokens.get('xWant', []),
                    'comet_need_ids': comet_tokens.get('xNeed', []),
                    'comet_effect_ids': comet_tokens.get('xEffect', []),
                }
                inputs.append(res)
            sample_idx += 1  # always increment for comet_by_target indexing

        if dialog[i]['speaker'] == 'sys':
            # B condition: no strategy token in encoder context
            if not no_strat_in_seq:
                text = [strat_id] + text

        context = context + [text]

    return inputs


def convert_inputs_to_features(inputs, toker, **kwargs):
    if len(inputs) == 0:
        return []

    max_input_length = kwargs.get('max_input_length', 512)
    max_decoder_input_length = kwargs.get('max_decoder_input_length', 50)
    max_persona_length = kwargs.get('max_persona_length', 128)
    max_comet_length = kwargs.get('max_comet_length', 128)
    no_strat_in_seq = kwargs.get('no_strat_in_seq', False)

    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id

    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(
            bos, eos,
            ipt['context'], max_input_length,
            ipt['response'], max_decoder_input_length, ipt['strat_id'],
            ipt['persona_ids'], max_persona_length,
            ipt['comet_intent_ids'], ipt['comet_want_ids'],
            ipt['comet_need_ids'], ipt['comet_effect_ids'],
            max_comet_length,
            no_strat_in_seq=no_strat_in_seq,
        )
        features.append(feat)
    return features


class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker: PreTrainedTokenizer, infer=False):
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
        bos = toker.bos_token_id
        if bos is None:
            bos = toker.cls_token_id
        eos = toker.eos_token_id
        if eos is None:
            eos = toker.sep_token_id

        # Context
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features],
                                  batch_first=True, padding_value=pad)
        attention_mask = pad_sequence([torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
                                      batch_first=True, padding_value=0.)

        # Decoder
        if not infer:
            decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                                              batch_first=True, padding_value=pad)
            labels = pad_sequence([torch.tensor(f.labels, dtype=torch.long) for f in features],
                                  batch_first=True, padding_value=-100)
        else:
            decoder_input_ids = torch.tensor([[f.decoder_input_ids[0]] for f in features], dtype=torch.long)
            labels = None

        # B condition: strat_id stored as explicit field (labels[0] is response token)
        # A condition: strat_id derived from labels[0] (strategy token)
        if hasattr(features[0], 'strat_id') and features[0].strat_id is not None:
            strat_id = torch.tensor([f.strat_id for f in features], dtype=torch.long) - len(toker) + 8
        else:
            strat_id = torch.tensor([f.labels[0] for f in features], dtype=torch.long) - len(toker) + 8

        # Persona
        persona_ids = pad_sequence(
            [torch.tensor(f.persona_ids, dtype=torch.long) if f.persona_ids else torch.tensor([pad], dtype=torch.long)
             for f in features],
            batch_first=True, padding_value=pad)
        persona_mask = pad_sequence(
            [torch.tensor([1.] * max(f.persona_length, 1), dtype=torch.float) for f in features],
            batch_first=True, padding_value=0.)
        # Fix: if persona was empty, set mask to 0
        for i, f in enumerate(features):
            if f.persona_length == 0:
                persona_mask[i, :] = 0.

        # COMET relations
        def _pad_comet(attr_name, length_attr):
            ids_list = []
            masks = []
            for f in features:
                ids = getattr(f, attr_name)
                length = getattr(f, length_attr)
                if ids:
                    ids_list.append(torch.tensor(ids, dtype=torch.long))
                    masks.append(torch.tensor([1.] * length, dtype=torch.float))
                else:
                    ids_list.append(torch.tensor([pad], dtype=torch.long))
                    masks.append(torch.tensor([0.], dtype=torch.float))

            padded_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad)
            padded_masks = pad_sequence(masks, batch_first=True, padding_value=0.)
            return padded_ids, padded_masks

        comet_intent_ids, comet_intent_mask = _pad_comet('comet_intent_ids', 'comet_intent_length')
        comet_want_ids, comet_want_mask = _pad_comet('comet_want_ids', 'comet_want_length')
        comet_need_ids, comet_need_mask = _pad_comet('comet_need_ids', 'comet_need_length')
        comet_effect_ids, comet_effect_mask = _pad_comet('comet_effect_ids', 'comet_effect_length')

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            'strat_id': strat_id,
            'persona_ids': persona_ids,
            'persona_mask': persona_mask,
            'comet_intent_ids': comet_intent_ids,
            'comet_intent_mask': comet_intent_mask,
            'comet_want_ids': comet_want_ids,
            'comet_want_mask': comet_want_mask,
            'comet_need_ids': comet_need_ids,
            'comet_need_mask': comet_need_mask,
            'comet_effect_ids': comet_effect_ids,
            'comet_effect_mask': comet_effect_mask,
        }

        return res


class DynamicBatchingLoader(object):
    def __init__(self, corpus_file, toker, batch_size, **kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.kwargs = kwargs
        self.num_examples = self.get_len(corpus_file, kwargs.get('last_turn_only', False))

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()

            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating"):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.toker, **self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []

            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch

        except StopIteration:
            pass

    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)

    def get_len(self, corpus, last_turn_only=False):
        with open(corpus, 'r', encoding="utf-8") as file:
            if last_turn_only:
                return sum(1 for _ in file)
            else:
                count = 0
                for line in file:
                    data = json.loads(line)
                    dialog = data['dialog']
                    count += sum(1 for i in range(1, len(dialog))
                                 if dialog[i]['speaker'] == 'sys')
                return count


def prepare_infer_batch(features, toker, interact=None):
    res = FeatureDataset.collate(features, toker, True)

    res['batch_size'] = res['input_ids'].size(0)

    other_res = res['other_res'] = {}
    other_res['acc_map'] = {
        'cls_strat_id': 'pred_strat_id',
    }

    if interact is None and GOLDEN_TRUTH:
        other_res['cls_strat_id'] = res.get('strat_id')
    else:
        other_res['cls_strat_id'] = res.pop('strat_id')

    return res


def get_infer_batch(infer_input_file, toker, **kwargs):
    assert 'infer_batch_size' in kwargs
    infer_batch_size = kwargs.get('infer_batch_size')

    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()

    features = []
    sample_ids = []
    posts = []
    references = []
    for sample_id, line in tqdm.tqdm(enumerate(reader), total=len(reader), desc=f"inferring"):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, toker, **kwargs)
        tmp_features = convert_inputs_to_features(inputs, toker, **kwargs)
        for i in range(len(inputs)):
            features.append(tmp_features[i])
            ipt = inputs[i]
            posts.append(toker.decode(ipt['context'][-1]))
            references.append(toker.decode(ipt['response']))
            sample_ids.append(sample_id)

            if len(sample_ids) == infer_batch_size:
                yield prepare_infer_batch(features, toker), posts, references, sample_ids
                features = []
                sample_ids = []
                posts = []
                references = []

    if len(sample_ids) > 0:
        yield prepare_infer_batch(features, toker), posts, references, sample_ids
