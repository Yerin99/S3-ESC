# coding=utf-8
"""
GLHG + s3 Inputter: Multi-Source Encoder + Strategy ID for probe/LoRA training.

Extends glhg.py by also computing strat_id (0-7 range) from dialog strategy labels.
Strategy tokens are added to expanded vocab (via CONFIG expanded_vocab).
decoder_input_ids and labels do NOT include strategy tokens (GLHG style).
"""

import json
import tqdm
import torch
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
import numpy as np
import random
from functools import partial
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from inputters.inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader
from .PARAMS import GOLDEN_TRUTH

# COMET 모델 지연 로드 (glhg.py와 공유 가능)
tokenizer_gpt = None
model_gpt = None

COMET_BART_PATH = '<COMET_BART_PATH>'

PROBLEM_TYPES = [
    'academic pressure',
    'Alcohol Abuse',
    'Appearance Anxiety',
    'breakup with partner',
    'conflict with parents',
    'Issues with Children',
    'Issues with Parents',
    'job crisis',
    'ongoing depression',
    'problems with friends',
    'Procrastination',
    'School Bullying',
    'Sleep Problems',
]
PROBLEM_TO_ID = {p: i for i, p in enumerate(PROBLEM_TYPES)}


def _load_comet_model():
    global tokenizer_gpt, model_gpt
    if tokenizer_gpt is None:
        print("Loading COMET-BART-AI2 model...")
        tokenizer_gpt = BartTokenizer.from_pretrained(COMET_BART_PATH)
        model_gpt = BartForConditionalGeneration.from_pretrained(COMET_BART_PATH).cuda()
        model_gpt.eval()
        print("COMET-BART-AI2 loaded successfully")


def _generate_comet_inference(text: str, relation_type: str = 'xIntent') -> str:
    _load_comet_model()
    if not text.strip():
        return "none."
    inp_str = f"{text} {relation_type}"
    inp_ids = tokenizer_gpt.encode(inp_str, return_tensors='pt').cuda()
    with torch.no_grad():
        output = model_gpt.generate(
            inp_ids,
            max_length=20,
            num_beams=3,
            num_return_sequences=1,
            pad_token_id=tokenizer_gpt.pad_token_id,
            early_stopping=True,
        )
    intention = tokenizer_gpt.decode(output[0], skip_special_tokens=True).strip()
    intention = intention.split('.')[0].strip()
    return intention + "." if intention else "none."


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
        decoder_input_ids,
        labels,
        strat_id=None,
        situation_ids=None,
        intention_ids=None,
        problem_type_id=None,
        last_utterance_start=None,
    ):
        self.input_ids = input_ids
        self.input_length = len(input_ids)

        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels

        # s3 addition: strategy ID (0-7)
        self.strat_id = strat_id if strat_id is not None else 0

        # Multi-source info
        self.situation_ids = situation_ids if situation_ids is not None else []
        self.situation_length = len(self.situation_ids)

        self.intention_ids = intention_ids if intention_ids is not None else []
        self.intention_length = len(self.intention_ids)

        self.problem_type_id = problem_type_id if problem_type_id is not None else 0
        self.last_utterance_start = last_utterance_start if last_utterance_start is not None else 0

        self.input_len = self.input_length + self.decoder_input_length


def featurize(
    bos, eos,
    context, max_input_length,
    response, max_decoder_input_length,
    strat_id=None,
    situation=None, max_situation_length=64,
    intention=None, max_intention_length=32,
    problem_type_id=None,
    last_utterance_start=None,
):
    # Context
    context = [c + [eos] for c in context]
    input_ids = sum(context, [])[:-1]
    input_ids = input_ids[-max_input_length:]

    # Adjust last_utterance_start for truncation
    if last_utterance_start is not None:
        original_len = len(sum(context, [])) - 1
        truncated = original_len - len(input_ids)
        last_utterance_start = max(0, last_utterance_start - truncated)

    # Response (NO strategy token prepended — GLHG style)
    labels = (response + [eos])[:max_decoder_input_length]
    decoder_input_ids = [bos] + labels[:-1]

    # Situation
    situation_ids = []
    if situation is not None:
        situation_ids = situation[:max_situation_length]

    # Intention
    intention_ids = []
    if intention is not None:
        intention_ids = intention[:max_intention_length]

    return InputFeatures(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
        strat_id=strat_id,
        situation_ids=situation_ids,
        intention_ids=intention_ids,
        problem_type_id=problem_type_id,
        last_utterance_start=last_utterance_start,
    )


def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):
    _load_comet_model()
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))

    dialog = data['dialog']
    situation = data.get('situation', '')
    problem_type = data.get('problem_type', '')

    problem_type_id = PROBLEM_TO_ID.get(problem_type, 0)
    situation_ids = process(situation) if situation else []

    inputs = []
    context = []
    context_lengths = []

    for i in range(len(dialog)):
        text = _norm(dialog[i]['text'])
        text_ids = process(text)

        if dialog[i]['speaker'] == 'sys':
            # Strategy token (for strat_id computation)
            strategy_name = dialog[i].get('strategy', 'Others')
            strat_token_ids = process('[' + strategy_name + ']')
            assert len(strat_token_ids) == 1, f"Strategy token not found in vocab: [{strategy_name}]"
            strat_token_id = strat_token_ids[0]

        if i > 0 and dialog[i]['speaker'] == 'sys':
            # Find last help-seeker utterance for intention
            last_usr_idx = i - 1
            while last_usr_idx >= 0 and dialog[last_usr_idx]['speaker'] != 'usr':
                last_usr_idx -= 1

            if last_usr_idx >= 0:
                last_usr_text = _norm(dialog[last_usr_idx]['text'])
                intention_text = _generate_comet_inference(last_usr_text, 'xIntent')
                intention_ids = process(intention_text)
            else:
                intention_ids = []

            # Compute last_utterance_start
            if len(context_lengths) > 0:
                last_utterance_start = sum(context_lengths[:-1]) + len(context_lengths) - 1
            else:
                last_utterance_start = 0

            res = {
                'context': context.copy(),
                'response': text_ids,
                'strat_token_id': strat_token_id,  # raw vocab ID
                'situation_ids': situation_ids,
                'intention_ids': intention_ids,
                'problem_type_id': problem_type_id,
                'last_utterance_start': last_utterance_start,
            }
            inputs.append(res)

        context = context + [text_ids]
        context_lengths.append(len(text_ids))

    return inputs


def convert_inputs_to_features(inputs, toker, **kwargs):
    if len(inputs) == 0:
        return []

    max_input_length = kwargs.get('max_input_length', 128)
    max_decoder_input_length = kwargs.get('max_decoder_input_length', 50)
    max_situation_length = kwargs.get('max_situation_length', 64)
    max_intention_length = kwargs.get('max_intention_length', 32)

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
    for ipt in inputs:
        # Convert raw strat_token_id → 0-7 range
        strat_token_id = ipt['strat_token_id']
        strat_id = strat_token_id - len(toker) + 8  # 0-7

        feat = featurize(
            bos=bos,
            eos=eos,
            context=ipt['context'],
            max_input_length=max_input_length,
            response=ipt['response'],
            max_decoder_input_length=max_decoder_input_length,
            strat_id=strat_id,
            situation=ipt.get('situation_ids'),
            max_situation_length=max_situation_length,
            intention=ipt.get('intention_ids'),
            max_intention_length=max_intention_length,
            problem_type_id=ipt.get('problem_type_id'),
            last_utterance_start=ipt.get('last_utterance_start'),
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

        # Context
        input_ids = pad_sequence(
            [torch.tensor(f.input_ids, dtype=torch.long) for f in features],
            batch_first=True, padding_value=pad
        )
        attention_mask = pad_sequence(
            [torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
            batch_first=True, padding_value=0.
        )

        # Situation
        situation_ids = pad_sequence(
            [torch.tensor(f.situation_ids if f.situation_ids else [pad], dtype=torch.long) for f in features],
            batch_first=True, padding_value=pad
        )
        situation_attention_mask = pad_sequence(
            [torch.tensor([1.] * max(f.situation_length, 1), dtype=torch.float) for f in features],
            batch_first=True, padding_value=0.
        )

        # Intention
        intention_ids = pad_sequence(
            [torch.tensor(f.intention_ids if f.intention_ids else [pad], dtype=torch.long) for f in features],
            batch_first=True, padding_value=pad
        )
        intention_attention_mask = pad_sequence(
            [torch.tensor([1.] * max(f.intention_length, 1), dtype=torch.float) for f in features],
            batch_first=True, padding_value=0.
        )

        # Problem type
        problem_type_ids = torch.tensor(
            [f.problem_type_id for f in features], dtype=torch.long
        )

        # Last utterance starts
        last_utterance_starts = torch.tensor(
            [f.last_utterance_start for f in features], dtype=torch.long
        )

        # Strategy IDs (0-7)
        strat_id = torch.tensor(
            [f.strat_id for f in features], dtype=torch.long
        )

        # Decoder inputs
        if not infer:
            decoder_input_ids = pad_sequence(
                [torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                batch_first=True, padding_value=pad
            )
            labels = pad_sequence(
                [torch.tensor(f.labels, dtype=torch.long) for f in features],
                batch_first=True, padding_value=-100
            )
        else:
            decoder_input_ids = torch.tensor(
                [[f.decoder_input_ids[0]] for f in features], dtype=torch.long
            )
            labels = None

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'situation_ids': situation_ids,
            'situation_attention_mask': situation_attention_mask,
            'intention_ids': intention_ids,
            'intention_attention_mask': intention_attention_mask,
            'problem_type_ids': problem_type_ids,
            'last_utterance_starts': last_utterance_starts,
            'strat_id': strat_id,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        }

        return res


class DynamicBatchingLoader(object):
    def __init__(self, corpus_file, toker, batch_size, **kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.num_examples = self.get_len(corpus_file)
        self.kwargs = kwargs

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
            for line in tqdm.tqdm(reader, total=len(reader), desc="validating"):
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

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog'][1:]))), reader))


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
    for sample_id, line in tqdm.tqdm(enumerate(reader), total=len(reader), desc="inferring"):
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
