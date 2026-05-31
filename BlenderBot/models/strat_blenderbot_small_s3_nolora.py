# coding=utf-8
"""
exp/s3 ablation: -LoRA condition.

Base strat model (no LoRA) with joint probe + scheduled sampling.
Strategy is reflected via decoder prefix token only (no LoRA expert routing).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel
from transformers.generation_utils import top_k_top_p_filtering
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.blenderbot_small import BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration

from .PARAMS import SAMPLE, TEMPERATURE


class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self._num_strategies = 8
        # s3 joint probe + SS (no LoRA)
        self._joint_probe_head = None
        self._joint_probe_source = "encoder"
        self._scheduled_sampler = None

    def apply_runtime_config(self, runtime_config):
        if bool(runtime_config.get("joint_probe", False)):
            hidden_dim = self.config.d_model
            self._joint_probe_head = nn.Linear(hidden_dim, self._num_strategies)
            self._joint_probe_source = str(runtime_config.get("joint_probe_source", "encoder")).lower()
            if self._joint_probe_source not in {"decoder", "encoder"}:
                raise ValueError('joint_probe_source must be "decoder" or "encoder"')

    def set_scheduled_sampler(self, sampler):
        self._scheduled_sampler = sampler

    def _strategy_token_start(self):
        return len(self.toker) - self._num_strategies

    def _compute_probe_logits(self, encoder_outputs, attention_mask, decoder_input_ids):
        if self._joint_probe_head is None:
            raise RuntimeError("No probe head available")

        if self._joint_probe_source == "encoder":
            encoder_hidden = (
                encoder_outputs.last_hidden_state
                if hasattr(encoder_outputs, 'last_hidden_state')
                else encoder_outputs[0]
            )
            mask = attention_mask.unsqueeze(-1).float()
            hidden = (encoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids[:, :1],
                encoder_hidden_states=(
                    encoder_outputs.last_hidden_state
                    if hasattr(encoder_outputs, 'last_hidden_state')
                    else encoder_outputs[0]
                ),
                encoder_attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            hidden = decoder_outputs.hidden_states[-1][:, 0, :]

        return self._joint_probe_head(hidden)

    def predict_strategy(self, logits, encoded_info):
        assert logits.dim() == 2
        strat_id = encoded_info.get("strat_id", None)

        if strat_id is not None:
            if not isinstance(strat_id, torch.Tensor):
                strat_id = torch.tensor(strat_id, device=logits.device)
            strat_id = strat_id.long()
            if strat_id.dim() == 0:
                strat_id = strat_id.unsqueeze(0)
            pred = strat_id
        else:
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)

        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1)[1]

        encoded_info.update({
            "pred_strat_id": pred,
            "pred_strat_id_top1": pred_top1,
            "pred_strat_id_top3": pred_top3,
            "pred_strat_id_dist": F.softmax(logits, dim=-1),
        })

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        validation=False,
        **kwargs,
    ):
        assert self.toker is not None
        assert (self.training or validation) == (labels is not None)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Inference mode
        if not self.training and not validation:
            use_cache = True
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
            return Seq2SeqLMOutput(
                loss=None,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        if validation:
            labels[:, 0] = -100

        strat_id = kwargs.get("strat_id", None)

        # Joint probe + scheduled sampling (training only)
        probe_loss = None
        if self.training and self._joint_probe_head is not None and strat_id is not None:
            if encoder_outputs is None:
                encoder_outputs = self.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

            encoder_hidden = (
                encoder_outputs.last_hidden_state
                if hasattr(encoder_outputs, 'last_hidden_state')
                else encoder_outputs[0]
            )

            with torch.no_grad():
                if self._joint_probe_source == "encoder":
                    mask = attention_mask.unsqueeze(-1).float()
                    probe_hidden = (encoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                else:
                    rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                    probe_dec = self.model.decoder(
                        input_ids=decoder_input_ids[:, :1],
                        encoder_hidden_states=encoder_hidden,
                        encoder_attention_mask=attention_mask,
                        return_dict=True,
                        output_hidden_states=True,
                    )
                    if rng_state is not None:
                        torch.cuda.set_rng_state(rng_state)
                    probe_hidden = probe_dec.hidden_states[-1][:, 0, :]

            probe_logits = self._joint_probe_head(probe_hidden)
            probe_loss = F.cross_entropy(probe_logits, strat_id)

            # Scheduled sampling: replace strategy token in decoder_input_ids
            if self._scheduled_sampler is not None:
                routing_strat_id = self._scheduled_sampler.sample_strategy_tokens(
                    gt_strategy_ids=strat_id,
                    strategy_logits=probe_logits.detach(),
                )
                strategy_tokens = routing_strat_id + self._strategy_token_start()
                decoder_input_ids = decoder_input_ids.clone()
                decoder_input_ids[:, 1] = strategy_tokens

        # Standard forward (strategy prefix token included in decoder_input_ids)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        if validation:
            lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()

        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
        loss = loss.view(labels.size(0), labels.size(1))
        label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
        masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
        ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        if self.training:
            res = {'all': masked_lm_loss, 'ppl': ppl_value}
            if probe_loss is not None:
                res['probe'] = probe_loss
            return res

        # Validation
        return loss, label_size

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        return_dict=None,
        **kwargs,
    ):
        assert not self.training
        assert self.toker is not None

        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        strat_id = kwargs.get("strat_id", None)
        if strat_id is None:
            if self._joint_probe_head is not None:
                probe_logits = self._compute_probe_logits(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                self.predict_strategy(probe_logits, encoded_info)
            else:
                # Fallback: BB strategy logits (like original strat model)
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs[0],
                    encoder_attention_mask=attention_mask,
                    return_dict=return_dict,
                )
                lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
                strategy_logits = lm_logits[:, 0, -self._num_strategies:]
                self.predict_strategy(strategy_logits, encoded_info)
            strat_id = encoded_info['pred_strat_id']

        if not isinstance(strat_id, torch.Tensor):
            strat_id = torch.tensor(strat_id, device=decoder_input_ids.device)
        strat_id = strat_id.long()
        if strat_id.dim() == 0:
            strat_id = strat_id.unsqueeze(0)

        # Append strategy token to decoder_input_ids (prefix-based conditioning)
        strategy_token_ids = strat_id + self._strategy_token_start()
        decoder_input_ids = torch.cat([decoder_input_ids, strategy_token_ids.unsqueeze(-1)], dim=-1)

        generation_kwargs = dict(kwargs)
        generation_kwargs.pop("labels", None)
        generation_kwargs.pop("strat_id", None)
        generation_kwargs.pop("other_res", None)
        generation_kwargs.pop("batch_size", None)
        generation_kwargs["max_length"] = kwargs["max_length"] + decoder_input_ids.size(1)
        generation_kwargs["use_cache"] = True

        if len(self.toker) > self.toker.vocab_size:
            generation_kwargs["bad_words_ids"] = [[i] for i in range(self.toker.vocab_size, len(self.toker))]

        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **generation_kwargs,
        )
        return encoded_info, generations[:, decoder_input_ids.size(1):]
