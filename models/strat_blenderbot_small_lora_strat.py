# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lora_expert_layers import LoRAExpertContext, disable_expert_lora, wrap_modules_with_lora
from models.model_utils import BaseModel
from transformers.generation_utils import top_k_top_p_filtering
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.blenderbot_small import BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration

from .PARAMS import SAMPLE, TEMPERATURE


class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self._lora_initialized = False
        self._decoder_lora_context = LoRAExpertContext()
        self._lora_num_experts = 8
        self._lora_total_experts = self._lora_num_experts + 1
        self._lora_decoder_shared = False
        self._lora_use_encoder_shared = True
        self._lora_base_trainable = True
        self._lora_expert_select_by = "full"
        self._lora_strategy_mode = "control_no_lora"
        self._lora_strategy_fixed_expert_id = self._lora_num_experts
        self._lora_strategy_block_base_grad = False
        self._lora_strategy_pred_use_lora = False
        self._lora_use_strategy_token_prefix = False
        self._lora_disable_strategy_loss = False
        self._probe_head = None

    def apply_runtime_config(self, runtime_config):
        if self._lora_initialized:
            return
        if self.toker is None:
            raise RuntimeError("Tokenizer must be tied before LoRA initialization")

        self._lora_num_experts = int(runtime_config.get("lora_num_experts", 8))
        lora_rank = int(runtime_config.get("lora_rank", 8))
        lora_alpha = int(runtime_config.get("lora_alpha", 16))
        lora_dropout = float(runtime_config.get("lora_dropout", 0.05))
        self._lora_decoder_shared = bool(runtime_config.get("lora_decoder_shared", False))
        self._lora_use_encoder_shared = bool(runtime_config.get("lora_use_encoder_shared", True))
        self._lora_base_trainable = bool(runtime_config.get("lora_base_trainable", True))
        requested_expert_select_by = str(runtime_config.get("lora_expert_select_by", "full")).lower()
        if requested_expert_select_by not in {"accum", "full"}:
            raise ValueError('lora_expert_select_by must be one of {"accum", "full"}')
        self._lora_expert_select_by = "full"

        self._lora_strategy_mode = str(runtime_config.get("lora_strategy_mode", "control_no_lora")).lower()
        if self._lora_strategy_mode not in {"control_no_lora", "with_lora_fixed"}:
            raise ValueError('lora_strategy_mode must be one of {"control_no_lora", "with_lora_fixed"}')
        self._lora_strategy_fixed_expert_id = self._lora_num_experts
        self._lora_strategy_block_base_grad = bool(runtime_config.get("lora_strategy_block_base_grad", False))
        self._lora_strategy_pred_use_lora = bool(
            runtime_config.get(
                "lora_strategy_pred_use_lora",
                self._lora_strategy_mode == "with_lora_fixed",
            )
        )
        self._lora_use_strategy_token_prefix = bool(
            runtime_config.get(
                "lora_use_strategy_token_prefix",
                self._lora_strategy_mode == "with_lora_fixed",
            )
        )
        self._lora_disable_strategy_loss = bool(runtime_config.get("lora_disable_strategy_loss", False))
        self._lora_total_experts = self._lora_num_experts + 1

        target_proj = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
        is_target = lambda n, m: isinstance(m, nn.Linear) and n.split(".")[-1] in target_proj

        decoder_wrapped = wrap_modules_with_lora(
            root=self.model.decoder,
            predicate=is_target,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            num_experts=None if self._lora_decoder_shared else self._lora_total_experts,
            context=None if self._lora_decoder_shared else self._decoder_lora_context,
        )
        if not decoder_wrapped:
            raise RuntimeError("No decoder modules were wrapped by LoRA in BlenderBot")

        if self._lora_use_encoder_shared:
            encoder_wrapped = wrap_modules_with_lora(
                root=self.model.encoder,
                predicate=is_target,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                num_experts=None,
                context=None,
            )
            if not encoder_wrapped:
                raise RuntimeError("No encoder modules were wrapped by shared LoRA in BlenderBot")

        self._set_trainability(self._lora_base_trainable)
        self._lora_initialized = True

    def _set_trainability(self, base_trainable: bool):
        if base_trainable:
            for p in self.parameters():
                p.requires_grad = True
            return

        for p in self.parameters():
            p.requires_grad = False
        for name, p in self.named_parameters():
            if ".lora_a" in name or ".lora_b" in name:
                p.requires_grad = True

    def _strategy_token_start(self):
        return len(self.toker) - 8

    def _begin_strategy_grad_scope(self):
        kept_param_ids = set()
        input_embeddings = self.get_input_embeddings()
        if input_embeddings is not None and hasattr(input_embeddings, "weight"):
            kept_param_ids.add(id(input_embeddings.weight))
        lm_head_weight = getattr(getattr(self, "lm_head", None), "weight", None)
        if lm_head_weight is not None:
            kept_param_ids.add(id(lm_head_weight))

        restore = []
        for name, p in self.named_parameters():
            keep = (".lora_a" in name) or (".lora_b" in name) or (id(p) in kept_param_ids)
            if keep:
                continue
            if p.requires_grad:
                restore.append((p, True))
                p.requires_grad = False
        return restore

    def _end_strategy_grad_scope(self, restore):
        for p, original_requires_grad in restore:
            p.requires_grad = original_requires_grad

    def _set_expert_ids(self, strat_id: torch.Tensor):
        self._decoder_lora_context.expert_ids = strat_id.long()
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = False

    def _set_fixed_expert_id(self, expert_id: int):
        self._decoder_lora_context.expert_ids = None
        self._decoder_lora_context.fixed_expert_id = int(expert_id)
        self._decoder_lora_context.disable_lora = False

    def _disable_expert_lora(self):
        self._decoder_lora_context.expert_ids = None
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = True

    def _clear_expert_context(self):
        self._decoder_lora_context.expert_ids = None
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = False

    def load_probe(self, path):
        state = torch.load(path, map_location="cpu")
        hidden_dim = state["weight"].size(1)
        num_classes = state["weight"].size(0)
        self._probe_head = nn.Linear(hidden_dim, num_classes)
        self._probe_head.load_state_dict(state)
        self._probe_head.to(next(self.parameters()).device)
        self._probe_head.eval()

    def _compute_probe_logits(self, encoder_outputs, attention_mask, decoder_input_ids):
        with disable_expert_lora(self):
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids[:, :1],
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        hidden = decoder_outputs.hidden_states[-1][:, 0, :]
        return self._probe_head(hidden)

    def _strip_strategy_token(self, decoder_input_ids=None, labels=None):
        if decoder_input_ids is not None and decoder_input_ids.size(1) > 1:
            decoder_input_ids = torch.cat([decoder_input_ids[:, :1], decoder_input_ids[:, 2:]], dim=1)
        if labels is not None and labels.size(1) > 0:
            labels = labels[:, 1:]
        return decoder_input_ids, labels

    def _assert_no_strategy_token(self, decoder_input_ids=None, labels=None):
        strategy_start = self._strategy_token_start()
        strategy_end = len(self.toker)
        if decoder_input_ids is not None:
            strategy_mask = (decoder_input_ids >= strategy_start) & (decoder_input_ids < strategy_end)
            assert not strategy_mask.any(), "strategy token must be excluded from decoder_input_ids"
        if labels is not None:
            strategy_mask = (labels >= strategy_start) & (labels < strategy_end)
            assert not strategy_mask.any(), "strategy token must be excluded from labels"

    def _compute_generated_ppl(
        self,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_outputs,
        generations: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        if generations.size(1) == 0:
            return torch.full((generations.size(0),), float("inf"), device=generations.device)

        if generations.size(1) > 1:
            score_decoder_input_ids = torch.cat([decoder_input_ids, generations[:, :-1]], dim=-1)
        else:
            score_decoder_input_ids = decoder_input_ids

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=score_decoder_input_ids,
            encoder_outputs=encoder_outputs,
            use_cache=False,
            return_dict=True,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        lm_logits = lm_logits.contiguous()

        labels = generations.clone()
        valid_mask = labels.ne(pad_token_id) & labels.ne(eos_token_id)
        labels = labels.masked_fill(~valid_mask, -100)

        loss = F.cross_entropy(
            lm_logits.reshape(-1, lm_logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        )
        loss = loss.reshape(labels.size(0), labels.size(1))

        token_count = valid_mask.sum(dim=-1)
        ppl = torch.full((labels.size(0),), float("inf"), device=labels.device, dtype=loss.dtype)
        valid_rows = token_count > 0
        if valid_rows.any():
            ppl[valid_rows] = torch.exp(loss[valid_rows].sum(dim=-1) / token_count[valid_rows].float())
        return ppl

    def _compute_generated_ppl_from_scores(
        self,
        scores,
        generations: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        if len(scores) == 0 or generations.size(1) == 0:
            return torch.full((generations.size(0),), float("inf"), device=generations.device)

        steps = min(len(scores), generations.size(1))
        token_logps = []
        for t in range(steps):
            step_log_probs = F.log_softmax(scores[t], dim=-1)
            step_tokens = generations[:, t].unsqueeze(-1)
            step_logp = step_log_probs.gather(-1, step_tokens).squeeze(-1)
            token_logps.append(step_logp)
        token_logps = torch.stack(token_logps, dim=1)

        valid_tokens = generations[:, :steps].ne(pad_token_id)
        if eos_token_id is not None:
            valid_tokens = valid_tokens & generations[:, :steps].ne(eos_token_id)

        token_count = valid_tokens.sum(dim=-1)
        ppl = torch.full((generations.size(0),), float("inf"), device=generations.device, dtype=token_logps.dtype)
        valid_rows = token_count > 0
        if valid_rows.any():
            nll_sum = -(token_logps[valid_rows] * valid_tokens[valid_rows].float()).sum(dim=-1)
            ppl[valid_rows] = torch.exp(nll_sum / token_count[valid_rows].float())
        return ppl

    def _validate_strategy_ids(self, strat_id: torch.Tensor, batch_size: int):
        model_device = next(self.parameters()).device
        if not isinstance(strat_id, torch.Tensor):
            strat_id = torch.tensor(strat_id, device=model_device)
        strat_id = strat_id.long()
        if strat_id.dim() == 0:
            strat_id = strat_id.unsqueeze(0)
        if strat_id.dim() != 1 or strat_id.size(0) != batch_size:
            raise RuntimeError("strategy id must be [batch] and match batch size")
        if (strat_id < 0).any() or (strat_id >= self._lora_num_experts).any():
            raise RuntimeError(f"strategy id must be in [0, {self._lora_num_experts - 1}]")
        return strat_id

    def _compute_strategy_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_outputs,
        return_dict: bool,
    ) -> torch.Tensor:
        if decoder_input_ids.size(1) < 1:
            raise RuntimeError("decoder_input_ids must have at least 1 token for strategy prediction")
        if input_ids is None and encoder_outputs is None:
            raise RuntimeError("either input_ids or encoder_outputs must be provided for strategy prediction")

        strategy_input_ids = decoder_input_ids[:, :1]

        if not self._lora_decoder_shared:
            if self._lora_strategy_pred_use_lora:
                self._set_fixed_expert_id(self._lora_strategy_fixed_expert_id)
            else:
                self._disable_expert_lora()

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=strategy_input_ids,
                encoder_outputs=encoder_outputs,
                use_cache=False,
                return_dict=return_dict,
            )
        finally:
            if not self._lora_decoder_shared:
                self._clear_expert_context()

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        strategy_start = self._strategy_token_start()
        strategy_end = len(self.toker)
        return lm_logits[:, 0, strategy_start:strategy_end].contiguous()

    def predict_strategy(self, logits: torch.Tensor, encoded_info):
        assert logits.dim() == 2
        strat_id = encoded_info.get("strat_id", None)

        if strat_id is not None:
            if not isinstance(strat_id, torch.Tensor):
                strat_id = torch.tensor(strat_id, device=logits.device)
            strat_id = strat_id.long()
            if strat_id.dim() == 0:
                strat_id = strat_id.unsqueeze(0)
            if strat_id.dim() != 1 or strat_id.size(0) != logits.size(0):
                raise RuntimeError("strat_id must be [batch] and match logits batch size")
            pred = strat_id
        else:
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)

        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1)[1]

        encoded_info.update(
            {
                "pred_strat_id": pred,
                "pred_strat_id_top1": pred_top1,
                "pred_strat_id_top3": pred_top3,
                "pred_strat_id_dist": F.softmax(logits, dim=-1),
            }
        )

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
        if not self._lora_initialized:
            raise RuntimeError("LoRA runtime config is not initialized")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not self.training and not validation:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
            lm_logits = lm_logits.contiguous()

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

        strat_id = kwargs.get("strat_id", None)
        if not self._lora_decoder_shared and strat_id is None:
            raise RuntimeError("strat_id is required for expert routing during train/validation")
        if strat_id is not None:
            strat_id = self._validate_strategy_ids(strat_id, decoder_input_ids.size(0))

        strategy_logits = None
        strategy_token_loss = None
        if not self._lora_disable_strategy_loss:
            strategy_grad_restore = []
            if self.training and self._lora_strategy_block_base_grad:
                strategy_grad_restore = self._begin_strategy_grad_scope()
            try:
                strategy_logits = self._compute_strategy_logits(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    encoder_outputs=encoder_outputs,
                    return_dict=return_dict,
                )
            finally:
                if strategy_grad_restore:
                    self._end_strategy_grad_scope(strategy_grad_restore)
            if strat_id is not None:
                strategy_token_loss = F.cross_entropy(strategy_logits, strat_id, reduction="none")

        use_strategy_token_context = self._lora_use_strategy_token_prefix
        if use_strategy_token_context:
            # Keep strategy token in decoder inputs so response decoding is conditioned on it.
            # Response loss still excludes strategy token by aligning with labels[:, 1:].
            response_decoder_input_ids = decoder_input_ids
            response_labels = labels[:, 1:] if labels is not None else None
        else:
            response_decoder_input_ids, response_labels = self._strip_strategy_token(
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            )
            self._assert_no_strategy_token(decoder_input_ids=response_decoder_input_ids, labels=response_labels)

        if not self._lora_decoder_shared:
            self._set_expert_ids(strat_id)

        try:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=response_decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=False,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
            if validation:
                lm_logits = lm_logits[..., : self.toker.vocab_size].contiguous()
            else:
                lm_logits = lm_logits.contiguous()
            response_lm_logits = lm_logits[:, 1:, :] if use_strategy_token_context else lm_logits

            response_token_loss = F.cross_entropy(
                response_lm_logits.reshape(-1, response_lm_logits.size(-1)),
                response_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            )
            response_token_loss = response_token_loss.reshape(response_labels.size(0), response_labels.size(1))
            response_label_size = torch.sum(response_labels.ne(-100), dim=1).type_as(response_token_loss)
            response_loss = torch.sum(response_token_loss) / torch.sum(response_label_size)

            strategy_valid = labels[:, :1].ne(-100).type_as(response_token_loss).squeeze(1)
            if strategy_token_loss is None:
                strategy_valid = strategy_valid.new_zeros(strategy_valid.size())
                strategy_token_loss = response_token_loss.new_zeros(response_token_loss.size(0))
            else:
                strategy_token_loss = strategy_token_loss * strategy_valid
            if validation:
                strategy_valid = strategy_valid.new_zeros(strategy_valid.size())
                strategy_token_loss = response_token_loss.new_zeros(response_token_loss.size(0))

            total_token_loss = torch.sum(response_token_loss, dim=1) + strategy_token_loss
            total_label_size = response_label_size + strategy_valid
            total_loss = torch.sum(total_token_loss) / torch.sum(total_label_size)
            ppl_value = torch.exp(torch.mean(total_token_loss.float() / total_label_size.float()))

            if self.training:
                res = {
                    "all": total_loss,
                    "ppl": ppl_value,
                    "resp": response_loss,
                }
                if strategy_valid.sum().item() > 0:
                    res["strat"] = torch.sum(strategy_token_loss) / torch.sum(strategy_valid)
                return res

            loss = torch.cat([strategy_token_loss.unsqueeze(1), response_token_loss], dim=1)
            return loss, total_label_size
        finally:
            self._clear_expert_context()

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
        if not self._lora_initialized:
            raise RuntimeError("LoRA runtime config is not initialized")
        if kwargs.get("num_return_sequences", 1) != 1:
            raise ValueError("LoRA expert inference currently supports num_return_sequences=1 only")

        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        self._assert_no_strategy_token(decoder_input_ids=decoder_input_ids, labels=None)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self._lora_decoder_shared:
            generation_kwargs = dict(kwargs)
            generation_kwargs.pop("labels", None)
            generation_kwargs.pop("strat_id", None)
            generation_kwargs.pop("other_res", None)
            generation_kwargs.pop("batch_size", None)
            generation_kwargs["max_length"] = kwargs["max_length"] + decoder_input_ids.size(1)
            generation_kwargs["use_cache"] = True
            if len(self.toker) > self.toker.vocab_size:
                generation_kwargs["bad_words_ids"] = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
            generations = super().generate(
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                **generation_kwargs,
            )
            responses = generations[:, decoder_input_ids.size(1):]
            return encoded_info, responses

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        base_max_length = kwargs["max_length"] + decoder_input_ids.size(1)
        eos_token_id = kwargs["eos_token_id"]
        pad_token_id = kwargs["pad_token_id"]

        strat_id = kwargs.get("strat_id", None)
        if strat_id is None:
            if self._probe_head is not None:
                probe_logits = self._compute_probe_logits(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                self.predict_strategy(probe_logits, encoded_info)
            else:
                strategy_logits = self._compute_strategy_logits(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    encoder_outputs=encoder_outputs,
                    return_dict=return_dict,
                )
                self.predict_strategy(strategy_logits, encoded_info)
            predicted_expert_id = encoded_info["pred_strat_id"].long()
            predicted_expert_id = self._validate_strategy_ids(predicted_expert_id, decoder_input_ids.size(0))
            strat_id = predicted_expert_id
        if strat_id is not None:
            if not isinstance(strat_id, torch.Tensor):
                strat_id = torch.tensor(strat_id, device=decoder_input_ids.device)
            strat_id = strat_id.long()
            if strat_id.dim() == 0:
                strat_id = strat_id.unsqueeze(0)
            if strat_id.dim() != 1 or strat_id.size(0) != decoder_input_ids.size(0):
                raise RuntimeError("strat_id must be [batch] and match decoder_input_ids batch size")
            if (strat_id < 0).any() or (strat_id >= self._lora_num_experts).any():
                raise RuntimeError(f"strat_id must be in [0, {self._lora_num_experts - 1}]")
            if self._lora_use_strategy_token_prefix:
                strategy_token_ids = strat_id + self._strategy_token_start()
                decoder_with_strategy = torch.cat([decoder_input_ids, strategy_token_ids.unsqueeze(-1)], dim=-1)
            else:
                decoder_with_strategy = decoder_input_ids
            max_length = kwargs["max_length"] + decoder_with_strategy.size(1)
            self._set_expert_ids(strat_id)
            try:
                generation_kwargs = dict(kwargs)
                generation_kwargs.pop("labels", None)
                generation_kwargs.pop("strat_id", None)
                generation_kwargs.pop("other_res", None)
                generation_kwargs.pop("batch_size", None)
                generation_kwargs["max_length"] = max_length
                generation_kwargs["use_cache"] = True
                if len(self.toker) > self.toker.vocab_size:
                    generation_kwargs["bad_words_ids"] = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
                generations = super().generate(
                    attention_mask=attention_mask,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_with_strategy,
                    **generation_kwargs,
                )
                responses = generations[:, decoder_with_strategy.size(1):]
                used_expert_id = strat_id
                encoded_info["used_expert_id"] = used_expert_id
                encoded_info["used_expert_id_top1"] = used_expert_id.unsqueeze(-1)
                encoded_info["used_expert_id_full"] = used_expert_id
                encoded_info["used_expert_id_accum"] = used_expert_id
                encoded_info["used_expert_id_changed"] = torch.zeros_like(used_expert_id)
                encoded_info[f"used_expert_id_{self._lora_expert_select_by}"] = used_expert_id
                return encoded_info, responses
            finally:
                self._clear_expert_context()

        all_generations = []
        all_select_ppl = []
        try:
            for expert_id in range(self._lora_num_experts):
                self._set_fixed_expert_id(expert_id)
                generation_kwargs = dict(kwargs)
                generation_kwargs["max_length"] = base_max_length
                generation_kwargs["use_cache"] = True
                generation_kwargs["return_dict_in_generate"] = True
                generation_kwargs["output_scores"] = True
                if len(self.toker) > self.toker.vocab_size:
                    generation_kwargs["bad_words_ids"] = [[i] for i in range(self.toker.vocab_size, len(self.toker))]

                generation_output = super().generate(
                    attention_mask=attention_mask,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    **generation_kwargs,
                )
                generations = generation_output.sequences
                responses = generations[:, decoder_input_ids.size(1):]
                if self._lora_expert_select_by == "full":
                    select_ppl = self._compute_generated_ppl(
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        encoder_outputs=encoder_outputs,
                        generations=responses,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                    )
                else:
                    select_ppl = self._compute_generated_ppl_from_scores(
                        scores=generation_output.scores,
                        generations=responses,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                    )
                all_generations.append(responses)
                all_select_ppl.append(select_ppl)

            ppl_mat = torch.stack(all_select_ppl, dim=0)
            topk = min(3, ppl_mat.size(0))
            ppl_work = ppl_mat.clone()
            expert_batch_idx = torch.arange(ppl_work.size(1), device=ppl_work.device)
            topk_experts = []
            for _ in range(topk):
                next_expert = torch.argmin(ppl_work, dim=0)
                topk_experts.append(next_expert)
                ppl_work[next_expert, expert_batch_idx] = float("inf")
            used_expert_id_top3 = torch.stack(topk_experts, dim=1)
            used_expert_id = used_expert_id_top3[:, 0]
            used_expert_id_top1 = used_expert_id.unsqueeze(-1)
            stack_pad_id = eos_token_id if eos_token_id is not None else pad_token_id
            max_gen_len = max(g.size(1) for g in all_generations)
            padded_generations = []
            for g in all_generations:
                if g.size(1) < max_gen_len:
                    pad = g.new_full((g.size(0), max_gen_len - g.size(1)), stack_pad_id)
                    g = torch.cat([g, pad], dim=1)
                padded_generations.append(g)
            generation_mat = torch.stack(padded_generations, dim=0)
            batch_idx = torch.arange(generation_mat.size(1), device=generation_mat.device)
            best_generations = generation_mat[used_expert_id, batch_idx]
            encoded_info["used_expert_id"] = used_expert_id
            encoded_info["used_expert_id_top1"] = used_expert_id_top1
            encoded_info["used_expert_id_top3"] = used_expert_id_top3
            encoded_info["used_expert_id_full"] = used_expert_id
            encoded_info["used_expert_id_accum"] = used_expert_id
            encoded_info["used_expert_id_changed"] = torch.zeros_like(used_expert_id)
            encoded_info[f"used_expert_id_{self._lora_expert_select_by}"] = used_expert_id
            return encoded_info, best_generations
        finally:
            self._clear_expert_context()
