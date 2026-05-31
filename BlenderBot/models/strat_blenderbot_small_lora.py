# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lora_expert_layers import LoRAExpertContext, wrap_modules_with_lora
from models.model_utils import BaseModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.blenderbot_small import BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration


class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self._lora_initialized = False
        self._decoder_lora_context = LoRAExpertContext()
        self._lora_num_experts = 8
        self._lora_decoder_shared = False
        self._lora_use_encoder_shared = True
        self._lora_base_trainable = True
        self._lora_expert_select_by = "accum"

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
        self._lora_expert_select_by = str(runtime_config.get("lora_expert_select_by", "accum")).lower()
        if self._lora_expert_select_by not in {"accum", "full"}:
            raise ValueError('lora_expert_select_by must be one of {"accum", "full"}')

        target_proj = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
        is_target = lambda n, m: isinstance(m, nn.Linear) and n.split(".")[-1] in target_proj

        decoder_wrapped = wrap_modules_with_lora(
            root=self.model.decoder,
            predicate=is_target,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            num_experts=None if self._lora_decoder_shared else self._lora_num_experts,
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

    def _set_expert_ids(self, strat_id: torch.Tensor):
        self._decoder_lora_context.expert_ids = strat_id.long()
        self._decoder_lora_context.fixed_expert_id = None

    def _set_fixed_expert_id(self, expert_id: int):
        self._decoder_lora_context.expert_ids = None
        self._decoder_lora_context.fixed_expert_id = int(expert_id)

    def _clear_expert_context(self):
        self._decoder_lora_context.expert_ids = None
        self._decoder_lora_context.fixed_expert_id = None

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
        **kwargs
    ):
        assert self.toker is not None
        if not self._lora_initialized:
            raise RuntimeError("LoRA runtime config is not initialized")

        strat_id = kwargs.get("strat_id", None)
        assert (self.training or validation) == (labels is not None)

        decoder_input_ids, labels = self._strip_strategy_token(
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        self._assert_no_strategy_token(decoder_input_ids=decoder_input_ids, labels=labels)

        if self.training or validation:
            if not self._lora_decoder_shared and strat_id is None:
                raise RuntimeError("strat_id is required for expert routing during train/validation")
            if not self._lora_decoder_shared:
                self._set_expert_ids(strat_id)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        try:
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
            if validation:
                lm_logits = lm_logits[..., : self.toker.vocab_size].contiguous()
            else:
                lm_logits = lm_logits.contiguous()

            masked_lm_loss = None
            ppl_value = None
            if labels is not None:
                loss = F.cross_entropy(
                    lm_logits.reshape(-1, lm_logits.size(-1)),
                    labels.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                )
                loss = loss.reshape(labels.size(0), labels.size(1))
                label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
                masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
                ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

            if not self.training and not validation:
                if not return_dict:
                    output = (lm_logits,) + outputs[1:]
                    return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

                return Seq2SeqLMOutput(
                    loss=masked_lm_loss,
                    logits=lm_logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                )

            if self.training:
                return {"all": masked_lm_loss, "ppl": ppl_value}

            return loss, label_size
        finally:
            if self.training or validation:
                self._decoder_lora_context.expert_ids = None

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        return_dict=None,
        **kwargs
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

        max_length = kwargs["max_length"] + decoder_input_ids.size(1)
        eos_token_id = kwargs["eos_token_id"]
        pad_token_id = kwargs["pad_token_id"]

        strat_id = kwargs.get("strat_id", None)
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
                    decoder_input_ids=decoder_input_ids,
                    **generation_kwargs,
                )
                responses = generations[:, decoder_input_ids.size(1):]
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
                generation_kwargs["max_length"] = max_length
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
            # Iteratively pick minima for consistency with argmin-based expert choice.
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
