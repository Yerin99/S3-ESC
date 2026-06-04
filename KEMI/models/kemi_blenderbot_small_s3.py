# coding=utf-8
"""
KEMI + s3: Knowledge-Enhanced Emotional Support with LoRA Expert Routing,
Gradient-Isolated Parallel Probe, and Scheduled Sampling.

Extends KEMI by adding:
1. LoRA expert routing on decoder (8 experts, one per strategy)
2. Shared LoRA on encoder (optional)
3. Gradient-isolated parallel probe head (Linear, encoder mean-pool)
4. Scheduled sampling during training

Key design decisions:
- decoder_input_ids do NOT contain strategy tokens (s3 B-condition)
- Probe uses encoder mean-pool (simple encoder, no graph fusion like GLHG)
- strat_id comes from inputter (0-7 range)
- Knowledge acquisition pipeline (SBERT retrieval + COMET) preserved from KEMI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from models.model_utils import BaseModel
from models.kemi_blenderbot_small import Model as KEMIModel
from models.lora_expert_layers import (
    LoRAExpertContext,
    disable_expert_lora,
    wrap_modules_with_lora,
)
from transformers.models.blenderbot_small import (
    BlenderbotSmallConfig,
    BlenderbotSmallForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.generation_utils import top_k_top_p_filtering

from .PARAMS import SAMPLE, TEMPERATURE


class KEMIModelS3(KEMIModel):
    """
    KEMI + s3 model.

    Inherits KEMI's encoder-decoder architecture.
    Adds LoRA expert routing, gradient-isolated probe, and scheduled sampling.
    """

    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self._lora_initialized = False
        self._decoder_lora_context = LoRAExpertContext()
        self._lora_num_experts = 8
        self._lora_total_experts = 9  # 8 strategies + 1 shared slot (index 8)
        self._lora_use_encoder_shared = True
        self._lora_base_trainable = True

        # s3 probe and scheduled sampling
        self._joint_probe_head = None
        self._joint_probe_source = "encoder"
        self._scheduled_sampler = None

    def apply_runtime_config(self, runtime_config: dict):
        """Initialize LoRA from CONFIG JSON fields. Called once after tie_tokenizer."""
        if self._lora_initialized:
            return
        if self.toker is None:
            raise RuntimeError("Tokenizer must be tied before LoRA initialization")

        self._lora_num_experts = int(runtime_config.get("lora_num_experts", 8))
        lora_rank = int(runtime_config.get("lora_rank", 8))
        lora_alpha = int(runtime_config.get("lora_alpha", 16))
        lora_dropout = float(runtime_config.get("lora_dropout", 0.05))
        self._lora_use_encoder_shared = bool(runtime_config.get("lora_use_encoder_shared", True))
        self._lora_base_trainable = bool(runtime_config.get("lora_base_trainable", True))
        self._lora_total_experts = self._lora_num_experts + 1

        target_proj = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
        is_target = lambda n, m: isinstance(m, nn.Linear) and n.split(".")[-1] in target_proj

        # Decoder: expert LoRA (num_experts = 9, indices 0-7 used)
        decoder_wrapped = wrap_modules_with_lora(
            root=self.model.decoder,
            predicate=is_target,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            num_experts=self._lora_total_experts,
            context=self._decoder_lora_context,
        )
        if not decoder_wrapped:
            raise RuntimeError("No decoder modules were wrapped by LoRA")

        # Encoder: shared LoRA (no experts)
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
                raise RuntimeError("No encoder modules were wrapped by shared LoRA")

        # Joint probe head: Linear(d_model, 8)
        if bool(runtime_config.get("joint_probe", False)):
            hidden_dim = self.config.d_model  # 512 for BlenderBot-small
            self._joint_probe_head = nn.Linear(hidden_dim, self._lora_num_experts)
            self._joint_probe_source = runtime_config.get("joint_probe_source", "encoder")

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
        if self._joint_probe_head is not None:
            for p in self._joint_probe_head.parameters():
                p.requires_grad = True

    def set_scheduled_sampler(self, sampler):
        self._scheduled_sampler = sampler

    def _set_expert_ids(self, strat_id: torch.Tensor):
        self._decoder_lora_context.expert_ids = strat_id.long()
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = False

    def _clear_expert_context(self):
        self._decoder_lora_context.expert_ids = None
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = False

    def _disable_expert_lora(self):
        self._decoder_lora_context.expert_ids = None
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = True

    def _validate_strategy_ids(self, strat_id: torch.Tensor, batch_size: int) -> torch.Tensor:
        model_device = next(self.parameters()).device
        if not isinstance(strat_id, torch.Tensor):
            strat_id = torch.tensor(strat_id, device=model_device)
        strat_id = strat_id.long()
        if strat_id.dim() == 0:
            strat_id = strat_id.unsqueeze(0)
        if strat_id.dim() != 1 or strat_id.size(0) != batch_size:
            raise RuntimeError("strat_id must be [batch] and match batch size")
        if (strat_id < 0).any() or (strat_id >= self._lora_num_experts).any():
            raise RuntimeError(f"strat_id must be in [0, {self._lora_num_experts - 1}]")
        return strat_id

    def _compute_probe_logits_from_encoder(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute probe logits from encoder mean-pool (gradient-isolated)."""
        probe_head = self._joint_probe_head
        if probe_head is None:
            raise RuntimeError("No probe head available")
        mask = attention_mask.unsqueeze(-1).float()
        hidden = (encoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return probe_head(hidden)

    def predict_strategy(self, logits: torch.Tensor, encoded_info: dict):
        """Standard strategy prediction — argmax or sampling."""
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

    def load_probe(self, path, probe_source=None):
        """Load pre-trained probe head from checkpoint."""
        state = torch.load(path, map_location="cpu")
        hidden_dim = state["weight"].size(1)
        num_classes = state["weight"].size(0)
        self._joint_probe_head = nn.Linear(hidden_dim, num_classes)
        self._joint_probe_head.load_state_dict(state)
        self._joint_probe_head.to(next(self.parameters()).device)
        self._joint_probe_head.eval()

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
            raise RuntimeError("LoRA runtime config is not initialized. Check building_utils.py")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Inference mode (called by super().generate() step-by-step)
        if not self.training and not validation:
            if encoder_outputs is not None:
                encoder_hidden = encoder_outputs[0] if not isinstance(encoder_outputs, BaseModelOutput) else encoder_outputs.last_hidden_state
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
                return Seq2SeqLMOutput(
                    loss=None,
                    logits=lm_logits,
                    past_key_values=decoder_outputs.past_key_values,
                    decoder_hidden_states=decoder_outputs.hidden_states,
                    decoder_attentions=decoder_outputs.attentions,
                    cross_attentions=decoder_outputs.cross_attentions,
                    encoder_last_hidden_state=encoder_hidden,
                    encoder_hidden_states=None,
                    encoder_attentions=None,
                )

            # Fresh encode
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            encoder_hidden = encoder_outputs.last_hidden_state
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
            return Seq2SeqLMOutput(
                loss=None,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_hidden,
                encoder_hidden_states=None,
                encoder_attentions=None,
            )

        # Training / Validation
        strat_id = kwargs.get("strat_id", None)
        if strat_id is None:
            raise RuntimeError("strat_id is required for LoRA expert routing during train/validation")
        strat_id = self._validate_strategy_ids(strat_id, decoder_input_ids.size(0))

        # Encoder
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden = encoder_outputs.last_hidden_state

        # s3: Joint probe computation + scheduled sampling
        probe_loss = None
        routing_strat_id = strat_id

        if self.training and self._joint_probe_head is not None:
            # Compute probe hidden with gradient blocked from backbone
            with torch.no_grad():
                mask = attention_mask.unsqueeze(-1).float()
                probe_hidden = (encoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

            # probe_hidden is detached; only probe_head gradients flow
            probe_logits = self._joint_probe_head(probe_hidden)
            probe_loss = F.cross_entropy(probe_logits, strat_id)

            # Scheduled sampling: replace strat_id for routing
            if self._scheduled_sampler is not None:
                routing_strat_id = self._scheduled_sampler.sample_strategy_tokens(
                    gt_strategy_ids=strat_id,
                    strategy_logits=probe_logits.detach(),
                )

        # Set LoRA expert IDs for decoder
        self._set_expert_ids(routing_strat_id)

        try:
            # Decoder forward
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=False,
                return_dict=True,
            )
            lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias

            if validation:
                # s3: NO labels[:, 0] = -100 (labels don't have strategy token)
                lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()

            # LM loss
            loss_lm_raw = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
                reduction='none',
            )
            loss_lm_raw = loss_lm_raw.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss_lm_raw)
            loss_lm = torch.sum(loss_lm_raw) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(
                torch.sum(loss_lm_raw, dim=1).float() / label_size.float().clamp(min=1)
            ))

        finally:
            self._clear_expert_context()

        if self.training:
            res = {
                'all': loss_lm,
                'ppl': ppl_value,
            }
            if probe_loss is not None:
                res['probe'] = probe_loss
            return res

        # Validation
        return loss_lm_raw, label_size

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

        encoded_info = kwargs.copy()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).float()

        # Encoder
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden = encoder_outputs.last_hidden_state

        # Strategy prediction via probe
        strat_id = encoded_info.get("strat_id", None)
        if strat_id is None:
            if self._joint_probe_head is not None:
                if self._joint_probe_source == "decoder":
                    # Decoder BOS hidden (disable LoRA since no expert routing set yet)
                    bos_ids = decoder_input_ids[:, :1]
                    with disable_expert_lora(self):
                        dec_out = self.model.decoder(
                            input_ids=bos_ids,
                            encoder_hidden_states=encoder_hidden,
                            encoder_attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True,
                        )
                    probe_hidden = dec_out.last_hidden_state[:, 0, :]
                else:
                    # Encoder mean-pool probe (default)
                    mask = attention_mask.unsqueeze(-1).float()
                    probe_hidden = (encoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                probe_logits = self._joint_probe_head(probe_hidden)
                self.predict_strategy(probe_logits, encoded_info)
                strat_id = encoded_info["pred_strat_id"].long()
                strat_id = self._validate_strategy_ids(strat_id, decoder_input_ids.size(0))
            else:
                raise RuntimeError("No probe head available for strategy prediction")
        else:
            if not isinstance(strat_id, torch.Tensor):
                strat_id = torch.tensor(strat_id, device=decoder_input_ids.device)
            strat_id = strat_id.long()
            if strat_id.dim() == 0:
                strat_id = strat_id.unsqueeze(0)
            strat_id = self._validate_strategy_ids(strat_id, decoder_input_ids.size(0))

        # Set LoRA expert routing
        self._set_expert_ids(strat_id)

        try:
            kwargs_gen = {k: v for k, v in kwargs.items()
                          if k not in ('strat_id', 'labels', 'other_res', 'batch_size')}
            if 'max_length' in kwargs_gen:
                kwargs_gen['max_length'] = kwargs_gen['max_length'] + decoder_input_ids.size(1)
            kwargs_gen['use_cache'] = True

            # Block all expanded vocab tokens from generation
            if len(self.toker) > self.toker.vocab_size:
                kwargs_gen['bad_words_ids'] = [[i] for i in range(self.toker.vocab_size, len(self.toker))]

            generations = BlenderbotSmallForConditionalGeneration.generate(
                self,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                **kwargs_gen,
            )
        finally:
            self._clear_expert_context()

        encoded_info["used_expert_id"] = strat_id
        encoded_info["used_expert_id_top1"] = strat_id.unsqueeze(-1)
        return encoded_info, generations[:, decoder_input_ids.size(1):]

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """Override for generation step-by-step (cached generation)."""
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {
            'input_ids': None,
            'encoder_outputs': encoder_outputs,
            'past_key_values': past,
            'decoder_input_ids': decoder_input_ids,
            'attention_mask': attention_mask,
            'use_cache': True,
        }


# Alias
Model = KEMIModelS3
