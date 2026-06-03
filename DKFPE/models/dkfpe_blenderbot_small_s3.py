# coding=utf-8
"""
DKFPE + s3 Method Integration

Base: dkfpe_blenderbot_small.py (DKFPE, COLING 2025)
Added: LoRA Expert Routing, Gradient-Isolated Probe, Scheduled Sampling

Architecture (unchanged from DKFPE):
1. BlenderBot encoder → H_C (Eq. 1)
2. Lightweight encoders → H_P, H_com (Eq. 2-4)
3. Knowledge Refiner + Cross-Attention + Fusion → H_final (Eq. 5-13)
4. BlenderBot decoder with H_final + LoRA Expert Routing (Eq. 14-15)

s3 additions:
- LoRA Expert Routing on decoder (8 experts) + shared LoRA on encoder
- Gradient-isolated probe: Linear(512, 8) on raw encoder output (H_C, before fusion)
- Scheduled sampling: probe predictions replace GT strategy during training
- generate() uses probe for strategy prediction → LoRA expert routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel
from models.lora_expert_layers import LoRAExpertContext, disable_expert_lora, wrap_modules_with_lora
from models.fusion_encoder import (
    LightweightEncoder,
    KnowledgeRefiner,
    BidirectionalCrossAttention,
    WeightedFusion,
)
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
from transformers.models.blenderbot_small import (
    BlenderbotSmallConfig,
    BlenderbotSmallForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers import GenerationConfig
from .PARAMS import SAMPLE, TEMPERATURE

_VALID_GEN_PARAMS = set(GenerationConfig().to_dict().keys())


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )
    if 0.0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )
    return logits


class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        d_model = config.d_model  # 512 for BlenderBot-small

        # --- DKFPE components (unchanged) ---
        self.persona_encoder = LightweightEncoder(d_model=d_model, nhead=8, num_layers=2)
        self.comet_encoders = nn.ModuleDict({
            'xIntent': LightweightEncoder(d_model=d_model, nhead=8, num_layers=2),
            'xWant': LightweightEncoder(d_model=d_model, nhead=8, num_layers=2),
            'xNeed': LightweightEncoder(d_model=d_model, nhead=8, num_layers=2),
            'xEffect': LightweightEncoder(d_model=d_model, nhead=8, num_layers=2),
        })
        self.comet_combined_encoder = LightweightEncoder(d_model=d_model, nhead=8, num_layers=2)
        self.knowledge_refiner = KnowledgeRefiner(d_model=d_model, nhead=8, num_layers=2)
        self.persona_cross_attn = BidirectionalCrossAttention(d_model=d_model)
        self.comet_cross_attn = BidirectionalCrossAttention(d_model=d_model)
        self.fusion = WeightedFusion(num_components=5)

        # --- s3 components ---
        self._lora_initialized = False
        self._decoder_lora_context = LoRAExpertContext()
        self._lora_num_experts = 8
        self._lora_total_experts = self._lora_num_experts + 1  # +1 for shared slot (unused under control_no_lora)

        self._joint_probe_head = None
        self._joint_probe_source = "encoder"  # encoder mean-pool (recommended)
        self._scheduled_sampler = None

        # Initialize custom modules properly
        self._init_custom_weights()

    def _init_custom_weights(self):
        """Re-initialize custom (non-pretrained) modules with Xavier uniform."""
        custom_modules = [
            self.persona_encoder, self.comet_encoders,
            self.comet_combined_encoder, self.knowledge_refiner,
            self.persona_cross_attn, self.comet_cross_attn,
            self.fusion,
        ]
        for module in custom_modules:
            for name, p in module.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                elif 'bias' in name:
                    nn.init.zeros_(p)
                elif 'weight' in name and ('norm' in name or 'LayerNorm' in name or 'layer_norm' in name):
                    nn.init.ones_(p)

    def apply_runtime_config(self, runtime_config):
        """Initialize LoRA expert routing and probe head from config JSON."""
        if self._lora_initialized:
            return
        if self.toker is None:
            raise RuntimeError("Tokenizer must be tied before LoRA initialization")

        self._lora_num_experts = int(runtime_config.get("lora_num_experts", 8))
        lora_rank = int(runtime_config.get("lora_rank", 8))
        lora_alpha = int(runtime_config.get("lora_alpha", 16))
        lora_dropout = float(runtime_config.get("lora_dropout", 0.05))
        lora_base_trainable = bool(runtime_config.get("lora_base_trainable", True))
        lora_use_encoder_shared = bool(runtime_config.get("lora_use_encoder_shared", True))
        self._lora_total_experts = self._lora_num_experts + 1

        target_proj = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
        is_target = lambda n, m: isinstance(m, nn.Linear) and n.split(".")[-1] in target_proj

        # Decoder: expert LoRA
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

        # Encoder: shared LoRA (no expert routing)
        if lora_use_encoder_shared:
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

        # Joint probe head
        if bool(runtime_config.get("joint_probe", False)):
            hidden_dim = self.config.d_model
            self._joint_probe_head = nn.Linear(hidden_dim, self._lora_num_experts)
            self._joint_probe_source = str(runtime_config.get("joint_probe_source", "encoder")).lower()
            if self._joint_probe_source not in {"decoder", "encoder"}:
                raise ValueError('joint_probe_source must be "decoder" or "encoder"')

        # Trainability
        if lora_base_trainable:
            for p in self.parameters():
                p.requires_grad = True
        else:
            for p in self.parameters():
                p.requires_grad = False
            for name, p in self.named_parameters():
                if ".lora_a" in name or ".lora_b" in name:
                    p.requires_grad = True
            if self._joint_probe_head is not None:
                for p in self._joint_probe_head.parameters():
                    p.requires_grad = True

        # B condition: strategy info only via LoRA routing, not in sequence
        self._no_strat_in_seq = bool(runtime_config.get("no_strat_in_seq", False))

        self._lora_initialized = True

    def set_scheduled_sampler(self, sampler):
        self._scheduled_sampler = sampler

    # --- Expert routing helpers ---
    def _set_expert_ids(self, strat_id: torch.Tensor):
        self._decoder_lora_context.expert_ids = strat_id.long()
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = False

    def _disable_expert_lora(self):
        self._decoder_lora_context.expert_ids = None
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = True

    def _clear_expert_context(self):
        self._decoder_lora_context.expert_ids = None
        self._decoder_lora_context.fixed_expert_id = None
        self._decoder_lora_context.disable_lora = False

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

    # --- Probe ---
    def _compute_probe_logits(self, encoder_hidden, attention_mask, decoder_input_ids=None):
        """Compute probe logits from encoder hidden states (gradient-blocked).

        Args:
            encoder_hidden: raw encoder output H_C (batch, seq, d_model) — before fusion
            attention_mask: (batch, seq)
            decoder_input_ids: (batch, seq) — only needed for decoder probe source
        """
        if self._joint_probe_head is None:
            raise RuntimeError("No probe head available")

        if self._joint_probe_source == "encoder":
            mask = attention_mask.unsqueeze(-1).float()
            hidden = (encoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            # Decoder BOS position-0
            with disable_expert_lora(self):
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids[:, :1],
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )
            hidden = decoder_outputs.hidden_states[-1][:, 0, :]

        return self._joint_probe_head(hidden)

    def predict_strategy(self, logits, encoded_info):
        """Predict strategy from probe logits (inference)."""
        assert not self.training
        assert logits.dim() == 2  # (batch, 8)
        strat_id = encoded_info.get('strat_id', None)

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
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': F.softmax(logits, dim=-1),
        })

    def _predict_strategy_from_decoder_logits(self, lm_logits, encoded_info):
        """Original DKFPE strategy prediction from decoder logits (fallback)."""
        assert not self.training
        assert not self._no_strat_in_seq, (
            "B condition (no_strat_in_seq) cannot use decoder-logits strategy prediction. "
            "Use joint_probe instead."
        )
        strat_id = encoded_info.get('strat_id', None)
        logits = lm_logits[:, 0, -8:]

        if strat_id is not None:
            pred = strat_id
        else:
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)

        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]

        encoded_info.update({
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': F.softmax(logits, dim=-1),
        })

    # --- DKFPE encode/fuse helpers (unchanged from base) ---
    def _get_shared_embeddings(self):
        return self.model.encoder.embed_tokens

    def _encode_auxiliary(self, input_ids, attention_mask, encoder):
        embeddings = self._get_shared_embeddings()
        hidden = embeddings(input_ids)
        hidden = encoder(hidden, attention_mask)
        return hidden

    def _encode_context(self, input_ids, attention_mask, return_dict=True):
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        return encoder_outputs

    def _build_H_com(self, comet_intent_ids, comet_intent_mask,
                     comet_want_ids, comet_want_mask,
                     comet_need_ids, comet_need_mask,
                     comet_effect_ids, comet_effect_mask):
        relation_data = {
            'xIntent': (comet_intent_ids, comet_intent_mask),
            'xWant': (comet_want_ids, comet_want_mask),
            'xNeed': (comet_need_ids, comet_need_mask),
            'xEffect': (comet_effect_ids, comet_effect_mask),
        }
        all_hidden = []
        all_masks = []
        for relation, (ids, mask) in relation_data.items():
            if ids is not None and ids.size(1) > 0:
                hidden = self._encode_auxiliary(ids, mask, self.comet_encoders[relation])
                all_hidden.append(hidden)
                all_masks.append(mask)
        if not all_hidden:
            batch_size = comet_intent_ids.size(0) if comet_intent_ids is not None else 1
            d = self.config.d_model
            H_com = torch.zeros(batch_size, 1, d, device=self._get_shared_embeddings().weight.device)
            com_mask = torch.zeros(batch_size, 1, device=self._get_shared_embeddings().weight.device)
            return H_com, com_mask
        E_com_all = torch.cat(all_hidden, dim=1)
        com_mask = torch.cat(all_masks, dim=1)
        H_com = self.comet_combined_encoder(E_com_all, com_mask)
        return H_com, com_mask

    def _fuse(self, H_C, H_P, H_com, ctx_mask, persona_mask, com_mask):
        d = H_C.size(-1)
        l_c = H_C.size(1)
        E_r_com = self.knowledge_refiner(H_com, H_C, com_mask, ctx_mask)
        H_tilde_P, H_tilde_C_p = self.persona_cross_attn(H_C, H_P, ctx_mask, persona_mask)
        H_tilde_com, H_tilde_C_com = self.comet_cross_attn(H_C, E_r_com, ctx_mask, com_mask)
        if persona_mask is not None:
            p_mask = persona_mask.unsqueeze(-1).float()
            H_tilde_C_p_pooled = (H_tilde_C_p * p_mask).sum(dim=1, keepdim=True) / p_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        else:
            H_tilde_C_p_pooled = H_tilde_C_p.mean(dim=1, keepdim=True)
        H_tilde_C_p_pooled = H_tilde_C_p_pooled.expand(-1, l_c, -1)
        if com_mask is not None:
            c_mask = com_mask.unsqueeze(-1).float()
            H_tilde_C_com_pooled = (H_tilde_C_com * c_mask).sum(dim=1, keepdim=True) / c_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        else:
            H_tilde_C_com_pooled = H_tilde_C_com.mean(dim=1, keepdim=True)
        H_tilde_C_com_pooled = H_tilde_C_com_pooled.expand(-1, l_c, -1)
        components = [H_C, H_tilde_P, H_tilde_C_p_pooled, H_tilde_com, H_tilde_C_com_pooled]
        H_final = self.fusion(components)
        return H_final

    def _extract_auxiliary_kwargs(self, kwargs):
        """Extract DKFPE-specific auxiliary inputs from kwargs."""
        return {
            'persona_ids': kwargs.get('persona_ids', None),
            'persona_mask': kwargs.get('persona_mask', None),
            'comet_intent_ids': kwargs.get('comet_intent_ids', None),
            'comet_intent_mask': kwargs.get('comet_intent_mask', None),
            'comet_want_ids': kwargs.get('comet_want_ids', None),
            'comet_want_mask': kwargs.get('comet_want_mask', None),
            'comet_need_ids': kwargs.get('comet_need_ids', None),
            'comet_need_mask': kwargs.get('comet_need_mask', None),
            'comet_effect_ids': kwargs.get('comet_effect_ids', None),
            'comet_effect_mask': kwargs.get('comet_effect_mask', None),
        }

    def _encode_and_fuse(self, input_ids, attention_mask, aux, return_dict=True):
        """Run full DKFPE encode + fuse pipeline.

        Returns:
            H_C: raw encoder output (batch, l_c, d) — for probe
            H_final: fused representation (batch, l_c, d) — for decoder
            fused_encoder_outputs: BaseModelOutput wrapping H_final
        """
        encoder_outputs = self._encode_context(input_ids, attention_mask, return_dict=return_dict)
        H_C = encoder_outputs[0]

        persona_ids = aux['persona_ids']
        persona_mask = aux['persona_mask']
        if persona_ids is not None and persona_ids.size(1) > 0:
            H_P = self._encode_auxiliary(persona_ids, persona_mask, self.persona_encoder)
        else:
            H_P = torch.zeros(H_C.size(0), 1, H_C.size(-1), device=H_C.device)
            persona_mask = torch.zeros(H_C.size(0), 1, device=H_C.device)

        H_com, com_mask = self._build_H_com(
            aux['comet_intent_ids'], aux['comet_intent_mask'],
            aux['comet_want_ids'], aux['comet_want_mask'],
            aux['comet_need_ids'], aux['comet_need_mask'],
            aux['comet_effect_ids'], aux['comet_effect_mask'],
        )

        H_final = self._fuse(H_C, H_P, H_com, attention_mask, persona_mask, com_mask)

        fused_encoder_outputs = BaseModelOutput(
            last_hidden_state=H_final,
            hidden_states=None,
            attentions=None,
        )
        return H_C, H_final, fused_encoder_outputs

    # --- Forward (training/validation) ---
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
        if self._lora_initialized:
            assert True  # LoRA is ready
        else:
            raise RuntimeError("LoRA runtime config is not initialized. Call apply_runtime_config() first.")

        encoded_info = kwargs
        assert (self.training or validation) == (labels is not None)
        if validation:
            # A condition: labels[0] is strategy token → mask it
            # B condition: labels[0] is real response token → don't mask
            if not self._no_strat_in_seq:
                labels[:, 0] = -100

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- Non-training pass (for HuggingFace generate internal calls) ---
        if not self.training and not validation:
            outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
            from transformers.modeling_outputs import Seq2SeqLMOutput
            return Seq2SeqLMOutput(
                loss=None,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.hidden_states,
                decoder_attentions=outputs.attentions,
                encoder_last_hidden_state=encoder_outputs[0],
            )

        # --- Training / validation pass ---
        strat_id = kwargs.get('strat_id', None)
        if strat_id is not None:
            strat_id = self._validate_strategy_ids(strat_id, decoder_input_ids.size(0))

        aux = self._extract_auxiliary_kwargs(kwargs)

        if encoder_outputs is None:
            H_C, H_final, fused_encoder_outputs = self._encode_and_fuse(
                input_ids, attention_mask, aux, return_dict=return_dict
            )
            encoder_outputs = fused_encoder_outputs
        else:
            H_C = None  # encoder_outputs already provided (shouldn't happen during training)

        # --- s3: Joint probe computation + scheduled sampling ---
        probe_loss = None
        routing_strat_id = strat_id  # default: GT strategy

        if self.training and self._joint_probe_head is not None and strat_id is not None and H_C is not None:
            with torch.no_grad():
                if self._joint_probe_source == "encoder":
                    mask = attention_mask.unsqueeze(-1).float()
                    probe_hidden = (H_C * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                else:
                    rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                    with disable_expert_lora(self):
                        probe_dec = self.model.decoder(
                            input_ids=decoder_input_ids[:, :1],
                            encoder_hidden_states=H_C,
                            encoder_attention_mask=attention_mask,
                            return_dict=True,
                            output_hidden_states=True,
                        )
                    if rng_state is not None:
                        torch.cuda.set_rng_state(rng_state)
                    probe_hidden = probe_dec.hidden_states[-1][:, 0, :]

            # Explicit detach: only probe_head gradients flow, backbone is isolated
            probe_hidden = probe_hidden.detach()
            probe_logits = self._joint_probe_head(probe_hidden)
            probe_loss = F.cross_entropy(probe_logits, strat_id)

            # Scheduled sampling
            if self._scheduled_sampler is not None:
                routing_strat_id = self._scheduled_sampler.sample_strategy_tokens(
                    gt_strategy_ids=strat_id,
                    strategy_logits=probe_logits.detach(),
                )

        # --- Expert routing for decoder ---
        if routing_strat_id is not None:
            self._set_expert_ids(routing_strat_id)
        else:
            self._disable_expert_lora()

        try:
            outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

            if validation:
                lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()

            masked_lm_loss = None
            if labels is not None:
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
            else:
                # validation
                return loss, label_size
        finally:
            self._clear_expert_context()

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

        encoded_info = kwargs.copy()
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        aux = self._extract_auxiliary_kwargs(kwargs)

        # Encode + fuse
        H_C, H_final, fused_encoder_outputs = self._encode_and_fuse(
            input_ids, attention_mask, aux, return_dict=return_dict
        )

        # Strategy prediction
        strat_id = kwargs.get('strat_id', None)
        if strat_id is None:
            if self._joint_probe_head is not None:
                # s3: probe-based strategy prediction using raw encoder output
                probe_logits = self._compute_probe_logits(
                    encoder_hidden=H_C,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                self.predict_strategy(probe_logits, encoded_info)
            else:
                # Fallback: original DKFPE decoder-logits strategy prediction
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=H_final,
                    encoder_attention_mask=attention_mask,
                    return_dict=return_dict,
                )
                lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
                self._predict_strategy_from_decoder_logits(lm_logits, encoded_info)

            strat_id = encoded_info['pred_strat_id'].long()
            strat_id = self._validate_strategy_ids(strat_id, decoder_input_ids.size(0))

        if not isinstance(strat_id, torch.Tensor):
            strat_id = torch.tensor(strat_id, device=decoder_input_ids.device)
        strat_id = strat_id.long()
        if strat_id.dim() == 0:
            strat_id = strat_id.unsqueeze(0)

        # Append strategy token to decoder input (A condition only)
        # B condition: decoder starts with [bos] only; strategy via LoRA routing
        if not self._no_strat_in_seq:
            decoder_input_ids = torch.cat(
                [decoder_input_ids, strat_id[..., None] + len(self.toker) - 8],
                dim=-1
            )

        # Set expert routing
        self._set_expert_ids(strat_id)

        try:
            gen_kwargs = {k: v for k, v in kwargs.items() if k in _VALID_GEN_PARAMS}
            assert 'max_length' in gen_kwargs
            gen_kwargs['max_length'] = gen_kwargs['max_length'] + decoder_input_ids.size(1)
            gen_kwargs['use_cache'] = True

            if len(self.toker) > self.toker.vocab_size:
                bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
                gen_kwargs['bad_words_ids'] = bad_words_ids

            generations = BlenderbotSmallForConditionalGeneration.generate(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=fused_encoder_outputs,
                **gen_kwargs
            )
            return encoded_info, generations[:, decoder_input_ids.size(1):]
        finally:
            self._clear_expert_context()
