# coding=utf-8
"""
DKFPE Model (COLING 2025)

Architecture:
1. BlenderBot encoder for dialogue context → H_C (Eq. 1)
2. Lightweight encoders for persona → H_P (Eq. 2) and COMET per-relation → E_com^r (Eq. 3)
3. Concatenate relation encodings → E_com^all, encode → H_com (Eq. 4)
4. Knowledge Refiner: context-guided gating → E_r_com (Eq. 5-7)
5. Bidirectional Cross-Attention: persona↔context, commonsense↔context (Eq. 8-11)
6. Weighted Fusion → H_final (Eq. 12-13)
7. BlenderBot decoder with H_final (Eq. 14-15)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel
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

        # Lightweight encoders for persona and COMET (Eq. 2-3)
        self.persona_encoder = LightweightEncoder(d_model=d_model, nhead=8, num_layers=2)
        self.comet_encoders = nn.ModuleDict({
            'xIntent': LightweightEncoder(d_model=d_model, nhead=8, num_layers=2),
            'xWant': LightweightEncoder(d_model=d_model, nhead=8, num_layers=2),
            'xNeed': LightweightEncoder(d_model=d_model, nhead=8, num_layers=2),
            'xEffect': LightweightEncoder(d_model=d_model, nhead=8, num_layers=2),
        })

        # Combined commonsense encoder (Eq. 4)
        self.comet_combined_encoder = LightweightEncoder(d_model=d_model, nhead=8, num_layers=2)

        # Knowledge Refiner (Eq. 5-7)
        self.knowledge_refiner = KnowledgeRefiner(d_model=d_model, nhead=8, num_layers=2)

        # Bidirectional Cross-Attention (Eq. 8-11)
        self.persona_cross_attn = BidirectionalCrossAttention(d_model=d_model)
        self.comet_cross_attn = BidirectionalCrossAttention(d_model=d_model)

        # Weighted Fusion (Eq. 12-13)
        self.fusion = WeightedFusion(num_components=5)

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

    def _get_shared_embeddings(self):
        """Get the shared embedding layer from BlenderBot encoder."""
        return self.model.encoder.embed_tokens

    def _encode_auxiliary(self, input_ids, attention_mask, encoder):
        """Encode auxiliary text (persona/COMET) using shared embeddings + lightweight encoder.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            encoder: LightweightEncoder instance

        Returns:
            hidden_states: (batch, seq_len, d_model)
        """
        embeddings = self._get_shared_embeddings()
        hidden = embeddings(input_ids)  # (batch, seq_len, d_model)
        hidden = encoder(hidden, attention_mask)
        return hidden

    def _encode_context(self, input_ids, attention_mask, return_dict=True):
        """Encode dialogue context using BlenderBot encoder (Eq. 1).

        Returns:
            H_C: (batch, l_c, d_model)
            encoder_outputs: for passing to decoder
        """
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
        """Build combined commonsense representation (Eq. 3-4).

        Returns:
            H_com: (batch, l_com, d)
            com_mask: (batch, l_com)
        """
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
                # Eq. 3: E_com^r = Encoder(P_com^r)
                hidden = self._encode_auxiliary(ids, mask, self.comet_encoders[relation])
                all_hidden.append(hidden)
                all_masks.append(mask)

        if not all_hidden:
            # No commonsense available — return zeros
            batch_size = comet_intent_ids.size(0) if comet_intent_ids is not None else 1
            d = self.config.d_model
            H_com = torch.zeros(batch_size, 1, d, device=self._get_shared_embeddings().weight.device)
            com_mask = torch.zeros(batch_size, 1, device=self._get_shared_embeddings().weight.device)
            return H_com, com_mask

        # Eq. 4: E_com^all = E_com^Intent ⊕ E_com^Want ⊕ E_com^Need ⊕ E_com^Effect
        E_com_all = torch.cat(all_hidden, dim=1)  # (batch, l_total, d)
        com_mask = torch.cat(all_masks, dim=1)     # (batch, l_total)

        # Eq. 4: H_com = Encoder(E_com^all)
        H_com = self.comet_combined_encoder(E_com_all, com_mask)

        return H_com, com_mask

    def _fuse(self, H_C, H_P, H_com, ctx_mask, persona_mask, com_mask):
        """Apply Knowledge Refiner + Cross-Attention + Fusion (Eq. 5-13).

        Returns:
            H_final: (batch, l_c, d)
        """
        d = H_C.size(-1)
        l_c = H_C.size(1)

        # Eq. 5-7: Knowledge Refiner
        E_r_com = self.knowledge_refiner(H_com, H_C, com_mask, ctx_mask)  # (batch, l_com, d)

        # Eq. 8-11: Bidirectional Cross-Attention (persona ↔ context)
        H_tilde_P, H_tilde_C_p = self.persona_cross_attn(H_C, H_P, ctx_mask, persona_mask)
        # H_tilde_P: (batch, l_c, d) — persona-informed context
        # H_tilde_C_p: (batch, l_p, d) — context-informed persona

        # Eq. 8-11: Bidirectional Cross-Attention (commonsense ↔ context)
        H_tilde_com, H_tilde_C_com = self.comet_cross_attn(H_C, E_r_com, ctx_mask, com_mask)
        # H_tilde_com: (batch, l_c, d) — commonsense-informed context
        # H_tilde_C_com: (batch, l_com, d) — context-informed commonsense

        # Dimension alignment for fusion (Eq. 12)
        # H_C, H_tilde_P, H_tilde_com: all (batch, l_c, d) → ok
        # H_tilde_C_p: (batch, l_p, d) → mean pool → (batch, 1, d) → broadcast
        # H_tilde_C_com: (batch, l_com, d) → mean pool → (batch, 1, d) → broadcast

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

        # Eq. 12-13: Weighted Fusion
        components = [H_C, H_tilde_P, H_tilde_C_p_pooled, H_tilde_com, H_tilde_C_com_pooled]
        H_final = self.fusion(components)

        return H_final

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

        encoded_info = kwargs
        assert (self.training or validation) == (labels is not None)
        if validation:
            labels[:, 0] = -100

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Extract auxiliary inputs from kwargs
        persona_ids = kwargs.get('persona_ids', None)
        persona_mask = kwargs.get('persona_mask', None)
        comet_intent_ids = kwargs.get('comet_intent_ids', None)
        comet_intent_mask = kwargs.get('comet_intent_mask', None)
        comet_want_ids = kwargs.get('comet_want_ids', None)
        comet_want_mask = kwargs.get('comet_want_mask', None)
        comet_need_ids = kwargs.get('comet_need_ids', None)
        comet_need_mask = kwargs.get('comet_need_mask', None)
        comet_effect_ids = kwargs.get('comet_effect_ids', None)
        comet_effect_mask = kwargs.get('comet_effect_mask', None)

        if encoder_outputs is None:
            # Eq. 1: H_C = Encoder(U)
            encoder_outputs = self._encode_context(input_ids, attention_mask, return_dict=return_dict)
            H_C = encoder_outputs[0]  # (batch, l_c, d)

            # Eq. 2: H_P = Encoder(P_s)
            if persona_ids is not None and persona_ids.size(1) > 0:
                H_P = self._encode_auxiliary(persona_ids, persona_mask, self.persona_encoder)
            else:
                H_P = torch.zeros(H_C.size(0), 1, H_C.size(-1), device=H_C.device)
                persona_mask = torch.zeros(H_C.size(0), 1, device=H_C.device)

            # Eq. 3-4: H_com
            H_com, com_mask = self._build_H_com(
                comet_intent_ids, comet_intent_mask,
                comet_want_ids, comet_want_mask,
                comet_need_ids, comet_need_mask,
                comet_effect_ids, comet_effect_mask,
            )

            # Eq. 5-13: Fusion
            H_final = self._fuse(H_C, H_P, H_com, attention_mask, persona_mask, com_mask)

            # Replace encoder outputs with fused representation
            encoder_outputs = BaseModelOutput(
                last_hidden_state=H_final,
                hidden_states=None,
                attentions=None,
            )

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

        if not self.training and not validation:
            from transformers.modeling_outputs import Seq2SeqLMOutput
            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.hidden_states,
                decoder_attentions=outputs.attentions,
                encoder_last_hidden_state=encoder_outputs[0],
            )
        elif self.training:
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value}
            return res
        else:
            assert not self.training
            return loss, label_size

    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        strat_id = encoded_info.get('strat_id', None)
        logits = logits[:, 0, -8:]

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

        encoded_info = kwargs.copy()
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Extract auxiliary inputs
        persona_ids = kwargs.get('persona_ids', None)
        persona_mask = kwargs.get('persona_mask', None)
        comet_intent_ids = kwargs.get('comet_intent_ids', None)
        comet_intent_mask = kwargs.get('comet_intent_mask', None)
        comet_want_ids = kwargs.get('comet_want_ids', None)
        comet_want_mask = kwargs.get('comet_want_mask', None)
        comet_need_ids = kwargs.get('comet_need_ids', None)
        comet_need_mask = kwargs.get('comet_need_mask', None)
        comet_effect_ids = kwargs.get('comet_effect_ids', None)
        comet_effect_mask = kwargs.get('comet_effect_mask', None)

        # Eq. 1: Encode context
        encoder_outputs = self._encode_context(input_ids, attention_mask, return_dict=return_dict)
        H_C = encoder_outputs[0]

        # Eq. 2: Encode persona
        if persona_ids is not None and persona_ids.size(1) > 0:
            H_P = self._encode_auxiliary(persona_ids, persona_mask, self.persona_encoder)
        else:
            H_P = torch.zeros(H_C.size(0), 1, H_C.size(-1), device=H_C.device)
            persona_mask = torch.zeros(H_C.size(0), 1, device=H_C.device)

        # Eq. 3-4: COMET
        H_com, com_mask = self._build_H_com(
            comet_intent_ids, comet_intent_mask,
            comet_want_ids, comet_want_mask,
            comet_need_ids, comet_need_mask,
            comet_effect_ids, comet_effect_mask,
        )

        # Eq. 5-13: Fusion
        H_final = self._fuse(H_C, H_P, H_com, attention_mask, persona_mask, com_mask)

        # Strategy prediction
        fused_encoder_outputs = BaseModelOutput(
            last_hidden_state=H_final,
            hidden_states=None,
            attentions=None,
        )

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=H_final,
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        self.predict_strategy(lm_logits, encoded_info)

        # Prepare for generation
        gen_kwargs = {k: v for k, v in kwargs.items() if k in _VALID_GEN_PARAMS}

        decoder_input_ids = torch.cat(
            [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 8],
            dim=-1
        )

        assert 'max_length' in gen_kwargs
        gen_kwargs['max_length'] = gen_kwargs['max_length'] + decoder_input_ids.size(1)
        gen_kwargs['use_cache'] = True

        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            gen_kwargs['bad_words_ids'] = bad_words_ids

        # Use fused H_final as encoder output for generation
        generations = BlenderbotSmallForConditionalGeneration.generate(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=fused_encoder_outputs,
            **gen_kwargs
        )
        return encoded_info, generations[:, decoder_input_ids.size(1):]
