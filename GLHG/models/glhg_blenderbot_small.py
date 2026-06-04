# coding=utf-8
"""
GLHG (Global-to-Local Hierarchical Graph Network) for Emotional Support Conversation

논문 구현: "Control Globally, Understand Locally: A Global-to-Local Hierarchical 
Graph Network for Emotional Support Conversation" (IJCAI 2022)

구성 요소:
1. Multi-Source Encoder: Global cause, Local intention, Dialog history 인코딩
2. Hierarchical Graph Reasoner: 다중 소스 정보 간의 계층적 관계 모델링
3. Global-guide Decoder: 응답 생성 및 problem type 분류

저자: Wei Peng et al.
구현: GLHG 논문 기반 완전 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from models.model_utils import BaseModel
from models.hierarchical_graph import HierarchicalGraphReasoner, GlobalSemanticClassifier
from transformers.generation_utils import top_k_top_p_filtering
from transformers.models.blenderbot_small import (
    BlenderbotSmallConfig, 
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallModel,
)
from transformers.modeling_outputs import (
    BaseModelOutput, 
    Seq2SeqModelOutput, 
    Seq2SeqLMOutput,
)
from .PARAMS import SAMPLE, TEMPERATURE


class GLHGModel(BaseModel, BlenderbotSmallForConditionalGeneration):
    """
    GLHG Model
    
    BlenderBot-small 기반의 GLHG 구현
    
    Architecture:
    1. Multi-Source Encoder
       - Contextual Encoder (Enc_ctx): Dialog history encoding
       - Global Encoder (Enc_glo): Situation encoding with max-pooling
       - Local Encoder (Enc_loc): Intention encoding with max-pooling
    
    2. Hierarchical Graph Reasoner
       - Graph Attention Network for multi-source interaction
       - K layers of message passing
    
    3. Global-guide Decoder
       - Cross-attention with updated graph nodes
       - Problem type classification for global semantic monitoring
    """
    
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        
        hidden_dim = config.d_model  # 512 for blenderbot-small
        
        # Hierarchical Graph Reasoner
        self.graph_reasoner = HierarchicalGraphReasoner(
            hidden_dim=hidden_dim,
            num_layers=2,  # K=2 layers
            num_heads=4,
            dropout=config.dropout,
            window_size=5,
        )
        
        # Global Semantic Classifier (problem type classification)
        # ESConv has 13 problem types
        self.global_classifier = GlobalSemanticClassifier(
            hidden_dim=hidden_dim,
            num_classes=13,
            dropout=config.dropout,
        )
        
        # Projection layers for multi-source fusion
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)
        self.local_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Gate for combining graph-updated representations with encoder outputs
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Loss weights (λ1 = 0.5, λ2 = 0.5 as per paper)
        self.lambda1 = 0.5
        self.lambda2 = 0.5
    
    def _encode_global(
        self,
        situation_ids: torch.Tensor,
        situation_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Global Encoder: Situation 정보 인코딩
        
        논문 Equation (2): g = Max-pooling(Enc_glo(s_1, ..., s_P))
        
        Args:
            situation_ids: Situation token IDs (batch, situation_len)
            situation_attention_mask: Attention mask (batch, situation_len)
        
        Returns:
            Global embedding (batch, hidden_dim)
        """
        # Use the same encoder for global cause
        global_outputs = self.model.encoder(
            input_ids=situation_ids,
            attention_mask=situation_attention_mask,
            return_dict=True,
        )
        
        global_hidden = global_outputs.last_hidden_state  # (batch, situation_len, hidden_dim)
        
        # Max-pooling over sequence (with mask)
        mask_expanded = situation_attention_mask.unsqueeze(-1).float()
        global_hidden = global_hidden * mask_expanded
        global_hidden = global_hidden.masked_fill(mask_expanded == 0, float('-inf'))
        global_embedding = global_hidden.max(dim=1)[0]  # (batch, hidden_dim)
        
        # Handle all-masked case
        global_embedding = torch.where(
            torch.isinf(global_embedding),
            torch.zeros_like(global_embedding),
            global_embedding
        )
        
        return self.global_projection(global_embedding)
    
    def _encode_local(
        self,
        intention_ids: torch.Tensor,
        intention_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Local Encoder: Intention 정보 인코딩
        
        논문 Equation (3): l = Max-pooling(Enc_loc(ms_1, ..., ms_L))
        
        Args:
            intention_ids: Intention token IDs (batch, intention_len)
            intention_attention_mask: Attention mask (batch, intention_len)
        
        Returns:
            Local embedding (batch, hidden_dim)
        """
        # Use the same encoder for local intention
        local_outputs = self.model.encoder(
            input_ids=intention_ids,
            attention_mask=intention_attention_mask,
            return_dict=True,
        )
        
        local_hidden = local_outputs.last_hidden_state  # (batch, intention_len, hidden_dim)
        
        # Max-pooling over sequence (with mask)
        mask_expanded = intention_attention_mask.unsqueeze(-1).float()
        local_hidden = local_hidden * mask_expanded
        local_hidden = local_hidden.masked_fill(mask_expanded == 0, float('-inf'))
        local_embedding = local_hidden.max(dim=1)[0]  # (batch, hidden_dim)
        
        # Handle all-masked case
        local_embedding = torch.where(
            torch.isinf(local_embedding),
            torch.zeros_like(local_embedding),
            local_embedding
        )
        
        return self.local_projection(local_embedding)
    
    def _fuse_graph_outputs(
        self,
        encoder_hidden_states: torch.Tensor,
        updated_tokens: torch.Tensor,
        updated_global: torch.Tensor,
        updated_local: torch.Tensor,
    ) -> torch.Tensor:
        """
        Graph 출력과 인코더 출력 융합
        
        Gating mechanism으로 원본 인코더 출력과 그래프 업데이트된 표현 결합
        
        Args:
            encoder_hidden_states: 원본 인코더 출력 (batch, seq_len, hidden_dim)
            updated_tokens: 그래프 업데이트된 토큰 표현 (batch, seq_len, hidden_dim)
            updated_global: 그래프 업데이트된 전역 표현 (batch, hidden_dim)
            updated_local: 그래프 업데이트된 지역 표현 (batch, hidden_dim)
        
        Returns:
            융합된 hidden states (batch, seq_len + 2, hidden_dim)
        """
        # Gate for token fusion
        gate_input = torch.cat([encoder_hidden_states, updated_tokens], dim=-1)
        gate = self.fusion_gate(gate_input)  # (batch, seq_len, hidden_dim)
        
        fused_tokens = gate * updated_tokens + (1 - gate) * encoder_hidden_states
        
        # Prepend global and append local for decoder cross-attention
        # v^(K) = {v_1^(K), v_2^(K), ..., v_{T+2}^(K)}
        batch_size = encoder_hidden_states.size(0)
        global_expanded = updated_global.unsqueeze(1)  # (batch, 1, hidden_dim)
        local_expanded = updated_local.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # [global, tokens, local]
        fused_output = torch.cat([global_expanded, fused_tokens, local_expanded], dim=1)
        
        return fused_output
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        situation_ids: Optional[torch.Tensor] = None,
        situation_attention_mask: Optional[torch.Tensor] = None,
        intention_ids: Optional[torch.Tensor] = None,
        intention_attention_mask: Optional[torch.Tensor] = None,
        problem_type_ids: Optional[torch.Tensor] = None,
        last_utterance_starts: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[Tuple] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        validation: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass
        
        Args:
            input_ids: Dialog history token IDs (batch, seq_len)
            attention_mask: Dialog attention mask (batch, seq_len)
            situation_ids: Situation token IDs (batch, situation_len)
            situation_attention_mask: Situation attention mask (batch, situation_len)
            intention_ids: Intention token IDs (batch, intention_len)
            intention_attention_mask: Intention attention mask (batch, intention_len)
            problem_type_ids: Problem type labels (batch,)
            last_utterance_starts: Last utterance start positions (batch,)
            decoder_input_ids: Decoder input IDs (batch, tgt_len)
            encoder_outputs: Pre-computed encoder outputs (for generation)
            past_key_values: Cached key-values for generation
            labels: Target labels for loss computation (batch, tgt_len)
            use_cache: Whether to use cache
            return_dict: Whether to return dict
            validation: Whether in validation mode
        
        Returns:
            훈련 시: loss dict {'all': total_loss, 'ppl': ppl, 'cls_loss': cls_loss}
            추론 시: Seq2SeqLMOutput
        """
        assert self.toker is not None
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Encode if needed
        if encoder_outputs is None:
            # Default attention mask (only needed when encoding)
            if attention_mask is None and input_ids is not None:
                attention_mask = input_ids.ne(self.config.pad_token_id).float()
            
            # Default last_utterance_starts
            if last_utterance_starts is None and input_ids is not None:
                last_utterance_starts = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
            # 1. Contextual Encoder: Dialog history
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            
            # 2. Global Encoder: Situation
            if situation_ids is not None and situation_attention_mask is not None:
                global_embedding = self._encode_global(situation_ids, situation_attention_mask)
            else:
                # Fallback: use mean of encoder states
                global_embedding = (encoder_hidden_states * attention_mask.unsqueeze(-1)).sum(1)
                global_embedding = global_embedding / attention_mask.sum(1, keepdim=True).clamp(min=1)
            
            # 3. Local Encoder: Intention
            if intention_ids is not None and intention_attention_mask is not None:
                local_embedding = self._encode_local(intention_ids, intention_attention_mask)
            else:
                # Fallback: use mean of last tokens
                local_embedding = encoder_hidden_states[:, -1, :]
            
            # 4. Hierarchical Graph Reasoner
            updated_global, updated_tokens, updated_local = self.graph_reasoner(
                global_embedding=global_embedding,
                token_embeddings=encoder_hidden_states,
                local_embedding=local_embedding,
                attention_mask=attention_mask,
                last_utterance_starts=last_utterance_starts,
            )
            
            # 5. Fuse graph outputs with encoder outputs
            fused_hidden_states = self._fuse_graph_outputs(
                encoder_hidden_states=encoder_hidden_states,
                updated_tokens=updated_tokens,
                updated_global=updated_global,
                updated_local=updated_local,
            )
            
            # Create extended attention mask for fused output (includes global and local)
            batch_size = input_ids.size(0)
            extended_attention_mask = torch.cat([
                torch.ones(batch_size, 1, device=attention_mask.device),  # global
                attention_mask,  # tokens
                torch.ones(batch_size, 1, device=attention_mask.device),  # local
            ], dim=1)
            
            # Update encoder_outputs with fused states
            encoder_outputs = BaseModelOutput(
                last_hidden_state=fused_hidden_states,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
            
            # Store for decoder
            attention_mask = extended_attention_mask
            
            # Store updated_global for classification
            self._updated_global = updated_global
        else:
            # Use pre-computed encoder outputs (generation mode)
            fused_hidden_states = encoder_outputs.last_hidden_state
        
        # Decoder
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=fused_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        
        # LM head
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        
        # Compute losses
        loss_lm = None
        loss_cls = None
        ppl_value = None
        
        if labels is not None:
            # Language modeling loss (L1)
            loss_lm_raw = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
                reduction='none'
            )
            loss_lm_raw = loss_lm_raw.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss_lm_raw)
            loss_lm = torch.sum(loss_lm_raw) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss_lm_raw, dim=1).float() / label_size.float().clamp(min=1)))
            
            # Problem type classification loss (L2)
            if problem_type_ids is not None and hasattr(self, '_updated_global'):
                cls_logits = self.global_classifier(self._updated_global)
                loss_cls = F.cross_entropy(cls_logits, problem_type_ids)
        
        # Inference mode
        if not self.training and not validation:
            if not return_dict:
                output = (lm_logits,) + decoder_outputs[1:]
                return ((loss_lm,) + output) if loss_lm is not None else output
            
            return Seq2SeqLMOutput(
                loss=loss_lm,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        
        # Training mode
        if self.training:
            # Joint loss (Equation 13): L = λ1 * L1 + λ2 * L2
            total_loss = self.lambda1 * loss_lm
            if loss_cls is not None:
                total_loss = total_loss + self.lambda2 * loss_cls
            
            res = {
                'all': total_loss,
                'ppl': ppl_value,
            }
            if loss_cls is not None:
                res['cls_loss'] = loss_cls
            
            return res
        
        # Validation mode
        return loss_lm_raw, label_size
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        encoder_outputs=None,
        **kwargs
    ):
        """Generation을 위한 입력 준비"""
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
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        situation_ids: Optional[torch.Tensor] = None,
        situation_attention_mask: Optional[torch.Tensor] = None,
        intention_ids: Optional[torch.Tensor] = None,
        intention_attention_mask: Optional[torch.Tensor] = None,
        last_utterance_starts: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        응답 생성
        
        Args:
            input_ids: Dialog history token IDs
            attention_mask: Dialog attention mask
            situation_ids: Situation token IDs
            situation_attention_mask: Situation attention mask
            intention_ids: Intention token IDs
            intention_attention_mask: Intention attention mask
            last_utterance_starts: Last utterance start positions
            decoder_input_ids: Initial decoder input IDs
            **kwargs: Additional generation arguments
        
        Returns:
            생성된 응답
        """
        assert not self.training
        assert self.toker is not None
        
        encoded_info = kwargs.copy()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Default attention mask
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).float()
        
        # Default last_utterance_starts
        if last_utterance_starts is None:
            last_utterance_starts = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
        
        # 1. Encode context
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # 2. Global encoding
        if situation_ids is not None and situation_attention_mask is not None:
            global_embedding = self._encode_global(situation_ids, situation_attention_mask)
        else:
            global_embedding = (encoder_hidden_states * attention_mask.unsqueeze(-1)).sum(1)
            global_embedding = global_embedding / attention_mask.sum(1, keepdim=True).clamp(min=1)
        
        # 3. Local encoding
        if intention_ids is not None and intention_attention_mask is not None:
            local_embedding = self._encode_local(intention_ids, intention_attention_mask)
        else:
            local_embedding = encoder_hidden_states[:, -1, :]
        
        # 4. Graph reasoning
        updated_global, updated_tokens, updated_local = self.graph_reasoner(
            global_embedding=global_embedding,
            token_embeddings=encoder_hidden_states,
            local_embedding=local_embedding,
            attention_mask=attention_mask,
            last_utterance_starts=last_utterance_starts,
        )
        
        # 5. Fuse outputs
        fused_hidden_states = self._fuse_graph_outputs(
            encoder_hidden_states=encoder_hidden_states,
            updated_tokens=updated_tokens,
            updated_global=updated_global,
            updated_local=updated_local,
        )
        
        # Extended attention mask
        batch_size = input_ids.size(0)
        extended_attention_mask = torch.cat([
            torch.ones(batch_size, 1, device=attention_mask.device),
            attention_mask,
            torch.ones(batch_size, 1, device=attention_mask.device),
        ], dim=1)
        
        encoder_outputs = BaseModelOutput(
            last_hidden_state=fused_hidden_states,
        )
        
        # Generation
        if 'max_length' in kwargs:
            kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True
        
        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids
        
        generations = BlenderbotSmallForConditionalGeneration.generate(
            self,
            attention_mask=extended_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        
        return encoded_info, generations[:, decoder_input_ids.size(1):]


# Alias for backward compatibility
Model = GLHGModel
