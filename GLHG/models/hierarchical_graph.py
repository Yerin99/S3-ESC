# coding=utf-8
"""
Hierarchical Graph Reasoner for GLHG

논문의 Section 3.3 구현:
- Global connection: 전역 노드가 모든 노드와 연결
- Local connection: 지역 노드가 마지막 발화의 토큰들과만 연결
- Contextual connection: 토큰 노드들 간의 연결

Graph Attention Network (GAT)를 사용하여 노드 간 상호작용 모델링
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT)
    
    논문 Equation (4), (5), (6) 구현:
    - F(v_1, v_j) = LeakyReLU(a^T [W * v_1 || W * v_j])
    - alpha_j = softmax(F(v_1, v_j))
    - v_1^(k+1) = σ(Σ alpha_j * W * v_j)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        """
        Args:
            hidden_dim: 은닉 차원 크기
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
            negative_slope: LeakyReLU의 negative slope
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        # Linear transformation W
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Attention weight a (2d -> 1 for concatenated features)
        self.a = nn.Parameter(torch.zeros(1, num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout_layer = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """파라미터 초기화"""
        nn.init.xavier_uniform_(self.W.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            query: Query 텐서 (batch, query_len, hidden_dim)
            key: Key 텐서 (batch, key_len, hidden_dim)
            value: Value 텐서 (batch, key_len, hidden_dim)
            mask: 어텐션 마스크 (batch, query_len, key_len)
        
        Returns:
            업데이트된 노드 특징 (batch, query_len, hidden_dim)
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        
        # Linear transformation
        Q = self.W(query)  # (batch, query_len, hidden_dim)
        K = self.W(key)    # (batch, key_len, hidden_dim)
        V = self.W(value)  # (batch, key_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim)
        
        # Compute attention scores: a^T [Q || K]
        # Q: (batch, query_len, num_heads, head_dim)
        # K: (batch, key_len, num_heads, head_dim)
        
        Q_expanded = Q.unsqueeze(2).expand(-1, -1, key_len, -1, -1)  # (batch, query_len, key_len, num_heads, head_dim)
        K_expanded = K.unsqueeze(1).expand(-1, query_len, -1, -1, -1)  # (batch, query_len, key_len, num_heads, head_dim)
        
        # Concatenate Q and K
        QK_concat = torch.cat([Q_expanded, K_expanded], dim=-1)  # (batch, query_len, key_len, num_heads, 2*head_dim)
        
        # Compute attention scores
        # a: (1, num_heads, 2*head_dim)
        attn_scores = (QK_concat * self.a.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (batch, query_len, key_len, num_heads)
        attn_scores = self.leaky_relu(attn_scores)
        
        # Transpose for softmax
        attn_scores = attn_scores.permute(0, 3, 1, 2)  # (batch, num_heads, query_len, key_len)
        
        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, query_len, key_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Handle NaN from all-masked positions
        attn_weights = torch.where(
            torch.isnan(attn_weights),
            torch.zeros_like(attn_weights),
            attn_weights
        )
        
        # Weighted sum of values
        V = V.permute(0, 2, 1, 3)  # (batch, num_heads, key_len, head_dim)
        output = torch.matmul(attn_weights, V)  # (batch, num_heads, query_len, head_dim)
        
        # Reshape back
        output = output.permute(0, 2, 1, 3).contiguous()  # (batch, query_len, num_heads, head_dim)
        output = output.view(batch_size, query_len, self.hidden_dim)  # (batch, query_len, hidden_dim)
        
        return output


class HierarchicalGraphReasoner(nn.Module):
    """
    Hierarchical Graph Reasoner
    
    논문 Section 3.3 구현:
    - 다중 소스 정보(global cause, local intention, dialog history) 간의 
      계층적 관계를 Graph Attention Network로 모델링
    
    노드 구성: V = {g, h_1, ..., h_T, l}
    - g: Global cause embedding
    - h_t: Token embeddings from dialog history
    - l: Local psychological intention embedding
    
    엣지 구성:
    - Global connection: g <-> all nodes
    - Local connection: l <-> last utterance tokens, l <-> g
    - Contextual connection: h_t <-> neighboring tokens
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        window_size: int = 5,
    ):
        """
        Args:
            hidden_dim: 은닉 차원 크기 (BlenderBot-small: 512)
            num_layers: Graph layer 수 (K)
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
            window_size: Contextual connection의 윈도우 크기
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Graph Attention Layers for each layer
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks for each node type
        self.ffn_global = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.ffn_local = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.ffn_token = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Output layer norms
        self.output_norm_global = nn.LayerNorm(hidden_dim)
        self.output_norm_local = nn.LayerNorm(hidden_dim)
        self.output_norm_token = nn.LayerNorm(hidden_dim)
    
    def _build_adjacency_mask(
        self,
        batch_size: int,
        seq_len: int,
        last_utterance_starts: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        그래프 연결을 위한 adjacency mask 생성
        
        Args:
            batch_size: 배치 크기
            seq_len: 시퀀스 길이
            last_utterance_starts: 마지막 발화 시작 위치 (batch,)
            attention_mask: 패딩 마스크 (batch, seq_len)
            device: 디바이스
        
        Returns:
            global_mask: Global node의 adjacency mask
            local_mask: Local node의 adjacency mask
            token_mask: Token nodes의 adjacency mask
        """
        # Total nodes: 1 (global) + seq_len (tokens) + 1 (local) = seq_len + 2
        total_nodes = seq_len + 2
        
        # Global node (index 0) connects to all nodes
        global_mask = torch.ones(batch_size, 1, total_nodes, device=device)
        
        # Local node (index -1) connects to:
        # 1. Global node
        # 2. Last utterance tokens
        local_mask = torch.zeros(batch_size, 1, total_nodes, device=device)
        local_mask[:, :, 0] = 1  # Connect to global node
        local_mask[:, :, -1] = 1  # Self-connection
        
        for b in range(batch_size):
            start = last_utterance_starts[b].item() + 1  # +1 for global node offset
            end = seq_len + 1  # Token indices: 1 to seq_len
            if start < end:
                local_mask[b, :, start:end] = 1
        
        # Token nodes (indices 1 to seq_len) connect to:
        # 1. Global node
        # 2. Neighboring tokens within window
        # 3. Local node (for last utterance tokens)
        token_mask = torch.zeros(batch_size, seq_len, total_nodes, device=device)
        token_mask[:, :, 0] = 1  # All tokens connect to global
        
        # Contextual connections (within window)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            token_mask[:, i, start + 1:end + 1] = 1  # +1 for global node offset
        
        # Last utterance tokens connect to local node
        for b in range(batch_size):
            start = last_utterance_starts[b].item()
            token_mask[b, start:, -1] = 1
        
        # Apply attention mask to token positions
        attention_mask_expanded = attention_mask.unsqueeze(1)  # (batch, 1, seq_len)
        
        # Mask out padding positions
        padding_mask = torch.zeros(batch_size, seq_len, total_nodes, device=device)
        padding_mask[:, :, 1:seq_len + 1] = attention_mask_expanded.expand(-1, seq_len, -1)
        padding_mask[:, :, 0] = 1  # Global always visible
        padding_mask[:, :, -1] = 1  # Local always visible
        
        token_mask = token_mask * padding_mask * attention_mask.unsqueeze(-1)
        
        return global_mask, local_mask, token_mask
    
    def forward(
        self,
        global_embedding: torch.Tensor,
        token_embeddings: torch.Tensor,
        local_embedding: torch.Tensor,
        attention_mask: torch.Tensor,
        last_utterance_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            global_embedding: Global cause embedding (batch, hidden_dim)
            token_embeddings: Token embeddings from encoder (batch, seq_len, hidden_dim)
            local_embedding: Local intention embedding (batch, hidden_dim)
            attention_mask: Attention mask (batch, seq_len)
            last_utterance_starts: Last utterance start positions (batch,)
        
        Returns:
            updated_global: Updated global embedding (batch, hidden_dim)
            updated_tokens: Updated token embeddings (batch, seq_len, hidden_dim)
            updated_local: Updated local embedding (batch, hidden_dim)
        """
        batch_size = token_embeddings.size(0)
        seq_len = token_embeddings.size(1)
        device = token_embeddings.device
        
        # Build adjacency masks
        global_mask, local_mask, token_mask = self._build_adjacency_mask(
            batch_size, seq_len, last_utterance_starts, attention_mask, device
        )
        
        # Initialize node representations
        g = global_embedding.unsqueeze(1)  # (batch, 1, hidden_dim)
        h = token_embeddings  # (batch, seq_len, hidden_dim)
        l = local_embedding.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Concatenate all nodes: [g, h_1, ..., h_T, l]
        all_nodes = torch.cat([g, h, l], dim=1)  # (batch, seq_len + 2, hidden_dim)
        
        # Graph reasoning through K layers
        for k in range(self.num_layers):
            # Update global node (Equation 4)
            g_new = self.gat_layers[k](g, all_nodes, all_nodes, global_mask)
            
            # Update token nodes (Equation 7)
            h_new = self.gat_layers[k](h, all_nodes, all_nodes, token_mask)
            
            # Update local node (Equation 8)
            l_new = self.gat_layers[k](l, all_nodes, all_nodes, local_mask)
            
            # 논문 Eq. 6: v_i^(k+1) = σ(Σ α_j * W * v_j) — σ = ELU
            g_new = F.elu(g_new)
            h_new = F.elu(h_new)
            l_new = F.elu(l_new)

            # Residual connections and layer norm
            g = self.layer_norms[k](g + g_new)
            h = self.layer_norms[k](h + h_new)
            l = self.layer_norms[k](l + l_new)
            
            # Update all_nodes for next layer
            all_nodes = torch.cat([g, h, l], dim=1)
        
        # Final feed-forward processing
        g = g + self.ffn_global(g)
        h = h + self.ffn_token(h)
        l = l + self.ffn_local(l)
        
        # Output normalization
        updated_global = self.output_norm_global(g.squeeze(1))  # (batch, hidden_dim)
        updated_tokens = self.output_norm_token(h)  # (batch, seq_len, hidden_dim)
        updated_local = self.output_norm_local(l.squeeze(1))  # (batch, hidden_dim)
        
        return updated_global, updated_tokens, updated_local


class GlobalSemanticClassifier(nn.Module):
    """
    Global Semantic Information Classifier
    
    논문 Equation (10) 구현:
    p(o) = Softmax(MLP(v_1^(K)))
    
    Global node representation을 사용하여 problem type 분류
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 13,  # ESConv has 13 problem types
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: 은닉 차원 크기
            num_classes: 분류할 클래스 수 (problem types)
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, global_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            global_embedding: Global node embedding (batch, hidden_dim)
        
        Returns:
            logits: Classification logits (batch, num_classes)
        """
        return self.classifier(global_embedding)
