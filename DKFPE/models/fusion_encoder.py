# coding=utf-8
"""
DKFPE Multi-perspective Fusion Encoder

Components:
- LightweightEncoder: 2-layer Transformer encoder for persona/COMET
- KnowledgeRefiner: Eq. 5-7, refines commonsense with context
- BidirectionalCrossAttention: Eq. 8-11, cross-attention between two representations
- WeightedFusion: Eq. 12-13, learnable weighted combination
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightEncoder(nn.Module):
    """2-layer Transformer encoder for persona and COMET text encoding.

    Uses the same d_model as BlenderBot-small (512) for compatibility.
    """

    def __init__(self, d_model=512, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            (batch, seq_len, d_model)
        """
        if attention_mask is not None:
            # TransformerEncoder expects src_key_padding_mask: True for padding
            key_padding_mask = (attention_mask == 0)

            # Handle all-padding rows: softmax(all -inf) = NaN
            # Identify batch elements where ALL positions are padding
            all_pad = key_padding_mask.all(dim=1)  # (batch,)
            if all_pad.any():
                # Temporarily unmask first position for all-pad rows to avoid NaN
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad, 0] = False
        else:
            key_padding_mask = None

        output = self.encoder(hidden_states, src_key_padding_mask=key_padding_mask)

        # Zero out output for all-padding rows
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).float()

        return output


class KnowledgeRefiner(nn.Module):
    """Knowledge Refiner (Eq. 5-7).

    Refines commonsense knowledge by incorporating context information.

    Eq. 5: H_mix = H_com ⊕ H_C (feature concatenation along hidden dim)
    Eq. 6: H_ref = Enc_ref(H_mix)
    Eq. 7: E_r_com = MLP(σ(H_ref) ⊙ H_ref)
    """

    def __init__(self, d_model=512, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        # Enc_ref: Transformer encoder operating on 2*d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model * 2,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.refiner_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP: Linear(2d → d) + ReLU
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, H_com, H_C, com_mask=None, ctx_mask=None):
        """
        Args:
            H_com: (batch, l_com, d) — commonsense representation
            H_C: (batch, l_c, d) — context representation
            com_mask: (batch, l_com) — 1 for real, 0 for pad
            ctx_mask: (batch, l_c) — 1 for real, 0 for pad

        Returns:
            E_r_com: (batch, l_com, d)
        """
        d = H_C.size(-1)
        l_com = H_com.size(1)

        # Eq. 5: H_mix = H_com ⊕ H_C
        # Mean pool H_C → (batch, 1, d), broadcast to (batch, l_com, d)
        if ctx_mask is not None:
            ctx_mask_float = ctx_mask.unsqueeze(-1).float()
            H_C_pooled = (H_C * ctx_mask_float).sum(dim=1, keepdim=True) / ctx_mask_float.sum(dim=1, keepdim=True).clamp(min=1e-9)
        else:
            H_C_pooled = H_C.mean(dim=1, keepdim=True)

        H_C_expanded = H_C_pooled.expand(-1, l_com, -1)  # (batch, l_com, d)
        H_mix = torch.cat([H_com, H_C_expanded], dim=-1)  # (batch, l_com, 2d)

        # Eq. 6: H_ref = Enc_ref(H_mix)
        if com_mask is not None:
            key_padding_mask = (com_mask == 0)
            # Handle all-padding rows to avoid NaN from softmax
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad, 0] = False
        else:
            key_padding_mask = None
        H_ref = self.refiner_encoder(H_mix, src_key_padding_mask=key_padding_mask)

        # Eq. 7: E_r_com = MLP(σ(H_ref) ⊙ H_ref)
        gate = torch.sigmoid(H_ref)
        E_r_com = self.mlp(gate * H_ref)  # (batch, l_com, d)

        # Zero out output for all-padding rows
        if com_mask is not None:
            E_r_com = E_r_com * com_mask.unsqueeze(-1).float()

        return E_r_com


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional Cross-Attention (Eq. 8-11).

    Computes cross-attention in both directions between two sequences.

    Eq. 8:  Z_P = Softmax(H_C · H_P^T / √d) · H_P
    Eq. 9:  Z_C = Softmax(H_P · H_C^T / √d) · H_C
    Eq. 10: H̃_P = LayerNorm(Z_P + H_C)
    Eq. 11: H̃_C^p = LayerNorm(Z_C + H_P)
    """

    def __init__(self, d_model=512):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.layer_norm_a = nn.LayerNorm(d_model)
        self.layer_norm_b = nn.LayerNorm(d_model)

    def forward(self, H_A, H_B, mask_A=None, mask_B=None):
        """
        Computes bidirectional cross-attention between H_A and H_B.

        Args:
            H_A: (batch, l_a, d) — e.g., context H_C
            H_B: (batch, l_b, d) — e.g., persona H_P or commonsense H_com
            mask_A: (batch, l_a) — 1=real, 0=pad
            mask_B: (batch, l_b) — 1=real, 0=pad

        Returns:
            H_tilde_B: (batch, l_a, d) — B-informed A (like H̃_P in Eq. 10)
            H_tilde_A: (batch, l_b, d) — A-informed B (like H̃_C^p in Eq. 11)
        """
        # Eq. 8: Z_B = Softmax(H_A · H_B^T / √d) · H_B
        scores_AB = torch.bmm(H_A, H_B.transpose(1, 2)) / self.scale  # (batch, l_a, l_b)
        if mask_B is not None:
            scores_AB = scores_AB.masked_fill(mask_B.unsqueeze(1) == 0, float('-inf'))
        attn_AB = F.softmax(scores_AB, dim=-1)
        attn_AB = attn_AB.masked_fill(torch.isnan(attn_AB), 0.0)
        Z_B = torch.bmm(attn_AB, H_B)  # (batch, l_a, d)

        # Eq. 9: Z_A = Softmax(H_B · H_A^T / √d) · H_A
        scores_BA = torch.bmm(H_B, H_A.transpose(1, 2)) / self.scale  # (batch, l_b, l_a)
        if mask_A is not None:
            scores_BA = scores_BA.masked_fill(mask_A.unsqueeze(1) == 0, float('-inf'))
        attn_BA = F.softmax(scores_BA, dim=-1)
        attn_BA = attn_BA.masked_fill(torch.isnan(attn_BA), 0.0)
        Z_A = torch.bmm(attn_BA, H_A)  # (batch, l_b, d)

        # Eq. 10: H̃_B = LayerNorm(Z_B + H_A)
        H_tilde_B = self.layer_norm_a(Z_B + H_A)  # (batch, l_a, d)

        # Eq. 11: H̃_A = LayerNorm(Z_A + H_B)
        H_tilde_A = self.layer_norm_b(Z_A + H_B)  # (batch, l_b, d)

        return H_tilde_B, H_tilde_A


class WeightedFusion(nn.Module):
    """Weighted Fusion (Eq. 12-13).

    H_final = λ1·H_C + λ2·H̃_P + λ3·H̃_C^p_pooled + λ4·H̃_com + λ5·H̃_C^com_pooled
    λ_i = exp(w_i) / Σ_j exp(w_j)  (softmax-normalized learnable scalars)
    """

    def __init__(self, num_components=5):
        super().__init__()
        # Initialize all weights equally (softmax → uniform)
        self.w = nn.Parameter(torch.zeros(num_components))

    def forward(self, components, masks=None):
        """
        Args:
            components: list of (batch, l_c, d) tensors
            masks: not used, all components should already be (batch, l_c, d)

        Returns:
            H_final: (batch, l_c, d)
        """
        lambdas = F.softmax(self.w, dim=0)
        H_final = sum(l * c for l, c in zip(lambdas, components))
        return H_final
