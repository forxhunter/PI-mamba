"""
Linear Attention (Performer-style) Backbone for Baseline Comparison

This module implements a baseline backbone using kernel-based linear attention
instead of Mamba. This is used to validate that Mamba's advantages come from
its selective state updates, not just linear complexity.

Reference:
    Choromanski et al. (2021). Rethinking Attention with Performers. ICLR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Shared components from PI-Mamba
from .backbone_pi_mamba import AdaptiveLayerNorm, FlowBackbone


class LinearAttention(nn.Module):
    """
    Linear Attention using ReLU kernel (simplified Performer).
    
    Complexity: O(L * d^2) instead of O(L^2 * d)
    
    Uses the kernel trick: softmax(Q @ K.T) ≈ φ(Q) @ φ(K).T
    where φ(x) = ReLU(x) (simple positive features)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for numerical stability
        self.scale = self.head_dim ** -0.25
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input
            
        Returns:
            y: (B, L, D) output
        """
        B, L, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, L, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # (B, L, D) each
        
        # Reshape to heads
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.n_heads)
        
        # Apply kernel feature map: φ(x) = ReLU(x)
        # This gives us positive features for the linear attention formula
        q = F.relu(q * self.scale)
        k = F.relu(k * self.scale)
        
        # Linear attention: O(L * d^2)
        # Compute K^T @ V first: (d, d) per head
        kv = torch.einsum('bhld, bhle -> bhde', k, v)  # (B, H, d, d)
        
        # Then Q @ (K^T V): (L, d)
        out = torch.einsum('bhld, bhde -> bhle', q, kv)  # (B, H, L, d)
        
        # Normalize by sum of K
        k_sum = k.sum(dim=2, keepdim=True)  # (B, H, 1, d)
        normalizer = torch.einsum('bhld, bhkd -> bhlk', q, k_sum)  # (B, H, L, 1)
        out = out / (normalizer + 1e-6)
        
        # Reshape back
        out = rearrange(out, 'b h l d -> b l (h d)')
        
        return self.out_proj(self.dropout(out))


class LinearAttentionBlock(nn.Module):
    """Linear Attention block with residual and FFN."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        expand_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LinearAttention(d_model, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expand_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expand_factor, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PerformerBackbone(FlowBackbone):
    """
    Performer-based backbone for protein structure generation.

    Uses Linear Attention instead of Mamba for the sequence encoder.
    All other components (embeddings, output heads, sampler, bond projection)
    are inherited from ``FlowBackbone`` and identical to PI-Mamba.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 8,
        n_structure_layers: int = 4,
        max_length: int = 2048,
        dropout: float = 0.1,
        coord_scale: float = 10.0,
    ):
        super().__init__(
            d_model=d_model,
            n_structure_layers=n_structure_layers,
            max_length=max_length,
            coord_scale=coord_scale,
        )

        self.n_layers = n_layers

        # Linear Attention layers (instead of PI-Mamba)
        self.attn_layers = nn.ModuleList([
            LinearAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                expand_factor=4,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Adaptive layer norms (time-conditioned)
        self.adaptive_norms = nn.ModuleList([
            AdaptiveLayerNorm(d_model, d_cond=d_model)
            for _ in range(n_layers)
        ])

    def encode(self, h, t_emb, return_physics: bool = False):
        for attn_layer, adaptive_norm in zip(self.attn_layers, self.adaptive_norms):
            # Same residual pattern as PI-Mamba: h <- h + f(AdaLN(h, t)). The
            # attention block has its own internal skip, which is removed here.
            h_norm = adaptive_norm(h, t_emb)
            h = h + (attn_layer(h_norm) - h_norm)
        # Performer has no physics-informed parameters
        return h, ({'note': 'Performer baseline (no physics)'} if return_physics else None)
