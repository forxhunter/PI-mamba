"""
Transformer Backbone for Baseline Comparison

Standard full self-attention encoder (O(L^2)) with the same embeddings, output
heads and sampler as PI-Mamba.
"""

import torch.nn as nn

from .backbone_pi_mamba import AdaptiveLayerNorm, FlowBackbone


class TransformerBackbone(FlowBackbone):
    """
    Transformer-based backbone for protein structure generation.

    Args:
        d_model: Hidden dimension
        n_layers: Number of encoder layers
        n_heads: Attention heads per layer
        n_structure_layers: Number of structure module layers
        max_length: Length of the precomputed position encoding table
        dropout: Dropout inside each encoder layer
        coord_scale: Angstrom per model unit
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

        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
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
        return h, ({'note': 'Transformer baseline (no physics)'} if return_physics else None)
