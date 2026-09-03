"""Hybrid encoder for PI-Mamba.

This module defines a `HybridEncoder` that fuses a Physics‑Informed Mamba block
(with Rouse‑derived spectral initialization) and a lightweight sparse‑attention
branch. The two streams are combined via a gated residual connection, providing
both the linear‑time inductive bias of Mamba and the ability to capture long‑
range interactions that may be missed by a purely convolutional state‑space
model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi_mamba_layer import PhysicsInformedMambaBlock


class SparseAttentionBranch(nn.Module):
    """A simple sparse‑attention module.

    For efficiency we restrict attention to a sliding window of size `window`
    around each residue. This mimics the local‑pair updates used in AlphaFold
    while keeping the cost O(L·window).
    """

    def __init__(self, d_model: int, n_heads: int = 8, window: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window = window
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # Linear projection to inject positional bias for the window
        self.pos_bias = nn.Parameter(torch.randn(2 * window + 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        # Pad for windowed attention
        pad = self.window
        x_padded = F.pad(x, (0, 0, pad, pad), mode="constant", value=0.0)
        # Build queries, keys, values for each position using sliding windows
        queries = []
        keys = []
        values = []
        for i in range(L):
            start = i
            end = i + 2 * self.window + 1
            window_slice = x_padded[:, start:end, :]
            queries.append(x[:, i : i + 1, :])
            keys.append(window_slice)
            values.append(window_slice)
        # Stack to shape (B, L, 1, D) -> (B*L, 1, D)
        Q = torch.cat(queries, dim=1).view(B * L, 1, D)
        K = torch.cat(keys, dim=1).view(B * L, 2 * self.window + 1, D)
        V = torch.cat(values, dim=1).view(B * L, 2 * self.window + 1, D)
        # Add positional bias
        K = K + self.pos_bias.unsqueeze(0)
        V = V + self.pos_bias.unsqueeze(0)
        # Multihead attention (single query per position)
        attn_out, _ = self.attn(Q, K, V)
        attn_out = attn_out.view(B, L, D)
        return attn_out


class GatedFusion(nn.Module):
    """Gated residual fusion of two streams.

    `gate = sigmoid(W_g * x_mamba + b_g)` and the output is
    `x_fused = gate * x_mamba + (1 - gate) * x_sparse`.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_mamba: torch.Tensor, x_sparse: torch.Tensor) -> torch.Tensor:
        gate = self.sigmoid(self.gate_proj(x_mamba))
        return gate * x_mamba + (1 - gate) * x_sparse


class HybridEncoder(nn.Module):
    """Combine Physics‑Informed Mamba with sparse attention.

    The module can be dropped into the existing `PIMambaBackbone` in place of the
    original `self.pi_mamba_layers` list. It returns a single hidden representation
    that can be fed to the structure module.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 12,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        n_groups: int = 8,
        dropout: float = 0.1,
        use_physics: bool = True,
        sparse_window: int = 32,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.mamba_layers = nn.ModuleList(
            [
                PhysicsInformedMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand_factor=expand_factor,
                    max_length=2048,
                    n_groups=n_groups,
                    dropout=dropout,
                    use_physics=use_physics,
                )
                for _ in range(n_layers)
            ]
        )
        self.sparse_branch = SparseAttentionBranch(d_model, n_heads=8, window=sparse_window)
        self.fusion = GatedFusion(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        h = x
        for layer in self.mamba_layers:
            h = layer(h, return_physics=False)[0]  # discard physics info
        # Sparse attention stream (operates on the same input)
        h_sparse = self.sparse_branch(x)
        # Fuse the two streams
        h_fused = self.fusion(h, h_sparse)
        return self.norm(h_fused)

# End of hybrid_encoder.py
