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

from .pi_mamba_layer import PhysicsInformedMambaBlock


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
        W = 2 * self.window + 1
        # Sliding windows of keys/values around every position: (B, L, W, D)
        x_padded = F.pad(x, (0, 0, self.window, self.window), mode="constant", value=0.0)
        windows = x_padded.unfold(1, W, 1).permute(0, 1, 3, 2)
        K = windows.reshape(B * L, W, D) + self.pos_bias.unsqueeze(0)
        Q = x.reshape(B * L, 1, D)
        # Window slots that fall outside the chain are masked out
        idx = torch.arange(L, device=x.device).unsqueeze(1) + torch.arange(W, device=x.device) - self.window
        pad_mask = ((idx < 0) | (idx >= L)).repeat(B, 1)  # (B*L, W)
        attn_out, _ = self.attn(Q, K, K, key_padding_mask=pad_mask)
        return attn_out.view(B, L, D)


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

    Optional encoder variant. It is not used by `PIMambaBackbone` or by the
    training script. It maps embedded inputs (B, L, D) to hidden states
    (B, L, D), the same contract as `FlowBackbone.encode` without the time
    conditioning.
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
        **layer_kwargs,
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
                    **layer_kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        self.sparse_branch = SparseAttentionBranch(d_model, n_heads=8, window=sparse_window)
        self.fusion = GatedFusion(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
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
