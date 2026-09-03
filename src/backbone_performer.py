"""
Linear Attention (Performer-style) Backbone for Baseline Comparison

This module implements a baseline backbone using kernel-based linear attention
instead of Mamba. This is used to validate that Mamba's advantages come from
its selective state updates, not just linear complexity.

Reference:
    Choromanski et al. (2021). Rethinking Attention with Performers. ICLR.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from einops import rearrange

# Import shared components from PI-Mamba
from backbone_pi_mamba import (
    AdaptiveLayerNorm,
    TimeEmbedding,
    StructureModule,
    OmegaHead,
    NeRFProjection,
)


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


class PerformerBackbone(nn.Module):
    """
    Performer-based backbone for protein structure generation.
    
    Uses Linear Attention instead of Mamba for the sequence encoder.
    All other components (Structure Module, Chain Retraction, OmegaHead)
    are identical to PI-Mamba for fair comparison.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 8,
        n_structure_layers: int = 4,
        max_length: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Time embedding
        self.time_embed = TimeEmbedding(d_model)
        
        # Input projection
        self.input_proj = nn.Linear(3, d_model)
        
        # Position encoding
        self.register_buffer(
            'pos_encoding',
            self._create_pos_encoding(max_length, d_model)
        )
        
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
        
        # Structure module (same as PI-Mamba)
        self.structure_module = StructureModule(
            d_model=d_model,
            n_layers=n_structure_layers,
        )
        
        # Omega Prediction Head
        self.omega_head = OmegaHead(d_model)
        
        # Chain retraction
        self.kinematic_proj = NeRFProjection()
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 3)
        
    def _create_pos_encoding(self, max_length: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
        return_physics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass: predict velocity field for flow matching.
        """
        B, L, _ = coords.shape
        
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Input projection
        h = self.input_proj(coords)
        
        # Add position encoding
        h = h + self.pos_encoding[:, :L, :]
        
        # Pass through Linear Attention layers
        for i, (attn_layer, adaptive_norm) in enumerate(
            zip(self.attn_layers, self.adaptive_norms)
        ):
            h = adaptive_norm(h, t_emb)
            h = attn_layer(h)
        
        # Structure module
        coords_pred = self.structure_module(h, t_emb)
        
        # Output projection
        velocity = self.output_proj(h)
        
        # Omega logits
        omega_logits = self.omega_head(h)
        
        if return_physics:
            # Performer has no physics-informed parameters
            return velocity, omega_logits, {'note': 'Performer baseline (no physics)'}
        
        return velocity, omega_logits
        
    def sample(
        self,
        length: int,
        n_samples: int = 1,
        n_steps: int = 100,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Generate protein backbones using flow matching."""
        if device is None:
            device = next(self.parameters()).device
        
        coords = torch.randn(n_samples, length, 3, device=device) * 0.1
        timesteps = torch.linspace(1.0, 0.0, n_steps, device=device)
        
        with torch.no_grad():
            for i in range(len(timesteps) - 1):
                t = timesteps[i].unsqueeze(0).expand(n_samples)
                dt = timesteps[i] - timesteps[i + 1]
                
                velocity, omega_logits = self.forward(coords, t)
                
                # Get target distances from omega predictions
                cis_probs = F.softmax(omega_logits, dim=-1)[:, :, 1]
                is_cis = cis_probs > 0.5
                
                target_dists = torch.ones(n_samples, length - 1, 1, device=device) * 3.8
                cis_mask = is_cis[:, 1:]
                target_dists[cis_mask] = 2.96
                
                coords = coords + velocity * dt
                
                # Chain retraction every 10 steps
                if (i + 1) % 10 == 0:
                    coords = self.kinematic_proj._project_bonds(coords, target_lengths=target_dists)
        
        coords = self.kinematic_proj._project_bonds(coords, target_lengths=target_dists)
        
        return coords


if __name__ == "__main__":
    print("Testing Performer Backbone")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = PerformerBackbone(
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_structure_layers=2,
    ).to(device)
    
    # Test forward
    coords = torch.randn(2, 100, 3, device=device)
    t = torch.rand(2, device=device)
    
    velocity, omega_logits, info = model(coords, t, return_physics=True)
    
    print(f"Input shape: {coords.shape}")
    print(f"Output shape: {velocity.shape}")
    print(f"Omega logits: {omega_logits.shape}")
    print(f"Info: {info}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # Test sampling
    print("\nTesting sampling...")
    samples = model.sample(length=50, n_samples=2, n_steps=20, device=device)
    print(f"Sample shape: {samples.shape}")
    
    print("\n✓ All tests passed!")
