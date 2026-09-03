#!/usr/bin/env python3
"""
PI-Mamba Architecture Comparison Sweep

Runs PI-Mamba vs Transformer vs Performer and logs C_k metrics.
Generates data for P3 (Architecture Ordering) validation.

Usage:
    python run_ck_sweep.py --steps 100 --output sweep_results.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from backbone_pi_mamba import PIMambaBackbone
from backbone_performer import PerformerBackbone
from measure_mode_concentration import compute_mode_concentration
from rouse_physics import compute_rouse_eigenvectors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class TransformerBackbone(nn.Module):
    """Simple Transformer baseline for comparison."""
    def __init__(self, d_model=256, n_layers=6, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(3, d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(64, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, 3)
        self.last_hidden = None
        
    def _time_encoding(self, t, dim=64):
        half = dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
    def forward(self, x, t):
        B, L, _ = x.shape
        h = self.input_proj(x)
        t_emb = self.time_embed(self._time_encoding(t))
        h = h + t_emb.unsqueeze(1)
        h = self.encoder(h)
        self.last_hidden = h.detach()
        return self.output_proj(h), h


def compute_dummy_sctm(coords: torch.Tensor) -> float:
    """Placeholder scTM - in real use, call ESMFold pipeline."""
    # Use compactness as proxy: lower Rg = more structured = higher scTM
    B, L, _ = coords.shape
    centroid = coords.mean(dim=1, keepdim=True)
    rg = torch.sqrt(((coords - centroid)**2).sum(dim=-1).mean(dim=1))
    # Normalize: Rg ~ sqrt(L) for random, much smaller for structured
    expected_rg = np.sqrt(L) * 3.8 / np.sqrt(6)  # Gaussian chain Rg
    compactness = 1 - (rg.mean().item() / expected_rg)
    return max(0.0, min(1.0, 0.5 + compactness * 0.5))


def train_and_measure(model_type: str, steps: int = 100, seed: int = 0) -> dict:
    """Train a model and measure C_k over training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training {model_type} on {device} (seed={seed})")
    
    # Initialize model
    if model_type == "pi_mamba":
        model = PIMambaBackbone(d_model=256, n_layers=6, d_state=64, use_physics=True)
    elif model_type == "transformer":
        model = TransformerBackbone(d_model=256, n_layers=6)
    elif model_type == "performer":
        model = PerformerBackbone(d_model=256, n_layers=6)
    else:
        raise ValueError(f"Unknown model: {model_type}")
        
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Tracking
    ck_history = []
    loss_history = []
    
    model.train()
    for step in range(steps):
        B, L = 8, 100
        coords = torch.randn(B, L, 3, device=device)
        t = torch.rand(B, device=device)
        
        # Forward
        velocity_pred, hidden = model(coords, t)
        target = -coords  # Simple target: move towards origin
        loss = F.mse_loss(velocity_pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Log every 10 steps
        if step % 10 == 0:
            with torch.no_grad():
                ck_result = compute_mode_concentration(hidden, k=10)
                ck_history.append({
                    "step": step,
                    "C_10": ck_result["C_k"],
                    "loss": loss.item()
                })
            logger.info(f"[{model_type}] Step {step}: Loss={loss.item():.4f}, C_10={ck_result['C_k']:.4f}")
            
        loss_history.append(loss.item())
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Generate sample
        coords = torch.randn(16, 100, 3, device=device)
        t = torch.ones(16, device=device) * 0.0
        pred, hidden = model(coords, t)
        
        final_ck = compute_mode_concentration(hidden, k=10)
        final_sctm = compute_dummy_sctm(pred)
        
    return {
        "model": model_type,
        "seed": seed,
        "final_C_10": final_ck["C_k"],
        "final_loss": loss_history[-1],
        "final_scTM": final_sctm,
        "history": ck_history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=str, default="sweep_results.json")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    models = ["pi_mamba", "transformer", "performer"]
    
    all_results = []
    
    for model_type in models:
        for seed in seeds:
            try:
                result = train_and_measure(model_type, steps=args.steps, seed=seed)
                all_results.append(result)
                
                # Save incrementally
                with open(args.output, 'w') as f:
                    json.dump(all_results, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Failed {model_type} seed={seed}: {e}")
                all_results.append({
                    "model": model_type,
                    "seed": seed,
                    "error": str(e)
                })
    
    # Summary
    print("\n" + "="*60)
    print("SWEEP RESULTS SUMMARY")
    print("="*60)
    
    for model_type in models:
        model_results = [r for r in all_results if r.get("model") == model_type and "error" not in r]
        if model_results:
            ck_vals = [r["final_C_10"] for r in model_results]
            sctm_vals = [r["final_scTM"] for r in model_results]
            print(f"{model_type:15s}: C_10={np.mean(ck_vals):.3f}±{np.std(ck_vals):.3f}, scTM={np.mean(sctm_vals):.3f}±{np.std(sctm_vals):.3f}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
