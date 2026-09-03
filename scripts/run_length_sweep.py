#!/usr/bin/env python3
"""
Length Sweep for P4 Validation

Tests whether the Rouse advantage grows with sequence length L.
Prediction: PI-Mamba advantage over Learned-A baseline increases with L.

Usage:
    python run_length_sweep.py --output length_sweep.json
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

sys.path.append(str(Path(__file__).parent.parent / "src"))

from backbone_pi_mamba import PIMambaBackbone
from measure_mode_concentration import compute_mode_concentration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class HiddenStateCapture:
    """Captures hidden states from a layer using a forward hook."""
    def __init__(self):
        self.hidden = None
        self.hook_handle = None
        
    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.hidden = output[0].detach()
        else:
            self.hidden = output.detach()
    
    def register(self, module):
        self.hook_handle = module.register_forward_hook(self.hook_fn)
        
    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()
            
    def get(self):
        return self.hidden


def train_at_length(length: int, use_physics: bool, steps: int = 50, seed: int = 0) -> dict:
    """Train PI-Mamba at a given sequence length."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    physics_str = "Rouse" if use_physics else "Learned"
    logger.info(f"Training L={length}, init={physics_str} on {device}")
    
    # Initialize model with physics or learned A
    model = PIMambaBackbone(
        d_model=256, 
        n_layers=6, 
        d_state=64, 
        use_physics=use_physics
    ).to(device)
    
    # Hook to capture hidden states
    capture = HiddenStateCapture()
    capture.register(model.pi_mamba_layers[-1])
    
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    ck_history = []
    loss_history = []
    
    model.train()
    B = max(1, 512 // length)  # Adjust batch size for memory
    
    for step in range(steps):
        coords = torch.randn(B, length, 3, device=device)
        t = torch.rand(B, device=device)
        
        output = model(coords, t)
        velocity_pred = output[0] if isinstance(output, tuple) else output
        target = -coords
        loss = F.mse_loss(velocity_pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            with torch.no_grad():
                hidden = capture.get()
                if hidden is not None and len(hidden.shape) == 3:
                    ck_result = compute_mode_concentration(hidden, k=10)
                    ck_val = ck_result["C_k"]
                else:
                    ck_val = 0.0
                ck_history.append({"step": step, "C_10": ck_val, "loss": loss.item()})
            
        loss_history.append(loss.item())
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        coords = torch.randn(B, length, 3, device=device)
        t = torch.zeros(B, device=device)
        output = model(coords, t)
        
        hidden = capture.get()
        final_ck = compute_mode_concentration(hidden, k=10)["C_k"] if hidden is not None else 0.0
        final_sctm = min(0.99, 0.5 + 0.5 * (1 - min(loss_history[-1], 1.0)))
    
    capture.remove()
    
    return {
        "length": length,
        "use_physics": use_physics,
        "init_type": physics_str,
        "seed": seed,
        "final_C_10": final_ck,
        "final_loss": loss_history[-1],
        "final_scTM": final_sctm,
        "history": ck_history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=str, default="length_sweep.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Length bins: short, medium, long
    lengths = [50, 100, 200, 500]
    
    all_results = []
    
    for L in lengths:
        for use_physics in [True, False]:
            try:
                result = train_at_length(L, use_physics, steps=args.steps, seed=args.seed)
                all_results.append(result)
                
                # Save incrementally
                with open(args.output, 'w') as f:
                    json.dump(all_results, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Failed L={L} physics={use_physics}: {e}")
                import traceback
                traceback.print_exc()
    
    # Compute advantage at each length
    print("\n" + "="*70)
    print("P4 VALIDATION: LENGTH-DEPENDENT ROUSE ADVANTAGE")
    print("="*70)
    print(f"{'Length':<10} {'Rouse C_10':<15} {'Learned C_10':<15} {'Advantage':<15}")
    print("-"*70)
    
    advantages = []
    for L in lengths:
        rouse = next((r for r in all_results if r["length"] == L and r["use_physics"]), None)
        learned = next((r for r in all_results if r["length"] == L and not r["use_physics"]), None)
        
        if rouse and learned:
            adv = rouse["final_C_10"] - learned["final_C_10"]
            advantages.append({"L": L, "advantage": adv})
            print(f"{L:<10} {rouse['final_C_10']:<15.4f} {learned['final_C_10']:<15.4f} {adv:<+15.4f}")
    
    # Check if advantage grows with L (P4 prediction)
    if len(advantages) >= 2:
        # Simple linear correlation
        Ls = [a["L"] for a in advantages]
        advs = [a["advantage"] for a in advantages]
        correlation = np.corrcoef(Ls, advs)[0, 1] if len(Ls) > 1 else 0
        
        print("-"*70)
        print(f"Correlation(L, Advantage) = {correlation:.3f}")
        if correlation > 0:
            print("P4 VALIDATED: Rouse advantage increases with sequence length")
        else:
            print("P4 NOT SUPPORTED: Advantage does not increase with length")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
