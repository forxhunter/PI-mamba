#!/usr/bin/env python3
"""
Validation Callback for Rouse Mode Concentration.

Integrates with PyTorch training loop to log C_k metrics during validation.

Usage:
    from validation_callback import ModeConcentrationCallback
    callback = ModeConcentrationCallback(k=10)
    
    # In validation loop:
    for batch in val_loader:
        hidden_states = model.get_hidden_states(batch)
        callback.accumulate(hidden_states)
    
    metrics = callback.compute()
    wandb.log(metrics)
"""

import torch
import numpy as np
from typing import Dict, Optional
from measure_mode_concentration import compute_mode_concentration, compute_rouse_eigenvectors


class ModeConcentrationCallback:
    """Accumulates hidden states and computes C_k at validation time."""
    
    def __init__(self, k: int = 10, max_samples: int = 1000):
        self.k = k
        self.max_samples = max_samples
        self.hidden_states_buffer = []
        self.total_samples = 0
        
    def reset(self):
        """Reset buffer for new epoch."""
        self.hidden_states_buffer = []
        self.total_samples = 0
        
    def accumulate(self, hidden_states: torch.Tensor):
        """
        Add hidden states to buffer.
        
        Args:
            hidden_states: (B, L, D) tensor of Mamba hidden states
        """
        if self.total_samples >= self.max_samples:
            return
            
        # Detach and move to CPU to avoid GPU memory issues
        h = hidden_states.detach().cpu()
        self.hidden_states_buffer.append(h)
        self.total_samples += h.shape[0]
        
    def compute(self) -> Dict[str, float]:
        """
        Compute C_k and related metrics from accumulated states.
        
        Returns:
            Dict with C_k, cumulative explained variance, etc.
        """
        if not self.hidden_states_buffer:
            return {"C_k": 0.0, "warning": "no_data"}
            
        # Concatenate all hidden states
        all_hidden = torch.cat(self.hidden_states_buffer, dim=0)
        
        # Compute mode concentration
        result = compute_mode_concentration(all_hidden, k=self.k)
        
        return {
            f"C_{self.k}": result["C_k"],
            "mode_total_variance": result["total_variance"],
            "mode_cumulative_10": result["cumulative_explained"][-1] if result["cumulative_explained"] else 0.0,
        }


def create_training_hook(model, callback: ModeConcentrationCallback):
    """
    Create a forward hook to capture hidden states during forward pass.
    
    Args:
        model: PI-Mamba model with a 'mamba_layers' attribute
        callback: ModeConcentrationCallback instance
        
    Returns:
        hook_handle for removal
    """
    def hook_fn(module, input, output):
        # Assuming output is (B, L, D) hidden states
        if isinstance(output, torch.Tensor) and len(output.shape) == 3:
            callback.accumulate(output)
    
    # Register on the last Mamba layer
    if hasattr(model, 'mamba_layers'):
        last_layer = model.mamba_layers[-1]
        return last_layer.register_forward_hook(hook_fn)
    else:
        print("Warning: Could not find mamba_layers attribute")
        return None


# Integration example for train.py
INTEGRATION_TEMPLATE = '''
# Add to train.py validation loop:

from validation_callback import ModeConcentrationCallback

# Initialize
ck_callback = ModeConcentrationCallback(k=10)

def validate(model, val_loader):
    model.eval()
    ck_callback.reset()
    
    with torch.no_grad():
        for batch in val_loader:
            # Forward pass
            output = model(batch)
            
            # Get hidden states (model-specific)
            hidden_states = model.last_hidden_state  # or similar accessor
            ck_callback.accumulate(hidden_states)
            
            # ... compute scTM, loss, etc.
    
    # Log C_k
    ck_metrics = ck_callback.compute()
    wandb.log({
        "val/C_10": ck_metrics["C_10"],
        "val/scTM": mean_sctm,
        "val/loss": mean_loss,
    })
    
    return ck_metrics
'''

if __name__ == "__main__":
    print("Integration template:")
    print(INTEGRATION_TEMPLATE)
