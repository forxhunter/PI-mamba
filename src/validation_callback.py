#!/usr/bin/env python3
"""
Validation Callback for Rouse Mode Concentration.

Integrates with PyTorch training loop to log C_k metrics during validation.

All hidden states accumulated between two resets must have the same sequence
length, because the Rouse basis depends on it.

Usage:
    from src.validation_callback import ModeConcentrationCallback, create_training_hook
    callback = ModeConcentrationCallback(k=10)
    handle = create_training_hook(model, callback)

    # In validation loop (batches of one length):
    for batch in val_loader:
        model(coords, t)          # the hook stores the encoder output

    metrics = callback.compute()
    handle.remove()

scripts/train.py --log_ck logs the same quantity without the hook, through
``model.hidden_states``.
"""

from typing import Dict

import torch

from .measure_mode_concentration import compute_mode_concentration


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
    Create a forward hook that captures the output of the last encoder layer.

    Args:
        model: a ``FlowBackbone`` (PI-Mamba, Performer or Transformer)
        callback: ModeConcentrationCallback instance

    Returns:
        hook handle; call ``.remove()`` on it when done
    """
    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
            callback.accumulate(hidden)

    for name in ("pi_mamba_layers", "attn_layers"):
        if hasattr(model, name):
            return getattr(model, name)[-1].register_forward_hook(hook_fn)
    raise AttributeError("model has neither pi_mamba_layers nor attn_layers")
