#!/usr/bin/env python3
"""
Rouse Mode Concentration Measurement (Theory Contract Object).

Computes C_k: fraction of Mamba hidden-state variance captured by the first k Rouse modes.

Usage:
    python measure_mode_concentration.py --checkpoint path/to/ckpt.pt --output metrics.json
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path

# Import from v12 package
from rouse_physics import RouseTransform, compute_rouse_eigenvectors


def compute_mode_concentration(
    hidden_states: torch.Tensor,
    k: int = 10,
    rouse_transform: RouseTransform = None,
) -> dict:
    """
    Compute Rouse Mode Concentration C_k.
    
    Args:
        hidden_states: (B, L, D) Mamba hidden states from a batch.
        k: Number of top modes to consider.
        rouse_transform: Precomputed RouseTransform object.
        
    Returns:
        dict with:
            - C_k: Fraction of variance in top-k modes.
            - eigenvalues: Top-k eigenvalues of projected covariance.
            - total_variance: Tr(Sigma_h).
    """
    B, L, D = hidden_states.shape
    device = hidden_states.device
    
    if rouse_transform is None:
        rouse_transform = RouseTransform(max_length=L)
    
    # Get Rouse eigenvectors V: (L, L)
    V = compute_rouse_eigenvectors(L, device=device, normalize=True)
    
    # Reshape hidden states: (B*D, L)
    h_flat = hidden_states.permute(0, 2, 1).reshape(-1, L)  # (B*D, L)
    
    # Project onto Rouse modes: h_mode = h @ V
    h_mode = h_flat @ V  # (B*D, L) modes
    
    # Compute variance per mode: var(h_mode[:, p]) for each p
    mode_variance = h_mode.var(dim=0)  # (L,)
    
    # Total variance
    total_var = mode_variance.sum().item()
    
    # Top-k variance
    top_k_var = mode_variance[:k].sum().item()
    
    # C_k
    C_k = top_k_var / (total_var + 1e-8)
    
    # Cumulative explained variance curve
    cumulative = torch.cumsum(mode_variance, dim=0) / (total_var + 1e-8)
    
    return {
        "C_k": float(C_k),
        "k": k,
        "total_variance": float(total_var),
        "top_k_variance": float(top_k_var),
        "mode_variances": mode_variance[:k].cpu().tolist(),
        "cumulative_explained": cumulative[:k].cpu().tolist(),
    }


def measure_from_checkpoint(
    checkpoint_path: str,
    model_class: str = "PIMambaBackbone",
    data_loader = None,
    k: int = 10,
    n_batches: int = 10,
) -> dict:
    """
    Load a model checkpoint and measure C_k on validation data.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        model_class: Name of model class to instantiate.
        data_loader: DataLoader for validation data.
        k: Number of top modes.
        n_batches: Number of batches to average over.

    Returns:
        Aggregated metrics dict.
    """
    import torch
    from pathlib import Path
    import sys

    # Add src to path
    sys.path.append(str(Path(__file__).parent))

    # Import model
    if model_class == "PIMambaBackbone":
        from backbone_pi_mamba import PIMambaBackbone as ModelClass
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Initialize model
    model = ModelClass(
        d_model=checkpoint.get('d_model', 256),
        n_layers=checkpoint.get('n_layers', 12),
        d_state=checkpoint.get('d_state', 64),
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # If no dataloader provided, create a simple one
    if data_loader is None:
        print("Warning: No dataloader provided. Using random data for demonstration.")
        # Create dummy data
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return n_batches * 8
            def __getitem__(self, idx):
                return torch.randn(100, 3)

        data_loader = torch.utils.data.DataLoader(
            DummyDataset(),
            batch_size=8,
            shuffle=False
        )

    # Collect hidden states
    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= n_batches:
                break

            # Get batch data
            if isinstance(batch, (list, tuple)):
                coords = batch[0].to(device)
            else:
                coords = batch.to(device)

            B, L, _ = coords.shape

            # Forward pass to get hidden states
            # We need to access intermediate hidden states from Mamba layers
            # For now, we'll use the final hidden state as a proxy
            t = torch.zeros(B, device=device)
            t_emb = model.time_embed(t)

            h = model.input_proj(coords)
            h = h + model.pos_encoding[:, :L, :]

            # Pass through PI-Mamba layers and collect hidden states
            for mamba_layer, adaptive_norm in zip(model.pi_mamba_layers, model.adaptive_norms):
                h = adaptive_norm(h, t_emb)
                h, _ = mamba_layer(h, return_physics=False)

            # Compute mode concentration on hidden states
            result = compute_mode_concentration(h, k=k)
            all_results.append(result)

    # Aggregate results
    aggregated = {
        'C_k': sum(r['C_k'] for r in all_results) / len(all_results),
        'k': k,
        'total_variance': sum(r['total_variance'] for r in all_results) / len(all_results),
        'top_k_variance': sum(r['top_k_variance'] for r in all_results) / len(all_results),
        'n_batches': len(all_results),
    }

    return aggregated


def demo():
    """Demo with random hidden states."""
    print("=== Rouse Mode Concentration Demo ===")
    
    B, L, D = 8, 100, 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simulate hidden states with Rouse structure (low modes dominate)
    V = compute_rouse_eigenvectors(L, device=device)
    
    # Create hidden states where low modes have higher weight
    mode_weights = 1.0 / (torch.arange(L, device=device, dtype=torch.float32) + 1)
    mode_weights = mode_weights.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
    
    random_modes = torch.randn(B, L, D, device=device)
    h_structured = V @ (random_modes * mode_weights)  # Weighted inverse transform
    
    # Compute C_k
    result = compute_mode_concentration(h_structured, k=10)
    
    print(f"C_10 (structured): {result['C_k']:.4f}")
    print(f"Cumulative explained (modes 1-10): {result['cumulative_explained']}")
    
    # Compare to random (unstructured)
    h_random = torch.randn(B, L, D, device=device)
    result_random = compute_mode_concentration(h_random, k=10)
    
    print(f"\nC_10 (random): {result_random['C_k']:.4f}")
    print(f"Cumulative explained (modes 1-10): {result_random['cumulative_explained']}")
    
    print(f"\n==> Structured has {result['C_k'] / result_random['C_k']:.1f}x higher concentration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="mode_concentration.json")
    parser.add_argument("--k", type=int, default=10, help="Number of top modes")
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.checkpoint:
        result = measure_from_checkpoint(args.checkpoint, k=args.k)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        parser.print_help()
