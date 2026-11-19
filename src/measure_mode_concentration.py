#!/usr/bin/env python3
"""
Rouse Mode Concentration Measurement (Theory Contract Object).

Computes C_k: fraction of Mamba hidden-state variance captured by the first k Rouse modes.

Usage (from the repository root):
    python -m src.measure_mode_concentration --demo
    python -m src.measure_mode_concentration --checkpoint path/to/ckpt.pt \
        --data_dir Distilled_trainset --output metrics.json
"""

import argparse
import json

import torch

from .rouse_physics import compute_rouse_eigenvectors


def compute_mode_concentration(
    hidden_states: torch.Tensor,
    k: int = 10,
) -> dict:
    """
    Compute Rouse Mode Concentration C_k.
    
    Args:
        hidden_states: (B, L, D) Mamba hidden states from a batch.
        k: Number of lowest Rouse modes to consider.
        
    Returns:
        dict with:
            - C_k: Fraction of variance in the k lowest modes. A signal with
              no structure along the chain gives about k / L.
            - total_variance: Sum of the per-mode variances.
            - mode_variances / cumulative_explained: per-mode values for the
              k lowest modes.
    """
    B, L, D = hidden_states.shape
    device = hidden_states.device
    
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
    data_dir: str,
    k: int = 10,
    n_batches: int = 10,
    batch_size: int = 16,
) -> dict:
    """
    Load a model checkpoint and measure C_k on real structures.

    The encoder output of the last layer is taken on clean structures (t = 1).

    Args:
        checkpoint_path: Path to a checkpoint written by scripts/train.py.
        data_dir: Directory with .pdb files.
        k: Number of lowest modes.
        n_batches: Number of batches to average over.
        batch_size: Chains per batch (all chains in a batch have one length).

    Returns:
        Aggregated metrics dict.
    """
    from torch.utils.data import DataLoader

    from .data import CADataset, LengthBatchSampler, collate_same_length
    from .model_io import load_checkpoint

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt = load_checkpoint(checkpoint_path, device=device)
    scale = ckpt["config"].get("coord_scale", 10.0)

    dataset = CADataset(data_dir)
    sampler = LengthBatchSampler(dataset.lengths, batch_size, shuffle=True, seed=0)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_same_length)

    all_results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break
            coords = batch["ca"].to(device) / scale
            t = torch.ones(coords.shape[0], device=device)
            h = model.hidden_states(coords, t)
            all_results.append(compute_mode_concentration(h, k=k))

    n = len(all_results)
    return {
        'C_k': sum(r['C_k'] for r in all_results) / n,
        'k': k,
        'total_variance': sum(r['total_variance'] for r in all_results) / n,
        'top_k_variance': sum(r['top_k_variance'] for r in all_results) / n,
        'n_batches': n,
    }


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
    parser.add_argument("--data_dir", type=str, help="Directory with .pdb files")
    parser.add_argument("--output", type=str, default="mode_concentration.json")
    parser.add_argument("--k", type=int, default=10, help="Number of top modes")
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.checkpoint:
        if not args.data_dir:
            parser.error("--data_dir is required with --checkpoint")
        result = measure_from_checkpoint(args.checkpoint, args.data_dir, k=args.k)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        parser.print_help()
