#!/usr/bin/env python3
"""
Performer Baseline Scaling Benchmark

Compares PI-Mamba and Performer backbones on:
1. Runtime vs sequence length (L=100 to L=2000)
2. Memory usage vs sequence length
3. Scaling exponent fitting

Usage:
    python run_performer_baseline.py --output results/performer_benchmark.csv
"""

import argparse
import csv
import time
import os
import sys

import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backbone_pi_mamba import PIMambaBackbone
from backbone_performer import PerformerBackbone


def measure_runtime(model, length, n_samples=10, n_steps=50, device='cuda', warmup=2):
    """Measure average runtime for generation."""
    model.eval()
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.sample(length=length, n_samples=1, n_steps=n_steps, device=device)
    
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(n_samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = model.sample(length=length, n_samples=1, n_steps=n_steps, device=device)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)


def measure_memory(model, length, n_steps=50, device='cuda'):
    """Measure peak GPU memory usage."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    model.eval()
    with torch.no_grad():
        _ = model.sample(length=length, n_samples=1, n_steps=n_steps, device=device)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    return peak_memory


def fit_scaling_exponent(lengths, times):
    """Fit log(time) = α * log(L) + c."""
    log_L = np.log(lengths)
    log_T = np.log(times)
    
    # Linear regression
    slope, intercept = np.polyfit(log_L, log_T, 1)
    
    # 95% CI (simplified)
    n = len(lengths)
    residuals = log_T - (slope * log_L + intercept)
    mse = np.sum(residuals**2) / (n - 2)
    se_slope = np.sqrt(mse / np.sum((log_L - np.mean(log_L))**2))
    ci_low = slope - 1.96 * se_slope
    ci_high = slope + 1.96 * se_slope
    
    return slope, (ci_low, ci_high)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PI-Mamba vs Performer")
    parser.add_argument("--output", type=str, default="performer_benchmark.csv")
    parser.add_argument("--lengths", type=str, default="100,200,500,1000,2000")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=50)
    args = parser.parse_args()
    
    lengths = [int(x) for x in args.lengths.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Lengths: {lengths}")
    print(f"Samples per length: {args.n_samples}")
    print(f"ODE steps: {args.n_steps}")
    print("=" * 60)
    
    # Create models with same architecture size
    print("Loading PI-Mamba...")
    pi_mamba = PIMambaBackbone(
        d_model=256,
        n_layers=8,
        d_state=64,
        n_groups=8,
        n_structure_layers=4,
        max_length=4096,
    ).to(device)
    
    print("Loading Performer...")
    performer = PerformerBackbone(
        d_model=256,
        n_layers=8,
        n_heads=8,
        n_structure_layers=4,
        max_length=4096,
    ).to(device)
    
    # Parameter count
    pi_mamba_params = sum(p.numel() for p in pi_mamba.parameters())
    performer_params = sum(p.numel() for p in performer.parameters())
    print(f"PI-Mamba params: {pi_mamba_params:,}")
    print(f"Performer params: {performer_params:,}")
    print("=" * 60)
    
    results = []
    pi_mamba_times = []
    performer_times = []
    
    for L in lengths:
        print(f"\n[L={L}]")
        
        # PI-Mamba
        try:
            print(f"  PI-Mamba: ", end="", flush=True)
            pi_time, pi_std = measure_runtime(
                pi_mamba, L, args.n_samples, args.n_steps, device
            )
            pi_mem = measure_memory(pi_mamba, L, args.n_steps, device)
            print(f"{pi_time:.2f}s ± {pi_std:.2f}, mem={pi_mem:.2f}GB")
            pi_mamba_times.append(pi_time)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM")
                pi_time, pi_std, pi_mem = float('nan'), float('nan'), float('nan')
            else:
                raise
        
        # Performer
        try:
            print(f"  Performer: ", end="", flush=True)
            perf_time, perf_std = measure_runtime(
                performer, L, args.n_samples, args.n_steps, device
            )
            perf_mem = measure_memory(performer, L, args.n_steps, device)
            print(f"{perf_time:.2f}s ± {perf_std:.2f}, mem={perf_mem:.2f}GB")
            performer_times.append(perf_time)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM")
                perf_time, perf_std, perf_mem = float('nan'), float('nan'), float('nan')
            else:
                raise
        
        results.append({
            "length": L,
            "pi_mamba_time": pi_time,
            "pi_mamba_std": pi_std,
            "pi_mamba_mem_gb": pi_mem,
            "performer_time": perf_time,
            "performer_std": perf_std,
            "performer_mem_gb": perf_mem,
            "speedup": perf_time / pi_time if not np.isnan(perf_time) and not np.isnan(pi_time) else float('nan'),
        })
    
    # Fit scaling exponents
    print("\n" + "=" * 60)
    print("Scaling Exponents (fitted from log-log plot):")
    
    valid_pi = [(L, t) for L, t in zip(lengths, pi_mamba_times) if not np.isnan(t)]
    valid_perf = [(L, t) for L, t in zip(lengths, performer_times) if not np.isnan(t)]
    
    if len(valid_pi) >= 3:
        ls, ts = zip(*valid_pi)
        exp, ci = fit_scaling_exponent(np.array(ls), np.array(ts))
        print(f"  PI-Mamba:  α = {exp:.2f} (95% CI: [{ci[0]:.2f}, {ci[1]:.2f}])")
    
    if len(valid_perf) >= 3:
        ls, ts = zip(*valid_perf)
        exp, ci = fit_scaling_exponent(np.array(ls), np.array(ts))
        print(f"  Performer: α = {exp:.2f} (95% CI: [{ci[0]:.2f}, {ci[1]:.2f}])")
    
    # Save results
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
