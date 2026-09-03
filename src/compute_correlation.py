#!/usr/bin/env python3
"""
Compute Correlation: C_k vs scTM

Analyzes sweep results to compute Spearman correlation with bootstrap CIs.
Generates publication-ready figures and tables.

Usage:
    python compute_correlation.py --input sweep_results/ --output figures/
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple


def load_sweep_results(input_dir: str) -> pd.DataFrame:
    """Load all JSON results from a sweep directory."""
    records = []
    
    for json_file in glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Extract relevant metrics
            record = {
                "file": os.path.basename(json_file),
                "model": data.get("model", "unknown"),
                "seed": data.get("seed", 0),
                "lr": data.get("lr", 0),
                "batch_size": data.get("batch_size", 0),
                "C_10": data.get("final_C_10", data.get("C_10", 0)),
                "scTM": data.get("final_scTM", data.get("scTM", 0)),
                "val_loss": data.get("final_val_loss", data.get("val_loss", 0)),
                "grad_norm": data.get("final_grad_norm", data.get("grad_norm", 0)),
            }
            records.append(record)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            
    return pd.DataFrame(records)


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = 1000) -> Tuple[float, float, float]:
    """Compute Spearman correlation with bootstrap 95% CI."""
    rho, _ = stats.spearmanr(x, y)
    
    boot_rhos = []
    for _ in range(n_boot):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boot_rhos.append(r)
    
    ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
    return rho, ci_low, ci_high


def compute_predictor_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations of various predictors with scTM."""
    predictors = ["C_10", "val_loss", "grad_norm"]
    results = []
    
    for pred in predictors:
        if pred not in df.columns or df[pred].isna().all():
            continue
            
        mask = ~(df[pred].isna() | df["scTM"].isna())
        if mask.sum() < 5:
            continue
            
        rho, ci_low, ci_high = bootstrap_spearman(
            df.loc[mask, pred].values,
            df.loc[mask, "scTM"].values
        )
        
        results.append({
            "Predictor": pred,
            "Spearman_rho": rho,
            "CI_low": ci_low,
            "CI_high": ci_high,
        })
    
    return pd.DataFrame(results)


def generate_correlation_figure(df: pd.DataFrame, output_path: str):
    """Generate scatter plot of C_10 vs scTM."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    predictors = [("C_10", "$\\mathcal{C}_{10}$"), 
                  ("val_loss", "Validation Loss"),
                  ("grad_norm", "Gradient Norm")]
    
    for ax, (pred, label) in zip(axes, predictors):
        if pred not in df.columns:
            continue
            
        mask = ~(df[pred].isna() | df["scTM"].isna())
        x = df.loc[mask, pred].values
        y = df.loc[mask, "scTM"].values
        
        ax.scatter(x, y, alpha=0.6, s=50)
        
        # Regression line
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8)
        
        rho, ci_low, ci_high = bootstrap_spearman(x, y)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("scTM", fontsize=12)
        ax.set_title(f"$\\rho$ = {rho:.2f} [{ci_low:.2f}, {ci_high:.2f}]", fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with sweep results")
    parser.add_argument("--output", default=".", help="Output directory for figures")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.input}...")
    df = load_sweep_results(args.input)
    print(f"Loaded {len(df)} runs")
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(df.head())
    
    # Compute correlations
    print("\n=== Predictor Correlations ===")
    corr_df = compute_predictor_correlations(df)
    print(corr_df.to_string(index=False))
    
    # Save table
    table_path = os.path.join(args.output, "correlation_table.csv")
    corr_df.to_csv(table_path, index=False)
    print(f"\nSaved: {table_path}")
    
    # Generate figure
    fig_path = os.path.join(args.output, "correlation_plot.png")
    generate_correlation_figure(df, fig_path)
    
    # Architecture comparison (P3)
    if "model" in df.columns and df["model"].nunique() > 1:
        print("\n=== Architecture Comparison (P3) ===")
        arch_summary = df.groupby("model").agg({
            "C_10": ["mean", "std"],
            "scTM": ["mean", "std"]
        }).round(3)
        print(arch_summary)


if __name__ == "__main__":
    main()
