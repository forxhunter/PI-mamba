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
            
            # Only per-run summaries written by scripts/train.py are used
            if not isinstance(data, dict) or "final_val_loss" not in data:
                continue

            def metric(*keys):
                # A metric that was not measured is NaN, never 0
                for key in keys:
                    if data.get(key) is not None:
                        return float(data[key])
                return float("nan")

            record = {
                "file": os.path.relpath(json_file, input_dir),
                "model": data.get("model", "unknown"),
                "seed": data.get("seed"),
                "lr": data.get("lr"),
                "batch_size": data.get("batch_size"),
                "C_10": metric("final_C_10", "C_10"),
                "scTM": metric("final_scTM", "scTM"),
                "val_loss": metric("final_val_loss", "val_loss"),
                "grad_norm": metric("final_grad_norm", "grad_norm"),
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
        if len(x) < 5:
            ax.set_title(f"{label}: fewer than 5 runs")
            continue
        
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
    if df["scTM"].isna().all():
        print("No run has a measured scTM. Run the sweep with --proteinmpnn_dir "
              "(or scripts/evaluate.py --mode eval) to measure it; "
              "correlations with scTM are skipped.")
    corr_df = compute_predictor_correlations(df)
    print(corr_df.to_string(index=False) if len(corr_df) else "(no predictor has 5 or more runs with scTM)")
    
    # Save table
    table_path = os.path.join(args.output, "correlation_table.csv")
    corr_df.to_csv(table_path, index=False)
    print(f"\nSaved: {table_path}")
    
    # Generate figure
    if len(corr_df):
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
