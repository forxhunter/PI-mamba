#!/usr/bin/env python3
"""Hyper-parameter sweep for the loss weights lambda_FAPE and lambda_bond.

For every pair of weights in the grid a model is trained with
scripts/train.py, and the script records the validation flow-matching loss
and the structure metrics of samples (bond-length spread before projection,
clashes, secondary-structure content). scTM is added when --proteinmpnn_dir
is given.

The validation flow-matching loss does not include the auxiliary terms, so it
is comparable across the grid.

Usage:
    python scripts/loss_weight_sweep.py --max_steps 5000 --out sweep_results.csv

Options that this script does not know are forwarded to scripts/train.py.
"""

import argparse
import csv
import itertools
from pathlib import Path

from sweep_utils import add_common_args, attach_sctm, base_overrides, fmt, parse_args, run_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loss-weight sweep for PI-Mamba")
    parser.add_argument("--lambda_fape_values", type=str, default="0,0.5,1.0,2.0")
    parser.add_argument("--lambda_bond_values", type=str, default="0,0.5,1.0,2.0")
    parser.add_argument("--out", type=str, default=None, help="CSV output file")
    add_common_args(parser, output_dir="sweep_results/loss_weights")
    args = parse_args(parser)

    fape_values = [float(v) for v in args.lambda_fape_values.split(",")]
    bond_values = [float(v) for v in args.lambda_bond_values.split(",")]

    results = []
    for lam_f, lam_b in itertools.product(fape_values, bond_values):
        for seed in args.seeds:
            summary = run_training(
                f"fape{lam_f}_bond{lam_b}_seed{seed}", args.output_dir,
                {**base_overrides(args), "lambda_fape": lam_f, "lambda_bond": lam_b, "seed": seed},
                args.train_args, args.force,
            )
            summary = attach_sctm(summary, args, args.output_dir)
            row = {
                "lambda_fape": lam_f,
                "lambda_bond": lam_b,
                "seed": seed,
                "validation_fm_loss": summary["final_val_loss"],
                "validation_scTM": summary["final_scTM"],
            }
            for length, metrics in summary["samples"].items():
                for key in ("raw_bond_std", "clash_frac", "helix_frac", "strand_frac"):
                    row[f"{length}_{key}"] = metrics[key]
            results.append(row)
            print(f"lambda_FAPE={lam_f}, lambda_bond={lam_b}, seed={seed} => "
                  f"val FM loss={row['validation_fm_loss']:.4f}, scTM={fmt(row['validation_scTM'], '.3f')}")

    out_path = Path(args.out or Path(args.output_dir) / "sweep_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Sweep results saved to {out_path}")
