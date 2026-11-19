#!/usr/bin/env python3
"""
Length Sweep for P4 Validation

Tests whether the Rouse advantage grows with sequence length L.
Prediction: PI-Mamba advantage over Learned-A baseline increases with L.

For every length in the sweep, a Rouse model and a Learned-A model are
trained on the chains of exactly that length and compared on held-out chains
of the same length. The advantage is the difference in validation
flow-matching loss (Learned-A minus Rouse; positive means Rouse is better).

Usage:
    python scripts/run_length_sweep.py --lengths 100,150,200,250,300 --seeds 0,1,2

Options that this script does not know are forwarded to scripts/train.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sweep_utils import (
    add_common_args, attach_sctm, base_overrides, fmt, mean_std, parse_args, run_training,
    welch_p_value,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", type=str, default="100,150,200,250,300",
                        help="chain lengths to sweep (each needs enough chains in the data set)")
    add_common_args(parser, output_dir="sweep_results/length")
    args = parse_args(parser)
    lengths = [int(v) for v in args.lengths.split(",")]

    all_results = []
    for L in lengths:
        for use_physics, name in ((True, "rouse"), (False, "learned")):
            for seed in args.seeds:
                summary = run_training(
                    f"L{L}_{name}_seed{seed}", args.output_dir,
                    {**base_overrides(args), "use_physics": use_physics, "min_len": L, "max_len": L,
                     "sample_lengths": L, "seed": seed, "log_ck": True},
                    args.train_args, args.force,
                )
                summary.update(length=L, init_type=name)
                all_results.append(attach_sctm(summary, args, args.output_dir))
                with open(Path(args.output_dir) / "length_sweep.json", "w") as f:
                    json.dump(all_results, f, indent=2)

    print("\n" + "=" * 86)
    print("P4 VALIDATION: LENGTH-DEPENDENT ROUSE ADVANTAGE (validation flow-matching loss)")
    print("=" * 86)
    print(f"{'Length':<10}{'Rouse':<14}{'Learned-A':<14}{'Advantage':<14}{'p (Welch)':<12}"
          f"{'Rouse C_10':<12}{'Learned C_10':<12}")
    print("-" * 86)

    advantages = []
    for L in lengths:
        rouse = [r for r in all_results if r["length"] == L and r["init_type"] == "rouse"]
        learned = [r for r in all_results if r["length"] == L and r["init_type"] == "learned"]
        loss_r = mean_std([r["final_val_loss"] for r in rouse])["mean"]
        loss_l = mean_std([r["final_val_loss"] for r in learned])["mean"]
        adv = loss_l - loss_r
        p = welch_p_value([r["final_val_loss"] for r in rouse], [r["final_val_loss"] for r in learned])
        advantages.append({"L": L, "advantage": adv, "p_value": p})
        print(f"{L:<10}{loss_r:<14.4f}{loss_l:<14.4f}{adv:<+14.4f}{fmt(p):<12}"
              f"{fmt(mean_std([r['final_C_10'] for r in rouse])['mean'], '.3f'):<12}"
              f"{fmt(mean_std([r['final_C_10'] for r in learned])['mean'], '.3f'):<12}")

    if len(advantages) >= 3:
        correlation = float(np.corrcoef([a["L"] for a in advantages],
                                        [a["advantage"] for a in advantages])[0, 1])
        print("-" * 86)
        print(f"Pearson correlation between L and the advantage: {correlation:.3f} "
              f"({len(advantages)} lengths)")
        print("A positive correlation is consistent with P4. Check the per-length p-values "
              "before reading anything into it.")

    with open(Path(args.output_dir) / "length_sweep_advantage.json", "w") as f:
        json.dump(advantages, f, indent=2)
    print(f"\nResults saved to: {Path(args.output_dir) / 'length_sweep.json'}")


if __name__ == "__main__":
    main()
