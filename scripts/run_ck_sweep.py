#!/usr/bin/env python3
"""
PI-Mamba Architecture Comparison Sweep

Trains PI-Mamba, a Transformer and a Performer with identical embeddings,
heads, data and optimisation settings, and records the validation loss and
the Rouse mode concentration C_10 of the encoder output for each.
Generates data for P3 (Architecture Ordering).

C_10 is measured on the hidden states of the last encoder layer, on clean
validation structures. scTM is measured only when --proteinmpnn_dir is given.

Usage:
    python scripts/run_ck_sweep.py --max_steps 5000 --seeds 0,1,2 --output_dir sweep_results/architecture

Options that this script does not know are forwarded to scripts/train.py.
The per-run summaries in <output_dir>/<run>/summary.json can be analysed with
    python src/compute_correlation.py --input <output_dir>
"""

import argparse
import json
from pathlib import Path

from sweep_utils import (
    add_common_args, attach_sctm, base_overrides, fmt, mean_std, parse_args, run_training,
)

MODELS = ("pi_mamba", "transformer", "performer")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=str, default=",".join(MODELS))
    add_common_args(parser, output_dir="sweep_results/architecture")
    args = parse_args(parser)

    all_results = []
    for model in args.models.split(","):
        for seed in args.seeds:
            summary = run_training(
                f"{model}_seed{seed}", args.output_dir,
                {**base_overrides(args), "model": model, "seed": seed, "log_ck": True},
                args.train_args, args.force,
            )
            all_results.append(attach_sctm(summary, args, args.output_dir))
            with open(Path(args.output_dir) / "sweep_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    print("\n" + "=" * 78)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 78)
    print(f"{'model':<14}{'params':>12}{'val loss':>20}{'C_10':>18}{'scTM':>14}")
    for model in args.models.split(","):
        runs = [r for r in all_results if r["model"] == model]
        loss = mean_std([r["final_val_loss"] for r in runs])
        ck = mean_std([r["final_C_10"] for r in runs])
        sctm = mean_std([r["final_scTM"] for r in runs])
        print(f"{model:<14}{runs[0]['n_params']:>12,}"
              f"{fmt(loss['mean']):>12} ± {fmt(loss['std']):<6}"
              f"{fmt(ck['mean'], '.3f'):>9} ± {fmt(ck['std'], '.3f'):<6}"
              f"{fmt(sctm['mean'], '.3f'):>10}")
    print(f"\nResults saved to: {Path(args.output_dir) / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
