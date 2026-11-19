#!/usr/bin/env python3
"""
Counterfactual Experiments for PI-Mamba Paper
==============================================

This script runs the three counterfactual experiments for Section S16:
  - S16.1: Scrambled Rouse spectrum control
  - S16.2: Rouse interpolation curve
  - S16.3: Short-length counterfactual

Each experiment trains real models with scripts/train.py on the training set
and reports measured quantities: validation flow-matching loss, Rouse mode
concentration C_10, sample structure metrics, and the learned relaxation time
tau per secondary-structure class. scTM is measured only when
--proteinmpnn_dir is given; otherwise it is reported as null.

Usage:
    python scripts/run_counterfactual_experiments.py --experiment scrambled
    python scripts/run_counterfactual_experiments.py --experiment interpolation
    python scripts/run_counterfactual_experiments.py --experiment short_length
    python scripts/run_counterfactual_experiments.py --experiment all --seeds 0,1,2

Options that this script does not know are forwarded to scripts/train.py,
for example ``--batch_size 16 --lr 5e-4``.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from sweep_utils import (
    add_common_args, attach_sctm, base_overrides, fmt, mean_std, parse_args,
    run_training, tau_by_secondary_structure, welch_p_value,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

METRICS = ("final_val_loss", "final_C_10", "final_scTM")


def train_variant(name: str, args, overrides: dict) -> dict:
    """Train one model variant and attach the post-hoc measurements."""
    summary = run_training(
        name, args.output_dir, {**base_overrides(args), "log_ck": True, **overrides},
        args.train_args, args.force,
    )
    if summary.get("model") == "pi_mamba":
        summary.update(tau_by_secondary_structure(summary["checkpoint"], args.data_dir))
    return attach_sctm(summary, args, args.output_dir)


def aggregate(runs: list) -> dict:
    """Mean and standard deviation over seeds of the reported quantities."""
    keys = METRICS + ("tau_helix", "tau_strand", "tau_coil")
    return {key: mean_std([r.get(key) for r in runs]) for key in keys}


def save(results: dict, args, filename: str) -> None:
    path = Path(args.output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {path}")


# ============================================================================
# EXPERIMENT S16.1: Scrambled Rouse Spectrum Control
# ============================================================================

def run_scrambled_experiment(args) -> dict:
    """
    Compares PI-Mamba with the Rouse spectrum (the d_state slowest modes)
    against PI-Mamba whose state channels take modes drawn at random from the
    whole Rouse spectrum.

    A pure reordering of the rates among the state channels is not a control:
    it is a symmetry of the layer, because the input and output projections of
    the state are learned. The scrambled spectrum therefore changes which
    modes are used, and with them the time scales of the layer.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT S16.1: Scrambled Rouse Spectrum Control")
    logger.info("=" * 60)

    results = {"experiment": "scrambled_rouse", "date": datetime.now().isoformat(), "models": {}}
    for spectrum in ("rouse", "scrambled"):
        runs = [
            train_variant(f"scrambled/{spectrum}_seed{seed}", args, {"spectrum": spectrum, "seed": seed})
            for seed in args.seeds
        ]
        results["models"][spectrum] = {"runs": runs, **aggregate(runs)}

    rouse, scrambled = results["models"]["rouse"], results["models"]["scrambled"]
    results["p_value_val_loss"] = welch_p_value(
        [r["final_val_loss"] for r in rouse["runs"]], [r["final_val_loss"] for r in scrambled["runs"]]
    )
    save(results, args, "scrambled_rouse_results.json")

    logger.info("Summary:")
    for name, data in results["models"].items():
        logger.info(
            f"  {name}: val_loss={fmt(data['final_val_loss']['mean'])}, "
            f"C_10={fmt(data['final_C_10']['mean'], '.3f')}, "
            f"tau helix/strand/coil={fmt(data['tau_helix']['mean'], '.3f')}/"
            f"{fmt(data['tau_strand']['mean'], '.3f')}/{fmt(data['tau_coil']['mean'], '.3f')}, "
            f"scTM={fmt(data['final_scTM']['mean'], '.3f')}"
        )
    logger.info(f"  Welch p-value (val loss): {fmt(results['p_value_val_loss'])}")
    return results


# ============================================================================
# EXPERIMENT S16.2: Rouse Interpolation Curve
# ============================================================================

def run_interpolation_experiment(args) -> dict:
    """
    Trains models whose spectrum is alpha * Rouse + (1 - alpha) * random for
    alpha in {0, 0.25, 0.5, 0.75, 1}. The random rates cover the same range as
    the Rouse rates at each chain length, so this curve tests the spacing of
    the spectrum at a fixed range of time scales.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT S16.2: Rouse Interpolation Curve")
    logger.info("=" * 60)

    results = {"experiment": "rouse_interpolation", "date": datetime.now().isoformat(), "alphas": {}}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        runs = [
            train_variant(
                f"interpolation/alpha{alpha}_seed{seed}", args,
                {"spectrum": "interpolated", "spectrum_alpha": alpha, "seed": seed},
            )
            for seed in args.seeds
        ]
        results["alphas"][str(alpha)] = {"runs": runs, **aggregate(runs)}
        data = results["alphas"][str(alpha)]
        logger.info(f"  alpha={alpha}: val_loss={fmt(data['final_val_loss']['mean'])}, "
                    f"scTM={fmt(data['final_scTM']['mean'], '.3f')}")

    save(results, args, "interpolation_results.json")
    return results


# ============================================================================
# EXPERIMENT S16.3: Short-Length Counterfactual
# ============================================================================

def run_short_length_experiment(args) -> dict:
    """
    Compares Rouse against Learned-A when training and validating on chains of
    different length ranges. The learned rates start uniformly distributed
    over the range of the Rouse rates of a 128-residue chain.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT S16.3: Short-Length Counterfactual")
    logger.info("=" * 60)

    bins = [tuple(int(v) for v in b.split("-")) for b in args.length_bins.split(",")]
    results = {"experiment": "short_length", "date": datetime.now().isoformat(), "length_bins": {}}

    for l_min, l_max in bins:
        bin_key = f"{l_min}-{l_max}"
        entry = {}
        for use_physics, name in ((True, "rouse"), (False, "learned_a")):
            runs = [
                train_variant(
                    f"short_length/L{bin_key}_{name}_seed{seed}", args,
                    {"use_physics": use_physics, "min_len": l_min, "max_len": l_max, "seed": seed,
                     # sample at a length inside the bin
                     "sample_lengths": l_max},
                )
                for seed in args.seeds
            ]
            entry[name] = {"runs": runs, **aggregate(runs)}
        entry["p_value_val_loss"] = welch_p_value(
            [r["final_val_loss"] for r in entry["rouse"]["runs"]],
            [r["final_val_loss"] for r in entry["learned_a"]["runs"]],
        )
        results["length_bins"][bin_key] = entry
        logger.info(
            f"  L in [{l_min},{l_max}]: val_loss Rouse={fmt(entry['rouse']['final_val_loss']['mean'])}, "
            f"Learned-A={fmt(entry['learned_a']['final_val_loss']['mean'])}, "
            f"p={fmt(entry['p_value_val_loss'])}"
        )

    save(results, args, "short_length_results.json")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run counterfactual experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["scrambled", "interpolation", "short_length", "all"],
                        help="Which experiment to run")
    parser.add_argument("--length_bins", type=str, default="50-100,125-200,250-300",
                        help="comma-separated min-max length ranges for short_length")
    add_common_args(parser, output_dir="counterfactual_results")
    args = parse_args(parser)

    if args.experiment in ("all", "scrambled"):
        run_scrambled_experiment(args)
    if args.experiment in ("all", "interpolation"):
        run_interpolation_experiment(args)
    if args.experiment in ("all", "short_length"):
        run_short_length_experiment(args)

    logger.info("=" * 60)
    logger.info("All experiments completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
