"""Nearest-neighbour TM-score of generated backbones against the training set.

For every structure in a sample directory the script reports the highest
TM-score against any training structure of the same length. This separates two
reasons a sample set can have a high self-consistency score: the backbones are
designable, or they are close to structures the model was trained on.

Both numbers are needed to compare two sample sets fairly. Running it on a set
produced by a different pipeline shows whether its scTM advantage comes with a
nearest-neighbour advantage as well.

Usage:
    python scripts/novelty_check.py \
        --samples eval_output/paper_512x16/pdbs \
        --samples PImamba_L100 \
        --reference Distilled_trainset

CA coordinates only, CPU, no external binaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import read_ca_coords  # noqa: E402
from src.metrics import tm_score  # noqa: E402


def load_dir(path: Path, limit: int | None = None) -> Dict[int, List[tuple]]:
    """CA traces grouped by length, as (name, tensor) pairs."""
    by_len: Dict[int, List[tuple]] = {}
    files = sorted(path.glob("*.pdb"))
    for f in files:
        try:
            ca = read_ca_coords(f)
        except Exception:  # noqa: BLE001
            continue
        if ca is None or len(ca) < 20:
            continue
        group = by_len.setdefault(len(ca), [])
        if limit is not None and len(group) >= limit:
            continue
        group.append((f.stem, torch.as_tensor(ca, dtype=torch.float32)))
    return by_len


def max_tm(query: torch.Tensor, pool: List[tuple], device: str, chunk: int = 64):
    """Highest TM-score of one structure against a pool, and which one."""
    best, best_name = -1.0, ""
    q = query.to(device)
    for i in range(0, len(pool), chunk):
        block = pool[i:i + chunk]
        ref = torch.stack([c for _, c in block]).to(device)
        scores = tm_score(q.unsqueeze(0).expand(len(block), -1, -1), ref)
        j = int(torch.argmax(scores))
        if float(scores[j]) > best:
            best = float(scores[j])
            best_name = block[j][0]
    return best, best_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", action="append", required=True,
                   help="directory of sample PDB files (repeatable)")
    p.add_argument("--reference", default="Distilled_trainset",
                   help="directory of training structures")
    p.add_argument("--max_reference", type=int, default=None,
                   help="cap on training structures per length (default: all)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    ref = load_dir(Path(args.reference), limit=args.max_reference)
    print(f"reference {args.reference}: "
          + ", ".join(f"L{k}={len(v)}" for k, v in sorted(ref.items()))[:200])

    report: Dict[str, dict] = {}
    rows = []

    for sample_dir in args.samples:
        path = Path(sample_dir)
        name = path.name if path.name != "pdbs" else path.parent.name
        samples = load_dir(path)
        scores: List[float] = []
        print(f"\n{name}: {sum(len(v) for v in samples.values())} structures")
        for length, items in sorted(samples.items()):
            pool = ref.get(length, [])
            if not pool:
                print(f"  L{length}: no reference structures of this length, skipped")
                continue
            for stem, ca in items:
                best, which = max_tm(ca, pool, args.device)
                scores.append(best)
                rows.append({"set": name, "sample": stem, "length": length,
                             "max_tm_to_train": best, "nearest": which})
        if not scores:
            continue
        a = np.array(scores)
        report[name] = {
            "n": len(a),
            "max_tm_mean": float(a.mean()),
            "max_tm_std": float(a.std()),
            "max_tm_median": float(np.median(a)),
            "max_tm_max": float(a.max()),
            "frac_above_0.5": float((a > 0.5).mean()),
            "frac_above_0.7": float((a > 0.7).mean()),
            "frac_above_0.9": float((a > 0.9).mean()),
        }
        r = report[name]
        print(f"  nearest-neighbour TM to training set: "
              f"mean {r['max_tm_mean']:.3f}, median {r['max_tm_median']:.3f}, "
              f"max {r['max_tm_max']:.3f}")
        print(f"  above 0.5 {r['frac_above_0.5']:.2f}, "
              f"0.7 {r['frac_above_0.7']:.2f}, 0.9 {r['frac_above_0.9']:.2f}")

    out = args.output or "novelty_check.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    csv_path = str(Path(out).with_suffix(".csv"))
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "sample", "length",
                                           "max_tm_to_train", "nearest"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} and {csv_path}")


if __name__ == "__main__":
    main()
