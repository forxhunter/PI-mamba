"""Paired comparison of register retention between two checkpoints.

``sheet_recovery.py`` noises the same reference structures with the same seed
for every checkpoint, so structure i of one run and structure i of another are
the same protein given the same noise. The retention of the two can therefore
be compared pair by pair, which removes the variation between structures. The
mean alone cannot: a run of 50 structures holds only a few dozen ladders, so
two means can differ by several per cent from counting noise.

Reports, for each feature, the two means, the mean of the paired differences,
its standard error, a two-sided sign test on the pairs, and a paired bootstrap
interval.

Usage:
    python scripts/compare_recovery.py a.json b.json --key t0=0.9
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import List

import numpy as np

FEATURES = ("registered_frac", "n_ladders", "strand_frac")


def sign_test(diff: np.ndarray) -> float:
    """Two-sided p-value for the pairs that changed."""
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def bootstrap_ci(diff: np.ndarray, n_boot: int = 20000,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", help="sheet_recovery.py output for the older checkpoint")
    p.add_argument("candidate", help="output for the newer checkpoint")
    p.add_argument("--key", default="t0=0.9",
                   help="which arm of the recovery report to compare")
    args = p.parse_args()

    a = json.load(open(args.baseline))
    b = json.load(open(args.candidate))
    for name, d in (("baseline", a), ("candidate", b)):
        if args.key not in d:
            raise SystemExit(f"{name} has no arm {args.key!r}; "
                             f"it has {sorted(d)}")

    ref_a = a["reference"]["n_ladders"]
    ref_b = b["reference"]["n_ladders"]
    if abs(ref_a - ref_b) > 1e-6:
        print(f"warning: the two runs used different reference structures "
              f"({ref_a:.3f} against {ref_b:.3f} ladders), so the pairs do "
              f"not line up")

    print(f"arm {args.key}, {Path(args.baseline).name} -> "
          f"{Path(args.candidate).name}\n")
    header = (f"{'feature':<18}{'baseline':>10}{'candidate':>11}"
              f"{'diff':>9}{'sem':>8}{'95% CI':>18}{'better':>8}{'p':>8}")
    print(header)
    for f in FEATURES:
        key = f"{f}_per_structure"
        if key not in a[args.key] or key not in b[args.key]:
            print(f"{f:<18}  no per-structure values in one of the files; "
                  f"rerun sheet_recovery.py")
            continue
        va = np.array(a[args.key][key], dtype=float)
        vb = np.array(b[args.key][key], dtype=float)
        if len(va) != len(vb):
            print(f"{f:<18}  {len(va)} against {len(vb)} structures, "
                  f"cannot pair")
            continue
        d = vb - va
        sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        lo, hi = bootstrap_ci(d)
        better = int((d > 0).sum())
        worse = int((d < 0).sum())
        print(f"{f:<18}{va.mean():>10.3f}{vb.mean():>11.3f}{d.mean():>+9.3f}"
              f"{sem:>8.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>18}"
              f"{f'{better}/{better + worse}':>8}{sign_test(d):>8.3f}")

    print(f"\n{len(np.array(a[args.key]['registered_frac_per_structure']))} "
          f"paired structures. 'better' counts the pairs that improved out of "
          f"those that changed at all.")


if __name__ == "__main__":
    main()
