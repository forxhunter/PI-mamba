"""Diagnose what limits scTM of generated backbones.

Two questions are answered from files that already exist:

1. Within one set of samples, which structural features separate a high-scTM
   backbone from a low-scTM one?  (Spearman correlation of each per-sample
   feature with its scTM from ``info.csv``.)
2. How do the feature distributions of two sample sets differ?  Comparing our
   samples with a reference set that scores higher under the same scTM
   pipeline shows which direction to push.

Usage:
    python scripts/diagnose_sctm.py \
        --eval_dir eval_output/paper_512x16 \
        --compare PImamba_L100 --compare Distilled_trainset

Only CA coordinates are used, so the script runs on CPU in seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import read_ca_coords  # noqa: E402
from src.metrics import (  # noqa: E402
    assign_secondary_structure,
    bond_lengths,
    clash_fraction,
    radius_of_gyration,
)


# ---------------------------------------------------------------- features


def contact_order(ca: torch.Tensor, cutoff: float = 8.0) -> float:
    """Mean sequence separation of residue pairs in contact, divided by L.

    Low values mean local, helical contacts; high values mean the chain folds
    back on itself.  Natural domains of 100 residues sit around 0.1 to 0.2.
    """
    d = torch.cdist(ca, ca)
    L = ca.shape[0]
    idx = torch.arange(L)
    sep = (idx[:, None] - idx[None, :]).abs().float()
    mask = (d < cutoff) & (sep >= 3)
    if mask.sum() == 0:
        return 0.0
    return float((sep[mask].mean() / L))


def n_contacts(ca: torch.Tensor, cutoff: float = 8.0) -> float:
    """Long-range contacts per residue (separation >= 12)."""
    d = torch.cdist(ca, ca)
    L = ca.shape[0]
    idx = torch.arange(L)
    sep = (idx[:, None] - idx[None, :]).abs()
    mask = (d < cutoff) & (sep >= 12)
    return float(mask.sum()) / (2.0 * L)


def ss_segments(ss: torch.Tensor) -> Dict[str, float]:
    """Number and mean length of helix and strand segments."""
    labels = ss.tolist()
    out: Dict[str, float] = {}
    for name, code in (("helix", 0), ("strand", 1)):
        runs: List[int] = []
        cur = 0
        for v in labels:
            if v == code:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        out[f"n_{name}_seg"] = float(len(runs))
        out[f"mean_{name}_len"] = float(np.mean(runs)) if runs else 0.0
    return out


def sample_features(ca: torch.Tensor) -> Dict[str, float]:
    """Per-structure features, all computed from the CA trace alone."""
    x = ca - ca.mean(dim=0, keepdim=True)
    b = bond_lengths(x.unsqueeze(0))[0]
    ss = assign_secondary_structure(x.unsqueeze(0))[0]
    L = x.shape[0]
    rg = float(radius_of_gyration(x.unsqueeze(0))[0])

    feats = {
        "length": float(L),
        "rg": rg,
        "rg_ratio": rg / (2.2 * L ** 0.38),
        "bond_mean": float(b.mean()),
        "bond_std": float(b.std()),
        "bond_max_dev": float((b - 3.8).abs().max()),
        "clash_frac": float(clash_fraction(x.unsqueeze(0))[0]),
        "helix_frac": float((ss == 0).float().mean()),
        "strand_frac": float((ss == 1).float().mean()),
        "coil_frac": float((ss == 2).float().mean()),
        "contact_order": contact_order(x),
        "long_contacts": n_contacts(x),
    }
    feats.update(ss_segments(ss))
    return feats


# ---------------------------------------------------------------- statistics


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without a SciPy dependency."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def collect(pdb_dir: Path, limit: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """Features of every PDB file in a directory, keyed by file stem."""
    out: Dict[str, Dict[str, float]] = {}
    files = sorted(pdb_dir.glob("*.pdb"))
    if limit:
        files = files[:limit]
    for f in files:
        try:
            ca = read_ca_coords(f)
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {f.name}: {exc}")
            continue
        if ca is None or len(ca) < 20:
            continue
        out[f.stem] = sample_features(torch.as_tensor(ca, dtype=torch.float32))
    return out


def summarise(feats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted({k for v in feats.values() for k in v})
    return {
        k: {
            "mean": float(np.mean([v[k] for v in feats.values()])),
            "std": float(np.std([v[k] for v in feats.values()])),
        }
        for k in keys
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True, help="directory with pdbs/ and info.csv")
    p.add_argument(
        "--compare",
        action="append",
        default=[],
        help="another directory of PDB files to compare against (repeatable)",
    )
    p.add_argument("--max_compare", type=int, default=200)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    print(f"reading {eval_dir}/pdbs")
    ours = collect(eval_dir / "pdbs")
    print(f"  {len(ours)} structures")

    # --- scTM per sample, if the eval has been run
    sctm: Dict[str, float] = {}
    info = eval_dir / "info.csv"
    if info.exists():
        with open(info) as fh:
            for row in csv.DictReader(fh):
                if row.get("scTM"):
                    sctm[row["domain"]] = float(row["scTM"])

    report: Dict[str, object] = {"n_samples": len(ours)}

    if sctm:
        common = [k for k in ours if k in sctm]
        print(f"\nscTM available for {len(common)} of {len(ours)} structures")
        y = np.array([sctm[k] for k in common])
        feature_names = sorted(ours[common[0]])
        corr = {}
        for name in feature_names:
            x = np.array([ours[k][name] for k in common])
            if x.std() == 0:
                continue
            corr[name] = spearman(x, y)
        report["spearman_with_sctm"] = corr
        print("\nSpearman correlation of each feature with scTM:")
        for name, r in sorted(corr.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {name:18s} {r:+.3f}")

        # top vs bottom quartile
        order = np.argsort(y)
        n_q = max(1, len(common) // 4)
        low = [common[i] for i in order[:n_q]]
        high = [common[i] for i in order[-n_q:]]
        print(f"\nbottom {n_q} vs top {n_q} by scTM "
              f"(scTM {y[order[:n_q]].mean():.3f} vs {y[order[-n_q:]].mean():.3f}):")
        quart = {}
        for name in feature_names:
            lo = float(np.mean([ours[k][name] for k in low]))
            hi = float(np.mean([ours[k][name] for k in high]))
            quart[name] = {"low_sctm": lo, "high_sctm": hi}
            print(f"  {name:18s} {lo:8.3f} -> {hi:8.3f}")
        report["quartiles"] = quart

    report["summary"] = {"ours": summarise(ours)}

    for other in args.compare:
        path = Path(other)
        print(f"\nreading {path}")
        feats = collect(path, limit=args.max_compare)
        print(f"  {len(feats)} structures")
        if not feats:
            continue
        report["summary"][path.name] = summarise(feats)
        print(f"\n{'feature':18s} {'ours':>18s} {path.name[:18]:>18s}")
        s_ours = report["summary"]["ours"]
        s_other = report["summary"][path.name]
        for name in sorted(s_ours):
            if name not in s_other:
                continue
            a, b = s_ours[name], s_other[name]
            print(f"{name:18s} {a['mean']:8.3f}+-{a['std']:<7.3f} "
                  f"{b['mean']:8.3f}+-{b['std']:<7.3f}")

    out = args.output or str(eval_dir / "diagnosis.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
