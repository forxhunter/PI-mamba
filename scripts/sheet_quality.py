"""Is the beta content of generated backbones made of real sheets?

The secondary-structure assignment in ``src/metrics.py`` labels a residue a
strand from the local CA geometry alone. A residue can carry that label without
being part of a sheet: an extended stretch of chain with no partner strand gets
the same label. Only paired, in-register strands fold back under a
self-consistency evaluation.

This script measures pairing directly from the CA trace:

  paired_strand_frac  fraction of strand residues with a partner strand residue
                      at an inter-strand CA-CA distance, at least 4 positions
                      away in sequence
  registered_frac     fraction of strand residues inside a ladder of at least
                      ``--min_ladder`` consecutive pairs whose partners are also
                      consecutive (parallel or antiparallel register)
  n_ladders           ladders per structure
  mean_ladder_len     mean number of pairs per ladder
  parallel_frac       share of ladders in parallel register

Run it on generated samples and on the training set. A set whose strand label
fraction matches the training set but whose registered fraction does not has
extended coil, not sheets.

Usage:
    python scripts/sheet_quality.py \
        --samples eval_output/paper_512x16/pdbs \
        --samples PImamba_L100 \
        --samples Distilled_trainset
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import read_ca_coords  # noqa: E402
from src.metrics import STRAND, assign_secondary_structure  # noqa: E402

# Inter-strand CA-CA distance of a hydrogen-bonded pair in a beta sheet.
# Antiparallel pairs sit near 4.9 A and parallel pairs near 4.8 A; the window
# is wide enough for both and narrow enough to exclude a helix (i to i+3 is
# 5.0 A, which the minimum sequence separation removes) and a random contact.
PAIR_MIN = 4.0
PAIR_MAX = 5.8
MIN_SEP = 4


def find_pairs(ca: torch.Tensor, ss: torch.Tensor) -> Dict[int, int]:
    """Closest in-window partner of every strand residue, by index."""
    strand_idx = torch.nonzero(ss == STRAND).flatten().tolist()
    if len(strand_idx) < 2:
        return {}
    d = torch.cdist(ca, ca)
    pairs: Dict[int, int] = {}
    for i in strand_idx:
        best, best_d = None, 1e9
        for j in strand_idx:
            if abs(i - j) < MIN_SEP:
                continue
            dij = float(d[i, j])
            if PAIR_MIN <= dij <= PAIR_MAX and dij < best_d:
                best, best_d = j, dij
        if best is not None:
            pairs[i] = best
    return pairs


def find_ladders(pairs: Dict[int, int], min_ladder: int) -> List[Tuple[List[int], int]]:
    """Runs of consecutive residues whose partners are also consecutive.

    Returns a list of (residue indices, direction), direction -1 for
    antiparallel and +1 for parallel register.
    """
    ladders: List[Tuple[List[int], int]] = []
    used: set = set()
    for i in sorted(pairs):
        if i in used:
            continue
        for direction in (-1, 1):
            run = [i]
            k = i
            while (k + 1) in pairs and pairs[k + 1] == pairs[k] + direction:
                k += 1
                run.append(k)
            if len(run) >= min_ladder:
                ladders.append((run, direction))
                used.update(run)
                break
    return ladders


def sheet_features(ca: torch.Tensor, min_ladder: int) -> Dict[str, float]:
    x = ca - ca.mean(dim=0, keepdim=True)
    ss = assign_secondary_structure(x.unsqueeze(0))[0]
    n_strand = int((ss == STRAND).sum())
    L = x.shape[0]
    out = {
        "length": float(L),
        "strand_frac": n_strand / L,
        "paired_strand_frac": 0.0,
        "registered_frac": 0.0,
        "n_ladders": 0.0,
        "mean_ladder_len": 0.0,
        "parallel_frac": float("nan"),
    }
    if n_strand < 2 * min_ladder:
        return out

    pairs = find_pairs(x, ss)
    out["paired_strand_frac"] = len(pairs) / n_strand

    ladders = find_ladders(pairs, min_ladder)
    in_ladder = sum(len(r) for r, _ in ladders)
    out["registered_frac"] = in_ladder / n_strand
    out["n_ladders"] = float(len(ladders))
    if ladders:
        out["mean_ladder_len"] = float(np.mean([len(r) for r, _ in ladders]))
        out["parallel_frac"] = float(np.mean([d > 0 for _, d in ladders]))
    return out


def collect(path: Path, min_ladder: int, limit: int | None,
            length: int | None) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for f in sorted(path.glob("*.pdb")):
        if limit is not None and len(out) >= limit:
            break
        try:
            ca = read_ca_coords(f)
        except Exception:  # noqa: BLE001
            continue
        if ca is None or len(ca) < 20:
            continue
        if length is not None and len(ca) != length:
            continue
        out[f.stem] = sheet_features(
            torch.as_tensor(ca, dtype=torch.float32), min_ladder)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", action="append", required=True)
    p.add_argument("--min_ladder", type=int, default=3)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--length", type=int, default=None,
                   help="keep only structures of this length")
    p.add_argument("--sctm_from", default=None,
                   help="info.csv to correlate the sheet features with scTM")
    p.add_argument("--output", default="sheet_quality.json")
    args = p.parse_args()

    report: Dict[str, dict] = {}
    rows = []
    keys = ["strand_frac", "paired_strand_frac", "registered_frac",
            "n_ladders", "mean_ladder_len", "parallel_frac"]

    for sample_dir in args.samples:
        path = Path(sample_dir)
        name = path.name if path.name != "pdbs" else path.parent.name
        feats = collect(path, args.min_ladder, args.limit, args.length)
        if not feats:
            print(f"{name}: no structures")
            continue
        report[name] = {"n": len(feats)}
        for k in keys:
            vals = np.array([v[k] for v in feats.values()], dtype=float)
            vals = vals[~np.isnan(vals)]
            report[name][k] = float(vals.mean()) if len(vals) else float("nan")
            report[name][k + "_std"] = float(vals.std()) if len(vals) else float("nan")
        for stem, v in feats.items():
            rows.append({"set": name, "sample": stem, **v})

    if report:
        names = list(report)
        print(f"{'feature':20s}" + "".join(f"{n[:16]:>18s}" for n in names))
        print(f"{'n':20s}" + "".join(f"{report[n]['n']:>18d}" for n in names))
        for k in keys:
            print(f"{k:20s}" + "".join(
                f"{report[n][k]:>11.3f}+-{report[n][k+'_std']:<6.3f}" for n in names))

    if args.sctm_from:
        sctm = {}
        with open(args.sctm_from) as fh:
            for row in csv.DictReader(fh):
                if row.get("scTM"):
                    sctm[row["domain"]] = float(row["scTM"])
        first = names[0]
        sub = [r for r in rows if r["set"] == first and r["sample"] in sctm]
        if len(sub) > 5:
            y = np.array([sctm[r["sample"]] for r in sub])
            print(f"\nwithin {first}, Spearman with scTM (n={len(sub)}):")
            for k in keys:
                x = np.array([r[k] for r in sub], dtype=float)
                ok = ~np.isnan(x)
                if ok.sum() < 5 or x[ok].std() == 0:
                    continue
                ra = np.argsort(np.argsort(x[ok])).astype(float)
                rb = np.argsort(np.argsort(y[ok])).astype(float)
                ra -= ra.mean()
                rb -= rb.mean()
                r = float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))
                print(f"  {k:20s} {r:+.3f}")
                report.setdefault("spearman_with_sctm", {})[k] = r

    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    csv_path = str(Path(args.output).with_suffix(".csv"))
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "sample"] + ["length"] + keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
