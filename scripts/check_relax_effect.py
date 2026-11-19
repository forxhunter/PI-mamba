"""Does the end-of-sampling relaxation damage beta sheets?

``FlowBackbone.sample`` finishes with ``relax_chain``, which pulls every CA-CA
distance onto 3.8 A and pushes apart residues closer than a clash cutoff. The
clash term works on all pairs, and hydrogen-bonded partners in a beta sheet sit
about 4.8 A apart, so the question is whether the relaxation treats a real sheet
contact as a clash.

The test does not need a model. Training structures already contain sheets, so
the relaxation is applied to them and the ladders are counted before and after.
Whatever it does to a real sheet, it does to a generated one.

Usage:
    python scripts/check_relax_effect.py --reference Distilled_trainset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sheet_quality import sheet_features  # noqa: E402
from src.backbone_pi_mamba import project_bonds, relax_chain  # noqa: E402
from src.data import read_ca_coords  # noqa: E402
from src.metrics import clash_fraction  # noqa: E402
from src.geometry import CA_CA_TRANS  # noqa: E402


def summarise(feats: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["strand_frac", "paired_strand_frac", "registered_frac",
            "n_ladders", "mean_ladder_len"]
    out = {}
    for k in keys:
        v = np.array([f[k] for f in feats], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = float(v.mean()) if len(v) else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="Distilled_trainset",
                   help="directory of PDB files to relax; training structures "
                        "show whether sheets survive, raw model samples "
                        "(generated with --no_projection) show whether clashes "
                        "are still removed")
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--coord_scale", type=float, default=10.0)
    p.add_argument("--min_dist", type=float, default=4.0,
                   help="clash cutoff used by sample(), in Angstrom")
    p.add_argument("--output", default="relax_effect.json")
    args = p.parse_args()

    files = sorted(Path(args.reference).glob("*.pdb"))
    chains = []
    for f in files:
        if len(chains) >= args.limit:
            break
        try:
            ca = read_ca_coords(f)
        except Exception:  # noqa: BLE001
            continue
        if ca is None or len(ca) != args.length:
            continue
        chains.append(torch.as_tensor(ca, dtype=torch.float32))
    if not chains:
        print(f"no structures of length {args.length} in {args.reference}")
        return
    x = torch.stack(chains)
    x = x - x.mean(dim=1, keepdim=True)
    print(f"{len(chains)} training structures of length {args.length}")

    scale = args.coord_scale
    bond = CA_CA_TRANS / scale
    xs = x / scale

    variants = {
        "original": x,
        "project_bonds only": project_bonds(xs.clone(), bond) * scale,
        f"relax_chain (min_dist {args.min_dist} A)": relax_chain(
            xs.clone(), bond, min_dist=args.min_dist / scale,
            w_clash=2.0, n_steps=500) * scale,
        "relax_chain (min_dist 3.0 A)": relax_chain(
            xs.clone(), bond, min_dist=3.0 / scale,
            w_clash=2.0, n_steps=500) * scale,
        "relax_chain + contact restraint 1.0": relax_chain(
            xs.clone(), bond, min_dist=args.min_dist / scale,
            w_clash=2.0, n_steps=500, w_contact=1.0) * scale,
        "relax_chain + contact restraint 4.0": relax_chain(
            xs.clone(), bond, min_dist=args.min_dist / scale,
            w_clash=2.0, n_steps=500, w_contact=4.0) * scale,
    }

    report = {}
    for name, coords in variants.items():
        feats = [sheet_features(c, min_ladder=3) for c in coords]
        report[name] = summarise(feats)
        d = (coords[:, 1:] - coords[:, :-1]).norm(dim=-1)
        report[name]["bond_mean"] = float(d.mean())
        report[name]["bond_std"] = float(d.std())
        rms = float((coords - x).norm(dim=-1).pow(2).mean().sqrt())
        report[name]["rms_shift_from_original"] = rms
        report[name]["clash_frac"] = float(clash_fraction(coords).mean())

    cols = ["strand_frac", "registered_frac", "n_ladders", "clash_frac",
            "bond_mean", "bond_std", "rms_shift_from_original"]
    width = max(len(n) for n in report) + 2
    print(f"\n{'variant':<{width}}" + "".join(f"{c[:12]:>14s}" for c in cols))
    for name, r in report.items():
        print(f"{name:<{width}}" + "".join(f"{r[c]:>14.3f}" for c in cols))

    base = report["original"]["n_ladders"]
    for name, r in report.items():
        if name == "original" or base == 0:
            continue
        print(f"\n{name}: {r['n_ladders'] / base * 100:.0f} per cent of the "
              f"ladders survive, CA shifted {r['rms_shift_from_original']:.2f} A RMS")

    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
