"""Benchmark motif scaffolding with PI-Mamba.

A motif is a set of residue ranges cut out of a real structure. For every
motif the script generates scaffolds with ``inpainting.scaffold_motif`` and
reports, per sample:

  * ``motif_rmsd``  RMSD (Angstrom) between the motif in the generated chain
                    and the reference motif after optimal superposition.
  * ``clash_frac``  fraction of residues with a non-neighbour CA closer than 3 A.
  * ``scTM``        self-consistency TM-score, only with --proteinmpnn_dir.
  * ``success``     motif_rmsd < 1 A (and scTM > 0.5 when scTM is measured).

Scaffolds made by other methods (for example RFdiffusion or FrameDiff) can
be scored with the same metrics: pass ``--baseline NAME=DIR`` with a
directory of PDB files named ``<motif>_*.pdb`` that contain the motif at the
same residue positions. No baseline is run or simulated by this script.

Usage (from the repository root):
    python -m src.benchmark_scaffold --ckpt checkpoints/best_model.pt \
        --motif helix:Distilled_trainset/sample_000000_L100_scTM0.965.pdb:10-24 \
        --motif two_seg:Distilled_trainset/sample_000000_L100_scTM0.965.pdb:5-14,60-69 \
        --n_samples 8 --out benchmark_results.csv

A motif is written NAME:PDB:RANGES[:LENGTH]. RANGES are 1-based inclusive
residue ranges in the source structure; the motif keeps these positions in a
scaffold of LENGTH residues (default: the length of the source structure).
"""

import argparse
import csv
from pathlib import Path

import torch

from .data import read_ca_coords, write_backbone_pdb
from .geometry import kabsch_rmsd
from .inpainting import scaffold_motif
from .metrics import clash_fraction
from .model_io import load_checkpoint


def parse_motif(spec: str):
    """Parse NAME:PDB:RANGES[:LENGTH] into (name, motif coords (M, 3), mask (L,))."""
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"motif must be NAME:PDB:RANGES[:LENGTH], got {spec!r}")
    name, pdb, ranges = parts[:3]
    ca = torch.from_numpy(read_ca_coords(pdb))
    length = int(parts[3]) if len(parts) == 4 else len(ca)
    mask = torch.zeros(length, dtype=torch.bool)
    for item in ranges.split(","):
        start, end = (int(v) for v in item.split("-"))
        if not 1 <= start <= end <= min(length, len(ca)):
            raise ValueError(f"range {item} is outside the structure or the scaffold")
        mask[start - 1:end] = True
    return name, ca[mask[:len(ca)].nonzero().squeeze(-1)], mask


def score(scaffolds, motif, mask, evaluator, workdir: Path, prefix: str):
    """Metrics for scaffolds (B, L, 3) that contain the motif at ``mask``."""
    B = scaffolds.shape[0]
    motif_rmsd = kabsch_rmsd(scaffolds[:, mask], motif.expand(B, -1, -1))
    clashes = clash_fraction(scaffolds)
    sc_tm = [None] * B
    if evaluator is not None:
        files = []
        for i, ca in enumerate(scaffolds):
            files.append(str(workdir / f"{prefix}_{i:03d}.pdb"))
            write_backbone_pdb(ca, files[-1])
        sc_tm = [r["scTM"] for r in evaluator.evaluate(files)]
    rows = []
    for i in range(B):
        ok = motif_rmsd[i].item() < 1.0 and (sc_tm[i] is None or sc_tm[i] > 0.5)
        rows.append({
            "motif_rmsd": round(motif_rmsd[i].item(), 4),
            "clash_frac": round(clashes[i].item(), 4),
            "scTM": None if sc_tm[i] is None else round(sc_tm[i], 4),
            "success": ok,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to PI-Mamba checkpoint")
    parser.add_argument("--motif", action="append", required=True, help="NAME:PDB:RANGES[:LENGTH] (repeatable)")
    parser.add_argument("--baseline", action="append", default=[], help="NAME=DIR of PDB files (repeatable)")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--no_final_projection", action="store_true",
                        help="keep the motif exact instead of making all bond lengths exactly 3.8 A")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proteinmpnn_dir", type=str, default=None, help="also measure scTM")
    parser.add_argument("--out", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.ckpt, device=device)

    out_path = Path(args.out)
    workdir = out_path.parent / (out_path.stem + "_pdbs")
    workdir.mkdir(parents=True, exist_ok=True)

    evaluator = None
    if args.proteinmpnn_dir:
        from .designability import SelfConsistencyEvaluator
        evaluator = SelfConsistencyEvaluator(proteinmpnn_dir=args.proteinmpnn_dir, device=str(device))

    results = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for spec in args.motif:
        name, motif, mask = parse_motif(spec)
        L, B = len(mask), args.n_samples

        out = scaffold_motif(
            model, L, motif.expand(B, -1, -1), mask.expand(B, -1),
            n_steps=args.n_steps, final_projection=not args.no_final_projection,
            device=device, generator=generator,
        )
        scaffolds = out["scaffold"].cpu()
        if evaluator is None:
            for i, ca in enumerate(scaffolds):
                write_backbone_pdb(ca, workdir / f"{name}_pimamba_{i:03d}.pdb")
        methods = {"PI-Mamba": score(scaffolds, motif, mask, evaluator, workdir, f"{name}_pimamba")}

        for item in args.baseline:
            method, directory = item.split("=", 1)
            files = sorted(Path(directory).glob(f"{name}_*.pdb"))
            coords = [torch.from_numpy(read_ca_coords(f)) for f in files]
            coords = [c for c in coords if len(c) == L]
            if not coords:
                print(f"[Warning] no {name}_*.pdb of length {L} in {directory}; skipping {method}")
                continue
            methods[method] = score(torch.stack(coords), motif, mask, evaluator, workdir, f"{name}_{method}")

        for method, rows in methods.items():
            for i, row in enumerate(rows):
                results.append({"motif": name, "method": method, "length": L, "sample": i, **row})
            rmsd = sum(r["motif_rmsd"] for r in rows) / len(rows)
            success = sum(r["success"] for r in rows) / len(rows)
            print(f"{method} on {name} (L={L}, motif {int(mask.sum())} residues, n={len(rows)}): "
                  f"mean motif RMSD={rmsd:.3f} A, success rate={success:.2f}")

    with out_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["motif", "method", "length", "sample", "motif_rmsd", "clash_frac", "scTM", "success"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Benchmark results saved to {out_path}")


if __name__ == "__main__":
    main()
