#!/usr/bin/env python3
"""
PI-Mamba Evaluation Script

Three steps, selected with --mode:

  generate  Sample backbones from a checkpoint and write PDB files to
            <eval_dir>/pdbs. --length takes one length or a comma-separated
            list, and generates --n_samples backbones for each.
  metrics   CA-only structure metrics of the PDB files in <eval_dir>/pdbs
            (bond lengths, clashes, compactness, secondary structure,
            diversity). No external tools needed. Writes metrics.json.
  eval      Self-consistency evaluation (scTM, scRMSD, pLDDT) with
            ProteinMPNN (CA-only) + ESMFold. Needs a ProteinMPNN checkout
            (--proteinmpnn_dir or PROTEINMPNN_DIR) and the transformers
            package. Writes info.csv and summary.json.
  all       generate, metrics and eval in sequence.

Usage:
    python scripts/evaluate.py --mode generate --checkpoint checkpoints/best_model.pt \
        --n_samples 100 --length 100 --eval_dir eval_output
    python scripts/evaluate.py --mode metrics --eval_dir eval_output \
        --reference_dir Distilled_trainset
    python scripts/evaluate.py --mode eval --eval_dir eval_output \
        --proteinmpnn_dir /path/to/ProteinMPNN

--mode metrics and --mode eval also work on any directory of single-chain PDB
files of your own: put them in <eval_dir>/pdbs.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import read_ca_coords, write_backbone_pdb
from src.metrics import pairwise_diversity, structure_metrics
from src.model_io import load_checkpoint


def pdb_files(eval_dir: str):
    files = sorted((Path(eval_dir) / "pdbs").glob("*.pdb"))
    if not files:
        raise FileNotFoundError(f"No PDB files in {Path(eval_dir) / 'pdbs'}")
    return files


def generate_samples(args):
    """Generate protein backbone samples using a trained model."""
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_checkpoint(args.checkpoint, device=device, use_ema=not args.no_ema)
    config = ckpt["config"]
    print(f"Loaded {config.get('model', 'pi_mamba')}: d_model={config.get('d_model')}, "
          f"n_layers={config.get('n_layers')}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    pdb_dir = Path(args.eval_dir) / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    total = 0
    for length in (int(v) for v in str(args.length).split(",")):
        done = 0
        while done < args.n_samples:
            n = min(args.batch_size, args.n_samples - done)
            print(f"Generating samples {done}..{done + n - 1} (L={length})...", flush=True)
            coords = model.sample(
                length=length, n_samples=n, n_steps=args.n_steps, device=device,
                project=not args.no_projection, relax=not args.no_relax,
                generator=generator,
            ).cpu()
            for ca in coords:
                write_backbone_pdb(ca, pdb_dir / f"sample_L{length}_{done:04d}.pdb")
                done += 1
        total += done

    print(f"Generated {total} samples in {pdb_dir}")


def _metrics_by_length(coords_by_length: dict) -> dict:
    out = {}
    for length in sorted(coords_by_length):
        ca = torch.stack(coords_by_length[length])
        out[f"L{length}"] = {
            "n": len(ca),
            **structure_metrics(ca),
            "pairwise_tm": pairwise_diversity(ca),
        }
    return out


def run_structure_metrics(args):
    """CA-only structure metrics of the generated PDB files."""
    by_length = defaultdict(list)
    for path in pdb_files(args.eval_dir):
        ca = torch.from_numpy(read_ca_coords(path))
        by_length[len(ca)].append(ca)
    result = {"samples": _metrics_by_length(by_length)}

    if args.reference_dir:
        ref_by_length = defaultdict(list)
        for path in sorted(Path(args.reference_dir).glob("*.pdb")):
            ca = read_ca_coords(path)
            if len(ca) in by_length and len(ref_by_length[len(ca)]) < args.max_reference:
                ref_by_length[len(ca)].append(torch.from_numpy(ca))
        result["reference"] = _metrics_by_length(ref_by_length)

    for group, table in result.items():
        for length, m in table.items():
            print(f"[{group} {length}, n={m['n']}] "
                  + ", ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "n"))

    out_path = Path(args.eval_dir) / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Metrics saved to {out_path}")


def run_sctm_pipeline(args):
    """Run the ProteinMPNN + ESMFold + TM-score evaluation pipeline."""
    from src.designability import SelfConsistencyEvaluator, write_results

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = SelfConsistencyEvaluator(
        proteinmpnn_dir=args.proteinmpnn_dir, esmfold_model=args.esmfold_model,
        n_seqs=args.n_seqs, temperature=args.sampling_temperature, device=device,
    )
    files = [str(p) for p in pdb_files(args.eval_dir)]
    print(f"Evaluating {len(files)} backbones from {args.eval_dir}...")
    records = evaluator.evaluate(files, seed=args.seed, verbose=True)
    summary = write_results(records, args.eval_dir)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"N samples:          {summary['n_samples']}")
    print(f"scTM mean:          {summary['scTM_mean']:.4f} ± {summary['scTM_std']:.4f}")
    print(f"scTM median:        {summary['scTM_median']:.4f}")
    print(f"scRMSD mean:        {summary['scRMSD_mean']:.4f}")
    print(f"pLDDT mean:         {summary['pLDDT_mean']:.2f}")
    print(f"scTM > 0.5:         {summary['frac_scTM_above_0.5']:.3f}")
    print(f"scRMSD < 2 A:       {summary['frac_scRMSD_below_2']:.3f}")
    print("=" * 60)
    print(f"Per-sample results: {Path(args.eval_dir) / 'info.csv'}")
    print(f"Summary:            {Path(args.eval_dir) / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PI-Mamba Evaluation")
    parser.add_argument('--mode', choices=['generate', 'metrics', 'eval', 'all'], default='all')
    parser.add_argument('--eval_dir', type=str, default='eval_output')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)

    g = parser.add_argument_group('generate')
    g.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    g.add_argument('--n_samples', type=int, default=100)
    g.add_argument('--length', type=str, default='100',
                   help='sequence length, or a comma-separated list; n_samples per length')
    g.add_argument('--n_steps', type=int, default=200)
    g.add_argument('--batch_size', type=int, default=10)
    g.add_argument('--no_ema', action='store_true', help='use the raw weights instead of the EMA weights')
    g.add_argument('--no_projection', action='store_true', help='skip the CA-CA bond projection')
    g.add_argument('--no_relax', action='store_true',
                   help='project bond lengths only, without resolving clashes (relax_chain)')

    g = parser.add_argument_group('metrics')
    g.add_argument('--reference_dir', type=str, default=None,
                   help='directory of reference PDB files to report the same metrics for')
    g.add_argument('--max_reference', type=int, default=500)

    g = parser.add_argument_group('eval')
    g.add_argument('--proteinmpnn_dir', type=str, default=None,
                   help='ProteinMPNN checkout (default: $PROTEINMPNN_DIR)')
    g.add_argument('--esmfold_model', type=str, default='facebook/esmfold_v1')
    g.add_argument('--n_seqs', type=int, default=8)
    g.add_argument('--sampling_temperature', type=float, default=0.1)
    args = parser.parse_args()

    if args.mode in ('generate', 'all'):
        generate_samples(args)
    if args.mode in ('metrics', 'all'):
        run_structure_metrics(args)
    if args.mode in ('eval', 'all'):
        run_sctm_pipeline(args)
