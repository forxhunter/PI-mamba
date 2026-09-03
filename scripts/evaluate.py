#!/usr/bin/env python3
"""
PI-Mamba Evaluation Script

Generates samples, then runs ProteinMPNN + ESMFold + TMscore pipeline.

Usage (two-step, different conda envs):
  Step 1 - Generate samples (fm env):
    CUDA_VISIBLE_DEVICES=1 conda run -n fm python evaluate.py --mode generate \
        --checkpoint ../checkpoints_v13/best_model.pt --n_samples 100 --length 100

  Step 2 - Run scTM pipeline (Proteus env):
    CUDA_VISIBLE_DEVICES=1 conda run -n Proteus python evaluate.py --mode eval \
        --eval_dir ../eval_output
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch


def reconstruct_full_backbone(ca_coords):
    """Reconstruct N, CA, C, O from CA-only coordinates using ideal geometry.

    Args:
        ca_coords: (L, 3) numpy array of CA positions
    Returns:
        atoms: list of (atom_name, x, y, z) for each residue (N, CA, C, O)
    """
    L = len(ca_coords)
    atoms = []

    # Ideal bond lengths (Angstroms)
    d_n_ca = 1.458
    d_ca_c = 1.523
    d_c_o = 1.231

    for i in range(L):
        ca = ca_coords[i]

        # Determine local frame from neighboring CAs
        if i == 0:
            fwd = ca_coords[1] - ca
        elif i == L - 1:
            fwd = ca - ca_coords[i - 1]
        else:
            fwd = ca_coords[i + 1] - ca_coords[i - 1]
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)

        # Get a perpendicular direction
        if i == 0:
            ref = ca_coords[min(2, L-1)] - ca
        elif i == L - 1:
            ref = ca - ca_coords[max(0, i-2)]
        else:
            ref = ca_coords[i+1] - 2*ca + ca_coords[i-1]

        perp = ref - np.dot(ref, fwd) * fwd
        norm_perp = np.linalg.norm(perp)
        if norm_perp < 1e-8:
            # Fallback: arbitrary perpendicular
            perp = np.array([1.0, 0.0, 0.0])
            perp = perp - np.dot(perp, fwd) * fwd
            norm_perp = np.linalg.norm(perp)
            if norm_perp < 1e-8:
                perp = np.array([0.0, 1.0, 0.0])
                perp = perp - np.dot(perp, fwd) * fwd
                norm_perp = np.linalg.norm(perp)
        perp = perp / (norm_perp + 1e-8)

        # Place N (before CA along backbone)
        n_pos = ca - d_n_ca * (0.87 * fwd + 0.50 * perp)
        # Place C (after CA along backbone)
        c_pos = ca + d_ca_c * (0.87 * fwd - 0.50 * perp)
        # Place O (off C, perpendicular to CA-C bond)
        co_dir = np.cross(fwd, perp)
        co_dir = co_dir / (np.linalg.norm(co_dir) + 1e-8)
        o_pos = c_pos + d_c_o * (0.50 * co_dir + 0.87 * (c_pos - ca) / (np.linalg.norm(c_pos - ca) + 1e-8))

        atoms.append(('N', n_pos))
        atoms.append(('CA', ca))
        atoms.append(('C', c_pos))
        atoms.append(('O', o_pos))

    return atoms


def write_backbone_pdb(ca_coords, path, chain='A'):
    """Write full backbone PDB (N, CA, C, O) from CA coordinates.

    Places N, C, O using ideal bond geometry relative to the CA trace,
    preserving the original CA positions exactly.

    Args:
        ca_coords: (L, 3) numpy array of CA positions
        path: output PDB path
    """
    L = len(ca_coords)

    # Ideal bond lengths
    d_n_ca = 1.458
    d_ca_c = 1.523
    d_c_o = 1.231

    # Build local frames from CA trace
    backbone = []  # list of (N, CA, C, O) positions

    for i in range(L):
        ca = ca_coords[i]

        # Forward direction along chain
        if i == 0:
            fwd = ca_coords[1] - ca
        elif i == L - 1:
            fwd = ca - ca_coords[i-1]
        else:
            fwd = ca_coords[i+1] - ca_coords[i-1]
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm < 1e-8:
            fwd = np.array([1.0, 0.0, 0.0])
        else:
            fwd = fwd / fwd_norm

        # Curvature direction (for placing N/C on alternating sides)
        if 0 < i < L - 1:
            curve = ca_coords[i+1] - 2*ca + ca_coords[i-1]
        elif i == 0 and L > 2:
            curve = ca_coords[2] - 2*ca_coords[1] + ca
        else:
            curve = np.array([0.0, 1.0, 0.0])

        # Make perpendicular to fwd
        curve = curve - np.dot(curve, fwd) * fwd
        cn = np.linalg.norm(curve)
        if cn < 1e-8:
            # Find any perpendicular
            if abs(fwd[0]) < 0.9:
                curve = np.cross(fwd, [1, 0, 0])
            else:
                curve = np.cross(fwd, [0, 1, 0])
            cn = np.linalg.norm(curve)
        curve = curve / cn

        # Third axis
        binorm = np.cross(fwd, curve)

        # Place N before CA, C after CA (tetrahedral-like angles)
        # N-CA-C angle ~111 degrees
        n_pos = ca - d_n_ca * (0.82 * fwd + 0.57 * curve)
        c_pos = ca + d_ca_c * (0.82 * fwd - 0.57 * curve)

        # O placement: ~120 deg from CA-C-O, in the plane
        c_ca = ca - c_pos
        c_ca = c_ca / (np.linalg.norm(c_ca) + 1e-8)
        o_dir = -0.5 * c_ca + 0.866 * binorm
        o_pos = c_pos + d_c_o * o_dir

        backbone.append((n_pos, ca, c_pos, o_pos))

    with open(path, 'w') as f:
        f.write("REMARK  Generated with PI-Mamba, full backbone reconstructed\n")
        atom_idx = 1
        for i, (n, ca, c, o) in enumerate(backbone):
            for name, pos, elem in [('N', n, 'N'), ('CA', ca, 'C'),
                                     ('C', c, 'C'), ('O', o, 'O')]:
                f.write(
                    f"ATOM  {atom_idx:5d}  {name:<3s} ALA {chain}{i+1:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                    f"  1.00  0.00           {elem}  \n"
                )
                atom_idx += 1
        f.write("END\n")


def generate_samples(args):
    """Generate protein backbone samples using PI-Mamba."""
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from backbone_pi_mamba import PIMambaBackbone

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    d_model = ckpt.get('d_model', 256)
    n_layers = ckpt.get('n_layers', 8)

    model = PIMambaBackbone(
        d_model=d_model, n_layers=n_layers,
        d_state=64, n_groups=8, n_structure_layers=4,
        max_length=2048, use_physics=True,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded model: d_model={d_model}, n_layers={n_layers}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # Generate
    pdb_dir = os.path.join(args.eval_dir, 'pdbs')
    os.makedirs(pdb_dir, exist_ok=True)

    batch_size = min(args.n_samples, 10)
    n_batches = (args.n_samples + batch_size - 1) // batch_size
    sample_idx = 0

    for b in range(n_batches):
        n = min(batch_size, args.n_samples - sample_idx)
        print(f"Generating batch {b+1}/{n_batches} ({n} samples, L={args.length})...")
        with torch.no_grad():
            coords = model.sample(
                length=args.length, n_samples=n,
                n_steps=args.n_steps, device=device,
            )
        coords_np = coords.cpu().numpy()
        for i in range(n):
            pdb_path = os.path.join(pdb_dir, f'sample_{sample_idx:04d}.pdb')
            write_backbone_pdb(coords_np[i], pdb_path)
            sample_idx += 1

    print(f"Generated {sample_idx} samples in {pdb_dir}")


def run_sctm_pipeline(args):
    """Run ProteinMPNN + ESMFold + TMscore evaluation pipeline."""
    PIPELINE_ROOT = '/data2/2026_RNAAI/baselines/insilico_design_pipeline'
    sys.path.insert(0, PIPELINE_ROOT)
    orig_dir = os.getcwd()
    os.chdir(PIPELINE_ROOT)

    # Set ESMFold cache
    os.environ['TORCH_HOME'] = '/data2/2026_RNAAI/baselines/Proteus/.cache/torch'

    from pipeline.models.inverse_folds.proteinmpnn import ProteinMPNN
    from pipeline.models.folds.esmfold import ESMFold
    from pipeline.standard.unconditional import UnconditionalPipeline

    device = f'cuda:0'

    print("Loading ProteinMPNN (CA model)...")
    inverse_fold_model = ProteinMPNN(device=device, num_samples=8, sampling_temperature=0.1)

    print("Loading ESMFold...")
    fold_model = ESMFold(device=device)

    print("Building pipeline...")
    pipeline = UnconditionalPipeline(inverse_fold_model, fold_model)

    # Clean up any previous intermediate dirs to avoid assert errors
    rootdir = args.eval_dir
    for subdir in ['sequences', 'structures', 'scores', 'results', 'designs']:
        d = os.path.join(rootdir, subdir)
        if os.path.exists(d):
            import shutil
            shutil.rmtree(d)

    print(f"Evaluating {rootdir}...")
    pipeline.evaluate(rootdir, clean=False, verbose=True)

    # Parse results
    info_csv = os.path.join(rootdir, 'info.csv')
    if os.path.exists(info_csv):
        import pandas as pd
        df = pd.read_csv(info_csv)
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"N samples:     {len(df)}")
        print(f"scTM mean:     {df['scTM'].mean():.4f} ± {df['scTM'].std():.4f}")
        print(f"scTM median:   {df['scTM'].median():.4f}")
        if 'scRMSD' in df.columns:
            print(f"scRMSD mean:   {df['scRMSD'].mean():.4f}")
        if 'pLDDT' in df.columns:
            print(f"pLDDT mean:    {df['pLDDT'].mean():.4f}")
            designable = df[(df['scRMSD'] <= 2.0) & (df['pLDDT'] >= 70)]
            print(f"Designability: {len(designable)/len(df):.3f} ({len(designable)}/{len(df)})")
        print("=" * 60)

        # Save summary
        summary = {
            'n_samples': len(df),
            'scTM_mean': float(df['scTM'].mean()),
            'scTM_std': float(df['scTM'].std()),
            'scTM_median': float(df['scTM'].median()),
        }
        if 'scRMSD' in df.columns:
            summary['scRMSD_mean'] = float(df['scRMSD'].mean())
        if 'pLDDT' in df.columns:
            summary['pLDDT_mean'] = float(df['pLDDT'].mean())
            summary['designability'] = float(len(designable) / len(df))

        with open(os.path.join(rootdir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {rootdir}/summary.json")
    else:
        print(f"WARNING: {info_csv} not found. Pipeline may have failed.")

    os.chdir(orig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PI-Mamba Evaluation")
    parser.add_argument('--mode', choices=['generate', 'eval', 'both'], default='both')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints_v13/best_model.pt')
    parser.add_argument('--eval_dir', type=str, default='../eval_output')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=200)
    args = parser.parse_args()

    if args.mode in ('generate', 'both'):
        generate_samples(args)
    if args.mode in ('eval', 'both'):
        run_sctm_pipeline(args)
