"""
Shared helpers for the experiment scripts.

Every experiment in this directory is a set of real training runs of
scripts/train.py on real data, followed by measurements on the resulting
checkpoints. Nothing is simulated: a quantity that was not measured is
reported as null.
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def add_common_args(parser: argparse.ArgumentParser, output_dir: str, max_steps: int = 5000) -> None:
    """Options shared by all experiment scripts."""
    parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "Distilled_trainset"))
    parser.add_argument("--seeds", type=str, default="0", help="comma-separated seeds")
    parser.add_argument("--max_steps", type=int, default=max_steps, help="training steps per run")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="re-run runs that already have a summary")
    parser.add_argument("--proteinmpnn_dir", type=str, default=None,
                        help="if given, also measure scTM with ProteinMPNN + ESMFold")
    parser.add_argument("--n_sctm_samples", type=int, default=20)
    parser.add_argument("--sctm_length", type=int, default=100)


def parse_args(parser: argparse.ArgumentParser):
    """Parse known options; anything else is forwarded to scripts/train.py."""
    args, train_args = parser.parse_known_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.train_args = train_args
    return args


def base_overrides(args) -> dict:
    return {
        "data_dir": args.data_dir,
        "max_steps": args.max_steps,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
    }


def run_training(name: str, output_dir, overrides: dict, extra_args: Optional[List[str]] = None,
                 force: bool = False) -> dict:
    """
    Run scripts/train.py once and return its JSON summary.

    The run is skipped when its summary already exists, so an interrupted
    sweep can be restarted.
    """
    run_dir = Path(output_dir) / name
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"[{name}] found {summary_path}, skipping")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(TRAIN_SCRIPT)]
        for key, value in overrides.items():
            if isinstance(value, bool) and key == "log_ck":
                cmd += ["--log_ck"] if value else []
            else:
                cmd += [f"--{key}", str(value)]
        cmd += ["--checkpoint_dir", str(run_dir), "--output", str(summary_path)]
        cmd += extra_args or []
        print(f"[{name}] {' '.join(cmd)}", flush=True)
        with open(run_dir / "stdout.log", "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(summary_path) as f:
        summary = json.load(f)
    summary["run"] = name
    return summary


@torch.no_grad()
def tau_by_secondary_structure(checkpoint: str, data_dir: str, max_chains: int = 256) -> Dict[str, float]:
    """
    Mean learned relaxation time tau per secondary-structure class.

    The model is run on clean validation structures (t = 1). tau is averaged
    over layers and groups for every residue, and residues are grouped by the
    CA-based secondary-structure assignment of src.metrics.
    """
    from src.data import CADataset
    from src.metrics import SS_NAMES, assign_secondary_structure
    from src.model_io import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(checkpoint, device=device)
    config = ckpt["config"]

    dataset = CADataset(data_dir, min_len=config.get("min_len", 20), max_len=config.get("max_len", 512),
                        max_bond=config.get("max_bond", 4.8))
    _, val_idx = dataset.split(config.get("val_fraction", 0.05), seed=0)
    by_length = defaultdict(list)
    for i in val_idx[:max_chains]:
        by_length[dataset.lengths[i]].append(dataset.coords[i])

    sums, counts = torch.zeros(3), torch.zeros(3)
    for chains in by_length.values():
        ca = torch.stack(chains).to(device)
        t = torch.ones(ca.shape[0], device=device)
        physics = model(ca / model.coord_scale, t, return_physics=True)[2]
        tau = physics["tau_per_residue"].cpu()
        ss = assign_secondary_structure(ca).cpu()
        for code in range(3):
            sums[code] += tau[ss == code].sum()
            counts[code] += (ss == code).sum()
    return {
        f"tau_{name}": (sums[code] / counts[code]).item() if counts[code] > 0 else None
        for code, name in enumerate(SS_NAMES)
    }


def measure_sctm(checkpoint: str, out_dir, proteinmpnn_dir: str, n_samples: int = 20,
                 length: int = 100, n_steps: int = 200, seed: int = 0) -> dict:
    """Generate samples from a checkpoint and run the scTM pipeline on them."""
    from src.data import write_backbone_pdb
    from src.designability import SelfConsistencyEvaluator, write_results
    from src.model_io import load_checkpoint

    out_dir = Path(out_dir)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(checkpoint, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    coords = model.sample(length, n_samples=n_samples, n_steps=n_steps, device=device, generator=generator)
    del model

    pdb_dir = out_dir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, ca in enumerate(coords.cpu()):
        files.append(str(pdb_dir / f"sample_{i:04d}.pdb"))
        write_backbone_pdb(ca, files[-1])

    evaluator = SelfConsistencyEvaluator(proteinmpnn_dir=proteinmpnn_dir, device=device)
    records = evaluator.evaluate(files, seed=seed)
    del evaluator
    if device == "cuda":
        torch.cuda.empty_cache()
    return write_results(records, out_dir)


def attach_sctm(summary: dict, args, output_dir) -> dict:
    """Fill ``final_scTM`` of a run summary when the scTM pipeline is configured."""
    if args.proteinmpnn_dir:
        result = measure_sctm(
            summary["checkpoint"], Path(output_dir) / summary["run"] / "sctm",
            args.proteinmpnn_dir, n_samples=args.n_sctm_samples, length=args.sctm_length,
        )
        summary["final_scTM"] = result["scTM_mean"]
        summary["sctm"] = result
    return summary


def mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    t = torch.tensor(values, dtype=torch.double)
    return {"mean": t.mean().item(), "std": t.std().item() if len(values) > 1 else None, "n": len(values)}


def welch_p_value(a: List[float], b: List[float]) -> Optional[float]:
    """Two-sided Welch t-test p-value, or None with fewer than two values per group."""
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    from scipy import stats
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def fmt(value, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)
