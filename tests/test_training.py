"""
End-to-end tests of the training and evaluation scripts on tiny models (CPU).

    python tests/test_training.py
    pytest tests/
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train  # noqa: E402
from src.geometry import nerf_build_backbone  # noqa: E402
from src.model_io import load_checkpoint  # noqa: E402

TINY = [
    "--d_model", "32", "--n_layers", "2", "--d_state", "8", "--n_groups", "4",
    "--n_structure_layers", "1", "--n_heads", "4", "--batch_size", "4",
    "--warmup_steps", "2", "--log_every", "4", "--val_every", "8", "--val_batches", "2",
    "--n_val_samples", "2", "--sample_steps", "3", "--device", "cpu",
]


def _run(script: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _write_full_backbone_pdbs(directory: Path, n: int = 12, length: int = 24) -> None:
    """Synthetic chains with N, CA and C atoms, built by NeRF from helix-like torsions."""
    torch.manual_seed(0)
    phi = math.radians(-60) + 0.3 * torch.randn(n, length)
    psi = math.radians(-45) + 0.3 * torch.randn(n, length)
    omega = torch.full((n, length), math.pi)
    backbones = nerf_build_backbone(phi, psi, omega)
    for k, bb in enumerate(backbones):
        with open(directory / f"chain_{k:02d}.pdb", "w") as f:
            serial = 1
            for i, residue in enumerate(bb):
                for name, xyz in zip(("N", "CA", "C"), residue):
                    f.write(f"ATOM  {serial:5d}  {name:<3s} ALA A{i + 1:4d}    "
                            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           {name[0]}  \n")
                    serial += 1
            f.write("END\n")


def test_train_resume_generate_and_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = tmp / "run"
        common = [*TINY, "--max_len", "100", "--sample_lengths", "50", "--log_ck",
                  "--checkpoint_dir", str(run)]
        summary = train.main([*common, "--max_steps", "8", "--output", str(tmp / "summary.json")])
        assert summary["steps"] == 8 and math.isfinite(summary["final_val_loss"])
        assert summary["final_C_10"] is not None and summary["final_scTM"] is None
        assert json.loads((tmp / "summary.json").read_text())["model"] == "pi_mamba"
        assert (run / "best_model.pt").exists() and (run / "last.pt").exists()

        # Resume continues from the stored step
        summary = train.main([*common, "--max_steps", "12", "--resume", str(run / "last.pt")])
        assert summary["steps"] == 12

        model, ckpt = load_checkpoint(run / "best_model.pt")
        assert ckpt["config"]["d_model"] == 32 and not model.training

        eval_dir = tmp / "eval"
        _run("evaluate.py", "--mode", "generate", "--checkpoint", str(run / "best_model.pt"),
             "--n_samples", "3", "--length", "40", "--n_steps", "3", "--eval_dir", str(eval_dir),
             "--device", "cpu")
        assert len(list((eval_dir / "pdbs").glob("*.pdb"))) == 3
        _run("evaluate.py", "--mode", "metrics", "--eval_dir", str(eval_dir))
        metrics = json.loads((eval_dir / "metrics.json").read_text())["samples"]["L40"]
        # The sampler relaxes bond lengths against steric overlap, so they end
        # up near 3.8 rather than exactly on it
        assert metrics["n"] == 3 and abs(metrics["bond_mean"] - 3.8) < 0.5, metrics


def test_ablation_and_baselines_train():
    with tempfile.TemporaryDirectory() as tmp:
        for extra in (["--use_physics", "False"], ["--spectrum", "scrambled"],
                      ["--model", "transformer"], ["--model", "performer"]):
            summary = train.main([*TINY, "--max_len", "75", "--sample_lengths", "--max_steps", "3",
                                  "--val_every", "3", "--checkpoint_dir", str(Path(tmp) / "_".join(extra)),
                                  *extra])
            assert math.isfinite(summary["final_val_loss"]), extra


def test_torsion_head_training_on_full_backbone_data():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = tmp / "data"
        data.mkdir()
        _write_full_backbone_pdbs(data)
        train.main([*TINY, "--data_dir", str(data), "--min_len", "10", "--val_fraction", "0.2",
                    "--lambda_torsion", "1.0", "--lambda_omega", "0.1", "--lambda_hbond", "0.1",
                    "--lambda_rama", "0.1", "--lambda_rg", "0.1", "--aux_t_min", "0.0",
                    "--sample_lengths", "24", "--max_steps", "6", "--val_every", "6",
                    "--checkpoint_dir", str(tmp / "run")])
        log = [json.loads(line) for line in open(tmp / "run" / "train_log.jsonl")]
        train_rows = [row for row in log if row["type"] == "train"]
        assert train_rows and math.isfinite(train_rows[-1]["torsion"])

        model, _ = load_checkpoint(tmp / "run" / "last.pt")
        backbone = model.reconstruct_full_backbone(torch.randn(2, 24, 3) * 8)
        assert backbone.shape == (2, 24, 3, 3) and torch.isfinite(backbone).all()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            import traceback
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
