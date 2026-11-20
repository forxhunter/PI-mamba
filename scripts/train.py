#!/usr/bin/env python3
"""
PI-Mamba training script (flow matching on CA coordinates).

Defaults come from configs/model/pi_mamba_v12.yaml and
configs/train/single_gpu.yaml. Extra YAML files given with --config and
command-line flags override them, in that order.

Usage:
    # Full PI-Mamba model (paper configuration: 16 layers, d_model 512)
    python scripts/train.py --use_physics True --data_dir /path/to/pdb/files

    # Learned-A ablation
    python scripts/train.py --use_physics False

    # Small model, short run
    python scripts/train.py --d_model 256 --n_layers 6 --max_steps 2000

Flow matching convention: x_0 ~ N(0, I), x_1 = data (model units),
x_t = (1 - t) x_0 + t x_1, target velocity x_1 - x_0.
"""

import argparse
import copy
import functools
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Keep the log readable while it is being redirected to a file
print = functools.partial(print, flush=True)

from src.backbone_pi_mamba import project_bonds
from src.data import CADataset, LengthBatchSampler, collate_same_length
from src.geometry import CA_CA_TRANS, coords_to_torsions, random_rotations
from src.losses import (
    BondLengthLoss, DistogramLoss, HydrogenBondLoss, LocalDistanceLoss,
    RadiusOfGyrationLoss, RamachandranLoss, RegisterDistanceLoss,
)
from src.measure_mode_concentration import compute_mode_concentration
from src.metrics import structure_metrics
from src.model_io import CHECKPOINT_FORMAT, MODEL_CHOICES, build_model, save_checkpoint
from src.rouse_physics import SPECTRUM_CHOICES

DEFAULT_CONFIGS = (
    REPO_ROOT / "configs" / "model" / "pi_mamba_v12.yaml",
    REPO_ROOT / "configs" / "train" / "single_gpu.yaml",
)
LOSS_TERMS = ("fape", "bond", "dist", "rg", "hbond", "rama", "omega", "torsion",
               "register")


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

def str2bool(value: str) -> bool:
    if value.lower() in ("true", "t", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train PI-Mamba with flow matching",
        argument_default=argparse.SUPPRESS,  # only flags given on the command line override the YAML
    )
    p.add_argument("--config", action="append", default=[], help="extra YAML config (repeatable)")

    g = p.add_argument_group("model")
    g.add_argument("--model", choices=MODEL_CHOICES)
    g.add_argument("--d_model", type=int)
    g.add_argument("--n_layers", type=int)
    g.add_argument("--d_state", type=int)
    g.add_argument("--d_conv", type=int)
    g.add_argument("--expand_factor", type=int)
    g.add_argument("--n_groups", type=int)
    g.add_argument("--n_heads", type=int)
    g.add_argument("--n_structure_layers", type=int)
    g.add_argument("--max_length", type=int)
    g.add_argument("--dropout", type=float)
    g.add_argument("--use_physics", type=str2bool, metavar="BOOL",
                   help="True: Rouse-derived A; False: learned-A ablation")
    g.add_argument("--spectrum", choices=SPECTRUM_CHOICES)
    g.add_argument("--spectrum_alpha", type=float)
    g.add_argument("--spectrum_seed", type=int)
    g.add_argument("--coord_scale", type=float)
    g.add_argument("--scan_chunk", type=int)
    g.add_argument("--checkpoint_scan", type=str2bool, metavar="BOOL")

    g = p.add_argument_group("data")
    g.add_argument("--data_dir", type=str, help="directory with .pdb files")
    g.add_argument("--min_len", type=int)
    g.add_argument("--max_len", type=int)
    g.add_argument("--max_bond", type=float)
    g.add_argument("--val_fraction", type=float)

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int)
    g.add_argument("--max_steps", type=int)
    g.add_argument("--batch_size", type=int)
    g.add_argument("--max_residues", type=int)
    g.add_argument("--lr", type=float)
    g.add_argument("--lr_min", type=float)
    g.add_argument("--warmup_steps", type=int)
    g.add_argument("--weight_decay", type=float)
    g.add_argument("--grad_clip", type=float)
    g.add_argument("--ema_decay", type=float)
    g.add_argument("--seed", type=int)

    g = p.add_argument_group("losses")
    for term in LOSS_TERMS:
        g.add_argument(f"--lambda_{term}", type=float)
    g.add_argument("--aux_t_min", type=float)

    g = p.add_argument_group("logging")
    g.add_argument("--checkpoint_dir", type=str)
    g.add_argument("--resume", type=str, help="checkpoint to resume from")
    g.add_argument("--output", type=str, help="write a JSON summary of the run to this file")
    g.add_argument("--log_every", type=int)
    g.add_argument("--val_every", type=int)
    g.add_argument("--val_batches", type=int)
    g.add_argument("--sample_lengths", type=int, nargs="*")
    g.add_argument("--n_val_samples", type=int)
    g.add_argument("--sample_steps", type=int)
    g.add_argument("--log_ck", action="store_true", help="log the Rouse mode concentration C_10")
    g.add_argument("--device", type=str)
    return p


def load_config(argv=None) -> dict:
    """Merge the default YAML files, --config files and command-line flags."""
    cli = vars(build_parser().parse_args(argv))
    config = {"resume": None, "output": None, "device": None}
    for path in list(DEFAULT_CONFIGS) + [Path(p) for p in cli.pop("config")]:
        with open(path) as f:
            config.update(yaml.safe_load(f) or {})
    config.update(cli)

    for key in ("data_dir", "checkpoint_dir"):
        path = Path(config[key])
        config[key] = str(path if path.is_absolute() else REPO_ROOT / path)
    return config


# ----------------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------------

class EMA:
    """Exponential moving average of the model weights, kept as a second model."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.n_updates = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.n_updates += 1
        # Warm up the decay so that early averages are not dominated by the init
        decay = min(self.decay, (1 + self.n_updates) / (10 + self.n_updates))
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.lerp_(p.detach(), 1.0 - decay)


def lr_at(step: int, config: dict, total_steps: int) -> float:
    """Linear warm-up followed by cosine decay to lr_min."""
    warmup = config["warmup_steps"]
    if step < warmup:
        return config["lr"] * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return config["lr_min"] + 0.5 * (config["lr"] - config["lr_min"]) * (1 + math.cos(math.pi * progress))


class AuxiliaryLosses:
    """Auxiliary losses on the predicted clean structure x1_hat = x_t + (1 - t) v."""

    def __init__(self, config: dict):
        self.weights = {term: float(config[f"lambda_{term}"]) for term in LOSS_TERMS}
        self.scale = config["coord_scale"]
        self.t_min = config["aux_t_min"]
        self.local = LocalDistanceLoss(window=8)
        self.bond = BondLengthLoss(ideal_dist=CA_CA_TRANS)
        self.dist = DistogramLoss()
        self.rg = RadiusOfGyrationLoss()
        self.hbond = HydrogenBondLoss()
        self.rama = RamachandranLoss()
        # Distance supervision restricted to the pairs that carry a beta sheet.
        # The whole-map term above spreads its weight over every pair, and the
        # register pairs are 0.10 per cent of them; see
        # scripts/check_pair_dilution.py.
        self.register = RegisterDistanceLoss(weight=1.0, local_weight=1.0)

    @property
    def needs_hidden(self) -> bool:
        return self.weights["torsion"] > 0

    def __call__(self, model, batch, x1, x_t, t, velocity, omega_logits, hidden) -> dict:
        """
        Returns a dict of unweighted loss terms (only those with non-zero weight).

        Distance-based terms are computed in Angstrom and divided by the
        coordinate scale, so they are in model units like the flow loss.
        """
        losses = {}
        sel = t >= self.t_min
        if not sel.any() or not any(w > 0 for w in self.weights.values()):
            return losses

        w = self.weights
        scale = self.scale
        x1_hat = (x_t + (1.0 - t).view(-1, 1, 1) * velocity)[sel] * scale  # Angstrom
        target = x1[sel] * scale
        mask = torch.ones(target.shape[:2], device=target.device)

        if w["fape"] > 0:
            losses["fape"] = self.local(x1_hat, target, mask) / scale
        if w["bond"] > 0:
            losses["bond"] = self.bond(x1_hat, mask) / scale
        if w["dist"] > 0:
            losses["dist"] = self.dist(x1_hat, torch.cdist(target, target), mask) / scale
        if w["rg"] > 0:
            losses["rg"] = self.rg(x1_hat, mask) / scale
        if w["hbond"] > 0:
            losses["hbond"] = self.hbond(x1_hat)
        if w["rama"] > 0:
            losses["rama"] = self.rama(x1_hat)
        if w["register"] > 0:
            losses["register"] = self.register(x1_hat, target) / scale
        if w["omega"] > 0:
            # Label 1 (cis) where the CA-CA distance to the previous residue is short
            d = (target[:, 1:] - target[:, :-1]).norm(dim=-1)
            labels = (d < 0.5 * (CA_CA_TRANS + 2.96)).long()
            losses["omega"] = F.cross_entropy(omega_logits[sel][:, 1:].reshape(-1, 2), labels.reshape(-1))
        if w["torsion"] > 0:
            # The rotation augmentation does not change torsion angles
            phi, psi, omega = coords_to_torsions(batch["backbone"][sel])
            true = torch.stack([phi, psi, omega], dim=-1)
            pred = model.predict_torsions(hidden[sel], model.time_embed(t[sel]))
            valid = torch.ones_like(true)
            valid[:, 0, 0] = 0  # phi of the first residue
            valid[:, -1, 1] = 0  # psi of the last residue
            valid[:, 0, 2] = 0  # omega of the first residue
            losses["torsion"] = ((1.0 - torch.cos(pred - true)) * valid).sum() / valid.sum()
        return losses

    def total(self, losses: dict) -> torch.Tensor:
        return sum(self.weights[name] * value for name, value in losses.items())


def flow_matching_batch(x1: torch.Tensor, generator=None):
    """Draw t and noise and build (x_t, t, target velocity) for clean data x1."""
    B = x1.shape[0]
    if generator is None:
        t = torch.rand(B, device=x1.device)
        x0 = torch.randn_like(x1)
    else:
        # Stratified times and fixed noise for a low-variance validation loss
        t = ((torch.randperm(B, generator=generator) + torch.rand(B, generator=generator)) / B).to(x1.device)
        x0 = torch.randn(x1.shape, generator=generator).to(x1.device)
    x0 = x0 - x0.mean(dim=1, keepdim=True)
    x_t = (1.0 - t).view(-1, 1, 1) * x0 + t.view(-1, 1, 1) * x1
    return x_t, t, x1 - x0


@torch.no_grad()
def validate(model, loader, config: dict, device) -> dict:
    """Flow-matching loss on the validation split, with fixed rotations, times and noise."""
    model.eval()
    generator = torch.Generator().manual_seed(1234)
    total, count = 0.0, 0
    ck_sum, ck_n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= config["val_batches"]:
            break
        # Fixed random orientations: the model is not rotation-equivariant and
        # is trained on randomly rotated structures
        x1 = batch["ca"]
        rot = random_rotations(x1.shape[0], generator=generator)
        x1 = (x1 @ rot.transpose(1, 2)).to(device) / config["coord_scale"]
        x_t, t, target = flow_matching_batch(x1, generator)
        velocity = model(x_t, t)[0]
        total += F.mse_loss(velocity, target, reduction="sum").item() / 3.0
        count += x1.shape[0] * x1.shape[1]
        if config["log_ck"]:
            # Mode concentration of the encoder output on the clean structure
            hidden = model.hidden_states(x1, torch.ones(x1.shape[0], device=device))
            ck_sum += compute_mode_concentration(hidden, k=10)["C_k"]
            ck_n += 1
    metrics = {"val_loss": total / max(count, 1)}
    if ck_n:
        metrics["C_10"] = ck_sum / ck_n
    return metrics


@torch.no_grad()
def sample_metrics(model, config: dict, device) -> dict:
    """Structure metrics of samples, before and after the final bond projection."""
    model.eval()
    out = {}
    scale = config["coord_scale"]
    generator = torch.Generator(device=device).manual_seed(4321)
    for length in config["sample_lengths"] or []:
        raw = model.sample(
            length, n_samples=config["n_val_samples"], n_steps=config["sample_steps"],
            device=device, project=False, generator=generator,
        )
        projected = project_bonds(raw / scale, CA_CA_TRANS / scale) * scale
        m_raw, m_proj = structure_metrics(raw), structure_metrics(projected)
        out[f"L{length}"] = {
            "raw_bond_mean": m_raw["bond_mean"],
            "raw_bond_std": m_raw["bond_std"],
            **{k: m_proj[k] for k in ("clash_frac", "rg", "rg_ratio", "helix_frac", "strand_frac")},
        }
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv=None) -> dict:
    config = load_config(argv)
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    # ---- data
    need_backbone = config["lambda_torsion"] > 0
    dataset = CADataset(
        config["data_dir"], min_len=config["min_len"], max_len=config["max_len"],
        max_bond=config["max_bond"], with_backbone=need_backbone,
    )
    train_idx, val_idx = dataset.split(config["val_fraction"], seed=0)
    train_sampler = LengthBatchSampler(
        dataset.lengths, config["batch_size"], indices=train_idx, shuffle=True,
        seed=config["seed"], max_residues=config["max_residues"],
    )
    val_sampler = LengthBatchSampler(
        dataset.lengths, config["batch_size"], indices=val_idx, shuffle=False,
        max_residues=config["max_residues"],
    )
    train_loader = DataLoader(dataset, batch_sampler=train_sampler, collate_fn=collate_same_length)
    val_loader = DataLoader(dataset, batch_sampler=val_sampler, collate_fn=collate_same_length)
    print(f"Data: {len(train_idx)} train / {len(val_idx)} val chains from {config['data_dir']} "
          f"({dataset.n_dropped} dropped by the filters), {len(train_sampler)} batches per epoch")

    # ---- model
    model = build_model(config).to(device)
    ema = EMA(model, config["ema_decay"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model']}, d_model={config['d_model']}, n_layers={config['n_layers']}, "
          f"use_physics={config['use_physics']}, spectrum={config['spectrum']}, {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    aux = AuxiliaryLosses(config)

    total_steps = config["epochs"] * len(train_sampler)
    if config["max_steps"]:
        total_steps = min(total_steps, config["max_steps"])

    step, start_epoch, best_val = 0, 0, float("inf")
    if config["resume"]:
        ckpt = torch.load(config["resume"], map_location="cpu", weights_only=False)
        if ckpt.get("format_version", 1) != CHECKPOINT_FORMAT:
            raise RuntimeError(f"{config['resume']} was written by a different version of the "
                               f"architecture and cannot be resumed")
        model.load_state_dict(ckpt["model_state_dict"])
        ema.model.load_state_dict(ckpt["ema_state_dict"])
        ema.n_updates = ckpt.get("ema_updates", ckpt["step"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step, start_epoch, best_val = ckpt["step"], ckpt["epoch"], ckpt.get("best_val", best_val)
        # Continue the random streams instead of replaying them from the seed
        if "sampler_rng_state" in ckpt:
            train_sampler.rng.setstate(ckpt["sampler_rng_state"])
            torch.set_rng_state(ckpt["torch_rng_state"])
            if device.type == "cuda" and ckpt.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"], device)
        print(f"Resumed from {config['resume']} at step {step}")

    def save(name: str, epoch: int, **extra) -> None:
        save_checkpoint(
            ckpt_dir / name, model, config, ema_state=ema.model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), step=step, epoch=epoch,
            ema_updates=ema.n_updates, best_val=best_val,
            sampler_rng_state=train_sampler.rng.getstate(),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            **extra,
        )

    def log(record: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def run_validation(epoch: int) -> dict:
        nonlocal best_val
        metrics = validate(ema.model, val_loader, config, device)
        metrics["samples"] = sample_metrics(ema.model, config, device)
        is_best = metrics["val_loss"] < best_val
        if is_best:
            best_val = metrics["val_loss"]
            save("best_model.pt", epoch, val_metrics=metrics)
        save("last.pt", epoch, val_metrics=metrics)
        line = f"[val] step {step}: val_loss={metrics['val_loss']:.4f}{' (best)' if is_best else ''}"
        if "C_10" in metrics:
            line += f", C_10={metrics['C_10']:.3f}"
        print(line)
        for name, m in metrics["samples"].items():
            print(f"      samples {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
        log({"type": "val", "step": step, "epoch": epoch, **metrics})
        return metrics

    # ---- training loop
    running, last_val, grad_norm = {}, {}, float("nan")
    t_start = t_log = time.time()
    done = step >= total_steps
    epoch = start_epoch
    while not done:
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            x1 = batch["ca"]
            rot = random_rotations(x1.shape[0], device=device)
            x1 = (x1 @ rot.transpose(1, 2)) / config["coord_scale"]
            x_t, t, target = flow_matching_batch(x1)

            lr = lr_at(step, config, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            out = model(x_t, t, return_hidden=aux.needs_hidden)
            velocity, omega_logits = out[0], out[1]
            hidden = out[2] if aux.needs_hidden else None

            losses = {"fm": F.mse_loss(velocity, target)}
            aux_losses = aux(model, batch, x1, x_t, t, velocity, omega_logits, hidden)
            loss = losses["fm"] + aux.total(aux_losses)
            losses.update(aux_losses)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {losses}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["grad_clip"] or float("inf")
            ).item()
            optimizer.step()
            ema.update(model)
            step += 1

            running.setdefault("loss", []).append(loss.item())
            for name, value in losses.items():
                running.setdefault(name, []).append(value.item())

            if step % config["log_every"] == 0:
                now = time.time()
                means = {k: sum(v) / len(v) for k, v in running.items()}
                rate = config["log_every"] / (now - t_log)
                print(f"step {step}/{total_steps} (epoch {epoch}) "
                      + " ".join(f"{k}={v:.4f}" for k, v in means.items())
                      + f" grad={grad_norm:.2f} lr={lr:.2e} {rate:.2f} it/s")
                log({"type": "train", "step": step, "epoch": epoch, "lr": lr,
                     "grad_norm": grad_norm, "it_per_s": rate, **means})
                running, t_log = {}, now

            if step % config["val_every"] == 0:
                last_val = run_validation(epoch)
                model.train()

            if step >= total_steps:
                done = True
                break
        else:
            epoch += 1  # only an epoch that ran to its end is counted

    if not last_val or step % config["val_every"] != 0:
        last_val = run_validation(epoch)

    summary = {
        "model": config["model"],
        "use_physics": config["use_physics"],
        "spectrum": config["spectrum"],
        "spectrum_alpha": config["spectrum_alpha"],
        "seed": config["seed"],
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "weight_decay": config["weight_decay"],
        "d_model": config["d_model"],
        "n_layers": config["n_layers"],
        "min_len": config["min_len"],
        "max_len": config["max_len"],
        "n_params": n_params,
        "steps": step,
        "train_time_s": time.time() - t_start,
        "final_val_loss": last_val["val_loss"],
        "best_val_loss": best_val,
        "final_grad_norm": grad_norm,
        "final_C_10": last_val.get("C_10"),
        # scTM needs the external ProteinMPNN + ESMFold pipeline (scripts/evaluate.py)
        "final_scTM": None,
        "samples": last_val.get("samples", {}),
        "checkpoint": str(ckpt_dir / "best_model.pt"),
    }
    if config["output"]:
        Path(config["output"]).parent.mkdir(parents=True, exist_ok=True)
        with open(config["output"], "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {config['output']}")
    return summary


if __name__ == "__main__":
    main()
