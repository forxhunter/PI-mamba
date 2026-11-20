"""Can the model keep a beta sheet it is handed, or only fail to invent one?

Sampling starts from pure noise, so a model that never produces sheets could
either be unable to represent one, or able to represent one and unable to reach
it from noise. The two have opposite consequences: the first is an architecture
or capacity problem, the second is a coverage problem that more training or a
different sampler can address.

The test separates them. A training structure that contains ladders is noised
part of the way back along the flow-matching path,

    x_t = (1 - t) * x_0 + t * x_1,   x_0 Gaussian,  x_1 the real structure,

and the model integrates from t back to 1. At t close to 1 the model is handed
an almost correct sheet and only has to keep it. At lower t it has to rebuild
more of it. Ladders are counted before and after.

Usage:
    python scripts/sheet_recovery.py \
        --checkpoint checkpoints/paper_512x16/best_model.pt \
        --reference Distilled_trainset --n 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sheet_quality import sheet_features  # noqa: E402
from src.data import read_ca_coords  # noqa: E402
from src.model_io import load_checkpoint  # noqa: E402


def ladder_stats(coords: torch.Tensor) -> Dict[str, object]:
    """Mean features, plus the per-structure values.

    The per-structure arrays matter because two checkpoints are run on the
    same reference structures with the same noise seed, so their retention
    can be compared pair by pair. Comparing means alone cannot tell a real
    difference from the counting noise of a few dozen ladders.
    """
    feats = [sheet_features(c.cpu(), min_ladder=3) for c in coords]
    out: Dict[str, object] = {}
    for k in ("strand_frac", "registered_frac", "n_ladders"):
        v = np.array([f[k] for f in feats], dtype=float)
        out[k] = float(np.nanmean(v))
        out[f"{k}_per_structure"] = [float(x) for x in v]
    return out


@torch.no_grad()
def integrate_from(model, x1: torch.Tensor, t0: float, n_steps: int,
                   generator: torch.Generator) -> torch.Tensor:
    """Noise x1 to time t0 and integrate the flow back to t=1."""
    device = x1.device
    noise = torch.randn(x1.shape, device=device, generator=generator)
    noise = noise - noise.mean(dim=1, keepdim=True)
    x = (1.0 - t0) * noise + t0 * x1

    ts = torch.linspace(t0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t = ts[i].expand(x.shape[0])
        dt = ts[i + 1] - ts[i]
        x = x + model.forward(x, t)[0] * dt
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference", default="Distilled_trainset")
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--n", type=int, default=50,
                   help="structures with at least one ladder to use")
    p.add_argument("--t0", default="0.3,0.5,0.7,0.9",
                   help="comma-separated starting times")
    p.add_argument("--n_steps", type=int, default=200,
                   help="Euler steps for the full interval; each t0 uses the "
                        "same step size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="sheet_recovery.json")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_checkpoint(args.checkpoint, device=device, use_ema=True)
    model.eval()
    scale = ckpt["config"].get("coord_scale", 10.0)
    print(f"{args.checkpoint}: d_model={ckpt['config'].get('d_model')}, "
          f"n_layers={ckpt['config'].get('n_layers')}, coord_scale={scale}")

    # Keep only training structures that actually contain a ladder, otherwise
    # there is nothing for the model to preserve.
    chains: List[torch.Tensor] = []
    for f in sorted(Path(args.reference).glob("*.pdb")):
        if len(chains) >= args.n:
            break
        try:
            ca = read_ca_coords(f)
        except Exception:  # noqa: BLE001
            continue
        if ca is None or len(ca) != args.length:
            continue
        t = torch.as_tensor(ca, dtype=torch.float32)
        t = t - t.mean(dim=0, keepdim=True)
        if sheet_features(t, min_ladder=3)["n_ladders"] >= 1:
            chains.append(t)
    if not chains:
        print("no training structures with a ladder at this length")
        return

    x1 = torch.stack(chains).to(device)
    print(f"{len(chains)} training structures with at least one ladder")

    report = {"reference": ladder_stats(x1)}
    print(f"\n{'start':>8}  {'strand':>8} {'registered':>11} {'ladders':>9}  kept")
    r = report["reference"]
    base = r["n_ladders"]
    print(f"{'input':>8}  {r['strand_frac']:>8.3f} {r['registered_frac']:>11.3f} "
          f"{base:>9.3f}")

    for t0 in (float(v) for v in args.t0.split(",")):
        g = torch.Generator(device=device).manual_seed(args.seed)
        steps = max(1, int(round(args.n_steps * (1.0 - t0))))
        out = integrate_from(model, x1 / scale, t0, steps, g) * scale
        s = ladder_stats(out)
        report[f"t0={t0}"] = {**s, "euler_steps": steps}
        kept = s["n_ladders"] / base * 100 if base else float("nan")
        print(f"{t0:>8.2f}  {s['strand_frac']:>8.3f} {s['registered_frac']:>11.3f} "
              f"{s['n_ladders']:>9.3f}  {kept:>5.0f} %")

    # A pure sample from noise, for reference, with the same post-processing
    g = torch.Generator(device=device).manual_seed(args.seed)
    pure = model.sample(length=args.length, n_samples=len(chains),
                        n_steps=args.n_steps, device=device, project=False,
                        generator=g)
    s = ladder_stats(pure)
    report["from_noise"] = s
    print(f"{'noise':>8}  {s['strand_frac']:>8.3f} {s['registered_frac']:>11.3f} "
          f"{s['n_ladders']:>9.3f}")

    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
