"""Is the flow-matching loss coupled to beta register at all?

Training lowers the flow-matching loss. If more training is to fix the
missing beta sheets, the structures the model gets wrong in the loss have to
be the structures whose register it loses. If the two are uncorrelated,
lowering the loss further cannot be expected to raise register, and the
objective has to change rather than the training length.

The test uses the same construction as ``sheet_recovery.py``. Each reference
structure is noised to t0 with a fixed seed, so for one structure there is

  * the per-structure flow-matching error, the mean squared difference between
    the predicted velocity and the true velocity ``x1 - x0`` at that point;
  * the register kept after integrating from t0 to 1, as a fraction of the
    register the structure started with.

Both are per-structure, so they can be ranked against each other. A Spearman
correlation near zero says the loss carries no information about register.

Usage:
    python scripts/loss_vs_register.py \
        --checkpoint checkpoints/paper_512x16/last.pt --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sheet_quality import sheet_features  # noqa: E402
from src.data import read_ca_coords  # noqa: E402
from src.model_io import load_checkpoint  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, with average ranks for ties."""
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if len(a) < 3:
        return float("nan")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average the ranks of tied values
        vs = v[order]
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[j + 1] == vs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference", default="Distilled_trainset")
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--t0", type=float, default=0.9)
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="loss_vs_register.json")
    args = p.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_checkpoint(args.checkpoint, device=device, use_ema=True)
    model.eval()
    scale = ckpt["config"].get("coord_scale", 10.0)

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
        print("no reference structures with a ladder at this length")
        return

    x1 = torch.stack(chains).to(device) / scale
    print(f"{len(chains)} reference structures with at least one ladder, "
          f"t0={args.t0}")

    g = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(x1.shape, device=device, generator=g)
    noise = noise - noise.mean(dim=1, keepdim=True)
    x0 = noise

    # The flow-matching target does not depend on t: v = x1 - x0 everywhere.
    xt = (1.0 - args.t0) * x0 + args.t0 * x1
    t = torch.full((x1.shape[0],), args.t0, device=device)
    pred = model.forward(xt, t)[0]
    target = x1 - x0
    fm_err = ((pred - target) ** 2).mean(dim=(1, 2)).cpu().numpy()

    # Integrate the same points to t=1 and measure the register kept.
    steps = max(1, int(round(args.n_steps * (1.0 - args.t0))))
    ts = torch.linspace(args.t0, 1.0, steps + 1, device=device)
    x = xt.clone()
    for i in range(steps):
        tt = ts[i].expand(x.shape[0])
        x = x + model.forward(x, tt)[0] * (ts[i + 1] - ts[i])
    out = x * scale

    before = np.array([sheet_features(c.cpu(), min_ladder=3)["registered_frac"]
                       for c in x1 * scale], dtype=float)
    after = np.array([sheet_features(c.cpu(), min_ladder=3)["registered_frac"]
                      for c in out], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kept = np.where(before > 0, after / before, np.nan)

    rho_kept = spearman(fm_err, kept)
    rho_after = spearman(fm_err, after)
    rho_before = spearman(fm_err, before)

    print(f"\nflow-matching error   mean {fm_err.mean():.4f}, "
          f"spread {fm_err.std():.4f}, "
          f"{fm_err.min():.4f} to {fm_err.max():.4f}")
    print(f"register kept         mean {np.nanmean(kept):.3f}")
    print(f"\nSpearman of the per-structure flow-matching error against")
    print(f"  register kept          {rho_kept:+.3f}")
    print(f"  register after         {rho_after:+.3f}")
    print(f"  register before (data) {rho_before:+.3f}")

    # Split the structures by loss and compare their register, which is easier
    # to read than a correlation.
    order = np.argsort(fm_err)
    q = max(1, len(order) // 4)
    low, high = order[:q], order[-q:]
    print(f"\n{'quartile':<20}{'fm error':>10}{'kept':>8}{'after':>8}")
    print(f"{'lowest loss':<20}{fm_err[low].mean():>10.4f}"
          f"{np.nanmean(kept[low]):>8.3f}{after[low].mean():>8.3f}")
    print(f"{'highest loss':<20}{fm_err[high].mean():>10.4f}"
          f"{np.nanmean(kept[high]):>8.3f}{after[high].mean():>8.3f}")

    report = {
        "checkpoint": args.checkpoint,
        "t0": args.t0,
        "n": len(chains),
        "spearman_fm_vs_kept": rho_kept,
        "spearman_fm_vs_after": rho_after,
        "spearman_fm_vs_before": rho_before,
        "fm_error": [float(v) for v in fm_err],
        "register_before": [float(v) for v in before],
        "register_after": [float(v) for v in after],
    }
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
