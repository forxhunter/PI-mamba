"""Does the distance-map loss under-weight the pairs that carry a sheet?

``RegisterDistanceLoss`` restricts distance supervision to the pairs in
register in the reference structure. The reasoning is that those pairs are a
small part of the distance map, so a loss averaged over the whole map puts
little weight on them. Two things have to hold for that to matter:

  * the register pairs really are a small fraction of the map, and
  * the model's distance error on them is not smaller than its error
    elsewhere, otherwise there is nothing to gain by weighting them up.

Both are measured here on the training structures, using the endpoint the
model's velocity implies at a given ``t``, which is what the loss would see.

Usage:
    python scripts/check_pair_dilution.py \
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
from src.losses.register_loss import register_pair_mask, MIN_SEP  # noqa: E402
from src.model_io import load_checkpoint  # noqa: E402


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference", default="Distilled_trainset")
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--t", default="0.3,0.5,0.7,0.9",
                   help="comma-separated times at which to form the endpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="pair_dilution.json")
    args = p.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_checkpoint(args.checkpoint, device=device, use_ema=True)
    model.eval()
    scale = ckpt["config"].get("coord_scale", 10.0)

    chains: List[torch.Tensor] = []
    with_sheet = 0
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
        chains.append(t - t.mean(dim=0, keepdim=True))
        if sheet_features(chains[-1], min_ladder=3)["n_ladders"] >= 1:
            with_sheet += 1
    if not chains:
        print(f"no structures of length {args.length} in {args.reference}")
        return

    x1 = torch.stack(chains).to(device)
    print(f"{len(chains)} structures of length {args.length}, "
          f"{with_sheet} with at least one ladder")

    mask = register_pair_mask(x1)
    L = x1.shape[1]
    idx = torch.arange(L, device=device)
    scored = ((idx[None, :] - idx[:, None]).abs() >= MIN_SEP).float()
    n_pair = mask.sum(dim=(-1, -2))
    frac = float(n_pair.sum() / (scored.sum() * len(chains)))
    print(f"register pairs are {frac * 100:.2f} per cent of the pairs the "
          f"distance map scores, {float(n_pair.mean()):.1f} per structure")
    print(f"{int((n_pair > 0).sum())} of {len(chains)} structures have one")

    d_target = torch.cdist(x1, x1)
    report = {
        "n": len(chains),
        "register_pair_fraction": frac,
        "pairs_per_structure": float(n_pair.mean()),
        "structures_with_pairs": int((n_pair > 0).sum()),
        "by_t": {},
    }

    has = n_pair > 0
    print(f"\n{'t':>6}{'all pairs':>12}{'register':>11}{'ratio':>8}")
    for tv in (float(v) for v in args.t.split(",")):
        g = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn(x1.shape, device=device, generator=g)
        noise = noise - noise.mean(dim=1, keepdim=True)
        xt = ((1.0 - tv) * noise + tv * x1 / scale)
        tt = torch.full((x1.shape[0],), tv, device=device)
        v = model.forward(xt, tt)[0]
        x1_hat = (xt + (1.0 - tv) * v) * scale

        err = (torch.cdist(x1_hat, x1_hat) - d_target) ** 2
        all_err = float((err * scored).sum() / (scored.sum() * len(chains)))
        reg_err = float((err * mask)[has].sum() / n_pair[has].sum())
        ratio = reg_err / all_err if all_err > 0 else float("nan")
        report["by_t"][f"t={tv}"] = {"all_pairs": all_err,
                                     "register_pairs": reg_err,
                                     "ratio": ratio}
        print(f"{tv:>6.2f}{all_err:>12.3f}{reg_err:>11.3f}{ratio:>8.2f}")

    print("\nSquared CA-CA distance error in square Angstrom, averaged over "
          "the pairs the distance map scores and over the register pairs. A "
          "ratio above 1 means the model is worse on the sheet pairs than on "
          "the map as a whole, so weighting them up asks for the part it is "
          "getting wrong.")

    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
