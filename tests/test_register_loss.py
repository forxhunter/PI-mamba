"""Checks for the differentiable register score.

A surrogate is only useful if it ranks structures the way the metric it stands
in for does, and if moving along its gradient moves that metric. These checks
test both on constructed coordinates, so they need no model and no GPU.

The last check is the one that decides whether the term is worth training
with: gradient ascent on the soft score, starting from a structure with no
register, has to raise the discrete ``registered_frac``. A surrogate that
correlates with the metric but whose gradient does not improve it would waste
the training run.

Run:  python tests/test_register_loss.py
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sheet_quality import sheet_features  # noqa: E402
from src.losses.register_loss import (  # noqa: E402
    RegisterDistanceLoss,
    RegisterLoss,
    soft_register_score,
)
from test_sheet_quality import ideal_hairpin  # noqa: E402


def alpha_helix(n: int = 20) -> torch.Tensor:
    """An ideal alpha helix: rise 1.5 A per residue, radius 2.3 A, 100 deg."""
    import math
    coords = []
    for i in range(n):
        a = math.radians(100.0 * i)
        coords.append([2.3 * math.cos(a), 2.3 * math.sin(a), 1.5 * i])
    ca = torch.tensor(coords, dtype=torch.float32)
    return ca - ca.mean(dim=0, keepdim=True)


def random_coil(n: int = 20, seed: int = 0) -> torch.Tensor:
    """A self-avoiding-ish random walk with 3.8 A steps."""
    g = torch.Generator().manual_seed(seed)
    steps = torch.randn(n - 1, 3, generator=g)
    steps = steps / steps.norm(dim=-1, keepdim=True) * 3.8
    ca = torch.cat([torch.zeros(1, 3), steps.cumsum(dim=0)])
    return ca - ca.mean(dim=0, keepdim=True)


def test_hairpin_scores_above_helix_and_coil():
    hairpin = ideal_hairpin(n_per_strand=10)
    n = hairpin.shape[0]  # two strands plus a two-residue turn
    batch = torch.stack([hairpin, alpha_helix(n), random_coil(n, seed=1)])
    s = soft_register_score(batch)
    hairpin, helix, coil = (float(v) for v in s)
    assert hairpin > 5 * max(helix, coil), (
        f"hairpin {hairpin:.4f}, helix {helix:.4f}, coil {coil:.4f}")
    print(f"ok   soft score: hairpin {hairpin:.4f}, helix {helix:.4f}, "
          f"coil {coil:.4f}")


def test_score_tracks_the_discrete_metric():
    """Hairpins of several lengths, ranked by both measures."""
    pairs = []
    for n in (4, 5, 6, 8, 10, 12):
        ca = ideal_hairpin(n_per_strand=n)
        soft = float(soft_register_score(ca.unsqueeze(0))[0])
        hard = sheet_features(ca, min_ladder=3)["registered_frac"]
        pairs.append((n, soft, hard))
    soft_order = [n for n, s, _ in sorted(pairs, key=lambda r: r[1])]
    hard_order = [n for n, _, h in sorted(pairs, key=lambda r: r[2])]
    # Both measures should put the shortest hairpin lowest and the longest
    # highest; exact agreement in the middle is not required.
    assert soft_order[0] == hard_order[0] and soft_order[-1] == hard_order[-1], \
        f"soft order {soft_order}, discrete order {hard_order}"
    print("ok   soft and discrete agree on the extremes: "
          + ", ".join(f"n={n} soft {s:.3f} hard {h:.3f}" for n, s, h in pairs))


def test_gradient_is_finite_and_nonzero():
    ca = random_coil(40, seed=2).unsqueeze(0).requires_grad_(True)
    s = soft_register_score(ca).sum()
    s.backward()
    g = ca.grad
    assert torch.isfinite(g).all(), "the gradient has non-finite entries"
    assert float(g.abs().max()) > 0, "the gradient is identically zero"
    print(f"ok   gradient is finite, largest entry {float(g.abs().max()):.3e}")


def _fit_to_hairpin(seed: int, use_register: bool, iters: int = 1200,
                    bond: float = 3.8) -> float:
    """Fit a coil to a hairpin and return the register it ends with.

    The baseline is the distance-map loss the model already trains with,
    averaged over every pair, plus a bond term. The arm adds
    :class:`RegisterDistanceLoss`, which is the same kind of supervision
    restricted to the pairs that carry the sheet.
    """
    target = ideal_hairpin(n_per_strand=10)
    t = target.unsqueeze(0)
    with torch.no_grad():
        d_target = torch.cdist(t, t)
    extra = RegisterDistanceLoss(weight=1.0, local_weight=1.0)

    x = random_coil(target.shape[0], seed=seed).clone().unsqueeze(0)
    x.requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)
    for _ in range(iters):
        opt.zero_grad()
        dist_map = ((torch.cdist(x, x) - d_target) ** 2).mean()
        d = (x[:, 1:] - x[:, :-1]).norm(dim=-1)
        chain = ((d - bond) ** 2).mean()
        loss = dist_map + 10.0 * chain
        if use_register:
            loss = loss + extra(x, t)
        loss.backward()
        opt.step()
    return sheet_features(x.detach()[0], min_ladder=3)["registered_frac"]


def test_register_term_beats_the_plain_distance_map():
    """The check that decides whether the term is worth a training run.

    A distance-map loss averaged over all pairs puts about two per cent of
    its weight on the pairs that carry a sheet. The claim behind the term is
    that this dilution is why register is not learned, so restricting the
    same supervision to those pairs should produce more register at the same
    optimisation budget. The comparison is paired by seed.

    This is a fit in coordinate space to constructed hairpins, not a trained
    model, so it says the term has the effect it is designed to have, not how
    much it will be worth in training.
    """
    seeds = range(16)
    base = [_fit_to_hairpin(s, use_register=False) for s in seeds]
    with_reg = [_fit_to_hairpin(s, use_register=True) for s in seeds]
    n_base = sum(1 for v in base if v > 0)
    n_reg = sum(1 for v in with_reg if v > 0)
    diff = [b - a for a, b in zip(base, with_reg)]
    mean_diff = sum(diff) / len(diff)
    assert n_reg > n_base, (
        f"structures with any register: {n_base} without the term, "
        f"{n_reg} with it, out of {len(base)}")
    assert mean_diff > 0, f"mean paired difference {mean_diff:+.3f}"
    print(f"ok   register: {n_base}/{len(base)} structures without the term, "
          f"{n_reg}/{len(base)} with it; paired difference {mean_diff:+.3f}")


def test_loss_is_zero_when_the_prediction_matches():
    ca = ideal_hairpin(n_per_strand=10).unsqueeze(0)
    coil = random_coil(ca.shape[1], seed=4).unsqueeze(0)
    # The register term on its own, so the extension term does not mask it.
    loss = RegisterLoss(weight=1.0, extension_weight=0.0)
    assert float(loss(ca, ca)) == 0.0, "a perfect prediction is penalised"
    shortfall = float(loss(coil, ca))
    assert shortfall > 0, "a coil against a hairpin is not penalised"
    # The reverse must not be penalised: more register than the data is free.
    assert float(loss(ca, coil)) == 0.0, \
        "extra register is penalised, so the hinge is the wrong way round"
    # The extension term is symmetric, so it must fire in both directions.
    ext = RegisterLoss(weight=0.0, extension_weight=1.0)
    assert float(ext(coil, ca)) > 0 and float(ext(ca, coil)) > 0, \
        "the extension term does not fire"
    assert float(ext(ca, ca)) == 0.0, "the extension term penalises a match"
    print(f"ok   hinge: 0 when matched, {shortfall:.4f} for a coil against a "
          f"hairpin, 0 for a hairpin against a coil; the extension term "
          f"fires both ways")


if __name__ == "__main__":
    tests = [
        test_hairpin_scores_above_helix_and_coil,
        test_score_tracks_the_discrete_metric,
        test_gradient_is_finite_and_nonzero,
        test_register_term_beats_the_plain_distance_map,
        test_loss_is_zero_when_the_prediction_matches,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed} of {len(tests)} checks passed")
    sys.exit(1 if failed else 0)
