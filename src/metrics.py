"""
Structure-quality metrics that only need CA coordinates.

These metrics check whether a CA trace looks like a folded protein chain:
virtual bond lengths, steric clashes, compactness and secondary-structure
content. They do not measure designability; scTM / scRMSD need an external
inverse-folding and structure-prediction pipeline (see scripts/evaluate.py).
"""

import math
from typing import Dict

import torch

from .geometry import CA_CA_TRANS, ca_pseudo_dihedrals, kabsch_align

HELIX, STRAND, COIL = 0, 1, 2
SS_NAMES = ("helix", "strand", "coil")


def bond_lengths(ca: torch.Tensor) -> torch.Tensor:
    """CA-CA virtual bond lengths, (..., L-1)."""
    return (ca[..., 1:, :] - ca[..., :-1, :]).norm(dim=-1)


def radius_of_gyration(ca: torch.Tensor) -> torch.Tensor:
    """Radius of gyration (...,) of (..., L, 3) coordinates."""
    centred = ca - ca.mean(dim=-2, keepdim=True)
    return centred.pow(2).sum(dim=-1).mean(dim=-1).sqrt()


def clash_fraction(ca: torch.Tensor, cutoff: float = 3.0, min_sep: int = 2) -> torch.Tensor:
    """
    Fraction of residues (B,) that have another CA closer than ``cutoff``,
    counting only pairs at least ``min_sep`` apart along the chain.
    """
    L = ca.shape[-2]
    dist = torch.cdist(ca, ca)
    idx = torch.arange(L, device=ca.device)
    far = (idx[:, None] - idx[None, :]).abs() >= min_sep
    clashing = ((dist < cutoff) & far).any(dim=-1)
    return clashing.float().mean(dim=-1)


def _min_run(flags: torch.Tensor, min_len: int) -> torch.Tensor:
    """Keep only True runs of at least ``min_len`` along the last dimension."""
    if min_len <= 1:
        return flags
    x = flags.float().unsqueeze(1)  # (B, 1, L)
    # Windows that are fully True
    full = torch.nn.functional.avg_pool1d(x, min_len, stride=1) > 1 - 1e-6  # (B, 1, L-min_len+1)
    # A position is kept if it lies in at least one fully True window
    kept = torch.nn.functional.max_pool1d(
        torch.nn.functional.pad(full.float(), (min_len - 1, min_len - 1)), min_len, stride=1
    )
    return kept.squeeze(1) > 0.5


def assign_secondary_structure(ca: torch.Tensor) -> torch.Tensor:
    """
    Approximate secondary structure from CA geometry (after P-SEA, Labesse et
    al. 1997).

    Residue i is described by the virtual bond angle (i-1, i, i+1) and the
    pseudo-dihedral (i-1, i, i+1, i+2).
        helix:  angle 89 +- 15 deg, dihedral  50 +- 25 deg, runs of >= 4
        strand: angle 124 +- 20 deg, dihedral -170 +- 50 deg, runs of >= 3
    Everything else, including the chain ends, is coil.

    Args:
        ca: (B, L, 3) CA coordinates in Angstrom

    Returns:
        (B, L) long tensor with values HELIX, STRAND or COIL
    """
    B, L, _ = ca.shape
    ss = torch.full((B, L), COIL, dtype=torch.long, device=ca.device)
    if L < 7:  # a helix run needs 4 interior residues, i.e. L - 3 >= 4
        return ss

    v1 = ca[:, :-2] - ca[:, 1:-1]
    v2 = ca[:, 2:] - ca[:, 1:-1]
    cos = torch.nn.functional.cosine_similarity(v1, v2, dim=-1).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cos))[:, :-1]  # residues 1 .. L-3
    dihedral = torch.rad2deg(ca_pseudo_dihedrals(ca))  # residues 1 .. L-3

    def wrapped(x, centre):
        return ((x - centre + 180.0) % 360.0 - 180.0).abs()

    helix = (wrapped(dihedral, 50.0) <= 25.0) & ((angle - 89.0).abs() <= 15.0)
    strand = (wrapped(dihedral, -170.0) <= 50.0) & ((angle - 124.0).abs() <= 20.0)

    helix = _min_run(helix, 4)
    strand = _min_run(strand & ~helix, 3)

    inner = ss[:, 1:L - 2]
    inner[strand] = STRAND
    inner[helix] = HELIX
    return ss


def secondary_structure_fractions(ca: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-chain fraction (B,) of helix, strand and coil residues."""
    ss = assign_secondary_structure(ca)
    return {name: (ss == code).float().mean(dim=-1) for code, name in enumerate(SS_NAMES)}


def structure_metrics(ca: torch.Tensor) -> Dict[str, float]:
    """
    Summary metrics for a batch of CA traces of one length.

    Args:
        ca: (B, L, 3) CA coordinates in Angstrom

    Returns:
        dict of floats averaged over the batch
    """
    L = ca.shape[1]
    bonds = bond_lengths(ca)
    ss = secondary_structure_fractions(ca)
    rg = radius_of_gyration(ca)
    return {
        "bond_mean": bonds.mean().item(),
        "bond_std": bonds.std().item(),
        "bond_dev_frac": ((bonds - CA_CA_TRANS).abs() > 0.1).float().mean().item(),
        "clash_frac": clash_fraction(ca).mean().item(),
        "rg": rg.mean().item(),
        # Ratio to the compact-globule estimate 2.2 * L^0.38 Angstrom
        "rg_ratio": (rg / (2.2 * L ** 0.38)).mean().item(),
        "helix_frac": ss["helix"].mean().item(),
        "strand_frac": ss["strand"].mean().item(),
        "coil_frac": ss["coil"].mean().item(),
    }


def tm_score_superposed(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    TM-score (B,) of two CA traces of the same length with the residue
    correspondence i <-> i, evaluated after a single least-squares
    superposition.

    TM-align searches for the superposition that maximises the score, so this
    value is a lower bound of the TM-score for the same alignment.
    """
    L = pred.shape[1]
    d0 = max(1.24 * (L - 15) ** (1.0 / 3.0) - 1.8, 0.5) if L > 15 else 0.5
    aligned = kabsch_align(pred, ref)
    d = (aligned - ref).norm(dim=-1)
    return (1.0 / (1.0 + (d / d0) ** 2)).mean(dim=-1)


def pairwise_diversity(ca: torch.Tensor, max_pairs: int = 2000) -> float:
    """
    Mean pairwise ``tm_score_superposed`` within a batch of same-length
    samples. Lower means more diverse.
    """
    B = ca.shape[0]
    if B < 2:
        return math.nan
    i, j = torch.triu_indices(B, B, offset=1, device=ca.device)
    if len(i) > max_pairs:
        sel = torch.randperm(len(i), device=ca.device)[:max_pairs]
        i, j = i[sel], j[sel]
    return tm_score_superposed(ca[i], ca[j]).mean().item()


def tm_score(pred: torch.Tensor, ref: torch.Tensor, n_iter: int = 20) -> torch.Tensor:
    """
    TM-score of two CA traces of the same length with the residue
    correspondence i <-> i, normalised by that length.

    Follows the search of the TMscore program (Zhang & Skolnick 2004): the
    superposition is started from fragments of several sizes and positions and
    refined by repeatedly re-fitting on the residues that are currently close,
    keeping the best score found.

    Args:
        pred: (L, 3) or (B, L, 3)
        ref: (L, 3) or (B, L, 3)

    Returns:
        scalar tensor or (B,) tensor
    """
    if pred.dim() == 2:
        return tm_score(pred[None], ref[None], n_iter)[0]

    B, L, _ = pred.shape
    d0 = max(1.24 * (L - 15) ** (1.0 / 3.0) - 1.8, 0.5) if L > 15 else 0.5
    d_search = min(max(d0, 4.5), 8.0)

    # Seed fragments: lengths L, L/2, L/4, ... (at least 4), several start positions each
    seeds = []
    frag = L
    while True:
        starts = sorted({int(s) for s in torch.linspace(0, L - frag, min(L - frag + 1, 12)).tolist()})
        for start in starts:
            mask = torch.zeros(L, device=pred.device)
            mask[start:start + frag] = 1.0
            seeds.append(mask)
        if frag <= 4:
            break
        frag = max(frag // 2, 4)
    seed_mask = torch.stack(seeds)  # (S, L)
    S = seed_mask.shape[0]

    best = pred.new_zeros(B)
    for b in range(B):
        p = pred[b].expand(S, L, 3)
        r = ref[b].expand(S, L, 3)
        mask = seed_mask.clone()
        for _ in range(n_iter):
            d = (kabsch_align(p, r, mask) - r).norm(dim=-1)  # (S, L)
            score = (1.0 / (1.0 + (d / d0) ** 2)).mean(dim=-1)
            best[b] = torch.maximum(best[b], score.max())
            new_mask = (d < d_search).float()
            # Keep at least the three closest residues so that the fit is defined
            few = new_mask.sum(dim=-1) < 3
            if few.any():
                closest = d[few].topk(3, dim=-1, largest=False).indices
                new_mask[few] = torch.zeros_like(new_mask[few]).scatter_(1, closest, 1.0)
            if torch.equal(new_mask, mask):
                break
            mask = new_mask
    return best
