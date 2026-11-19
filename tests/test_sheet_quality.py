"""Checks for the beta-sheet pairing metric in scripts/sheet_quality.py.

The metric decides whether strand-labelled residues form paired, in-register
ladders. It is only useful if it says yes on an ideal sheet and no on a chain
with the same strand label fraction and no partner strand, so both cases are
constructed here from coordinates.

Run:  python tests/test_sheet_quality.py
"""

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sheet_quality import find_ladders, find_pairs, sheet_features  # noqa: E402
from src.metrics import STRAND, assign_secondary_structure  # noqa: E402

RISE = 3.3       # A along the strand axis per residue
PLEAT = 0.95     # A, alternating displacement perpendicular to the sheet
SPACING = 4.8    # A between hydrogen-bonded partners in neighbouring strands


def ideal_hairpin(n_per_strand: int = 10) -> torch.Tensor:
    """Antiparallel two-strand hairpin, built from ideal sheet geometry."""
    coords = []
    for i in range(n_per_strand):
        coords.append([0.0, RISE * i, PLEAT * (-1) ** i])
    # Two turn residues carrying the chain across to the second strand.
    # Their positions are chosen so that all three CA-CA distances across the
    # turn are 3.8 A: mirrored about the midpoint of the two strands, with the
    # pleat flipped and the rise solved from the bond length.
    top = RISE * (n_per_strand - 1)
    turn_x = (SPACING - 3.8) / 2.0
    dy = math.sqrt(3.8 ** 2 - turn_x ** 2 - (2 * PLEAT) ** 2)
    coords.append([turn_x, top + dy, PLEAT])
    coords.append([SPACING - turn_x, top + dy, PLEAT])
    for i in range(n_per_strand):
        coords.append([SPACING, top - RISE * i, PLEAT * (-1) ** (i + 1)])
    return torch.tensor(coords, dtype=torch.float32)


def straight_strand(n: int = 22) -> torch.Tensor:
    """One extended strand with no partner: strand labels, no pairing."""
    return torch.tensor(
        [[0.0, RISE * i, PLEAT * (-1) ** i] for i in range(n)],
        dtype=torch.float32,
    )


def test_bond_lengths_are_physical():
    for name, ca in (("hairpin", ideal_hairpin()), ("strand", straight_strand())):
        d = (ca[1:] - ca[:-1]).norm(dim=-1)
        assert 3.5 < float(d.min()) and float(d.max()) < 4.3, \
            f"{name}: CA-CA {float(d.min()):.2f} to {float(d.max()):.2f} A"
    print("ok   bond lengths of the constructed traces are 3.5 to 4.3 A")


def test_hairpin_is_labelled_strand():
    ca = ideal_hairpin()
    ss = assign_secondary_structure(ca.unsqueeze(0))[0]
    frac = float((ss == STRAND).float().mean())
    assert frac > 0.5, f"only {frac:.2f} of the hairpin is labelled strand"
    print(f"ok   {frac:.2f} of the ideal hairpin is labelled strand")


def test_hairpin_pairs_and_registers():
    ca = ideal_hairpin()
    ss = assign_secondary_structure(ca.unsqueeze(0))[0]
    pairs = find_pairs(ca, ss)
    n_strand = int((ss == STRAND).sum())
    assert len(pairs) >= 0.8 * n_strand, \
        f"only {len(pairs)} of {n_strand} strand residues found a partner"

    ladders = find_ladders(pairs, min_ladder=3)
    assert ladders, "no ladder found in an ideal hairpin"
    assert all(d < 0 for _, d in ladders), \
        f"hairpin register should be antiparallel, got {[d for _, d in ladders]}"

    feats = sheet_features(ca, min_ladder=3)
    assert feats["registered_frac"] > 0.7, \
        f"registered_frac {feats['registered_frac']:.2f} on an ideal hairpin"
    assert feats["parallel_frac"] == 0.0
    print(f"ok   ideal hairpin: paired {feats['paired_strand_frac']:.2f}, "
          f"registered {feats['registered_frac']:.2f}, "
          f"{int(feats['n_ladders'])} antiparallel ladder(s) of "
          f"{feats['mean_ladder_len']:.1f} pairs")


def test_lone_strand_has_no_pairing():
    ca = straight_strand()
    feats = sheet_features(ca, min_ladder=3)
    assert feats["strand_frac"] > 0.5, \
        f"the lone strand should still be labelled strand, got {feats['strand_frac']:.2f}"
    assert feats["registered_frac"] == 0.0, \
        f"a lone strand registered {feats['registered_frac']:.2f}"
    assert feats["paired_strand_frac"] == 0.0
    print(f"ok   lone strand: strand_frac {feats['strand_frac']:.2f}, "
          f"registered 0.00 (the metric separates label from pairing)")


def test_helix_is_not_counted_as_a_sheet():
    """An alpha-helix has i to i+3 CA distances near 5 A; they must not pair."""
    radius, rise, turn = 2.3, 1.5, math.radians(100.0)
    ca = torch.tensor(
        [[radius * math.cos(turn * i), radius * math.sin(turn * i), rise * i]
         for i in range(24)],
        dtype=torch.float32,
    )
    feats = sheet_features(ca, min_ladder=3)
    assert feats["registered_frac"] == 0.0, \
        f"a helix registered {feats['registered_frac']:.2f} as sheet"
    print(f"ok   alpha-helix: strand_frac {feats['strand_frac']:.2f}, "
          f"registered 0.00")


if __name__ == "__main__":
    tests = [
        test_bond_lengths_are_physical,
        test_hairpin_is_labelled_strand,
        test_hairpin_pairs_and_registers,
        test_lone_strand_has_no_pairing,
        test_helix_is_not_counted_as_a_sheet,
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
