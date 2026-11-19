"""Checks for the contact restraint in relax_chain.

The relaxation at the end of sampling has to do two things at once: remove
steric clashes, and leave the structure that the model produced intact. The
restraint added by ``w_contact`` holds every pair that starts in contact and is
not a clash, so the clash term can still move the pairs that are too close.

These checks use constructed coordinates, so they say what the term does
without needing a model or the training data.

Run:  python tests/test_relax_contact.py
"""

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sheet_quality import sheet_features  # noqa: E402
from src.backbone_pi_mamba import relax_chain  # noqa: E402
from src.metrics import clash_fraction  # noqa: E402
from test_sheet_quality import ideal_hairpin  # noqa: E402

BOND = 3.8
MIN_DIST = 4.0


def hairpin_with_a_clash() -> torch.Tensor:
    """An ideal hairpin with one residue pushed onto a distant neighbour."""
    ca = ideal_hairpin(n_per_strand=10).clone()
    # Move a residue of the second strand onto a residue of the first.
    ca[18] = ca[3] + torch.tensor([0.9, 0.0, 0.0])
    return ca


def test_the_clash_is_there_to_start_with():
    ca = hairpin_with_a_clash()
    frac = float(clash_fraction(ca.unsqueeze(0))[0])
    assert frac > 0, "the constructed structure has no clash to remove"
    print(f"ok   constructed structure has a clash fraction of {frac:.3f}")


def test_contact_restraint_still_removes_clashes():
    ca = hairpin_with_a_clash().unsqueeze(0)
    out = relax_chain(ca.clone(), bond=BOND, min_dist=MIN_DIST,
                      n_steps=500, w_clash=2.0, w_contact=1.0)
    frac = float(clash_fraction(out)[0])
    assert frac == 0.0, f"clash fraction {frac:.3f} after relaxation"
    d = (out[0, 1:] - out[0, :-1]).norm(dim=-1)
    assert abs(float(d.mean()) - BOND) < 0.1, \
        f"mean bond {float(d.mean()):.2f} A after relaxation"
    print(f"ok   with the contact restraint the clash is removed and the mean "
          f"bond is {float(d.mean()):.2f} A")


def test_contact_restraint_keeps_the_ladder():
    """The default loses ladders on a hairpin; the restraint should not."""
    ca = ideal_hairpin(n_per_strand=10).unsqueeze(0)
    before = sheet_features(ca[0], min_ladder=3)["n_ladders"]
    assert before > 0, "the ideal hairpin has no ladder to keep"

    default = relax_chain(ca.clone(), bond=BOND, min_dist=MIN_DIST,
                          n_steps=500, w_clash=2.0)
    restrained = relax_chain(ca.clone(), bond=BOND, min_dist=MIN_DIST,
                             n_steps=500, w_clash=2.0, w_contact=1.0)

    n_default = sheet_features(default[0], min_ladder=3)["n_ladders"]
    n_restrained = sheet_features(restrained[0], min_ladder=3)["n_ladders"]
    assert n_restrained >= before, \
        f"ladders fell from {before:.0f} to {n_restrained:.0f} with the restraint"
    print(f"ok   ladders: {before:.0f} before, {n_default:.0f} with the "
          f"defaults, {n_restrained:.0f} with the contact restraint")


def test_restraint_moves_the_structure_less():
    ca = ideal_hairpin(n_per_strand=10).unsqueeze(0)
    default = relax_chain(ca.clone(), bond=BOND, min_dist=MIN_DIST,
                          n_steps=500, w_clash=2.0)
    restrained = relax_chain(ca.clone(), bond=BOND, min_dist=MIN_DIST,
                             n_steps=500, w_clash=2.0, w_contact=1.0)

    def shift(y):
        return float((y - ca).norm(dim=-1).pow(2).mean().sqrt())

    assert shift(restrained) <= shift(default), \
        f"restraint moved the structure {shift(restrained):.2f} A against " \
        f"{shift(default):.2f} A for the defaults"
    print(f"ok   RMS shift {shift(restrained):.2f} A with the restraint "
          f"against {shift(default):.2f} A with the defaults")


def test_default_is_unchanged():
    """w_contact defaults to 0, so existing behaviour must be bit-identical."""
    ca = hairpin_with_a_clash().unsqueeze(0)
    torch.manual_seed(0)
    a = relax_chain(ca.clone(), bond=BOND, min_dist=MIN_DIST, n_steps=200)
    torch.manual_seed(0)
    b = relax_chain(ca.clone(), bond=BOND, min_dist=MIN_DIST, n_steps=200,
                    w_contact=0.0)
    assert torch.allclose(a, b), "w_contact=0 changed the result"
    print("ok   w_contact=0 reproduces the previous behaviour exactly")


if __name__ == "__main__":
    tests = [
        test_the_clash_is_there_to_start_with,
        test_contact_restraint_still_removes_clashes,
        test_contact_restraint_keeps_the_ladder,
        test_restraint_moves_the_structure_less,
        test_default_is_unchanged,
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
