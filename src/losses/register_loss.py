"""Supervision for beta-sheet register.

The sheet term in :class:`HydrogenBondLoss` counts residues that have any
partner at about 4.8 A and saturates once five per cent of them do. A residue
can satisfy it by sitting next to any single distant residue, which is what a
compact coil already does, so the term says nothing about whether the strands
are in register. Measurements on this model's samples show the strand label is
produced at roughly the right rate while register is not, so register is what
needs supervision.

This module holds two things.

:func:`soft_register_score` is a differentiable form of the ``registered_frac``
metric in ``scripts/sheet_quality.py``, built from a Gaussian window on the
CA-CA distance, a soft assignment of each residue to a partner, and the
bilinear amount by which consecutive residues choose consecutive partners. It
ranks structures the way the discrete metric does and is cheap, so it is
useful for monitoring. It is not useful as a loss on its own: the window has a
width of 0.6 A, so a residue 15 A from the partner it should have feels
nothing, and gradient descent on it from a coil leaves the discrete metric at
zero. A score with no range cannot assemble a sheet.

:class:`RegisterDistanceLoss` is the term meant for training. It supervises
the CA-CA distances of exactly the pairs that are in register in the reference
structure, and the local spans of the residues those pairs belong to. The
model already has a distance-map loss, but it averages over every pair.
Measured on 200 training structures of length 100 by
``scripts/check_pair_dilution.py``: the register pairs are 0.10 per cent of
the pairs the distance map scores, 9.1 per structure, and 54 of the 200
structures have any. So the existing term puts about a thousandth of its
weight on the geometry that is missing from the samples.

The same measurement shows the model is worse on those pairs than on the map
as a whole, by a factor of 1.9 at t=0.9 and 1.1 to 1.5 at t=0.5 to 0.7, so
weighting them up asks for the part the model is getting wrong. At t=0.3 the
ratio is 0.69, because there the endpoint is an extrapolation from noise and
the error is dominated by the long distances; that is the reason for the
``t_min`` argument of :func:`register_loss_from_velocity`.

Restricting the average costs no new geometry: the target is the real
structure's own distance, and the gradient has unlimited range, unlike the
soft score.

What it did in training
-----------------------

It did not work, and the default weight is 0 for that reason.

The term was run as a continuation from the 100000-step paper-config
checkpoint at ``lambda_register 1.0`` to 200000 steps, against a continuation
from the same checkpoint on the same learning-rate schedule with the term
off. The term itself fell from 0.173 to 0.094 over those 100000 steps, so it
was optimised throughout. Nothing else changed. From noise at length 100,
300 samples per arm at the end of both runs:

                        control    with the term
  registered_frac        0.010         0.011        (after relaxation)
  ladders per sample     0.080         0.077
  registered_frac        0.016         0.012        (bonds projected only)
  ladders per sample     0.150         0.103

Register retention at t=0.9, paired over the same 200 structures, differed by
-0.010 with a standard error of 0.011, so neither arm is ahead. The cost is
real and did not shrink: the flow-matching validation loss ended at 0.4478
against the control's 0.4375, 2.4 per cent higher, the same gap it had at
102500 steps.

So the model learned to place the supervised pairs correctly when it is given
a partly noised structure, and that did not carry over to sampling from
noise. The dilution the term was built to fix is real, and fixing it is not
sufficient.

One number here was wrong at the halfway point and is worth recording. At
25000 steps the register arm looked significantly worse on retention, -0.032
with a standard error of 0.012. That was transient and had gone by the end of
the run. A single paired comparison partway through a training run can show a
difference that does not survive, in either direction.

Anyone reusing this module should read the result as evidence about the
problem rather than about the implementation: a different weight, a different
``aux_t_min``, or training from scratch rather than as a continuation might
behave differently.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# The same window the discrete metric uses, as a centre and a width rather
# than an interval: PAIR_MIN 4.0 to PAIR_MAX 5.8 A.
IDEAL_PAIR_DIST = 4.9
PAIR_WIDTH = 0.6
MIN_SEP = 4

# An extended backbone spreads its own neighbours much further than a helix or
# a coil does. Measured on constructed geometry: an ideal strand has
# ||x[i+1] - x[i-1]|| about 6.5 A and ||x[i+2] - x[i-1]|| about 9.6 A, an ideal
# alpha helix 5.4 and 5.1, and a 3.8 A random walk about 5 and 5 to 6. The
# thresholds sit above the helix and below the strand.
EXT_SPAN2_MIN = 6.2
EXT_SPAN2_SOFT = 0.3
EXT_SPAN3_MIN = 8.0
EXT_SPAN3_SOFT = 1.0


def extension_gate(
    ca_coords: torch.Tensor,
    span2_min: float = EXT_SPAN2_MIN,
    span2_soft: float = EXT_SPAN2_SOFT,
    span3_min: float = EXT_SPAN3_MIN,
    span3_soft: float = EXT_SPAN3_SOFT,
) -> torch.Tensor:
    """(B, L) soft indicator that the backbone is locally extended.

    Register is only meaningful between strands. Without this gate the score
    can be raised by a compact coil, because a residue 3.8 A from its
    neighbour is close to that neighbour's partner as well, so consecutive
    residues choose consecutive partners from chain connectivity alone. The
    gate was added after the ungated score put a random coil at half the
    value of an ideal hairpin and gradient ascent on it left the discrete
    metric at zero.
    """
    B, L, _ = ca_coords.shape
    gate = ca_coords.new_zeros(B, L)
    if L < 4:
        return gate
    span2 = (ca_coords[:, 2:] - ca_coords[:, :-2]).norm(dim=-1)   # centre 1..L-2
    span3 = (ca_coords[:, 3:] - ca_coords[:, :-3]).norm(dim=-1)   # centre 1..L-3
    g2 = torch.sigmoid((span2 - span2_min) / span2_soft)
    g3 = torch.sigmoid((span3 - span3_min) / span3_soft)
    n = min(g2.shape[1], g3.shape[1])
    gate[:, 1:1 + n] = g2[:, :n] * g3[:, :n]
    return gate


def soft_register_score(
    ca_coords: torch.Tensor,
    ideal: float = IDEAL_PAIR_DIST,
    width: float = PAIR_WIDTH,
    tau: float = 0.3,
    min_sep: int = MIN_SEP,
) -> torch.Tensor:
    """Register per residue, differentiable, one value per structure.

    Args:
        ca_coords: (B, L, 3) CA coordinates in Angstrom.
        ideal: centre of the pairing window.
        width: standard deviation of the window.
        tau: temperature of the partner assignment. Small values make each
            residue commit to one partner, which is what the discrete metric
            does; large values spread the assignment over every candidate and
            flatten the score.
        min_sep: pairs closer than this in sequence are not pairing.

    Returns:
        (B,) score. A hairpin whose two strands are fully in register scores
        near 1; a helix or a coil scores near 0.
    """
    if ca_coords.dim() != 3:
        raise ValueError(f"expected (B, L, 3), got {tuple(ca_coords.shape)}")
    B, L, _ = ca_coords.shape
    if L < 2 * min_sep:
        return ca_coords.new_zeros(B)

    d = torch.cdist(ca_coords, ca_coords)
    idx = torch.arange(L, device=ca_coords.device)
    sep = (idx[None, :] - idx[:, None]).abs()
    allowed = (sep >= min_sep).unsqueeze(0)

    # Weight of each pair under the pairing window, restricted to pairs where
    # both residues sit in an extended stretch of backbone.
    e = extension_gate(ca_coords)
    k = torch.exp(-((d - ideal) ** 2) / (2.0 * width ** 2)) * allowed
    k = k * e.unsqueeze(-1) * e.unsqueeze(-2)

    # Soft choice of one partner per residue. The row is normalised by its own
    # mass rather than by a softmax, so a residue with one clear partner puts
    # almost all of its weight there whatever the chain length; a softmax
    # divides by the number of candidates instead, which pushed the score of
    # an ideal hairpin to a fortieth of its discrete value and left gradients
    # around 1e-5.
    mass = k.sum(dim=-1, keepdim=True)
    a = k / (mass + 1e-8)

    # Sharpen the row so that a residue spreading its weight over several
    # candidates, which is what a coil does, counts for less than one that
    # commits. tau = 1 leaves the assignment as it is.
    if tau != 1.0:
        a = a ** (1.0 / max(tau, 1e-3))
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)

    # A residue with no partner in the window should not contribute at all.
    present = 1.0 - torch.exp(-mass.squeeze(-1))
    a = a * present.unsqueeze(-1)

    # Consecutive residues choosing consecutive partners.
    a_i = a[:, :-1, :]
    a_next = a[:, 1:, :]
    antiparallel = (a_i[:, :, 1:] * a_next[:, :, :-1]).sum(dim=(-1, -2))
    parallel = (a_i[:, :, :-1] * a_next[:, :, 1:]).sum(dim=(-1, -2))
    return (antiparallel + parallel) / L


class RegisterLoss(nn.Module):
    """Supervised extension, plus a hinge on the soft register score.

    In flow matching the reference is the true endpoint ``x1`` and the
    prediction is the endpoint the velocity implies,
    ``x1_hat = x_t + (1 - t) * v``.

    Two terms, because one is not enough:

    ``extension``
        The per-residue extension gate of the prediction against the gate of
        the reference, as a squared error. This term is supervised residue by
        residue and has a gradient everywhere, including on a prediction with
        no sheet at all.

    ``register``
        ``relu(score(x1) - score(x1_hat))``, so a prediction with more
        register than the data is not penalised.

    The register term alone cannot start from a coil: its score is a product
    of gates that are all near zero there, so the gradient is about 1e-4 and
    gradient ascent on it leaves the discrete metric at zero. That is the same
    saturation that makes the existing sheet term in
    :class:`HydrogenBondLoss` ineffective, so repeating it would not be worth
    a training run. The extension term supplies the gradient that tells a
    residue which should be in a strand to straighten out, and the register
    term then decides where its partner goes.
    """

    def __init__(
        self,
        weight: float = 1.0,
        extension_weight: float = 1.0,
        ideal: float = IDEAL_PAIR_DIST,
        width: float = PAIR_WIDTH,
        tau: float = 0.3,
        min_sep: int = MIN_SEP,
        coord_scale: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.extension_weight = extension_weight
        self.ideal = ideal
        self.width = width
        self.tau = tau
        self.min_sep = min_sep
        self.coord_scale = coord_scale

    def score(self, ca_coords: torch.Tensor) -> torch.Tensor:
        return soft_register_score(
            ca_coords * self.coord_scale,
            ideal=self.ideal, width=self.width,
            tau=self.tau, min_sep=self.min_sep,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ):
        """
        Args:
            pred: (B, L, 3) predicted structure, in model units if
                ``coord_scale`` is set.
            target: (B, L, 3) reference structure in the same units.
        """
        s_pred = self.score(pred)
        e_pred = extension_gate(pred * self.coord_scale)
        with torch.no_grad():
            s_target = self.score(target)
            e_target = extension_gate(target * self.coord_scale)

        shortfall = F.relu(s_target - s_pred).mean()
        extension = ((e_pred - e_target) ** 2).mean()
        loss = self.weight * shortfall + self.extension_weight * extension
        if return_components:
            return {
                "total": loss,
                "register": shortfall.detach(),
                "extension": extension.detach(),
                "score_pred": s_pred.mean().detach(),
                "score_target": s_target.mean().detach(),
            }
        return loss


@torch.no_grad()
def register_pair_mask(
    ca_coords: torch.Tensor,
    pair_min: float = 4.0,
    pair_max: float = 5.8,
    min_sep: int = MIN_SEP,
    extension_min: float = 0.5,
    require_neighbour: bool = True,
) -> torch.Tensor:
    """(B, L, L) mask of the pairs that form sheet register in a structure.

    A pair qualifies when the two residues are at least ``min_sep`` apart in
    sequence, their distance is inside the pairing window, and both sit in an
    extended stretch of backbone. With ``require_neighbour`` the pair must
    also have a neighbouring pair offset by one in both chains, which is what
    distinguishes a ladder from an isolated contact.

    This is a mask taken from the reference structure, so it needs no
    gradient.
    """
    B, L, _ = ca_coords.shape
    d = torch.cdist(ca_coords, ca_coords)
    idx = torch.arange(L, device=ca_coords.device)
    sep = (idx[None, :] - idx[:, None]).abs().unsqueeze(0)
    e = extension_gate(ca_coords) > extension_min
    m = ((sep >= min_sep) & (d >= pair_min) & (d <= pair_max)
         & e.unsqueeze(-1) & e.unsqueeze(-2))
    if not require_neighbour:
        return m.float()

    # A pair (i, j) keeps its place only if (i+1, j-1) or (i+1, j+1) or the
    # same one step back also pairs, so that single contacts drop out.
    pad = F.pad(m.float(), (1, 1, 1, 1))
    anti = pad[:, 2:, :-2] + pad[:, :-2, 2:]     # (i+1, j-1) and (i-1, j+1)
    para = pad[:, 2:, 2:] + pad[:, :-2, :-2]     # (i+1, j+1) and (i-1, j-1)
    return (m & ((anti + para) > 0)).float()


class RegisterDistanceLoss(nn.Module):
    """Distance supervision restricted to the reference's sheet pairs.

    ``mean over the register pairs of (d_pred - d_ref) ** 2``, with the pairs
    taken from the reference structure. Structures with no sheet contribute
    nothing, so the term is averaged over the structures that have one rather
    than over the batch, which keeps its scale independent of how much sheet
    the batch happens to contain.

    ``huber`` bounds the term for pairs the prediction puts far away, which at
    low ``t`` is most of them.
    """

    def __init__(
        self,
        weight: float = 1.0,
        local_weight: float = 1.0,
        min_sep: int = MIN_SEP,
        coord_scale: float = 1.0,
        huber: Optional[float] = 2.0,
        require_neighbour: bool = True,
    ):
        super().__init__()
        self.weight = weight
        self.local_weight = local_weight
        self.min_sep = min_sep
        self.coord_scale = coord_scale
        self.huber = huber
        self.require_neighbour = require_neighbour

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ):
        """
        Args:
            pred: (B, L, 3) predicted structure, in model units if
                ``coord_scale`` is set.
            target: (B, L, 3) reference structure in the same units.
        """
        scale = self.coord_scale
        mask = register_pair_mask(target * scale, min_sep=self.min_sep,
                                  require_neighbour=self.require_neighbour)
        n_pair = mask.sum(dim=(-1, -2))
        d_pred = torch.cdist(pred * scale, pred * scale)
        with torch.no_grad():
            d_target = torch.cdist(target * scale, target * scale)

        diff = d_pred - d_target
        if self.huber is not None:
            per_pair = F.huber_loss(d_pred, d_target, reduction="none",
                                    delta=self.huber)
        else:
            per_pair = diff ** 2
        per_structure = (per_pair * mask).sum(dim=(-1, -2)) / n_pair.clamp_min(1)

        has_sheet = n_pair > 0
        if bool(has_sheet.any()):
            term = per_structure[has_sheet].mean()
        else:
            term = pred.new_zeros(())

        # The pair distances alone are not enough. Putting residue i at 4.9 A
        # from residue j can be done with a coil, and the discrete metric only
        # counts residues whose own backbone is extended, so a run that drove
        # the pair term to zero left registered_frac at zero. These are the
        # local spans of the residues the reference has in a strand, which is
        # the geometry the pair term cannot see.
        local = pred.new_zeros(())
        if self.local_weight != 0:
            strand = (extension_gate(target * scale) > 0.5).float()
            parts = []
            for span in (2, 3):
                dp = (pred[:, span:] - pred[:, :-span]).norm(dim=-1) * scale
                with torch.no_grad():
                    dt = (target[:, span:] - target[:, :-span]).norm(dim=-1) * scale
                w = strand[:, :-span] * strand[:, span:]
                n = w.sum(dim=-1)
                if self.huber is not None:
                    per = F.huber_loss(dp, dt, reduction="none", delta=self.huber)
                else:
                    per = (dp - dt) ** 2
                parts.append(((per * w).sum(dim=-1) / n.clamp_min(1), n > 0))
            vals = torch.stack([p for p, _ in parts])
            keep = torch.stack([k for _, k in parts])
            if bool(keep.any()):
                local = vals[keep].mean()

        loss = self.weight * term + self.local_weight * local

        if return_components:
            return {
                "total": loss,
                "pair_distance": term.detach(),
                "local_span": local.detach(),
                "pairs_per_structure": n_pair.mean().detach(),
                "structures_with_sheet": has_sheet.float().mean().detach(),
            }
        return loss


def register_loss_from_velocity(
    x_t: torch.Tensor,
    t: torch.Tensor,
    v_pred: torch.Tensor,
    x1: torch.Tensor,
    loss: RegisterLoss,
    t_min: float = 0.0,
) -> torch.Tensor:
    """Apply the register loss to the endpoint a predicted velocity implies.

    At small ``t`` the implied endpoint is a long extrapolation from a point
    that is almost pure noise, so the score there is uninformative and the
    gradient is large. ``t_min`` drops those samples from the term.
    """
    keep = t >= t_min
    if not bool(keep.any()):
        return x_t.new_zeros(())
    shape = (-1,) + (1,) * (x_t.dim() - 1)
    x1_hat = x_t + (1.0 - t.view(shape)) * v_pred
    return loss(x1_hat[keep], x1[keep])
