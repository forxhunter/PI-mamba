"""
FAPE-inspired local distance loss for CA-only protein backbones.

Matches local pairwise distance matrices between predicted and target structures,
plus enforces ideal CA-CA bond lengths (3.8 Å).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalDistanceLoss(nn.Module):
    """
    Local pairwise distance matching loss.

    For each pair of residues within a window, match the distance in the
    predicted structure to the distance in the target structure.
    """

    def __init__(self, window: int = 8, clamp: float = 10.0):
        super().__init__()
        self.window = window
        self.clamp = clamp

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, L, 3) predicted CA coordinates
            target: (B, L, 3) target CA coordinates
            mask: (B, L) validity mask (1=valid, 0=padding)
        Returns:
            loss: scalar
        """
        loss = 0.0
        count = 0.0
        for offset in range(1, self.window + 1):
            pred_dist = torch.norm(pred[:, offset:] - pred[:, :-offset], dim=-1)
            target_dist = torch.norm(target[:, offset:] - target[:, :-offset], dim=-1)
            m = mask[:, offset:] * mask[:, :-offset]
            diff = torch.clamp((pred_dist - target_dist).abs(), max=self.clamp)
            loss = loss + (diff * m).sum()
            count = count + m.sum()
        return loss / (count + 1e-8)


class BondLengthLoss(nn.Module):
    """Enforce ideal CA-CA bond length of 3.8 Å."""

    def __init__(self, ideal_dist: float = 3.8):
        super().__init__()
        self.ideal_dist = ideal_dist

    def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, L, 3) CA coordinates
            mask: (B, L) validity mask
        Returns:
            loss: scalar
        """
        dists = torch.norm(coords[:, 1:] - coords[:, :-1], dim=-1)
        bond_mask = mask[:, 1:] * mask[:, :-1]
        return ((dists - self.ideal_dist).abs() * bond_mask).sum() / (bond_mask.sum() + 1e-8)
