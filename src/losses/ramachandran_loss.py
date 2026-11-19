"""
Pseudo-Ramachandran loss on CA pseudo-dihedrals.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry import ca_pseudo_dihedrals


class RamachandranLoss(nn.Module):
    """
    Ramachandran-like loss for CA-based pseudo-dihedrals.

    Since the model outputs CA traces, we compute the pseudo-dihedral angles
    (CA_{i-1}, CA_i, CA_{i+1}, CA_{i+2}) and penalize deviations from the two
    regions populated by regular secondary structure: about +50 degrees
    (right-handed alpha helix) and about 180 degrees (beta strand).

    This is a prior on local backbone geometry, not a data term. It is off by
    default in train.py.
    """

    def __init__(self, weight: float = 1.0, threshold: float = 0.5):
        super().__init__()
        self.weight = weight
        self.threshold = threshold  # radians of free deviation around each mode

    def forward(self, ca_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ca_coords: [B, L, 3]
        """
        if ca_coords.shape[1] < 4:
            return ca_coords.new_zeros(())

        angles = ca_pseudo_dihedrals(ca_coords)  # [B, L-3]

        # Distance to the nearest allowed mode
        d_helix = torch.abs(torch.remainder(angles - math.radians(50.0) + math.pi, 2 * math.pi) - math.pi)
        d_sheet = math.pi - torch.abs(angles)  # distance to +-180 degrees
        min_dist = torch.min(d_helix, d_sheet)

        # Penalize only deviations beyond the threshold
        loss = F.relu(min_dist - self.threshold)

        return loss.mean() * self.weight
