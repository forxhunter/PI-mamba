"""
Secondary-Structure Prior Loss (hydrogen-bond geometry proxy)

This module implements an auxiliary loss that encourages the formation of
canonical secondary structure elements (alpha-helices and beta-sheets). With CA
coordinates only, hydrogen bonds cannot be evaluated directly, so the loss
uses the CA-CA distances that the hydrogen-bond patterns imply.

The loss is a prior, not a data term: it asks for a minimum amount of helix-
and sheet-like geometry in every structure. It is off by default in train.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HydrogenBondLoss(nn.Module):
    """
    Auxiliary loss for hydrogen bond satisfaction in protein backbones.
    
    Encourages:
    - α-helix patterns: H-bonds between residue i and i+4
    - β-sheet patterns: H-bonds between distant residues with parallel/antiparallel geometry
    
    The loss is computed from CA-CA distances. Satisfaction of a distance
    criterion is measured with a sigmoid of width ``temperature`` so that the
    loss has a gradient.
    """
    
    def __init__(
        self,
        helix_weight: float = 1.0,
        sheet_weight: float = 0.5,
        ideal_hbond_dist: float = 3.0,  # Å (N-O distance)
        distance_tolerance: float = 0.5,  # Å
        helix_i_to_i4_dist: float = 6.2,  # Å (CA_i to CA_{i+4} in α-helix)
        helix_tolerance: float = 0.8,
        temperature: float = 0.25,  # Å, softness of the satisfaction indicator
    ):
        super().__init__()
        self.helix_weight = helix_weight
        self.sheet_weight = sheet_weight
        self.ideal_hbond_dist = ideal_hbond_dist
        self.distance_tolerance = distance_tolerance
        self.helix_i_to_i4_dist = helix_i_to_i4_dist
        self.helix_tolerance = helix_tolerance
        self.temperature = temperature

    def _satisfied(self, deviation: torch.Tensor, tolerance: float) -> torch.Tensor:
        """Soft indicator in (0, 1) of deviation < tolerance."""
        return torch.sigmoid((tolerance - deviation) / self.temperature)
    
    def compute_helix_loss(self, ca_coords: torch.Tensor) -> torch.Tensor:
        """
        Encourage α-helix formation by targeting ideal i→i+4 CA distances.
        
        In an α-helix:
        - CA_i to CA_{i+4} distance ≈ 6.2 Å
        - CA_i to CA_{i+3} distance ≈ 5.4 Å
        
        Args:
            ca_coords: (B, L, 3) CA coordinates
            
        Returns:
            helix_loss: Scalar loss encouraging helix geometry
        """
        B, L, _ = ca_coords.shape
        
        if L < 5:
            return torch.tensor(0.0, device=ca_coords.device)
        
        # Compute i→i+4 distances
        i4_dists = torch.norm(ca_coords[:, 4:] - ca_coords[:, :-4], dim=-1)
        
        # Compute i→i+3 distances (for 3_10 helix compatibility)
        i3_dists = torch.norm(ca_coords[:, 3:] - ca_coords[:, :-3], dim=-1)
        
        # Soft loss: encourage some positions to adopt helix geometry
        # Using a smooth minimum to allow flexibility
        helix_4_deviation = torch.abs(i4_dists - self.helix_i_to_i4_dist)
        helix_3_deviation = torch.abs(i3_dists - 5.4)
        
        # Encourage at least 30% of residues to be in helix-like geometry
        helix_4_satisfied = self._satisfied(helix_4_deviation, self.helix_tolerance)
        helix_3_satisfied = self._satisfied(helix_3_deviation, self.helix_tolerance)
        
        # Combined helix satisfaction
        helix_frac = (helix_4_satisfied.mean() + helix_3_satisfied.mean()) / 2.0
        
        # Target: at least 30% helix content
        target_helix_frac = 0.30
        helix_loss = F.relu(target_helix_frac - helix_frac)
        
        return helix_loss
    
    def compute_sheet_loss(self, ca_coords: torch.Tensor) -> torch.Tensor:
        """
        Encourage β-sheet formation by detecting strand pairing patterns.
        
        In a β-sheet:
        - Parallel strands: CA_i to CA_j distance ≈ 4.8 Å (j >> i)
        - Antiparallel: alternating short-long pattern
        
        Args:
            ca_coords: (B, L, 3) CA coordinates
            
        Returns:
            sheet_loss: Scalar loss encouraging sheet geometry
        """
        B, L, _ = ca_coords.shape
        
        if L < 10:
            return torch.tensor(0.0, device=ca_coords.device)
        
        # Compute pairwise distance matrix (for long-range contacts)
        # Only consider pairs with sequence separation >= 5
        coords_i = ca_coords.unsqueeze(2)  # (B, L, 1, 3)
        coords_j = ca_coords.unsqueeze(1)  # (B, 1, L, 3)
        
        dist_matrix = torch.norm(coords_i - coords_j, dim=-1)  # (B, L, L)
        
        # Mask for sequence separation >= 5
        seq_sep = torch.abs(torch.arange(L, device=ca_coords.device).unsqueeze(0) - 
                           torch.arange(L, device=ca_coords.device).unsqueeze(1))
        long_range_mask = (seq_sep >= 5).float().unsqueeze(0)
        
        # Sheet contacts: d ≈ 4.8 Å
        ideal_sheet_dist = 4.8
        sheet_deviation = torch.abs(dist_matrix - ideal_sheet_dist)
        sheet_contacts = (self._satisfied(sheet_deviation, 1.0) * long_range_mask).sum(dim=(1, 2))

        # Number of sheet contacts per residue (a residue in the middle of a
        # sheet has two partners; each pair is counted twice in the sum)
        sheet_contact_frac = sheet_contacts / (2.0 * L)
        
        # Target: at least 5% of the residues paired with a strand partner
        target_sheet_frac = 0.05
        sheet_loss = F.relu(target_sheet_frac - sheet_contact_frac.mean())
        
        return sheet_loss
    
    def forward(
        self, 
        ca_coords: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute total hydrogen bond auxiliary loss.
        
        Args:
            ca_coords: (B, L, 3) CA coordinates
            return_components: If True, return dict with individual losses
            
        Returns:
            total_loss: Weighted sum of helix and sheet losses
        """
        helix_loss = self.compute_helix_loss(ca_coords)
        sheet_loss = self.compute_sheet_loss(ca_coords)
        
        total_loss = self.helix_weight * helix_loss + self.sheet_weight * sheet_loss
        
        if return_components:
            return {
                "total": total_loss,
                "helix": helix_loss,
                "sheet": sheet_loss,
            }
        
        return total_loss
