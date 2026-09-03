import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RamachandranLoss(nn.Module):
    """
    Ramachandran Loss for CA-based pseudo-dihedrals.
    
    Since the model outputs CA traces, we compute the pseudo-dihedral angles
    (CA_{i-1}, CA_i, CA_{i+1}, CA_{i+2}) and penalize deviations from 
    allowed protein regions.
    
    This acts as a "validity" constraint for the backbone geometry.
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
        # approximate centers of allowed regions for alpha and beta
        # in pseudo-dihedral space (which differs slightly from phi/psi)
        # But for generic "compactness", we can target standard regions
        self.register_buffer('alpha_center', torch.tensor([math.radians(-60.0), math.radians(-45.0)]))
        self.register_buffer('beta_center', torch.tensor([math.radians(-135.0), math.radians(135.0)]))
        
    def _dihedral(self, p0, p1, p2, p3):
        """Compute dihedral angle between 4 points."""
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        
        n1 = torch.cross(b1, b2)
        n2 = torch.cross(b2, b3)
        
        m1 = torch.cross(n1, F.normalize(b2, dim=-1))
        
        x = (n1 * n2).sum(dim=-1)
        y = (m1 * n2).sum(dim=-1)
        
        return torch.atan2(y, x)

    def forward(self, ca_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ca_coords: [B, L, 3]
        """
        B, L, _ = ca_coords.shape
        if L < 4:
            return torch.tensor(0.0, device=ca_coords.device)
            
        # Extract pseudo-dihedrals
        # We need 4 consecutive CA atoms
        # For simplicity, we can calculate one angle per residue i (using i-1, i, i+1, i+2)
        # Sequence of sliding windows
        
        # P0: 0..L-4
        # P1: 1..L-3
        # P2: 2..L-2
        # P3: 3..L-1
        
        p0 = ca_coords[:, 0:-3]
        p1 = ca_coords[:, 1:-2]
        p2 = ca_coords[:, 2:-1]
        p3 = ca_coords[:, 3:]
        
        angles = self._dihedral(p0, p1, p2, p3) # [B, L-3]
        
        # For CA pseudo-dihedrals, the distribution is roughly:
        # Alpha: ~50 degrees (Note: CA dihedrals are different from Phi/Psi!)
        # Actually, CA-pseudo-dihedrals for Alpha helix are around +50 deg or -50 deg?
        # Literature (Levitt, 1976) says CA torsion is ~50 for alpha, ~180 for beta.
        # Let's target these modes.
        
        # Allowed regions (very rough priors)
        # Helix: ~50 degrees (0.87 rad) or ~ -50? 
        # Standard Right-handed helix pseudo-dihedral is ~50 deg.
        # Beta sheet pseudo-dihedral is ~180 deg (3.14 rad).
        
        # Distance to nearest allowed mode
        d_helix = torch.min(torch.abs(angles - math.radians(50)), torch.abs(angles - math.radians(-50))) # approximate
        d_sheet = torch.abs(torch.abs(angles) - math.pi) # distance to 180 or -180
        
        # Soft minimum: distance to *either* helix or sheet
        min_dist = torch.min(d_helix, d_sheet)
        
        # Penalize if far from any allowed structure
        # e.g., if dist > 30 degrees (0.5 rad)
        threshold = 0.5
        loss = F.relu(min_dist - threshold)
        
        return loss.mean() * self.weight
