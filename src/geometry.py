"""
Geometry utilities for PI-Mamba.

This module implements:
1. Differential NeRF (Natural Extension Reference Frame) for backbone reconstruction.
2. Conversion between coordinates and torsion angles (phi, psi, omega).
3. Bond constraint definitions.

References:
    Parsons et al. (2005) "Practical conversion from torsion space to Cartesian space for in silico protein synthesis".
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

# Ideal geometry constants (Engh & Huber, 1991)
# Bond lengths in Angstroms
L_N_CA = 1.458
L_CA_C = 1.525
L_C_N = 1.329

# Bond angles in Radians
# Note: These are the interior angles. NeRF typically uses the supplementary angle or specific definition.
# We use standard definitions:
# Angle N-CA-C
A_N_CA_C = 1.939  # ~111.2 degrees
# Angle CA-C-N
A_CA_C_N = 2.028  # ~116.2 degrees
# Angle C-N-CA
A_C_N_CA = 2.124  # ~121.7 degrees


def normalize_vector(v: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.norm(v, dim=dim, keepdim=True) + eps)


def dihedral_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
    Compute dihedral angle for atoms a-b-c-d.
    Args:
        a, b, c, d: (..., 3) coordinates
    Returns:
        angle: (...,) in range [-pi, pi]
    """
    u1 = b - a
    u2 = c - b
    u3 = d - c
    
    u1 = normalize_vector(u1)
    u2 = normalize_vector(u2)
    u3 = normalize_vector(u3)
    
    # n1 = normal to plane (a,b,c)
    n1 = torch.cross(u1, u2, dim=-1)
    # n2 = normal to plane (b,c,d)
    n2 = torch.cross(u2, u3, dim=-1)
    
    n1 = normalize_vector(n1)
    n2 = normalize_vector(n2)
    
    # cos_phi = n1 . n2
    # sin_phi = (n1 x n2) . u2
    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(torch.cross(n1, n2, dim=-1) * u2, dim=-1)
    
    return torch.atan2(y, x)


def coords_to_torsions(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract backbone torsion angles (phi, psi, omega) from full backbone coordinates.

    Args:
        coords: (B, L, 3, 3) backbone coordinates [N, CA, C] per residue

    Returns:
        phi: (B, L) phi angles in radians
        psi: (B, L) psi angles in radians
        omega: (B, L) omega angles in radians
    """
    B, L, _, _ = coords.shape
    device = coords.device

    # Extract atoms
    N = coords[:, :, 0, :]   # (B, L, 3)
    CA = coords[:, :, 1, :]  # (B, L, 3)
    C = coords[:, :, 2, :]   # (B, L, 3)

    # Initialize angles
    phi = torch.zeros(B, L, device=device)
    psi = torch.zeros(B, L, device=device)
    omega = torch.zeros(B, L, device=device)

    # Phi: C_{i-1} - N_i - CA_i - C_i
    for i in range(1, L):
        phi[:, i] = dihedral_angle(
            C[:, i-1],   # C_{i-1}
            N[:, i],     # N_i
            CA[:, i],    # CA_i
            C[:, i]      # C_i
        )

    # Psi: N_i - CA_i - C_i - N_{i+1}
    for i in range(L-1):
        psi[:, i] = dihedral_angle(
            N[:, i],     # N_i
            CA[:, i],    # CA_i
            C[:, i],     # C_i
            N[:, i+1]    # N_{i+1}
        )

    # Omega: CA_{i-1} - C_{i-1} - N_i - CA_i
    for i in range(1, L):
        omega[:, i] = dihedral_angle(
            CA[:, i-1],  # CA_{i-1}
            C[:, i-1],   # C_{i-1}
            N[:, i],     # N_i
            CA[:, i]     # CA_i
        )

    return phi, psi, omega

def nerf_step(
    a: torch.Tensor, 
    b: torch.Tensor, 
    c: torch.Tensor, 
    bond_len: float, 
    bond_angle: float, 
    torsion: torch.Tensor
) -> torch.Tensor:
    """
    Place atom d given a, b, c and geometric params.
    
    Args:
        a, b, c: (B, 3) previous atoms
        bond_len: scalar length c-d
        bond_angle: scalar angle b-c-d (radians)
        torsion: (B,) angle a-b-c-d (radians)
        
    Returns:
        d: (B, 3) placed atom
    """
    # b -> c unit vector
    u_bc = normalize_vector(c - b)
    
    # Normal to abc plane
    n = normalize_vector(torch.cross(b - a, u_bc, dim=-1))
    
    # Vector in plane perpendicular to bc
    m = torch.cross(n, u_bc, dim=-1)
    
    # Place d in local frame (defined by c, u_bc, m, n)
    # x (along n), y (along m), z (along u_bc)
    # Wait, convention:
    # d = c + len * (cos(angle) * (-u_bc or u_bc?) + sin(angle) * (sin(tor)*n + cos(tor)*m)?)
    
    # Standard NeRF formulation:
    # D2 = [-r*cos(theta), r*cos(torsion)*sin(theta), r*sin(torsion)*sin(theta)]
    # This is in the frame where C is origin, BC is -x axis? No, that's complex.
    
    # Map to our frame u_bc, m, n
    # We want angle(b-c-d) = bond_angle
    # So d-c makes angle bond_angle with b-c (which is -u_bc)
    # Let theta = bond_angle.
    # component along -u_bc is cos(theta). Component along u_bc is -cos(theta).
    # actually usually 'bond_angle' is the interior angle (e.g. 120 deg).
    # So the deviation from u_bc extension is (180 - bond_angle).
    
    # Let's use the explicit basis construction from the paper/alphafold.
    
    # Re-normalize just in case
    bc = normalize_vector(c - b)
    n = normalize_vector(torch.cross(b - a, bc)) # Normal to plane ABC
    m = torch.cross(n, bc) # Perpendicular in plane
    
    # D position relative to C
    # We use the supplement of the bond angle for the rotation
    # theta_supp = pi - bond_angle
    
    # d_local = [
    #   len * sin(pi - angle) * cos(torsion),  -> along n? (depends on torsion ref)
    #   len * sin(pi - angle) * sin(torsion),  -> along m?
    #   len * cos(pi - angle)                  -> along bc
    # ]
    
    # Let's stick to a verified implementations (e.g. OpenFold/AlphaFold) logic
    # In frame M = [bc, m, n]^T (rows) ?
    
    # x coordinate (along bc): L * cos(pi - angle)
    d_x = bond_len * torch.cos(torch.tensor(torch.pi) - bond_angle)
    
    # radial r = L * sin(pi - angle)
    r = bond_len * torch.sin(torch.tensor(torch.pi) - bond_angle)
    
    # torsion determines x/y in the normal plane
    # y (along m): r * cos(torsion)
    # z (along n): r * sin(torsion)
    
    d_m = r * torch.cos(torsion)
    d_n = r * torch.sin(torsion)
    
    # d = c + d_x * bc + d_m * m + d_n * n
    
    # Note: Torch.cos/sin expect float or tensor
    d = c + d_x * bc + \
        (r * torch.cos(torsion)).unsqueeze(-1) * m + \
        (r * torch.sin(torsion)).unsqueeze(-1) * n
        
    return d

def nerf_build_backbone(
    phi: torch.Tensor,
    psi: torch.Tensor,
    omega: torch.Tensor,
    init_coords: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Reconstruct backbone (N, CA, C) from torsion angles using NeRF.
    
    Args:
        phi: (B, L)
        psi: (B, L)
        omega: (B, L)
        init_coords: (B, 3, 3) Initial N, CA, C for first residue? 
                     Or usually we place first residue at origin.
                     
    Returns:
        coords: (B, L, 3, 3) N, CA, C coordinates
    """
    B, L = phi.shape
    device = phi.device
    
    coords = torch.zeros(B, L, 3, 3, device=device)
    
    # Initialize first residue (arbitrary placement)
    # N at (-1.458, 0, 0)
    # CA at (0, 0, 0)
    # C at (1.525, 0, 0) ? angle?
    
    # Place N1, CA1, C1 in x-y plane
    n1 = torch.tensor([-L_N_CA, 0., 0.], device=device).view(1, 3).expand(B, 3)
    ca1 = torch.tensor([0., 0., 0.], device=device).view(1, 3).expand(B, 3)
    
    # C1 placement: Bond L_CA_C, angle N-CA-C
    # In x-y plane
    angle_c = torch.tensor(torch.pi) - A_N_CA_C
    c1_x = L_CA_C * torch.cos(angle_c)
    c1_y = L_CA_C * torch.sin(angle_c)
    c1 = torch.tensor([c1_x, c1_y, 0.], device=device).view(1, 3).expand(B, 3)
    
    coords[:, 0, 0] = n1
    coords[:, 0, 1] = ca1
    coords[:, 0, 2] = c1
    
    # Iterative reconstruction
    prev_n = n1
    prev_ca = ca1
    prev_c = c1
    
    for i in range(1, L):
        # 1. Place N_i from (N_i-1, CA_i-1, C_i-1) using psi_{i-1} ?? No.
        # Connection i-1 to i is the peptide bond.
        # Atoms: CA_{i-1}, C_{i-1}, N_i
        # Torsion: psi_{i-1} corresponds to N-CA-C-N
        # Bond: C_{i-1}-N_i (Length L_C_N)
        # Angle: CA-C-N (A_CA_C_N)
        # Torsion: psi_{i-1}
        
        # Wait, definitions:
        # phi_i: C_{i-1}-N_i-CA_i-C_i
        # psi_i: N_i-CA_i-C_i-N_{i+1}
        # omega_i: CA_{i-1}-C_{i-1}-N_i-CA_i
        
        # So to place N_i:
        # Ref atoms: CA_{i-1}, C_{i-1}
        # We need 3 previous atoms for NeRF.
        # Chain: ... N_{i-1} - CA_{i-1} - C_{i-1} - N_i
        # Torsion for C_{i-1}-N_i bond is rotation around C_{i-1}-CA_{i-1}? No.
        # Torsion is psi_{i-1} (N_{i-1}-CA_{i-1}-C_{i-1}-N_i)
        
        n_curr = nerf_step(
            coords[:, i-1, 0], # N_{i-1}
            coords[:, i-1, 1], # CA_{i-1}
            coords[:, i-1, 2], # C_{i-1}
            L_C_N,
            A_CA_C_N,
            psi[:, i-1]
        )
        
        # 2. Place CA_i
        # Chain: CA_{i-1} - C_{i-1} - N_i - CA_i
        # Torsion: omega_i (CA_{i-1}-C_{i-1}-N_i-CA_i)
        # Bond: N_i-CA_i (L_N_CA)
        # Angle: C-N-CA (A_C_N_CA)
        
        ca_curr = nerf_step(
            coords[:, i-1, 1], # CA_{i-1}
            coords[:, i-1, 2], # C_{i-1}
            n_curr,            # N_i
            L_N_CA,
            A_C_N_CA,
            omega[:, i]
        )
        
        # 3. Place C_i
        # Chain: C_{i-1} - N_i - CA_i - C_i
        # Torsion: phi_i (C_{i-1}-N_i-CA_i-C_i)
        # Bond: CA_i-C_i (L_CA_C)
        # Angle: N-CA-C (A_N_CA_C)
        
        c_curr = nerf_step(
            coords[:, i-1, 2], # C_{i-1}
            n_curr,            # N_i
            ca_curr,           # CA_i
            L_CA_C,
            A_N_CA_C,
            phi[:, i]
        )
        
        coords[:, i, 0] = n_curr
        coords[:, i, 1] = ca_curr
        coords[:, i, 2] = c_curr
        
    return coords

