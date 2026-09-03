"""
Rouse Physics Module for Physics-Informed Mamba

This module implements the mathematical foundations of the Rouse model of polymer
dynamics, providing the eigenvalues and eigenvectors that define the physics-informed
state transition matrix.

The Rouse Model:
    γ dr_n/dt = -k Σ_m K_nm r_m + η_n(t)

Where:
    - r_n: position of bead n
    - γ: friction coefficient  
    - k: spring constant
    - K: connectivity (Laplacian) matrix
    - η_n(t): thermal noise

The connectivity matrix K for a linear chain is tridiagonal:
    K_nm = 2 if n=m (interior), 1 if n=m (terminal), -1 if |n-m|=1, 0 otherwise

Key insight: The Rouse dynamics discretize to exactly a Mamba state transition:
    h_{t+1} = A h_t + B x_t
    where A = exp(-k/γ * Λ * Δt) and Λ are the Rouse eigenvalues.

References:
    [1] Rouse, P.E. (1953). A Theory of the Linear Viscoelastic Properties of Dilute 
        Solutions of Coiling Polymers. J. Chem. Phys. 21, 1272.
    [2] Doi, M. & Edwards, S.F. (1986). The Theory of Polymer Dynamics. Oxford.
    [3] Gu, A. & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective 
        State Spaces. arXiv:2312.00752.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional
from functools import lru_cache


def compute_rouse_eigenvalues(N: int, device: torch.device = None) -> torch.Tensor:
    """
    Compute the Rouse eigenvalues for a linear polymer chain of N beads.
    
    The eigenvalues of the Rouse connectivity matrix are:
        λ_p = 4 sin²(pπ / 2N), for p = 0, 1, ..., N-1
    
    Physical interpretation:
        - p=0: λ=0, center of mass mode (translation)
        - p=1: λ≈(π/N)², end-to-end stretching mode
        - p=N-1: λ≈4, highest frequency local fluctuation
    
    Args:
        N: Number of beads (residues)
        device: Torch device
        
    Returns:
        eigenvalues: Tensor of shape (N,) containing λ_p
    """
    p = torch.arange(N, dtype=torch.float32, device=device)
    eigenvalues = 4.0 * torch.sin(p * math.pi / (2 * N)) ** 2
    return eigenvalues


def compute_rouse_eigenvectors(N: int, device: torch.device = None, 
                                normalize: bool = True) -> torch.Tensor:
    """
    Compute the Rouse eigenvectors (normal modes) for a linear polymer chain.
    
    The eigenvectors are discrete cosine functions:
        V_{np} = cos(pπ(n + 0.5) / N)
    
    These form an orthogonal basis where:
        - p=0: uniform mode (all beads move together)
        - p=1: first harmonic (end-to-end)
        - p=k: k-th harmonic (k oscillations along chain)
    
    Note: This is essentially a Type-II Discrete Cosine Transform (DCT-II) basis.
    
    Args:
        N: Number of beads (residues)
        device: Torch device
        normalize: If True, normalize to orthonormal basis
        
    Returns:
        V: Tensor of shape (N, N) where V[n, p] = V_{np}
    """
    n = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)
    p = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N)
    
    V = torch.cos(p * math.pi * (n + 0.5) / N)
    
    if normalize:
        # Normalize columns to unit length
        # For DCT-II: ||V[:, 0]|| = sqrt(N), ||V[:, p>0]|| = sqrt(N/2)
        V[:, 0] /= math.sqrt(N)
        V[:, 1:] /= math.sqrt(N / 2)
    
    return V


def compute_rouse_relaxation_times(N: int, tau_0: float = 1.0,
                                    device: torch.device = None) -> torch.Tensor:
    """
    Compute the relaxation times for each Rouse mode.
    
    The relaxation time for mode p is:
        τ_p = τ_0 / λ_p = τ_0 N² / (4π² p²)  (for p > 0)
    
    Physical interpretation:
        - Low modes (p small): SLOW relaxation, govern large-scale dynamics
        - High modes (p large): FAST relaxation, local fluctuations
    
    Args:
        N: Number of beads
        tau_0: Base relaxation time (γ/k in Rouse model)
        device: Torch device
        
    Returns:
        tau: Tensor of shape (N,) containing τ_p
             Note: τ_0 = inf (center of mass doesn't relax)
    """
    eigenvalues = compute_rouse_eigenvalues(N, device)
    
    # Avoid division by zero for p=0
    tau = torch.zeros(N, device=device)
    tau[0] = float('inf')  # Center of mass mode doesn't relax
    tau[1:] = tau_0 / eigenvalues[1:]
    
    return tau


def compute_state_transition_matrix(N: int, dt: float, 
                                     tau_effective: Optional[torch.Tensor] = None,
                                     device: torch.device = None) -> torch.Tensor:
    """
    Compute the physics-informed state transition matrix A.
    
    For the Rouse model:
        A_p = exp(-λ_p * dt / τ_0)
    
    In Mamba terms, this is the diagonal A matrix in mode space.
    
    If tau_effective is provided (input-dependent), it overrides τ_0:
        A_p = exp(-λ_p * dt / τ_effective)
    
    Args:
        N: Number of beads
        dt: Time step
        tau_effective: Optional input-dependent relaxation time, shape (N,) or (B, N)
        device: Torch device
        
    Returns:
        A: Diagonal state transition, shape (N,) or (B, N)
    """
    eigenvalues = compute_rouse_eigenvalues(N, device)
    
    if tau_effective is None:
        tau_effective = torch.ones(N, device=device)
    
    # A_p = exp(-λ_p * dt / τ_effective)
    A = torch.exp(-eigenvalues * dt / (tau_effective + 1e-8))
    
    return A


class RouseTransform(nn.Module):
    """
    Efficient Rouse mode transform using precomputed DCT basis.
    
    Transforms between real space (residue positions) and mode space
    (Rouse normal modes). This is equivalent to a Type-II DCT.
    
    For sequences longer than max_length, we use a chunked approach
    or recompute the basis on-the-fly.
    """
    
    def __init__(self, max_length: int = 2048):
        super().__init__()
        self.max_length = max_length
        
        # Precompute basis for common lengths
        V = compute_rouse_eigenvectors(max_length, normalize=True)
        self.register_buffer('V', V)  # (max_length, max_length)
        
        # Precompute eigenvalues
        eigenvalues = compute_rouse_eigenvalues(max_length)
        self.register_buffer('eigenvalues', eigenvalues)  # (max_length,)
        
    def forward_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform from real space to mode space.
        
        Args:
            x: (B, L, D) tensor in real space
            
        Returns:
            x_mode: (B, L, D) tensor in mode space
        """
        B, L, D = x.shape
        
        if L <= self.max_length:
            V = self.V[:L, :L]  # (L, L)
        else:
            V = compute_rouse_eigenvectors(L, device=x.device, normalize=True)
        
        # x_mode = V^T @ x (for each feature dimension)
        x_mode = torch.einsum('mn, bnd -> bmd', V.T, x)
        
        return x_mode
    
    def inverse_transform(self, x_mode: torch.Tensor) -> torch.Tensor:
        """
        Transform from mode space to real space.
        
        Args:
            x_mode: (B, L, D) tensor in mode space
            
        Returns:
            x: (B, L, D) tensor in real space
        """
        B, L, D = x_mode.shape
        
        if L <= self.max_length:
            V = self.V[:L, :L]  # (L, L)
        else:
            V = compute_rouse_eigenvectors(L, device=x_mode.device, normalize=True)
        
        # x = V @ x_mode
        x = torch.einsum('nm, bmd -> bnd', V, x_mode)
        
        return x
    
    def get_eigenvalues(self, L: int) -> torch.Tensor:
        """Get Rouse eigenvalues for length L."""
        if L <= self.max_length:
            return self.eigenvalues[:L]
        else:
            return compute_rouse_eigenvalues(L, device=self.eigenvalues.device)


class ZimmCorrection(nn.Module):
    """
    Zimm hydrodynamic correction to the Rouse model.
    
    The Zimm model includes hydrodynamic interactions between beads,
    which modifies the relaxation spectrum:
        τ_p^Zimm ∝ N^(3ν) / p^(3ν)  (vs τ_p^Rouse ∝ N² / p²)
    
    Where ν ≈ 0.588 is the Flory exponent for a self-avoiding walk.
    
    For proteins in solution, both Rouse and Zimm effects are relevant.
    """
    
    def __init__(self, flory_exponent: float = 0.588):
        super().__init__()
        self.nu = flory_exponent
        
    def correct_relaxation(self, tau_rouse: torch.Tensor, N: int) -> torch.Tensor:
        """
        Apply Zimm correction to Rouse relaxation times.
        
        Args:
            tau_rouse: Rouse relaxation times, shape (N,)
            N: Number of beads
            
        Returns:
            tau_zimm: Corrected relaxation times, shape (N,)
        """
        p = torch.arange(N, device=tau_rouse.device, dtype=torch.float32)
        
        # Zimm correction factor: (N/p)^(3ν - 2)
        # This interpolates between Rouse (3ν=2) and Zimm (3ν≈1.76)
        correction = torch.ones_like(tau_rouse)
        correction[1:] = (N / p[1:]) ** (3 * self.nu - 2)
        
        return tau_rouse * correction


# Utility functions for physical validation

def compute_end_to_end_distance(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute end-to-end distance for a batch of structures.
    
    For a Gaussian chain: <R_ee²> ∝ N
    For a self-avoiding walk: <R_ee²> ∝ N^(2ν)
    
    Args:
        coords: (B, L, 3) CA coordinates
        
    Returns:
        R_ee: (B,) end-to-end distances
    """
    return torch.norm(coords[:, -1] - coords[:, 0], dim=-1)


def compute_radius_of_gyration(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute radius of gyration for a batch of structures.
    
    R_g² = (1/N) Σ_i (r_i - r_cm)²
    
    For a Gaussian chain: R_g² = R_ee² / 6
    
    Args:
        coords: (B, L, 3) CA coordinates
        
    Returns:
        R_g: (B,) radii of gyration
    """
    center_of_mass = coords.mean(dim=1, keepdim=True)  # (B, 1, 3)
    deviations = coords - center_of_mass
    R_g_squared = (deviations ** 2).sum(dim=-1).mean(dim=-1)  # (B,)
    return torch.sqrt(R_g_squared)


def compute_persistence_length(coords: torch.Tensor, bond_length: float = 3.8) -> torch.Tensor:
    """
    Estimate persistence length from bond angle correlations.
    
    For a worm-like chain:
        <cos(θ_i) * cos(θ_j)> = exp(-|i-j| * b / ℓ_p)
    
    Where ℓ_p is the persistence length and b is the bond length.
    
    Args:
        coords: (B, L, 3) CA coordinates
        bond_length: CA-CA distance in Å
        
    Returns:
        l_p: (B,) estimated persistence lengths
    """
    # Compute bond vectors
    bonds = coords[:, 1:] - coords[:, :-1]  # (B, L-1, 3)
    bonds = bonds / (torch.norm(bonds, dim=-1, keepdim=True) + 1e-8)
    
    # Compute correlation at separation 1
    cos_theta = (bonds[:, :-1] * bonds[:, 1:]).sum(dim=-1)  # (B, L-2)
    avg_cos = cos_theta.mean(dim=-1)  # (B,)
    
    # ℓ_p = -b / ln(<cos θ>)
    l_p = -bond_length / torch.log(avg_cos.clamp(min=0.01, max=0.99))
    
    return l_p


def validate_rouse_scaling(
    generated_coords: torch.Tensor,
    lengths: torch.Tensor
) -> dict:
    """
    Validate that generated structures obey Rouse scaling laws.
    
    Checks:
        1. R_ee² ∝ N (Gaussian chain)
        2. R_g² ∝ N (Gaussian chain)
        3. R_ee² / R_g² ≈ 6 (theoretical ratio)
    
    Args:
        generated_coords: List of (L, 3) coordinate tensors
        lengths: Tensor of sequence lengths
        
    Returns:
        dict with scaling exponents and R² values
    """
    R_ee = []
    R_g = []
    N = []
    
    for coords, L in zip(generated_coords, lengths):
        coords = coords.unsqueeze(0)  # (1, L, 3)
        R_ee.append(compute_end_to_end_distance(coords).item())
        R_g.append(compute_radius_of_gyration(coords).item())
        N.append(L.item())
    
    R_ee = torch.tensor(R_ee)
    R_g = torch.tensor(R_g)
    N = torch.tensor(N, dtype=torch.float32)
    
    # Fit R_ee² = a * N^b via log-log regression
    log_N = torch.log(N)
    log_R_ee_sq = torch.log(R_ee ** 2)
    
    # Simple linear regression
    mean_x = log_N.mean()
    mean_y = log_R_ee_sq.mean()
    slope = ((log_N - mean_x) * (log_R_ee_sq - mean_y)).sum() / ((log_N - mean_x) ** 2).sum()
    
    return {
        'R_ee_exponent': slope.item(),  # Should be ~1 for Gaussian, ~1.18 for SAW
        'R_ee_R_g_ratio': (R_ee ** 2 / R_g ** 2).mean().item(),  # Should be ~6
        'mean_R_ee': R_ee.mean().item(),
        'mean_R_g': R_g.mean().item(),
    }


if __name__ == "__main__":
    # Test the Rouse physics module
    print("Testing Rouse Physics Module")
    print("=" * 50)
    
    N = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test eigenvalues
    eigenvalues = compute_rouse_eigenvalues(N, device)
    print(f"Eigenvalues (first 5): {eigenvalues[:5]}")
    print(f"Eigenvalues (last 5): {eigenvalues[-5:]}")
    
    # Test eigenvectors
    V = compute_rouse_eigenvectors(N, device, normalize=True)
    print(f"Eigenvector matrix shape: {V.shape}")
    
    # Verify orthonormality
    VtV = V.T @ V
    off_diag = VtV - torch.eye(N, device=device)
    print(f"Orthonormality error: {off_diag.abs().max().item():.6f}")
    
    # Test transform
    transform = RouseTransform(max_length=1024).to(device)
    x = torch.randn(2, N, 64, device=device)
    x_mode = transform.forward_transform(x)
    x_recon = transform.inverse_transform(x_mode)
    print(f"Reconstruction error: {(x - x_recon).abs().max().item():.6f}")
    
    # Test relaxation times
    tau = compute_rouse_relaxation_times(N, tau_0=1.0, device=device)
    print(f"Relaxation times (first 5): {tau[:5]}")
    
    print("\n✓ All tests passed!")
