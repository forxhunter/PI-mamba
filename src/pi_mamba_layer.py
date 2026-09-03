"""
Physics-Informed Mamba Layer (PI-Mamba)

This module implements the PI-Mamba layer where the state transition matrix A
is derived from the Rouse model of polymer dynamics, not learned arbitrarily.

Key Innovation:
    Standard Mamba: A is a learned diagonal matrix
    PI-Mamba: A = exp(-λ_p * τ) where λ_p are Rouse eigenvalues
    
    The "selectivity" comes from learning the effective relaxation time τ,
    which modulates how fast each mode relaxes.

Physical Interpretation:
    - Low modes (p small): global shape, slow relaxation
    - High modes (p large): local fluctuations, fast relaxation
    - Helices: high stiffness → fast high-mode relaxation
    - Loops: low stiffness → all modes active

References:
    [1] Gu, A. & Dao, T. (2024). Mamba-2: Transformers are SSMs. arXiv:2405.21060.
    [2] Rouse, P.E. (1953). J. Chem. Phys. 21, 1272.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat

from rouse_physics import (
    RouseTransform,
    compute_rouse_eigenvalues,
    compute_state_transition_matrix,
)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (for PyTorch < 2.4 compatibility)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class PhysicsInformedMamba(nn.Module):
    """
    Physics-Informed Mamba layer with Rouse-derived state transition.
    
    The core recurrence is:
        h_p(t+1) = A_p * h_p(t) + B_p * x_p(t)
        y_p(t) = C_p * h_p(t)
    
    Where:
        - Operations happen in Rouse mode space (after DCT transform)
        - A_p = exp(-λ_p * dt / τ) with λ_p the Rouse eigenvalues
        - τ is input-dependent (learned), providing selectivity
        - B, C are learned projections
    
    Args:
        d_model: Model dimension
        d_state: State dimension (N in Mamba notation)
        d_conv: Convolution width for input projection
        expand_factor: Expansion factor for intermediate dimension
        max_length: Maximum sequence length for precomputed transforms
        dt_min: Minimum time step
        dt_max: Maximum time step
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        max_length: int = 2048,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        n_groups: int = 8,
        use_physics: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        self.d_conv = d_conv
        self.max_length = max_length
        self.n_groups = n_groups
        self.use_physics = use_physics
        
        # Rouse transform for mode space operations
        self.rouse_transform = RouseTransform(max_length=max_length)
        
        # Input projection (like Mamba's in_proj)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # Physics-informed parameter projections
        # B, C are standard (learned)
        self.B_proj = nn.Linear(self.d_inner, d_state * n_groups, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state * n_groups, bias=False)
        
        # τ (relaxation time) projection - THIS IS THE KEY PHYSICS PARAMETER
        # Input-dependent τ provides the "selectivity" while respecting physics
        self.tau_proj = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 4),
            nn.SiLU(),
            nn.Linear(self.d_inner // 4, n_groups),
            nn.Softplus(),  # τ must be positive
        )
        
        # Time step projection (dt in Mamba)
        self.dt_proj = nn.Linear(self.d_inner, n_groups, bias=True)
        
        # Initialize dt bias for good default range
        dt_init_std = 0.02
        nn.init.uniform_(self.dt_proj.bias, math.log(dt_min), math.log(dt_max))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Layer norm (RMSNorm in Mamba-2)
        self.norm = RMSNorm(self.d_inner)
        
        # D residual (like Mamba)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Non-physics A parameter (for ablation)
        if not use_physics:
            # Learned A parameter for ablation: (d_state, n_groups)
            # This replaces the Rouse eigenvalues λ_p (which mapped to d_state)
            self.A_log = nn.Parameter(torch.randn(d_state, n_groups))
        
    def _compute_physics_aware_A(
        self, 
        tau: torch.Tensor, 
        dt: torch.Tensor, 
        L: int
    ) -> torch.Tensor:
        """
        Compute the physics-informed state transition matrix A.
        
        A_p = exp(-λ_p * dt / τ)
        
        Where λ_p are the Rouse eigenvalues (precomputed).
        
        Args:
            tau: (B, L, n_groups) input-dependent relaxation times
            dt: (B, L, n_groups) time steps
            L: sequence length
            
        Returns:
            A: (B, L, d_state, n_groups) state transition values
        """
        # Get Rouse eigenvalues for this length
        eigenvalues = self.rouse_transform.get_eigenvalues(L)  # (L,)
        
        # We use d_state modes, so take first d_state eigenvalues
        # or tile if d_state > L
        if self.d_state <= L:
            lambda_p = eigenvalues[:self.d_state]  # (d_state,)
        else:
            # Tile eigenvalues for larger state
            repeats = (self.d_state // L) + 1
            lambda_p = eigenvalues.repeat(repeats)[:self.d_state]
        
        # Reshape for broadcasting
        # lambda_p: (d_state,) -> (1, 1, d_state, 1)
        lambda_p = lambda_p.view(1, 1, -1, 1)
        
        # tau: (B, L, n_groups) -> (B, L, 1, n_groups)
        tau = tau.unsqueeze(2)
        
        # dt: (B, L, n_groups) -> (B, L, 1, n_groups)
        dt = dt.unsqueeze(2)
        
        # A = exp(-λ * dt / τ)
        # Add small epsilon to tau for numerical stability
        A = torch.exp(-lambda_p * dt / (tau + 1e-6))
        
        return A  # (B, L, d_state, n_groups)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_physics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass of PI-Mamba layer.
        
        Args:
            x: (B, L, D) input tensor
            return_physics: If True, return physics diagnostics
            
        Returns:
            y: (B, L, D) output tensor
            physics_info: Optional dict with physics diagnostics
        """
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each
        
        # Convolution for local context
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Compute physics-informed parameters
        tau = self.tau_proj(x_conv)  # (B, L, n_groups) - relaxation time
        dt = F.softplus(self.dt_proj(x_conv))  # (B, L, n_groups) - time step
        
        # Project to B, C
        B_val = self.B_proj(x_conv)  # (B, L, d_state * n_groups)
        C_val = self.C_proj(x_conv)  # (B, L, d_state * n_groups)
        
        # Reshape B, C
        B_val = rearrange(B_val, 'b l (n g) -> b l n g', g=self.n_groups)
        C_val = rearrange(C_val, 'b l (n g) -> b l n g', g=self.n_groups)
        
        # Compute A
        if self.use_physics:
            # Physics-informed A
            A = self._compute_physics_aware_A(tau, dt, L)  # (B, L, d_state, n_groups)
        else:
            # Standard learned A (input-independent recurrence dynamics)
            # A_param: (d_state, n_groups)
            A_param = -torch.exp(self.A_log)
            
            # Broadcast to B, L: (1, 1, d_state, n_groups)
            A_param = A_param.unsqueeze(0).unsqueeze(0)
            
            # Discretize: exp(A * dt)
            # dt is (B, L, n_groups) -> (B, L, 1, n_groups)
            # Result A: (B, L, d_state, n_groups)
            A = torch.exp(A_param * dt.unsqueeze(2)) 

        
        # Transform input to mode space
        x_mode = self.rouse_transform.forward_transform(x_conv)  # (B, L, d_inner)
        
        # SSM recurrence in mode space
        # For efficiency, we implement a simplified parallel scan
        # In practice, use the CUDA kernels from Mamba-2
        y = self._ssm_forward(x_mode, A, B_val, C_val)  # (B, L, d_inner)
        
        # Transform back from mode space
        y = self.rouse_transform.inverse_transform(y)
        
        # Apply normalization and gating
        y = self.norm(y)
        y = y * F.silu(z)
        
        # Residual (D term)
        y = y + x_proj * self.D
        
        # Output projection
        y = self.out_proj(y)
        
        if return_physics:
            physics_info = {
                'tau': tau.detach(),  # Learned relaxation times
                'dt': dt.detach(),
                'A_mean': A.mean().item(),
                'A_min': A.min().item(),
                'A_max': A.max().item(),
            }
            return y, physics_info
        
        return y, None
    
    def _ssm_forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        SSM forward pass (simplified, sequential version).
        
        In production, replace with Mamba-2's efficient parallel scan.
        
        h(t+1) = A * h(t) + B * x(t)
        y(t) = C * h(t)
        
        Args:
            x: (B, L, d_inner) input in mode space
            A: (B, L, d_state, n_groups) state transition
            B: (B, L, d_state, n_groups) input projection
            C: (B, L, d_state, n_groups) output projection
            
        Returns:
            y: (B, L, d_inner) output
        """
        B_size, L, d_inner = x.shape
        d_state = self.d_state
        n_groups = self.n_groups
        
        # Reshape x for grouped processing
        # d_inner = n_groups * (d_inner // n_groups)
        group_dim = d_inner // n_groups
        x = rearrange(x, 'b l (g d) -> b l g d', g=n_groups)  # (B, L, n_groups, group_dim)
        
        # Initialize state
        h = torch.zeros(B_size, d_state, n_groups, group_dim, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(L):
            # State update: h = A * h + B * x
            # A: (B, 1, d_state, n_groups) for timestep t
            # B: (B, 1, d_state, n_groups)
            # x: (B, g, d) for timestep t
            
            A_t = A[:, t:t+1, :, :]  # (B, 1, d_state, n_groups)
            B_t = B[:, t, :, :]      # (B, d_state, n_groups)
            C_t = C[:, t, :, :]      # (B, d_state, n_groups)
            x_t = x[:, t, :, :]      # (B, n_groups, group_dim)
            
            # h = A * h + outer(B, x)
            h = A_t.squeeze(1).unsqueeze(-1) * h + torch.einsum('bng, bgd -> bngd', B_t, x_t)
            
            # y = inner(C, h)
            y_t = torch.einsum('bng, bngd -> bgd', C_t, h)  # (B, n_groups, group_dim)
            outputs.append(y_t)
        
        # Stack and reshape
        y = torch.stack(outputs, dim=1)  # (B, L, n_groups, group_dim)
        y = rearrange(y, 'b l g d -> b l (g d)')  # (B, L, d_inner)
        
        return y


class PhysicsInformedMambaBlock(nn.Module):
    """
    Full PI-Mamba block with residual connection and normalization.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        max_length: int = 2048,
        n_groups: int = 8,
        dropout: float = 0.0,
        use_physics: bool = True,
    ):
        super().__init__()
        
        self.norm = RMSNorm(d_model)
        self.pi_mamba = PhysicsInformedMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            max_length=max_length,
            n_groups=n_groups,
            use_physics=use_physics,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, return_physics: bool = False):
        """
        Args:
            x: (B, L, D) input
            return_physics: If True, return physics diagnostics
            
        Returns:
            y: (B, L, D) output
            physics_info: Optional physics diagnostics
        """
        residual = x
        x = self.norm(x)
        x, physics_info = self.pi_mamba(x, return_physics=return_physics)
        x = self.dropout(x)
        y = x + residual
        
        return y, physics_info


if __name__ == "__main__":
    # Test PI-Mamba layer
    print("Testing Physics-Informed Mamba Layer")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create layer
    layer = PhysicsInformedMambaBlock(
        d_model=256,
        d_state=64,
        n_groups=8,
    ).to(device)
    
    # Test forward
    x = torch.randn(2, 100, 256, device=device)
    y, physics = layer(x, return_physics=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Physics info: {physics}")
    
    # Check gradients
    loss = y.sum()
    loss.backward()
    
    print(f"tau_proj grad norm: {layer.pi_mamba.tau_proj[0].weight.grad.norm().item():.6f}")
    
    # Count parameters
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"Total parameters: {n_params:,}")
    
    print("\n✓ All tests passed!")
