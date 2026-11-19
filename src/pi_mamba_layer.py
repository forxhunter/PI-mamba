"""
Physics-Informed Mamba Layer (PI-Mamba)

This module implements the PI-Mamba layer where the state transition matrix A
is derived from the Rouse model of polymer dynamics instead of being a free
learned parameter.

    Standard Mamba: A is a learned diagonal matrix
    PI-Mamba:       A_n = exp(-lambda_n * dt / tau), lambda_n = Rouse eigenvalues

The selectivity comes from the input-dependent relaxation time tau and step dt,
which modulate how fast each state channel relaxes.

Data flow of one layer:
    1. Causal depthwise convolution along the chain (local context).
    2. tau, dt, B, C are predicted per residue from the convolved features.
    3. The features are moved to Rouse mode space (orthonormal DCT-II of the
       chain) and the selective recurrence is run along the mode index, from
       the slowest mode to the fastest.
    4. The result is moved back to real space, normalised, gated and projected.

Two properties of this design are worth stating explicitly.

    The scan runs over the mode index, not over the chain. Step 2 predicts
    tau, dt, B and C per residue, and step 3 uses the parameters of residue p
    at mode p. The index that the recurrence walks is therefore a frequency,
    ordered from the slowest mode to the fastest, and the selectivity
    parameters have no mode-space meaning of their own. The layer is not
    causal along the chain.

    Reordering the rates among the state channels is a symmetry of the layer,
    because B and C are learned: the kernel sum_n C_n B_n exp(-rate_n * u) is
    invariant under a permutation of n. A control for the Rouse spectrum has
    to change the set of rates, not their order (see RouseTransform).

References:
    [1] Gu, A. & Dao, T. (2024). Mamba-2: Transformers are SSMs. arXiv:2405.21060.
    [2] Rouse, P.E. (1953). J. Chem. Phys. 21, 1272.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .rouse_physics import RouseTransform


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
        h(t) = A(t) * h(t-1) + B(t) * x(t)
        y(t) = C(t) . h(t)

    Where:
        - A_n(t) = exp(-lambda_n * dt(t) / tau(t)), lambda_n the Rouse eigenvalues
        - tau, dt are input-dependent (learned), providing selectivity
        - B, C are learned, input-dependent projections

    Args:
        d_model: Model dimension
        d_state: State dimension (N in Mamba notation)
        d_conv: Convolution width for input projection
        expand_factor: Expansion factor for intermediate dimension
        max_length: Kept for backward compatibility (not a hard limit)
        dt_min: Lower end of the initial time-step range
        dt_max: Upper end of the initial time-step range
        n_groups: Number of channel groups that share one (tau, dt, B, C) set
        use_physics: If False, A is a free learned parameter (Learned-A ablation)
        spectrum: Eigenvalue spectrum when use_physics is True
            (``rouse``, ``scrambled`` or ``interpolated``)
        spectrum_alpha: Mixing weight for the ``interpolated`` spectrum
        spectrum_seed: Seed for the ``scrambled`` / ``interpolated`` spectrum
        scan_chunk: Chunk length of the vectorised scan
        checkpoint_scan: Recompute the scan in the backward pass instead of
            storing its intermediates (lower memory, slightly more compute)
        learned_a_ref_length: Chain length whose Rouse rates set the range of
            the initial learned rates when use_physics is False
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
        spectrum: str = "rouse",
        spectrum_alpha: float = 1.0,
        spectrum_seed: int = 42,
        scan_chunk: int = 32,
        checkpoint_scan: bool = True,
        learned_a_ref_length: int = 128,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        self.d_conv = d_conv
        self.max_length = max_length
        self.n_groups = n_groups
        self.use_physics = use_physics
        self.scan_chunk = scan_chunk
        self.checkpoint_scan = checkpoint_scan

        if self.d_inner % n_groups != 0:
            raise ValueError(
                f"d_model * expand_factor ({self.d_inner}) must be divisible by n_groups ({n_groups})"
            )

        # Rouse transform for mode space operations
        self.rouse_transform = RouseTransform(
            max_length=max_length,
            spectrum=spectrum,
            alpha=spectrum_alpha,
            seed=spectrum_seed,
        )

        # Input projection (like Mamba's in_proj)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # B, C are standard (learned)
        self.B_proj = nn.Linear(self.d_inner, d_state * n_groups, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state * n_groups, bias=False)

        # tau (relaxation time) projection. Input-dependent tau provides the
        # selectivity while the spectrum stays fixed by the physics.
        self.tau_proj = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 4),
            nn.SiLU(),
            nn.Linear(self.d_inner // 4, n_groups),
            nn.Softplus(),  # tau must be positive
        )

        # Time step projection (dt in Mamba)
        self.dt_proj = nn.Linear(self.d_inner, n_groups, bias=True)

        # Initialise the bias so that dt = softplus(bias) is log-uniform in
        # [dt_min, dt_max] at the start of training.
        dt_init = torch.exp(
            torch.rand(n_groups) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt_init + torch.log(-torch.expm1(-dt_init)))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm (RMSNorm in Mamba-2)
        self.norm = RMSNorm(self.d_inner)

        # D residual (like Mamba)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Free A parameter for the Learned-A ablation: (d_state, n_groups).
        # The initial rates exp(A_log) are uniform over the range that the
        # Rouse rates of a chain of ``learned_a_ref_length`` beads cover, so
        # the ablation starts with comparable time scales and differs from
        # the physics model by the structure of the spectrum, not by its scale.
        if not use_physics:
            max_rate = self.rouse_transform.get_state_rates(learned_a_ref_length, d_state).max()
            rates = (torch.rand(d_state, n_groups) * max_rate).clamp_min(1e-4)
            self.A_log = nn.Parameter(torch.log(rates))

    def _decay_rates(self, L: int) -> torch.Tensor:
        """
        Per-state decay rates, so that A = exp(-rate * u).

        Returns:
            rate: (d_state, 1) for the physics spectrum (shared by all groups),
                  (d_state, n_groups) for the learned ablation.
        """
        if self.use_physics:
            return self.rouse_transform.get_state_rates(L, self.d_state).unsqueeze(-1)
        return torch.exp(self.A_log)

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
        G, N = self.n_groups, self.d_state

        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # Convolution for local context
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Input-dependent parameters
        tau = self.tau_proj(x_conv)  # (B, L, G) relaxation time
        dt = F.softplus(self.dt_proj(x_conv))  # (B, L, G) time step
        B_val = self.B_proj(x_conv).view(B, L, N, G)
        C_val = self.C_proj(x_conv).view(B, L, N, G)

        # A = exp(-rate * u)
        rate = self._decay_rates(L).to(x.dtype)
        if self.use_physics:
            u = dt / (tau + 1e-6)  # A_n = exp(-lambda_n * dt / tau)
        else:
            u = dt  # A = exp(-exp(A_log) * dt)

        # Transform input to mode space
        x_mode = self.rouse_transform.forward_transform(x_conv)  # (B, L, d_inner)

        # SSM recurrence along the mode index
        if self.checkpoint_scan and self.training and torch.is_grad_enabled():
            y = checkpoint(self._ssm_scan, x_mode, rate, u, B_val, C_val, use_reentrant=False)
        else:
            y = self._ssm_scan(x_mode, rate, u, B_val, C_val)

        # Transform back from mode space
        y = self.rouse_transform.inverse_transform(y)

        # Apply normalization and gating
        y = self.norm(y)
        y = y * F.silu(z)

        # Direct term of the SSM, y = C h + D x, with x the SSM input
        y = y + x_conv * self.D

        # Output projection
        y = self.out_proj(y)

        if return_physics:
            with torch.no_grad():
                A = torch.exp(-rate.view(1, 1, N, -1) * u.unsqueeze(2))
                physics_info = {
                    'tau': tau.detach(),  # (B, L, G) relaxation times
                    'dt': dt.detach(),
                    'A_mean': A.mean().item(),
                    'A_min': A.min().item(),
                    'A_max': A.max().item(),
                }
            return y, physics_info

        return y, None

    def _ssm_scan(
        self,
        x: torch.Tensor,
        rate: torch.Tensor,
        u: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorised selective scan. Exactly equivalent to ``_ssm_forward``.

        Because A is diagonal with A_n(t) = exp(-rate_n * u(t)), the recurrence
        has the closed form

            y(t) = sum_{s <= t} K(t, s) x(s),
            K(t, s) = sum_n C_n(t) B_n(s) exp(-rate_n * (c(t) - c(s))),

        with c the cumulative sum of u. The kernel is evaluated in chunks of
        ``scan_chunk`` positions and the state is carried between chunks, so
        the memory cost is O(L * scan_chunk) instead of O(L^2). Inside a chunk
        c is accumulated from the start of that chunk, so the differences stay
        small and do not lose precision on long chains.

        Args:
            x: (B, L, d_inner) input
            rate: (d_state, 1) or (d_state, n_groups) non-negative decay rates
            u: (B, L, n_groups) non-negative step sizes
            B: (B, L, d_state, n_groups) input projection
            C: (B, L, d_state, n_groups) output projection

        Returns:
            y: (B, L, d_inner) output
        """
        Bsz, L, d_inner = x.shape
        G = self.n_groups
        T = self.scan_chunk

        xg = x.reshape(Bsz, L, G, d_inner // G)
        rate = rate.view(1, 1, self.d_state, -1)  # (1, 1, N, G or 1)
        causal = torch.ones(T, T, device=x.device, dtype=x.dtype).tril()

        h = None  # state at the end of the previous chunk, (B, N, G, d)
        outputs = []
        for i0 in range(0, L, T):
            i1 = min(i0 + T, L)
            n = i1 - i0
            xc, Bc, Cc = xg[:, i0:i1], B[:, i0:i1], C[:, i0:i1]
            # cc[t] is the sum of u over the chunk up to and including t, so
            # cc[t] - cc[s] = c[t] - c[s] and cc[t] = c[t] - c[i0 - 1]
            cc = torch.cumsum(u[:, i0:i1], dim=1)  # (B, t, G)

            # Contributions from inputs inside the chunk
            delta = (cc.unsqueeze(2) - cc.unsqueeze(1)).clamp_min(0.0)  # (B, t, s, G)
            decay = torch.exp(-delta.unsqueeze(3) * rate.unsqueeze(1))  # (B, t, s, N, G)
            K = torch.einsum('btng, btsng -> btsg', Cc, decay * Bc.unsqueeze(1))
            K = K * causal[:n, :n].view(1, n, n, 1)
            y = torch.einsum('btsg, bsgd -> btgd', K, xc)

            # Contribution of the state carried over from earlier chunks
            if h is not None:
                carry = torch.exp(-cc.unsqueeze(2) * rate)  # (B, t, N, G)
                y = y + torch.einsum('btng, bngd -> btgd', Cc * carry, h)

            # State at the end of this chunk
            if i1 < L:
                to_end = torch.exp(-(cc[:, -1:] - cc).unsqueeze(2) * rate)  # (B, s, N, G)
                h_new = torch.einsum('bsng, bsgd -> bngd', Bc * to_end, xc)
                if h is not None:
                    keep = torch.exp(-cc[:, -1].unsqueeze(1) * rate[0])  # (B, N, G)
                    h_new = h_new + keep.unsqueeze(-1) * h
                h = h_new

            outputs.append(y)

        return torch.cat(outputs, dim=1).reshape(Bsz, L, d_inner)

    def _ssm_forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sequential reference implementation of the scan (slow, used in tests).

        h(t) = A(t) * h(t-1) + B(t) * x(t)
        y(t) = C(t) . h(t)

        Args:
            x: (B, L, d_inner) input
            A: (B, L, d_state, n_groups) state transition
            B: (B, L, d_state, n_groups) input projection
            C: (B, L, d_state, n_groups) output projection

        Returns:
            y: (B, L, d_inner) output
        """
        B_size, L, d_inner = x.shape
        G = self.n_groups
        x = x.reshape(B_size, L, G, d_inner // G)

        h = torch.zeros(B_size, self.d_state, G, d_inner // G, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            h = A[:, t].unsqueeze(-1) * h + torch.einsum('bng, bgd -> bngd', B[:, t], x[:, t])
            outputs.append(torch.einsum('bng, bngd -> bgd', C[:, t], h))

        return torch.stack(outputs, dim=1).reshape(B_size, L, d_inner)


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
        **layer_kwargs,
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
            **layer_kwargs,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, return_physics: bool = False,
                residual: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, L, D) input
            return_physics: If True, return physics diagnostics
            residual: (B, L, D) tensor the layer output is added to. Defaults
                to ``x``. The backbone passes the un-normalised residual
                stream here and its time-conditioned normalisation as ``x``.

        Returns:
            y: (B, L, D) output
            physics_info: Optional physics diagnostics
        """
        if residual is None:
            residual = x
        x = self.norm(x)
        x, physics_info = self.pi_mamba(x, return_physics=return_physics)
        x = self.dropout(x)
        y = x + residual

        return y, physics_info
