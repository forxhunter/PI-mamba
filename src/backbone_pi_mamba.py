"""
PI-Mamba Backbone for Protein Structure Generation

This module implements the PI-Mamba backbone:
1. Physics-Informed Mamba layers (Rouse-derived state transition)
2. Flow matching on CA coordinates
3. Bond-length projection of the generated CA trace

Architecture:
    Input: (B, L, 3) noisy CA coordinates x_t and time t
    |
    [AdaptiveLayerNorm(t) -> PI-Mamba Block] x N_layers  (with Rouse physics)
    |
    Linear -> (B, L, 3) velocity field

Flow matching convention (used by training and sampling):
    x_0 ~ N(0, I) (noise), x_1 = data, x_t = (1 - t) x_0 + t x_1,
    target velocity v = x_1 - x_0, sampling integrates t from 0 to 1.

Coordinates inside the network are in model units (Angstrom / coord_scale).
``sample`` returns Angstrom.

The structure module carries a torsion head (phi, psi, omega) for full-backbone
reconstruction through NeRF. That head is only meaningful after it has been
trained with full-backbone supervision (``--lambda_torsion`` in train.py); the
CA-only pipeline does not use it.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pi_mamba_layer import PhysicsInformedMambaBlock
from .rouse_physics import (
    compute_end_to_end_distance,
    compute_radius_of_gyration,
    compute_persistence_length,
)
from .geometry import nerf_build_backbone, CA_CA_TRANS


class AdaptiveLayerNorm(nn.Module):
    """Time-conditioned adaptive layer normalization."""
    
    def __init__(self, d_model: int, d_cond: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, d_model * 2),
        )
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input
            cond: (B, d_cond) conditioning (e.g., time embedding)
        """
        scale_shift = self.cond_proj(cond)  # (B, 2*D)
        scale, shift = scale_shift.chunk(2, dim=-1)  # (B, D) each
        scale = scale.unsqueeze(1)  # (B, 1, D)
        shift = shift.unsqueeze(1)  # (B, 1, D)
        
        x = self.norm(x)
        return x * (1 + scale) + shift


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for flow matching.

    The flow time t in [0, 1] is multiplied by ``time_scale`` before the
    sinusoids, as in diffusion models with 1000 discrete steps. Without the
    scaling the largest phase over the whole time range is 1 radian and the
    embedding is close to a linear function of t.
    """

    def __init__(self, d_model: int, max_period: int = 10000, time_scale: float = 1000.0):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        self.time_scale = time_scale
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) time values in [0, 1]
            
        Returns:
            emb: (B, d_model) time embeddings
        """
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device) / half
        )
        args = self.time_scale * t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, d_model)
        return self.mlp(emb)


class StructureModule(nn.Module):
    """
    Structure Module for coordinate and torsion prediction.

    A small stack of self-attention layers on top of the encoder output with a
    coordinate head and a torsion head (phi, psi, omega). It is used by
    ``reconstruct_full_backbone``; the velocity prediction does not go through it.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_pair: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        n_points: int = 4,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.d_model = d_model

        # Refinement layers
        self.layers = nn.ModuleList([
            StructureLayer(d_model, d_pair, n_heads, n_points)
            for _ in range(n_layers)
        ])

        # Predict CA coordinates (for flow matching)
        self.coord_proj = nn.Linear(d_model, 3)

        # Predict torsion angles (for full backbone reconstruction)
        self.torsion_proj = nn.Linear(d_model, 3)  # phi, psi, omega

    def forward(
        self,
        s: torch.Tensor,
        t_emb: torch.Tensor,
        return_torsions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            s: (B, L, D) single representation
            t_emb: (B, D) time embedding
            return_torsions: If True, also return torsion angles

        Returns:
            coords: (B, L, 3) CA coordinates
            torsions: (B, L, 3) [phi, psi, omega] if return_torsions=True
        """
        # Refine through structure layers
        for layer in self.layers:
            s = layer(s, t_emb)

        # Predict CA coordinates
        coords = self.coord_proj(s)  # (B, L, 3)

        if return_torsions:
            # Predict torsion angles
            torsions = self.torsion_proj(s)  # (B, L, 3)
            return coords, torsions

        return coords


class StructureLayer(nn.Module):
    """Single layer of the structure module."""
    
    def __init__(
        self,
        d_model: int,
        d_pair: int,
        n_heads: int,
        n_points: int,
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)

        # Time conditioning, added to the input of the layer. Zero-initialised,
        # so the layer starts out independent of t.
        self.time_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)

        # Global self-attention over the chain
        self.local_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        
        # Point-wise FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, s: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (B, L, D) single representation
            t_emb: (B, D) time embedding
        """
        s = s + self.time_proj(t_emb).unsqueeze(1)

        # Self-attention with residual
        s_norm = self.norm(s)
        s_attn, _ = self.local_attn(s_norm, s_norm, s_norm)
        s = s + s_attn
        
        # FFN with residual
        s = s + self.ffn(self.ffn_norm(s))
        
        return s


class OmegaHead(nn.Module):
    """
    Predicts peptide bond planarity (cis vs trans).
    
    Output: Logits for [trans, cis] (2 classes)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2)  # [trans, cis] logits
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) state
        Returns:
            logits: (B, L, 2)
        """
        return self.mlp(x)


def project_bonds(coords: torch.Tensor, target_dist=CA_CA_TRANS) -> torch.Tensor:
    """
    Rescale every CA-CA virtual bond to the target length.

    The chain is rebuilt from the first residue by a cumulative sum of the
    rescaled bond vectors, so bond directions are kept and bond lengths become
    exact. The result is shifted to keep the centroid of the input.

    Args:
        coords: (B, L, 3) CA coordinates
        target_dist: scalar, or (B, L-1, 1) tensor of per-bond target lengths

    Returns:
        (B, L, 3) projected CA coordinates
    """
    bonds = coords[:, 1:] - coords[:, :-1]
    ideal_bonds = F.normalize(bonds, dim=-1) * target_dist
    new_coords = torch.cat(
        [coords[:, :1], coords[:, :1] + torch.cumsum(ideal_bonds, dim=1)], dim=1
    )
    return new_coords - new_coords.mean(dim=1, keepdim=True) + coords.mean(dim=1, keepdim=True)


def relax_chain(
    coords: torch.Tensor,
    bond: float = CA_CA_TRANS,
    min_dist: float = 4.0,
    min_sep: int = 2,
    local_window: int = 4,
    n_steps: int = 300,
    lr: float = 0.02,
    w_bond: float = 1.0,
    w_clash: float = 1.0,
    w_local: float = 1.0,
    w_contact: float = 0.0,
    contact_cutoff: Optional[float] = None,
) -> torch.Tensor:
    """
    Relax a CA trace so that its virtual bonds have the right length and no two
    distant residues overlap.

    ``project_bonds`` rebuilds the chain from its first residue, so one changed
    bond moves everything after it and distant parts of the chain can be pushed
    into each other; it fixes bond lengths and can make clashes worse. This
    routine instead minimises

        w_bond  * mean((|x_{i+1} - x_i| - bond)^2)
      + w_clash * sum(relu(min_dist - d_ij)^2) / (B L)     over |i - j| >= min_sep
      + w_local * mean((d_ij - d_ij at the start)^2)       over 2 <= |i - j| <= local_window
      + w_contact * mean((d_ij - d_ij at the start)^2)     over the pairs that
                                                           start in contact

    by gradient descent on the coordinates. The third term holds the local
    geometry in place: without it, pushing overlapping residues apart also
    straightens helices, because the distances that define secondary structure
    are free to change. Separating the terms this way is what structure
    refinement does, local restraints plus a global non-overlap term.

    The third term reaches only to ``local_window``, so nothing restrains a pair
    further apart along the chain, and the register of a beta ladder is free to
    drift while the bond and clash terms are satisfied. Measured on 200 training
    structures of length 100, the defaults keep 65 per cent of the ladders that
    the input contains, against 94 per cent for ``project_bonds``, although they
    move the coordinates less. ``w_contact`` adds a restraint on every pair that
    starts in contact and is not a clash, at any separation, which leaves the
    clash term free to fix real overlaps while holding tertiary contacts and
    sheet pairing in place. It is off by default.

    On training structures the restraint keeps every ladder, 104 per cent
    against 65 per cent for the defaults. It does not carry over to raw model
    output: on 100 unprocessed samples of length 100 the ladders kept were 7
    with the defaults and 8 with the restraint, out of 19 in the input, which
    is a difference of one ladder and not a result. Raw samples arrive with a
    bond standard deviation of 0.17 A and a clash fraction of 0.074, so many of
    their contacts are not structure, and holding them works against the clash
    term. The step is worth 0.12 ladders per structure in total, against the
    0.73 that separates the model from its training data, so it is not where
    sheet content is lost.

    The defaults for min_dist and min_sep come from the training structures:
    over pairs at least 2 apart along the chain, the smallest CA-CA distance
    that occurs is 3.65 A and the 0.01th percentile is 4.20 A, so a 4.0 A floor
    excludes almost nothing that real proteins do.

    It changes the structure, so it is a post-processing step, not a
    measurement: report metrics on the relaxed coordinates only, and keep the
    unrelaxed ones if you want to see what the model produced.

    Args:
        coords: (B, L, 3) CA coordinates, in the same unit as bond and min_dist
        bond: target CA-CA distance
        min_dist: closest allowed distance between residues min_sep apart
        min_sep: minimum separation along the chain for a pair to count
        local_window: separations from 2 up to this are held at their initial
            distance, unless that distance is already further than |i - j|
            bonds could reach; separation 1 is left to the bond term
        n_steps: gradient steps
        lr: step size, in units of the coordinates
        w_bond, w_clash, w_local: relative weight of the three terms
        w_contact: weight of the restraint on pairs that start in contact, at
            any separation along the chain; 0 disables it
        contact_cutoff: a pair counts as in contact when it starts at least
            min_dist and at most this far apart; defaults to 2.1 * bond, which
            is 8.0 A for a 3.8 A virtual bond

    Returns:
        (B, L, 3) relaxed coordinates, with the centroid of the input
    """
    B, L, _ = coords.shape
    if L < 2:
        return coords.clone()

    x = coords.detach().clone().requires_grad_(True)
    idx = torch.arange(L, device=coords.device)
    sep = (idx[:, None] - idx[None, :]).abs()
    far = (sep >= min_sep).unsqueeze(0)
    with torch.no_grad():
        target_local = torch.cdist(coords, coords)
    # Separation 1 is left to the bond term, so the two do not pull against each
    # other where the input has a bad bond. A pair further apart than a chain of
    # that many bonds could reach is not restrained either: its geometry is
    # already impossible, so holding it would block the bond term from fixing it.
    reachable = target_local <= sep.unsqueeze(0) * bond
    near = ((sep >= 2) & (sep <= local_window)).unsqueeze(0) & reachable

    # Pairs that start in contact without overlapping. Holding these keeps the
    # register of a sheet and the packing of the fold while the clash term
    # works on the pairs that really are too close.
    use_contact = w_contact > 0
    if use_contact:
        cutoff = contact_cutoff if contact_cutoff is not None else 2.1 * bond
        contact = (
            far
            & reachable
            & (target_local <= cutoff)
            & (target_local >= min_dist)
            & ~near
        )
        n_contact = contact.sum().clamp_min(1)

    optimizer = torch.optim.Adam([x], lr=lr)

    with torch.enable_grad():
        for _ in range(n_steps):
            lengths = (x[:, 1:] - x[:, :-1]).norm(dim=-1)
            bond_term = ((lengths - bond) ** 2).mean()
            dist = torch.cdist(x, x)
            overlap = torch.relu(min_dist - dist) * far
            clash_term = (overlap ** 2).sum() / (B * L)
            local_term = (((dist - target_local) * near) ** 2).sum() / near.sum().clamp_min(1)
            loss = w_bond * bond_term + w_clash * clash_term + w_local * local_term
            if use_contact:
                contact_term = (((dist - target_local) * contact) ** 2).sum() / n_contact
                loss = loss + w_contact * contact_term
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    x = x.detach()
    return x - x.mean(dim=1, keepdim=True) + coords.mean(dim=1, keepdim=True)


class ChainProjection(nn.Module):
    """
    Kinematic projection of a CA trace onto the fixed-bond-length manifold.

    The model generates CA coordinates only, so the constraint that can be
    enforced exactly is the CA-CA virtual bond length (3.8 A for trans
    peptides). Bond angles and torsions of the N, C and O atoms are not
    constrained here.
    """

    def __init__(self, target_dist: float = CA_CA_TRANS):
        super().__init__()
        self.target_dist = target_dist

    def forward(self, coords: torch.Tensor, target_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            coords: (B, L, 3) CA coordinates
            target_lengths: optional (B, L-1, 1) per-bond target lengths

        Returns:
            (B, L, 3) CA coordinates with exact bond lengths
        """
        return project_bonds(coords, self.target_dist if target_lengths is None else target_lengths)


def sinusoidal_position_encoding(length: int, d_model: int) -> torch.Tensor:
    """Sinusoidal position encodings, shape (1, length, d_model)."""
    position = torch.arange(length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class FlowBackbone(nn.Module):
    """
    Shared scaffolding of all flow-matching backbones in this repository.

    Subclasses implement ``encode`` (the sequence encoder). Embeddings, output
    heads and the sampler are identical across PI-Mamba and the baselines so
    that comparisons only differ in the encoder.

    Args:
        d_model: Hidden dimension
        n_structure_layers: Number of structure module layers
        max_length: Length of the precomputed position encoding table
        coord_scale: Angstrom per model unit
    """

    def __init__(
        self,
        d_model: int = 256,
        n_structure_layers: int = 4,
        max_length: int = 2048,
        coord_scale: float = 10.0,
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        self.d_model = d_model
        self.max_length = max_length
        self.coord_scale = coord_scale

        # Time embedding
        self.time_embed = TimeEmbedding(d_model)

        # Input projection (from 3D coordinates to hidden)
        self.input_proj = nn.Linear(3, d_model)

        # Position encoding (sinusoidal)
        self.register_buffer(
            'pos_encoding', sinusoidal_position_encoding(max_length, d_model), persistent=False
        )

        # Structure module (torsion head for full-backbone reconstruction)
        self.structure_module = StructureModule(
            d_model=d_model,
            n_layers=n_structure_layers,
        )

        # Omega Prediction Head (Cis vs Trans)
        self.omega_head = OmegaHead(d_model)

        # Kinematic projection (exact CA-CA bond lengths)
        self.kinematic_proj = ChainProjection()

        # The residual stream is not normalised between layers, so it is
        # normalised once before the output heads.
        self.final_norm = nn.LayerNorm(d_model)

        # Final output projection (velocity)
        self.output_proj = nn.Linear(d_model, 3)

    # ------------------------------------------------------------------ encoder

    def encode(
        self, h: torch.Tensor, t_emb: torch.Tensor, return_physics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Map embedded inputs (B, L, D) to hidden states (B, L, D).

        Every layer reads the time-conditioned normalisation of the residual
        stream and adds its update to the un-normalised stream:
        h <- h + f(AdaLN(h, t)).
        """
        raise NotImplementedError

    def _encode(self, h, t_emb, return_physics: bool = False):
        """``encode`` followed by the final normalisation."""
        h, physics_info = self.encode(h, t_emb, return_physics=return_physics)
        return self.final_norm(h), physics_info

    def _embed(self, coords: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        L = coords.shape[1]
        t_emb = self.time_embed(t)  # (B, d_model)
        h = self.input_proj(coords)  # (B, L, d_model)
        if L <= self.pos_encoding.shape[1]:
            pos = self.pos_encoding[:, :L]
        else:
            pos = sinusoidal_position_encoding(L, self.d_model).to(h.device)
        return h + pos.to(h.dtype), t_emb

    def hidden_states(self, coords: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Encoder output (B, L, D) for coordinates in model units."""
        h, t_emb = self._embed(coords, t)
        return self._encode(h, t_emb)[0]

    def forward(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
        return_physics: bool = False,
        return_hidden: bool = False,
    ):
        """
        Forward pass: predict velocity field for flow matching.

        Args:
            coords: (B, L, 3) noisy CA coordinates at time t, in model units
            t: (B,) time values in [0, 1]
            return_physics: If True, also return physics diagnostics
            return_hidden: If True, also return the encoder output (B, L, D)

        Returns:
            velocity: (B, L, 3) predicted velocity field
            omega_logits: (B, L, 2) [trans, cis] logits
            physics_info: dict, only if return_physics
            hidden: (B, L, D), only if return_hidden
        """
        h, t_emb = self._embed(coords, t)
        h, physics_info = self._encode(h, t_emb, return_physics=return_physics)

        velocity = self.output_proj(h)  # (B, L, 3)
        omega_logits = self.omega_head(h)  # (B, L, 2)

        out = (velocity, omega_logits)
        if return_physics:
            out = out + (physics_info,)
        if return_hidden:
            out = out + (h,)
        return out

    # ------------------------------------------------------------------ sampler

    @torch.no_grad()
    def sample(
        self,
        length: int,
        n_samples: int = 1,
        n_steps: int = 200,
        device: torch.device = None,
        project: bool = True,
        relax: bool = True,
        project_every: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate protein backbones using flow matching.

        Flow matching: x_t = (1-t)*x_0 + t*x_1, integrate from t=0 (noise) to t=1 (data).

        Args:
            length: Sequence length
            n_samples: Number of samples to generate
            n_steps: Number of Euler integration steps
            device: Device for generation
            project: Apply the CA-CA bond-length projection
            relax: Resolve steric clashes as well as bond lengths at the end
                (``relax_chain``). Ignored when project is False.
            project_every: Project bond lengths every this many steps over the
                last 20% of the trajectory (0 = never, the default)
            generator: Optional random generator for the initial noise

        Returns:
            coords: (n_samples, length, 3) generated CA coordinates in Angstrom
        """
        if device is None:
            device = next(self.parameters()).device
        bond = CA_CA_TRANS / self.coord_scale

        # Centred standard Gaussian noise, as in training
        coords = torch.randn(n_samples, length, 3, device=device, generator=generator)
        coords = coords - coords.mean(dim=1, keepdim=True)

        # Time steps: integrate from t=0 (noise) to t=1 (data)
        timesteps = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

        for i in range(n_steps):
            t = timesteps[i].expand(n_samples)
            dt = timesteps[i + 1] - timesteps[i]

            velocity = self.forward(coords, t)[0]

            # Euler step
            coords = coords + velocity * dt

            # Projecting bond lengths during the trajectory was found to cost
            # secondary structure (helix content fell from 0.67 to 0.50 on
            # L=100 samples) without improving the final bond lengths, which
            # the relaxation below fixes anyway. It is off unless asked for.
            if project_every and i > n_steps * 0.8 and (i + 1) % project_every == 0:
                coords = project_bonds(coords, bond)

        # Final projection onto correct bond lengths, and onto structures
        # without steric clashes when asked for
        if project:
            coords = relax_chain(
                coords, bond, min_dist=4.0 / self.coord_scale, w_clash=2.0, n_steps=500,
            ) if relax else project_bonds(coords, bond)

        return coords * self.coord_scale

    @torch.no_grad()
    def reconstruct_full_backbone(self, ca_coords: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct full backbone (N, CA, C) from CA coordinates with the
        torsion head and NeRF.

        The torsion head has to be trained with full-backbone supervision
        first. With a model trained on CA-only data the output is not
        meaningful; use ``geometry.ca_to_backbone`` in that case.

        Args:
            ca_coords: (B, L, 3) CA coordinates in Angstrom

        Returns:
            full_backbone: (B, L, 3, 3) full backbone [N, CA, C] in Angstrom,
            built from ideal bond geometry in its own reference frame
        """
        B = ca_coords.shape[0]
        x = ca_coords - ca_coords.mean(dim=1, keepdim=True)

        # The clean structure is the t=1 end of the flow
        t = torch.ones(B, device=ca_coords.device)
        h, t_emb = self._embed(x / self.coord_scale, t)
        h, _ = self._encode(h, t_emb)

        _, torsions = self.structure_module(h, t_emb, return_torsions=True)
        return nerf_build_backbone(torsions[..., 0], torsions[..., 1], torsions[..., 2])

    def predict_torsions(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Torsion head output (B, L, 3) = [phi, psi, omega] in radians."""
        return self.structure_module(h, t_emb, return_torsions=True)[1]


class PIMambaBackbone(FlowBackbone):
    """
    Full PI-Mamba backbone for protein structure generation.

    Physics-Informed Mamba layers with time-conditioned normalisation, trained
    with flow matching on CA coordinates.

    Args:
        d_model: Hidden dimension
        n_layers: Number of PI-Mamba layers
        d_state: SSM state dimension
        d_conv: Width of the causal convolution in each layer
        expand_factor: Inner expansion of each layer
        n_groups: Channel groups per layer
        n_structure_layers: Number of structure module layers
        max_length: Length of the precomputed position encoding table
        dropout: Dropout on each layer output
        use_physics: Rouse-derived A (True) or free learned A (False)
        spectrum: ``rouse``, ``scrambled`` or ``interpolated`` (counterfactuals)
        spectrum_alpha: Mixing weight of the ``interpolated`` spectrum
        spectrum_seed: Seed of the ``scrambled`` / ``interpolated`` spectrum
        coord_scale: Angstrom per model unit
        scan_chunk: Chunk length of the vectorised scan
        checkpoint_scan: Trade compute for memory in the scan during training
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 12,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        n_groups: int = 8,
        n_structure_layers: int = 4,
        max_length: int = 2048,
        dropout: float = 0.1,
        use_physics: bool = True,
        spectrum: str = "rouse",
        spectrum_alpha: float = 1.0,
        spectrum_seed: int = 42,
        coord_scale: float = 10.0,
        scan_chunk: int = 32,
        checkpoint_scan: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            n_structure_layers=n_structure_layers,
            max_length=max_length,
            coord_scale=coord_scale,
        )

        self.n_layers = n_layers

        # PI-Mamba layers (the physics-informed backbone)
        self.pi_mamba_layers = nn.ModuleList([
            PhysicsInformedMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                max_length=max_length,
                n_groups=n_groups,
                dropout=dropout,
                use_physics=use_physics,
                spectrum=spectrum,
                spectrum_alpha=spectrum_alpha,
                spectrum_seed=spectrum_seed,
                scan_chunk=scan_chunk,
                checkpoint_scan=checkpoint_scan,
            )
            for _ in range(n_layers)
        ])

        # Adaptive layer norms (time-conditioned)
        self.adaptive_norms = nn.ModuleList([
            AdaptiveLayerNorm(d_model, d_cond=d_model)
            for _ in range(n_layers)
        ])

    def encode(self, h, t_emb, return_physics: bool = False):
        all_physics = []
        for mamba_layer, adaptive_norm in zip(self.pi_mamba_layers, self.adaptive_norms):
            # PI-Mamba layer on the time-conditioned normalisation of the
            # stream; the update is added to the un-normalised stream.
            h, physics_info = mamba_layer(
                adaptive_norm(h, t_emb), return_physics=return_physics, residual=h
            )
            if physics_info is not None:
                all_physics.append(physics_info)

        if not return_physics:
            return h, None

        # Aggregate physics across layers
        n = len(all_physics)
        physics_summary = {
            'mean_tau': sum(p['tau'].mean().item() for p in all_physics) / n,
            'mean_A': sum(p['A_mean'] for p in all_physics) / n,
            # (B, L) relaxation time per residue, averaged over layers and groups
            'tau_per_residue': torch.stack([p['tau'].mean(dim=-1) for p in all_physics]).mean(dim=0),
        }
        return h, physics_summary


# Physics validation utilities

def validate_generated_structures(
    model: FlowBackbone,
    n_samples: int = 50,
    lengths: tuple = (50, 100, 200, 500),
    device: torch.device = None,
) -> Dict:
    """
    Validate that generated structures obey polymer physics.
    
    Checks:
    1. End-to-end distance scaling
    2. Radius of gyration
    3. Persistence length
    4. Learned relaxation time patterns
    """
    if device is None:
        device = next(model.parameters()).device
    
    results = {
        'lengths': [],
        'R_ee': [],
        'R_g': [],
        'l_p': [],
    }
    
    model.eval()
    
    for L in lengths:
        print(f"Generating L={L}...")
        
        coords = model.sample(L, n_samples=n_samples, device=device)
        
        R_ee = compute_end_to_end_distance(coords).mean().item()
        R_g = compute_radius_of_gyration(coords).mean().item()
        l_p = compute_persistence_length(coords).mean().item()
        
        results['lengths'].append(L)
        results['R_ee'].append(R_ee)
        results['R_g'].append(R_g)
        results['l_p'].append(l_p)
        
        print(f"  R_ee = {R_ee:.1f}Å, R_g = {R_g:.1f}Å, ℓ_p = {l_p:.1f}Å")
    
    # Fit R_g ~ N^nu. R_g is used because the end-to-end distance of a single
    # folded chain is much noisier.
    import numpy as np
    slope, _ = np.polyfit(np.log(results['lengths']), np.log(results['R_g']), 1)
    results['scaling_exponent'] = float(slope)

    print(f"\nScaling: R_g ~ N^{slope:.3f}")
    print("Reference: 1/3 (compact globule), 0.5 (Gaussian chain), 0.588 (SAW)")
    
    return results
