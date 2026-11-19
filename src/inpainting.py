"""In-painting (motif-scaffolding) utilities for PI-Mamba.

``scaffold_motif`` generates a chain of ``length`` residues that contains a
given motif at given positions, with replacement guidance:

  1. The motif is centred and placed in the model frame. The rest of the
     chain starts from Gaussian noise.
  2. At every Euler step of the flow (t from 0 to 1) the motif positions are
     overwritten with the motif interpolated to the current time,
     x_t = (1 - t) * noise + t * motif, so the network always sees the motif
     at the noise level it expects.
  3. Only the scaffold positions are integrated with the predicted velocity.
  4. At the end the CA-CA bond projection is applied to the whole chain.

The projection moves the motif slightly, so the motif is not re-inserted
afterwards. ``motif_rmsd`` reports how far the motif in the final chain is
from the reference motif after optimal superposition.

The motif is placed with its centroid at the origin, which is also the
centroid of the generated chain. This simple placement suits motifs that sit
in the core of the scaffold.
"""

import torch

from .backbone_pi_mamba import project_bonds
from .geometry import CA_CA_TRANS, kabsch_rmsd


@torch.no_grad()
def scaffold_motif(
    model,
    length: int,
    motif_coords: torch.Tensor,
    motif_mask: torch.Tensor,
    n_steps: int = 100,
    proj_every: int = 10,
    final_projection: bool = True,
    device: torch.device = None,
    generator: torch.Generator = None,
) -> dict:
    """Generate a scaffold around a fixed motif using PI-Mamba.

    Args:
        model: trained ``FlowBackbone`` (for example ``PIMambaBackbone``).
        length: total sequence length L (>= motif length).
        motif_coords: (B, M, 3) CA coordinates of the motif in Angstrom.
        motif_mask: (B, L) boolean mask where ``True`` marks motif positions.
            Every row must mark exactly M positions; the motif residues are
            assigned to them in order.
        n_steps: number of Euler integration steps.
        proj_every: bond projection every this many steps during the last 20%
            of the trajectory (0 disables it).
        final_projection: apply the bond projection to the finished chain. It
            makes every CA-CA distance exactly 3.8 A but moves the motif, which
            raises ``motif_rmsd``; set it to False to keep the motif exact and
            accept approximate bond lengths.
        device: torch device (default: the device of the model).
        generator: optional random generator for the noise.

    Returns:
        dict with keys:
            "scaffold": (B, L, 3) generated CA coordinates in Angstrom,
            "motif_rmsd": (B,) RMSD in Angstrom between the motif positions of
                the generated chain and the reference motif.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    B, M, _ = motif_coords.shape
    L = length
    motif_mask = motif_mask.to(device).bool()
    if not (motif_mask.sum(dim=1) == M).all():
        raise ValueError("every row of motif_mask must mark exactly M positions")

    scale = model.coord_scale
    bond = CA_CA_TRANS / scale
    motif = motif_coords.to(device).float()
    motif = (motif - motif.mean(dim=1, keepdim=True)) / scale  # model units, centred

    noise = torch.randn(B, L, 3, device=device, generator=generator)
    noise = noise - noise.mean(dim=1, keepdim=True)
    motif_noise = noise[motif_mask].view(B, M, 3)

    x = noise.clone()
    timesteps = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t_now, t_next = timesteps[i], timesteps[i + 1]
        # The motif at the current noise level
        x[motif_mask] = ((1.0 - t_now) * motif_noise + t_now * motif).reshape(-1, 3)

        velocity = model(x, t_now.expand(B))[0]
        x = x + velocity * (t_next - t_now)

        if proj_every and i > n_steps * 0.8 and (i + 1) % proj_every == 0:
            x = project_bonds(x, bond)

    x[motif_mask] = motif.reshape(-1, 3)
    if final_projection:
        x = project_bonds(x, bond)

    scaffold = x * scale
    motif_generated = scaffold[motif_mask].view(B, M, 3)
    return {
        "scaffold": scaffold,
        "motif_rmsd": kabsch_rmsd(motif_generated, motif * scale),
    }
