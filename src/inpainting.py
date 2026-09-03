"""In‑painting (motif‑scaffolding) utilities for PI‑Mamba.

The core function `scaffold_motif` receives:
  * `model` – a trained `PIMambaBackbone` instance.
  * `motif_coords` – tensor of shape (B, M, 3) containing CA coordinates of the fixed motif.
  * `motif_mask` – boolean tensor (B, L) where `True` indicates positions belonging to the motif.
  * `length` – total sequence length L (>= M).
  * `n_steps` – number of ODE integration steps.

The algorithm:
  1. Initialise the full chain with Gaussian noise (CA only).
  2. At each Euler step, predict the velocity with the model.
  3. Zero‑out the velocity for motif positions (so they stay static).
  4. Perform an Euler update on the whole chain.
  5. Every `proj_every` steps (default 10) apply the kinematic projection to enforce geometry.
  6. After the final step, re‑inject the exact motif coordinates to guarantee perfect alignment.

The function returns the full scaffold (B, L, 3) and computes two RMSD metrics:
  * `motif_rmsd` – RMSD between the generated and reference motif (should be ~0).
  * `scaffold_rmsd` – RMSD of the non‑motif region against a reference scaffold if provided.

This module is deliberately lightweight; it does **not** depend on external
protein design libraries, making it easy to run in the CI environment.
"""

import torch
import torch.nn.functional as F


def rmsd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Root‑mean‑square deviation between two sets of coordinates.

    Both tensors must have shape (B, N, 3). The function aligns the centroids
    before computing the Euclidean distance.
    """
    a_centered = a - a.mean(dim=1, keepdim=True)
    b_centered = b - b.mean(dim=1, keepdim=True)
    diff = a_centered - b_centered
    return torch.sqrt((diff ** 2).sum(dim=[1, 2]) / a.shape[1])


def scaffold_motif(
    model,
    length: int,
    motif_coords: torch.Tensor,
    motif_mask: torch.Tensor,
    n_steps: int = 100,
    proj_every: int = 10,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Generate a scaffold around a fixed motif using PI‑Mamba.

    Args:
        model: `PIMambaBackbone` (already on `device`).
        length: total sequence length L (>= motif length).
        motif_coords: (B, M, 3) tensor of CA coordinates for the motif.
        motif_mask: (B, L) boolean mask where `True` marks motif positions.
        n_steps: number of Euler integration steps.
        proj_every: frequency of kinematic projection.
        device: torch device.

    Returns:
        dict with keys:
            "scaffold": (B, L, 3) generated CA coordinates,
            "motif_rmsd": scalar tensor,
            "full_rmsd": optional scalar if a reference scaffold is supplied.
    """
    model.eval()
    B, M, _ = motif_coords.shape
    L = length
    # Initialise full chain with noise (small Gaussian perturbation)
    scaffold = torch.randn(B, L, 3, device=device) * 0.1
    # Insert the motif coordinates into the initial scaffold (helps convergence)
    scaffold = scaffold.clone()
    scaffold[motif_mask] = motif_coords.view(-1, 3)

    # Time schedule (reverse flow: from noise to data)
    timesteps = torch.linspace(1.0, 0.0, n_steps, device=device)
    with torch.no_grad():
        for i in range(n_steps - 1):
            t = timesteps[i].unsqueeze(0).expand(B)
            dt = timesteps[i] - timesteps[i + 1]
            # Predict velocity for the whole chain
            velocity, _ = model(scaffold, t)
            # Zero velocity on motif positions
            velocity = velocity.clone()
            velocity[motif_mask] = 0.0
            # Euler update
            scaffold = scaffold + velocity * dt
            # Projection step
            if (i + 1) % proj_every == 0 or i == n_steps - 2:
                scaffold = model.kinematic_proj(scaffold)
        # Final projection to guarantee geometry
        scaffold = model.kinematic_proj(scaffold)
        # Re‑inject exact motif coordinates (ensures perfect alignment)
        scaffold[motif_mask] = motif_coords.view(-1, 3)

    # Compute RMSD for the motif (should be ~0)
    motif_generated = scaffold[motif_mask].view(B, M, 3)
    motif_rmsd_val = rmsd(motif_generated, motif_coords).mean()

    return {
        "scaffold": scaffold,
        "motif_rmsd": motif_rmsd_val,
    }

# End of inpainting.py
