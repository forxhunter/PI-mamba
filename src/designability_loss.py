"""Designability proxy loss for PI-Mamba.

The loss encourages generated backbones to be designable by estimating the
self‑consistency TM‑score (scTM) using a frozen ProteinMPNN model. The frozen
model predicts a sequence for a given backbone, runs ESMFold on the predicted
sequence, and computes the TM‑score between the original backbone and the
ESMFold structure. The proxy loss is simply ``1 - scTM`` (higher scTM is better).

This module provides a lightweight wrapper that can be called during training
without back‑propagating through ProteinMPNN or ESMFold – only the scTM value is
used as a scalar target.
"""

import torch
import torch.nn as nn

# The frozen models are loaded once at import time to avoid repeated I/O.
# In a real training script you would ensure the models are on the same device
# as the backbone tensors.

# NOTE: The actual ProteinMPNN and ESMFold implementations are not part of the
# repository. Here we provide placeholder classes that illustrate the expected
# API. Replace them with the real implementations when integrating.

class FrozenProteinMPNN(nn.Module):
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        # Load pretrained weights (placeholder)
        self.device = device
        # In practice: self.model = load_proteinmpnn_weights(...).to(device)
        self.model = None

    @torch.no_grad()
    def forward(self, backbone_coords: torch.Tensor) -> torch.Tensor:
        """Predict a sequence given backbone coordinates.

        Args:
            backbone_coords: (B, L, 3) CA coordinates.
        Returns:
            seq_logits: (B, L, 20) logits over the 20 amino acids.
        """
        # Placeholder: return uniform logits
        B, L, _ = backbone_coords.shape
        return torch.full((B, L, 20), 0.0, device=self.device)


class FrozenESMFold(nn.Module):
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        # Placeholder for the actual ESMFold model
        self.model = None

    @torch.no_grad()
    def forward(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        """Predict backbone coordinates from a one‑hot sequence.

        Args:
            seq_onehot: (B, L, 20) one‑hot encoded sequence.
        Returns:
            pred_coords: (B, L, 3) predicted CA coordinates.
        """
        B, L, _ = seq_onehot.shape
        # Placeholder: return a random walk that roughly respects bond length
        steps = torch.randn(B, L, 3, device=self.device) * 0.1
        coords = torch.cumsum(steps, dim=1)
        return coords


def tm_score(coord_a: torch.Tensor, coord_b: torch.Tensor) -> torch.Tensor:
    """Compute a simplified TM‑score between two sets of CA coordinates.

    This implementation follows the TM‑score definition but uses a fast
    approximation suitable for a proxy loss. It returns a value in [0, 1].
    """
    # Center the structures
    a_centered = coord_a - coord_a.mean(dim=1, keepdim=True)
    b_centered = coord_b - coord_b.mean(dim=1, keepdim=True)
    # Compute RMSD
    diff = a_centered - b_centered
    rmsd = torch.norm(diff, dim=[1, 2]) / torch.sqrt(torch.tensor(coord_a.shape[1], dtype=torch.float32))
    # Length‑dependent scaling factor d0 (approximation)
    L = coord_a.shape[1]
    d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8
    d0 = max(d0, 0.5)  # avoid division by zero for very short proteins
    tm = 1.0 / (1.0 + (rmsd / d0) ** 2)
    return tm


class DesignabilityProxyLoss(nn.Module):
    """Loss that penalizes low scTM.

    The forward pass returns ``1 - scTM`` so that minimizing the loss pushes the
    model toward backbones that are more designable according to the frozen
    pipeline.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.proteinmpnn = FrozenProteinMPNN(device)
        self.esmfold = FrozenESMFold(device)
        # No learnable parameters – the loss is a scalar.

    @torch.no_grad()
    def forward(self, backbone_coords: torch.Tensor) -> torch.Tensor:
        # backbone_coords: (B, L, 3)
        logits = self.proteinmpnn(backbone_coords)  # (B, L, 20)
        # Convert logits to one‑hot by taking argmax (deterministic proxy)
        seq_idx = torch.argmax(logits, dim=-1)  # (B, L)
        seq_onehot = F.one_hot(seq_idx, num_classes=20).float()
        pred_coords = self.esmfold(seq_onehot)  # (B, L, 3)
        sc_tm = tm_score(backbone_coords, pred_coords)  # (B,)
        loss = 1.0 - sc_tm.mean()
        return loss

# Example usage inside a training loop (pseudo‑code):
#   design_loss = DesignabilityProxyLoss(device)
#   loss_total = loss_fm + lambda_fape * loss_fape + lambda_bond * loss_bond
#   loss_total += lambda_sc * design_loss(backbone_coords)

# End of designability_loss.py
