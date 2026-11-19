"""Designability (self-consistency) score for PI-Mamba backbones.

``DesignabilityProxyLoss`` returns ``1 - scTM`` for a batch of CA traces,
where scTM comes from the real pipeline in ``designability.py``: CA-only
ProteinMPNN designs sequences, ESMFold folds them, and the TM-score between
the input backbone and the prediction is taken.

The value has no gradient (the pipeline samples discrete sequences), so it
cannot be used as a training loss term. It is meant for monitoring and for
model selection, for example at validation time.

Requirements: a ProteinMPNN checkout (``proteinmpnn_dir`` or the
PROTEINMPNN_DIR environment variable) and the ``transformers`` package.
"""

import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .data import write_backbone_pdb
from .designability import SelfConsistencyEvaluator
from .metrics import tm_score  # noqa: F401  (re-exported)


class DesignabilityProxyLoss(nn.Module):
    """``1 - mean scTM`` of a batch of backbones (no gradient).

    Args:
        device: device for ESMFold
        proteinmpnn_dir: ProteinMPNN checkout (default: $PROTEINMPNN_DIR)
        n_seqs: ProteinMPNN sequences per backbone
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        proteinmpnn_dir: Optional[str] = None,
        n_seqs: int = 8,
    ):
        super().__init__()
        self.evaluator = SelfConsistencyEvaluator(
            proteinmpnn_dir=proteinmpnn_dir, n_seqs=n_seqs, device=str(device)
        )

    @torch.no_grad()
    def sc_tm(self, backbone_coords: torch.Tensor) -> torch.Tensor:
        """scTM (B,) of CA traces (B, L, 3) in Angstrom."""
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for i, ca in enumerate(backbone_coords.detach().cpu()):
                files.append(str(Path(tmp) / f"backbone_{i:04d}.pdb"))
                write_backbone_pdb(ca, files[-1])
            records = self.evaluator.evaluate(files)
        return torch.tensor([r["scTM"] for r in records])

    @torch.no_grad()
    def forward(self, backbone_coords: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.sc_tm(backbone_coords).mean()
