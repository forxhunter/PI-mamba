"""
Global structure losses: Distogram matching and Radius of Gyration.

These losses teach the model to form correct long-range contacts
and compact globular folds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistogramLoss(nn.Module):
    """
    Match predicted pairwise CA distance matrix to target distogram.

    Uses smooth L1 loss on pairwise distances, with optional binning
    for numerical stability. Subsamples pairs for efficiency.
    """

    def __init__(self, max_dist: float = 40.0, n_subsample: int = 512):
        super().__init__()
        self.max_dist = max_dist
        self.n_subsample = n_subsample

    def forward(
        self,
        pred: torch.Tensor,
        target_distogram: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, L, 3) predicted CA coordinates
            target_distogram: (B, L, L) target pairwise distances
            mask: (B, L) validity mask
        Returns:
            loss: scalar
        """
        B, L, _ = pred.shape

        # Compute predicted distance matrix
        # (B, L, 1, 3) - (B, 1, L, 3) -> (B, L, L)
        pred_dist = torch.cdist(pred, pred)  # (B, L, L)

        # Clamp both to max_dist
        pred_dist = pred_dist.clamp(max=self.max_dist)
        target_dist = target_distogram.clamp(max=self.max_dist)

        # Pair mask: both residues must be valid
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B, L, L)

        # Exclude diagonal and near-diagonal (|i-j| <= 3, already handled by local loss)
        idx = torch.arange(L, device=pred.device)
        sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        long_range_mask = (sep > 3).float().unsqueeze(0)  # (1, L, L)
        pair_mask = pair_mask * long_range_mask

        # Subsample for efficiency if needed
        if L > 64:
            # Flatten valid pairs and subsample
            flat_mask = pair_mask.view(B, -1)
            flat_pred = pred_dist.view(B, -1)
            flat_target = target_dist.view(B, -1)

            loss = 0.0
            count = 0
            for b in range(B):
                valid_idx = flat_mask[b].nonzero(as_tuple=True)[0]
                if len(valid_idx) == 0:
                    continue
                if len(valid_idx) > self.n_subsample:
                    perm = torch.randperm(len(valid_idx), device=pred.device)[:self.n_subsample]
                    valid_idx = valid_idx[perm]
                p = flat_pred[b][valid_idx]
                t = flat_target[b][valid_idx]
                loss = loss + F.smooth_l1_loss(p, t, reduction='sum')
                count += len(valid_idx)
            return loss / (count + 1e-8)
        else:
            diff = F.smooth_l1_loss(pred_dist, target_dist, reduction='none')
            return (diff * pair_mask).sum() / (pair_mask.sum() + 1e-8)


class RadiusOfGyrationLoss(nn.Module):
    """
    Encourage compact globular structures by matching Rg to expected value.

    Expected Rg for globular proteins: Rg ~ 2.0 * L^0.4 (Angstroms)
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, L, 3) predicted CA coordinates
            mask: (B, L) validity mask
        Returns:
            loss: scalar
        """
        B = pred.shape[0]
        loss = 0.0

        for b in range(B):
            m = mask[b].bool()
            coords = pred[b][m]  # (L_valid, 3)
            L = coords.shape[0]
            if L < 5:
                continue

            com = coords.mean(dim=0, keepdim=True)
            rg_sq = ((coords - com) ** 2).sum(dim=-1).mean()
            rg = torch.sqrt(rg_sq + 1e-8)

            # Expected Rg for globular protein
            rg_expected = 2.0 * (L ** 0.4)

            # Penalize if Rg is too large (don't penalize compact)
            loss = loss + F.relu(rg - rg_expected * 1.2)

        return loss / B
