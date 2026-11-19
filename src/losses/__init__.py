"""Auxiliary structure losses for PI-Mamba (all operate on CA coordinates)."""

from .distogram_loss import DistogramLoss, RadiusOfGyrationLoss
from .fape_loss import BondLengthLoss, LocalDistanceLoss
from .hbond_loss import HydrogenBondLoss
from .ramachandran_loss import RamachandranLoss

__all__ = [
    "DistogramLoss",
    "RadiusOfGyrationLoss",
    "BondLengthLoss",
    "LocalDistanceLoss",
    "HydrogenBondLoss",
    "RamachandranLoss",
]
