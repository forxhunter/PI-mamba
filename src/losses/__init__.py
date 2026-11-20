"""Auxiliary structure losses for PI-Mamba (all operate on CA coordinates)."""

from .distogram_loss import DistogramLoss, RadiusOfGyrationLoss
from .fape_loss import BondLengthLoss, LocalDistanceLoss
from .hbond_loss import HydrogenBondLoss
from .ramachandran_loss import RamachandranLoss
from .register_loss import (
    RegisterDistanceLoss,
    RegisterLoss,
    soft_register_score,
)

__all__ = [
    "DistogramLoss",
    "RadiusOfGyrationLoss",
    "BondLengthLoss",
    "LocalDistanceLoss",
    "HydrogenBondLoss",
    "RamachandranLoss",
    "RegisterDistanceLoss",
    "RegisterLoss",
    "soft_register_score",
]
