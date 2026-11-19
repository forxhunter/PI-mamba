"""
Physics-Informed Mamba (PI-Mamba) V12

A protein structure generation model where the Mamba state transition
is derived from the Rouse model of polymer dynamics.

Key Components:
    - rouse_physics: Rouse eigenvalues, eigenvectors, and transforms
    - pi_mamba_layer: Physics-Informed Mamba layer
    - backbone_pi_mamba: PI-Mamba backbone trained with flow matching
    - backbone_performer / backbone_transformer: baseline encoders
    - data, metrics, geometry, losses: training and evaluation utilities

The state transition of every layer is
    h_t = A_t h_{t-1} + B_t x_t,   A_n = exp(-lambda_n * dt / tau),
where lambda_n are Rouse eigenvalues and tau, dt are input-dependent.

References:
    [1] Rouse, P.E. (1953). J. Chem. Phys. 21, 1272.
    [2] Gu, A. & Dao, T. (2024). Mamba-2. arXiv:2405.21060.
"""

from .rouse_physics import (
    compute_rouse_eigenvalues,
    compute_rouse_eigenvectors,
    compute_rouse_relaxation_times,
    compute_state_transition_matrix,
    RouseTransform,
    ZimmCorrection,
    compute_end_to_end_distance,
    compute_radius_of_gyration,
    compute_persistence_length,
    validate_rouse_scaling,
)

from .pi_mamba_layer import (
    PhysicsInformedMamba,
    PhysicsInformedMambaBlock,
)

from .backbone_pi_mamba import (
    FlowBackbone,
    PIMambaBackbone,
    AdaptiveLayerNorm,
    TimeEmbedding,
    StructureModule,
    ChainProjection,
    project_bonds,
    relax_chain,
    validate_generated_structures,
)

from .backbone_performer import PerformerBackbone
from .backbone_transformer import TransformerBackbone
from .model_io import build_model, load_checkpoint, save_checkpoint, MODEL_CHOICES

__version__ = "12.1.0"
__all__ = [
    # Rouse physics
    "compute_rouse_eigenvalues",
    "compute_rouse_eigenvectors",
    "compute_rouse_relaxation_times",
    "compute_state_transition_matrix",
    "RouseTransform",
    "ZimmCorrection",
    # Physics validation
    "compute_end_to_end_distance",
    "compute_radius_of_gyration",
    "compute_persistence_length",
    "validate_rouse_scaling",
    "validate_generated_structures",
    # PI-Mamba layer
    "PhysicsInformedMamba",
    "PhysicsInformedMambaBlock",
    # Backbones
    "FlowBackbone",
    "PIMambaBackbone",
    "PerformerBackbone",
    "TransformerBackbone",
    "AdaptiveLayerNorm",
    "TimeEmbedding",
    "StructureModule",
    "ChainProjection",
    "project_bonds",
    "relax_chain",
    # Model construction and checkpoints
    "build_model",
    "load_checkpoint",
    "save_checkpoint",
    "MODEL_CHOICES",
]
