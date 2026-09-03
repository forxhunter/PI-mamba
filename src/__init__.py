"""
Physics-Informed Mamba (PI-Mamba) V12

A protein structure generation model where the Mamba state transition
is derived from the Rouse model of polymer dynamics.

Key Components:
    - rouse_physics: Rouse eigenvalues, eigenvectors, and transforms
    - pi_mamba_layer: Physics-Informed Mamba layer
    - backbone_pi_mamba: Full PI-Mamba backbone with SE(3) flow matching

The core insight is that Mamba's recurrence IS the polymer physics:
    h_{t+1} = A h_t + B x_t
    where A = exp(-λ_p * τ) and λ_p are Rouse eigenvalues

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
    PIMambaBackbone,
    AdaptiveLayerNorm,
    TimeEmbedding,
    StructureModule,
    NeRFProjection,
    validate_generated_structures,
)

__version__ = "12.0.0"
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
    # Backbone
    "PIMambaBackbone",
    "AdaptiveLayerNorm",
    "TimeEmbedding",
    "StructureModule",
    "NeRFProjection",
]
