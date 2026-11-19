"""
Model construction and checkpoint I/O shared by all scripts.

A checkpoint is a dict with
    ``config``: keyword arguments for ``build_model`` (includes ``model``)
    ``model_state_dict``: weights
    ``ema_state_dict``: exponential-moving-average weights (optional)
plus whatever training state the caller adds.
"""

from typing import Optional, Tuple

import torch

from .backbone_performer import PerformerBackbone
from .backbone_pi_mamba import FlowBackbone, PIMambaBackbone
from .backbone_transformer import TransformerBackbone

MODEL_CHOICES = ("pi_mamba", "performer", "transformer")

# Bumped whenever the architecture changes in a way that makes older weights
# load but compute something different.
#   2: time embedding scaled by 1000, un-normalised residual stream with a
#      final LayerNorm, direct SSM term on the convolved input, time-conditioned
#      structure layers.
CHECKPOINT_FORMAT = 2

_COMMON_ARGS = ("d_model", "n_layers", "n_structure_layers", "max_length", "dropout", "coord_scale")
_MODEL_ARGS = {
    "pi_mamba": _COMMON_ARGS + (
        "d_state", "d_conv", "expand_factor", "n_groups", "use_physics",
        "spectrum", "spectrum_alpha", "spectrum_seed", "scan_chunk", "checkpoint_scan",
    ),
    "performer": _COMMON_ARGS + ("n_heads",),
    "transformer": _COMMON_ARGS + ("n_heads",),
}
_MODEL_CLASSES = {
    "pi_mamba": PIMambaBackbone,
    "performer": PerformerBackbone,
    "transformer": TransformerBackbone,
}


def build_model(config: dict) -> FlowBackbone:
    """
    Build a backbone from a config dict.

    ``config["model"]`` selects the architecture (default ``pi_mamba``). Keys
    that the selected architecture does not use are ignored, so one config can
    describe a sweep over architectures.
    """
    name = config.get("model", "pi_mamba")
    if name not in _MODEL_CLASSES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_CHOICES}")
    kwargs = {k: config[k] for k in _MODEL_ARGS[name] if config.get(k) is not None}
    return _MODEL_CLASSES[name](**kwargs)


def save_checkpoint(path, model: FlowBackbone, config: dict, ema_state: Optional[dict] = None, **extra) -> None:
    """Write a checkpoint that ``load_checkpoint`` can restore without other inputs."""
    ckpt = {"format_version": CHECKPOINT_FORMAT, "config": dict(config),
            "model_state_dict": model.state_dict()}
    if ema_state is not None:
        ckpt["ema_state_dict"] = ema_state
    ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(
    path, device="cpu", use_ema: bool = True, strict: bool = True, **overrides
) -> Tuple[FlowBackbone, dict]:
    """
    Load a model from a checkpoint.

    Args:
        path: Checkpoint file
        device: Device to put the model on
        use_ema: Load the EMA weights when the checkpoint has them
        strict: Raise if weights are missing or unexpected.
        **overrides: Config entries to override (for example ``max_length``)

    Returns:
        (model in eval mode, checkpoint dict)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        ckpt = {"model_state_dict": ckpt}
    version = ckpt.get("format_version", 1)
    if version != CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"Checkpoint {path} has format version {version}, but this code reads version "
            f"{CHECKPOINT_FORMAT}. The architecture changed between the versions, so the weights "
            f"cannot be used; retrain the model with scripts/train.py."
        )

    config = dict(ckpt.get("config") or {})
    config.update(overrides)

    model = build_model(config)
    state = ckpt["ema_state_dict"] if use_ema and ckpt.get("ema_state_dict") else ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Checkpoint {path} does not match the model built from its config.\n"
            f"  missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}"
        )

    ckpt["config"] = config
    return model.to(device).eval(), ckpt
