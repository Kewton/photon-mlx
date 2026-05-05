"""Standalone PHOTON MLX runtime API.

``photon_mlx`` is intentionally independent from ``baseline_reporag``.  The
RAG integration layer may import PHOTON, but PHOTON-only users should be able
to import this package without pulling in indexing, retrieval, server, or
Streamlit code.

Heavy MLX-backed objects are exported lazily via ``__getattr__`` so
``import photon_mlx`` remains lightweight and does not probe Metal devices.
"""

from __future__ import annotations

from typing import Any

from torch_ref.config import (
    EarlyStoppingConfig,
    HierarchyConfig,
    ModelConfig,
    PhotonConfig,
    TokenizerConfig,
    TrainingConfig,
    load_photon_config,
)

__all__ = [
    "CheckpointState",
    "DriftMetrics",
    "EarlyStoppingConfig",
    "HierarchyConfig",
    "HierarchicalState",
    "ModelConfig",
    "PhotonConfig",
    "PhotonInference",
    "PhotonModel",
    "PhotonSessionState",
    "TokenizerConfig",
    "TrainingConfig",
    "WorkingMemoryConfig",
    "load_checkpoint",
    "load_photon_config",
    "save_checkpoint",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CheckpointState": ("photon_mlx.checkpoint", "CheckpointState"),
    "DriftMetrics": ("photon_mlx.session", "DriftMetrics"),
    "HierarchicalState": ("photon_mlx.session", "HierarchicalState"),
    "PhotonInference": ("photon_mlx.inference", "PhotonInference"),
    "PhotonModel": ("photon_mlx.model", "PhotonModel"),
    "PhotonSessionState": ("photon_mlx.session", "PhotonSessionState"),
    "WorkingMemoryConfig": ("photon_mlx.session", "WorkingMemoryConfig"),
    "load_checkpoint": ("photon_mlx.checkpoint", "load_checkpoint"),
    "save_checkpoint": ("photon_mlx.checkpoint", "save_checkpoint"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
