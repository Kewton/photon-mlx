"""Pure I/O surface for PHOTON checkpoints (Issue #135 / DR1-002 / DR3-001).

This module is the only ``photon_mlx`` import that ``baseline_reporag`` is
allowed to touch on the runtime path.  It must therefore avoid pulling in
training-only dependencies (``mlx.optimizers``, ``photon_mlx.loss``) so that
``pipeline_factory.py``'s lazy MLX import policy is preserved.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import mlx.core as mx

from .model import PhotonModel

_logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Runtime DTO for ``state.json``.

    Mirrors the schema written by ``photon_mlx.trainer.TrainState`` but does
    not depend on the training module — that decoupling is what lets
    ``baseline_reporag`` consume checkpoints without importing optimizers.
    """

    step: int = 0
    best_val_loss: float = float("inf")
    best_step: int = 0
    patience_counter: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def save_checkpoint(
    model: PhotonModel,
    state: CheckpointState,
    path: str | Path,
) -> None:
    """Write model weights + state.json under ``path``.

    state.json is written via ``tmp + os.replace`` so a crash mid-write
    cannot leave a partial file behind (matches the trainer's previous
    behaviour).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    weights = dict(model.parameters())
    mx.savez(str(path / "weights.npz"), **_flatten(weights))

    state_path = path / "state.json"
    tmp_path = path / "state.json.tmp"
    tmp_path.write_text(
        json.dumps(
            {
                "step": state.step,
                "best_val_loss": state.best_val_loss,
                "best_step": state.best_step,
                "patience_counter": state.patience_counter,
                "train_losses": state.train_losses[-100:],
                "val_losses": state.val_losses[-100:],
            }
        ),
        encoding="utf-8",
    )
    os.replace(tmp_path, state_path)


def load_checkpoint(
    model: PhotonModel,
    path: str | Path,
) -> CheckpointState:
    """Load model weights and return a ``CheckpointState``.

    Forward-compatible: unknown keys in state.json are dropped with a
    warning rather than raising, so a runtime built against an older schema
    can still read checkpoints written by a newer trainer.
    """
    path = Path(path)
    weights = mx.load(str(path / "weights.npz"))
    model.load_weights(list(weights.items()))

    state_data = json.loads((path / "state.json").read_text(encoding="utf-8"))
    if not isinstance(state_data, dict):
        raise ValueError(
            f"state.json must decode to a dict, got {type(state_data).__name__}"
        )
    known = {f.name for f in fields(CheckpointState)}
    unknown = set(state_data) - known
    if unknown:
        _logger.warning("Ignoring unknown state.json keys: %s", sorted(unknown))
    filtered = {k: v for k, v in state_data.items() if k in known}
    return CheckpointState(**filtered)


def _flatten(tree: Any, prefix: str = "") -> dict[str, mx.array]:
    """Flatten a nested ``model.parameters()`` tree to dot-separated keys."""
    flat: dict[str, mx.array] = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            flat.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            flat.update(_flatten(v, f"{prefix}{i}."))
    elif isinstance(tree, mx.array):
        flat[prefix.rstrip(".")] = tree
    return flat
