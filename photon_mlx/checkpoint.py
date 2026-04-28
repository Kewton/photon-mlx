"""Pure I/O surface for PHOTON checkpoints (Issue #135 / DR1-002 / DR3-001).

This module is the only ``photon_mlx`` import that ``baseline_reporag`` is
allowed to touch on the runtime path.  It must therefore avoid pulling in
training-only dependencies (``mlx.optimizers``, ``photon_mlx.loss``) so that
``pipeline_factory.py``'s lazy MLX import policy is preserved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx

from .model import PhotonModel

_logger = logging.getLogger(__name__)

INTEGRITY_FORMAT_VERSION = "1"


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


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of ``path`` (DR4-003)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_integrity(path: Path) -> None:
    """Write ``integrity.json`` recording SHA-256 of weights.npz / state.json."""
    integrity = {
        "format_version": INTEGRITY_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights_sha256": _sha256_file(path / "weights.npz"),
        "state_sha256": _sha256_file(path / "state.json"),
    }
    integrity_path = path / "integrity.json"
    tmp_path = path / "integrity.json.tmp"
    tmp_path.write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    os.replace(tmp_path, integrity_path)


def save_checkpoint(
    model: PhotonModel,
    state: CheckpointState,
    path: str | Path,
) -> None:
    """Write model weights + state.json + integrity.json under ``path``.

    state.json is written via ``tmp + os.replace`` so a crash mid-write
    cannot leave a partial file behind (matches the trainer's previous
    behaviour). ``integrity.json`` (DR4-003) records SHA-256 hashes of
    the weights and state files so ``load_checkpoint`` can detect bit
    flips or malicious overwrites in transit (e.g. through git LFS or
    external storage fetched by another PR).
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

    _write_integrity(path)


def _verify_integrity(path: Path, *, strict: bool) -> None:
    """DR4-003: verify ``integrity.json`` matches the on-disk files.

    When ``strict=True`` and the file is missing, raise. Otherwise, log a
    WARNING (legacy checkpoints predating Issue #135 won't have an
    integrity file but should still be loadable). A hash mismatch always
    raises, regardless of ``strict``.
    """
    integrity_path = path / "integrity.json"
    if not integrity_path.exists():
        if strict:
            raise FileNotFoundError(
                f"integrity.json missing at {integrity_path} "
                "and verify_integrity=True (DR4-003 strict mode)"
            )
        _logger.warning(
            "integrity.json missing at %s — checkpoint integrity cannot be "
            "verified. Issue #135 / DR4-003: regenerate the checkpoint via "
            "save_checkpoint to populate it.",
            integrity_path,
        )
        return

    try:
        manifest = json.loads(integrity_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"integrity.json at {integrity_path} is not valid JSON (DR4-003)"
        ) from e
    expected_w = manifest.get("weights_sha256")
    expected_s = manifest.get("state_sha256")
    if not expected_w or not expected_s:
        raise ValueError(
            f"integrity.json at {integrity_path} missing required hashes "
            "(weights_sha256 / state_sha256) — DR4-003"
        )

    actual_w = _sha256_file(path / "weights.npz")
    actual_s = _sha256_file(path / "state.json")
    if actual_w != expected_w:
        raise ValueError(
            f"checkpoint integrity check FAILED: weights.npz hash mismatch "
            f"at {path} (expected {expected_w[:12]}…, got {actual_w[:12]}…). "
            "Refusing to load potentially tampered weights (DR4-003)."
        )
    if actual_s != expected_s:
        raise ValueError(
            f"checkpoint integrity check FAILED: state.json hash mismatch "
            f"at {path} (expected {expected_s[:12]}…, got {actual_s[:12]}…). "
            "Refusing to load potentially tampered state (DR4-003)."
        )


def load_checkpoint(
    model: PhotonModel,
    path: str | Path,
    *,
    verify_integrity: bool = False,
) -> CheckpointState:
    """Load model weights and return a ``CheckpointState``.

    Integrity (DR4-003): when ``integrity.json`` is present it is checked
    against the on-disk weights/state hashes; mismatches always raise.
    Set ``verify_integrity=True`` to also require the file's presence
    (recommended for production checkpoints fetched from external
    storage). Forward-compatible: unknown keys in state.json are dropped
    with a warning rather than raising, so a runtime built against an
    older schema can still read checkpoints written by a newer trainer.
    """
    path = Path(path)
    _verify_integrity(path, strict=verify_integrity)

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
