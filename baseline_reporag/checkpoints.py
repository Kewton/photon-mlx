"""Checkpoint cache and download helpers for PHOTON runtime."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)

CHECKPOINT_ROOT_ENV = "PHOTON_CHECKPOINT_ROOT"
CHECKPOINT_REPO_ENV = "PHOTON_CHECKPOINT_REPO_ID"
CHECKPOINT_REVISION_ENV = "PHOTON_CHECKPOINT_REVISION"


def checkpoint_root() -> Path:
    """Return the approved checkpoint root.

    The value mirrors ``baseline_reporag.photon_pipeline``: an explicit
    ``PHOTON_CHECKPOINT_ROOT`` wins, otherwise ``./checkpoints`` is used.
    """

    return Path(os.environ.get(CHECKPOINT_ROOT_ENV, "checkpoints")).resolve()


def maybe_download_checkpoint(
    raw_checkpoint_path: str,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
) -> None:
    """Download a missing relative checkpoint directory when a source is set.

    Download is opt-in through ``model.checkpoint_repo_id`` or
    ``PHOTON_CHECKPOINT_REPO_ID``.  When no source is configured this function
    is a no-op so the existing fail-fast path reports the missing local
    checkpoint exactly where it is loaded.
    """

    root = checkpoint_root()
    raw_path = Path(raw_checkpoint_path).expanduser()
    if raw_path.is_absolute():
        return

    candidate = root / raw_path
    if candidate.exists():
        return

    source_repo = repo_id or os.environ.get(CHECKPOINT_REPO_ENV)
    if not source_repo:
        return

    source_revision = revision or os.environ.get(CHECKPOINT_REVISION_ENV) or None
    rel = raw_path.as_posix().strip("/")
    if not rel or rel == ".":
        allow_patterns = ["weights.npz", "state.json", "integrity.json"]
    else:
        allow_patterns = [
            f"{rel}/weights.npz",
            f"{rel}/state.json",
            f"{rel}/integrity.json",
        ]

    root.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=source_repo,
            revision=source_revision,
            allow_patterns=allow_patterns,
            local_dir=str(root),
        )
    except Exception as exc:  # noqa: BLE001 - normalize external boundary
        raise RuntimeError(
            "checkpoint download failed "
            f"({type(exc).__name__}). Check PHOTON_CHECKPOINT_REPO_ID, "
            "PHOTON_CHECKPOINT_REVISION, HF_TOKEN, and network access."
        ) from None

    _logger.info("Downloaded PHOTON checkpoint artifact %s", rel or ".")
