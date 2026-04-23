"""Eval-runner helpers for the Streamlit app (Issue #82 Wave 2 / 4).

This module is intentionally streamlit-free (see ``app/components/__init__.py``)
so the pure-Python path helpers can be unit-tested without a Streamlit
runtime.  Wave 2 provides just the path / id sanitization primitives; the
subprocess orchestration (``build_eval_job_cmd``, ``start_eval_job``,
``parse_eval_progress``, ``sync_eval_job``) lands in Wave 4.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

# --- D4-002 constants ---------------------------------------------------
# Only these two sub-trees of PROJECT_ROOT are allowed to receive eval
# artifacts. Both ``make_eval_paths`` and (in Wave 4) ``sync_eval_job``
# re-validate every path on entry so a tampered state file cannot trick
# the app into reading/writing outside these directories.
ALLOWED_EVAL_RESULTS_DIR_NAME = "reports/eval_runs"
ALLOWED_EVAL_LOG_DIR_NAME = "logs/eval"

# --- D4-003 constants ---------------------------------------------------
# Resource caps for the async eval runner. Wave 4 wires these into the UI;
# they are exported from Wave 2 so the wizard / state-load code can share
# them without a cross-wave import cycle.
MAX_CONCURRENT_EVAL = 1
EVAL_WALL_CLOCK_TIMEOUT_SEC = 3600

# --- D2-002 mapping -----------------------------------------------------
# Supported ``eval_type`` → ``scripts.*`` module name. ``baseline_compare``
# is a UI-level virtual type implemented as two sequential ``static`` runs,
# so it is deliberately absent from this mapping.
EVAL_SCRIPT_MAP = {
    "static": "scripts.run_baseline_eval",
    "multi_turn": "scripts.run_multi_turn_eval",
}

# Hex-only job ids (as produced by ``uuid.uuid4().hex``). We accept the
# slightly broader ``[A-Za-z0-9]+`` allowlist so that explicit ids passed
# in tests / debugging can also be validated without loosening to allow
# path metacharacters.
_SAFE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9]+$")


def sanitize_job_id(raw: str | None = None) -> str:
    """Return a safe job id (hex characters only).

    If ``raw`` is :data:`None`, a fresh ``uuid.uuid4().hex`` is returned
    (32 lowercase hex chars). Otherwise ``raw`` is validated against
    :data:`_SAFE_JOB_ID_RE` — any input containing path separators,
    traversal (``..``), whitespace, or other metacharacters raises
    :class:`ValueError`.
    """

    if raw is None:
        return uuid.uuid4().hex
    if not isinstance(raw, str) or not _SAFE_JOB_ID_RE.match(raw):
        raise ValueError(f"Invalid job_id: {raw!r}. Only [A-Za-z0-9]+ is allowed.")
    return raw


def make_eval_paths(
    job_id: str,
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """Return ``(result_json, log_file, marker_file)`` for an eval run.

    All three paths are composed under ``project_root`` in the two allowed
    sub-directories (see :data:`ALLOWED_EVAL_RESULTS_DIR_NAME` and
    :data:`ALLOWED_EVAL_LOG_DIR_NAME`).  As defense-in-depth the computed
    paths are then ``.resolve()``-checked to confirm they stay strictly
    inside ``project_root`` — any attempt to escape (e.g. via a crafted
    ``job_id``) raises :class:`ValueError`.

    Args:
        job_id: The sanitized job id. Callers SHOULD pass the return value
            of :func:`sanitize_job_id`; as a safety net this function
            re-validates via :func:`sanitize_job_id`.
        project_root: The absolute repository root.

    Returns:
        A 3-tuple ``(result_json, log_file, marker_file)``:
            * ``result_json``  → ``<root>/reports/eval_runs/<id>.json``
            * ``log_file``     → ``<root>/logs/eval/<id>.log``
            * ``marker_file``  → ``<root>/reports/eval_runs/<id>.done``

    Raises:
        ValueError: if ``job_id`` fails sanitization or any computed path
            escapes ``project_root``.
    """

    # Defense-in-depth: re-validate the job_id even though callers are
    # expected to have already done so.
    job_id = sanitize_job_id(job_id)

    root = Path(project_root).resolve()
    results_dir = (root / ALLOWED_EVAL_RESULTS_DIR_NAME).resolve()
    log_dir = (root / ALLOWED_EVAL_LOG_DIR_NAME).resolve()

    result_json = (results_dir / f"{job_id}.json").resolve()
    log_file = (log_dir / f"{job_id}.log").resolve()
    marker_file = (results_dir / f"{job_id}.done").resolve()

    for label, path, parent in (
        ("result_json", result_json, results_dir),
        ("log_file", log_file, log_dir),
        ("marker_file", marker_file, results_dir),
    ):
        if not path.is_relative_to(parent):
            raise ValueError(
                f"eval {label} escaped allowed dir: {path} not under {parent}"
            )
        if not path.is_relative_to(root):
            raise ValueError(
                f"eval {label} escaped project_root: {path} not under {root}"
            )

    return result_json, log_file, marker_file
