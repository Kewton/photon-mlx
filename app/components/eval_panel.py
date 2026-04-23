"""Eval-runner helpers for the Streamlit app (Issue #82 Wave 2 / 4).

This module is intentionally streamlit-free (see ``app/components/__init__.py``)
so the pure-Python path helpers can be unit-tested without a Streamlit
runtime.  Wave 2 provided the path / id sanitization primitives; Wave 4
adds the subprocess orchestration (``build_eval_job_cmd``,
``start_eval_job``, ``parse_eval_progress``, ``tail_log_bytes``) and leaves
the state-reconciliation (``_sync_eval_job``) in ``app/photon_app.py`` so
that the sync loop can import lazily without adding a Streamlit dep here.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

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


# --- W4-T1: subprocess orchestration ------------------------------------

# Reuse ``_SAFE_ID_RE``-style allowlist for project_name / repo_id values
# that flow into argv. This mirrors ``photon_app._safe_id`` but stays
# streamlit-free (and local, to avoid an app→components circular import).
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def build_eval_job_cmd(
    eval_type: str,
    project_name: str,
    repo_id: str,
    config_path: str,
    output_json: Path,
    marker_file: Path,
    python_exec: str | None = None,
) -> list[str]:
    """Build an argv list suitable for ``subprocess.Popen(..., shell=False)``.

    Args:
        eval_type: ``"static"`` or ``"multi_turn"`` (keys of
            :data:`EVAL_SCRIPT_MAP`).  ``baseline_compare`` is a UI-level
            virtual type and is not accepted here.
        project_name: For provenance only — validated with the same
            ``[A-Za-z0-9_-]+`` allowlist so the value is safe to log.
        repo_id: Passed as ``--repo-id``.  Same allowlist as
            ``project_name``.
        config_path: Passed as ``--config``.  Not added to the allowlist
            because it is an internal path composed by the UI.
        output_json: Passed as ``--output``.  Should be a path produced by
            :func:`make_eval_paths`.
        marker_file: Passed as ``--marker-file``.  The subprocess writes
            a sentinel file here on successful completion; the main app's
            ``_sync_eval_job`` polls for its existence to detect success.
        python_exec: Path to the Python executable; defaults to
            ``sys.executable``.

    Returns:
        An argv list starting with ``python_exec, "-u", "-m",
        "<script_module>"`` followed by CLI flags.

    Raises:
        ValueError: for unknown ``eval_type`` / malformed ``project_name``
            or ``repo_id``.
    """

    if eval_type not in EVAL_SCRIPT_MAP:
        raise ValueError(
            f"Unknown eval_type: {eval_type!r}. "
            f"Expected one of {sorted(EVAL_SCRIPT_MAP)}."
        )
    if not isinstance(project_name, str) or not _SAFE_ID_RE.match(project_name):
        raise ValueError(
            f"Invalid project_name: {project_name!r}. Only [A-Za-z0-9_-]+ is allowed."
        )
    if not isinstance(repo_id, str) or not _SAFE_ID_RE.match(repo_id):
        raise ValueError(
            f"Invalid repo_id: {repo_id!r}. Only [A-Za-z0-9_-]+ is allowed."
        )

    script_mod = EVAL_SCRIPT_MAP[eval_type]
    exe = python_exec or sys.executable
    return [
        exe,
        "-u",
        "-m",
        script_mod,
        "--config",
        str(config_path),
        "--repo-id",
        repo_id,
        "--output",
        str(output_json),
        "--marker-file",
        str(marker_file),
    ]


def start_eval_job(
    cmd: list[str],
    log_file: Path,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Spawn an eval subprocess with ``shell=False``.

    ``stdout`` and ``stderr`` are redirected (merged) into ``log_file``;
    the file is opened in append mode so a caller can safely re-use the
    same log between phases / restarts.  ``TRANSFORMERS_VERBOSITY=error``
    and ``TOKENIZERS_PARALLELISM=false`` are injected into the child env
    unless the caller supplies explicit overrides.

    .. warning::

       This function MUST NOT use ``shell=True``.  The
       ``TestPageIndexNoShellTrue`` guardrail in
       ``tests/test_photon_app_helpers.py`` scans this module with an
       AST walker and will fail CI if a regression introduces it.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if env_overrides:
        env.update(env_overrides)
    # Append mode so a re-spawn (e.g. on UI restart) does not truncate
    # the previous phase's output.  The file handle stays open for the
    # lifetime of the child process; the OS closes it on exit.
    log_fh = open(log_file, "a", encoding="utf-8")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        shell=False,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc


def parse_eval_progress(log_file: Path) -> dict[str, Any]:
    """Tail ``log_file`` for ``PROGRESS …`` lines and return the latest.

    The eval scripts emit machine-readable progress lines of the form::

        PROGRESS done=15 total=120 p50_ms=19300 nc=0.183

    The most recent such line wins.  Non-PROGRESS log content is ignored.

    Returns:
        ``{"done_q", "total_q", "p50_latency_ms", "nc_rate"}`` on success,
        or an empty dict when the file is missing or contains no PROGRESS
        lines.
    """

    try:
        if not log_file.exists():
            return {}
        text = log_file.read_text(encoding="utf-8", errors="replace")
        progress_lines = [ln for ln in text.splitlines() if ln.startswith("PROGRESS ")]
        if not progress_lines:
            return {}
        last = progress_lines[-1]
        kv: dict[str, str] = {}
        for part in last[len("PROGRESS ") :].split():
            if "=" in part:
                key, value = part.split("=", 1)
                kv[key] = value
        return {
            "done_q": int(kv.get("done", 0)),
            "total_q": int(kv.get("total", 0)),
            "p50_latency_ms": float(kv.get("p50_ms", 0.0)),
            "nc_rate": float(kv.get("nc", 0.0)),
        }
    except (OSError, ValueError):
        return {}


def tail_log_bytes(log_file: Path, max_bytes: int = 2048) -> str:
    """Return the trailing ``max_bytes`` of ``log_file`` as decoded text.

    Intended for populating ``EvalJob.error_message`` when a subprocess
    dies without writing the marker file — we want the last few KB of
    stderr/stdout so the UI can surface the crash reason without having
    to read the full log.

    Empty string is returned when the file is missing or unreadable.
    """

    try:
        if not log_file.exists():
            return ""
        size = log_file.stat().st_size
        with open(log_file, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""
