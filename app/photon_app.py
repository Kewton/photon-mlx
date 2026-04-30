"""PHOTON-RepoRAG Management App (Streamlit)

Launch:
    streamlit run app/photon_app.py --server.port 3012 --server.baseUrlPath /proxy/photon
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
# Ensure baseline_reporag is importable when this module is launched via
# ``streamlit run app/photon_app.py`` (cwd may be the project root, but the
# app directory itself is not on sys.path by default).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_reporag.config import load_config  # noqa: E402
from baseline_reporag.pipeline_factory import (  # noqa: E402
    build_pipeline,
    override_repo_for_pipeline,
)


# Issue #82 Wave 3: drift + turn-history panels (streamlit-free helpers).
# The ``app`` directory has no ``__init__.py`` (see design note on the
# MySwiftAgent ``app/`` namespace collision in
# ``tests/test_photon_app_components.py``) so we load each component by
# absolute file path. Mirrors the loader pattern used by the unit tests.
def _load_component(mod_name: str):
    import importlib.util

    full_name = f"_photon_app_component_{mod_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    path = PROJECT_ROOT / "app" / "components" / f"{mod_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, path)
    assert spec and spec.loader, f"cannot spec component {mod_name} at {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


_drift_panel = _load_component("drift_panel")
_turn_history_panel = _load_component("turn_history_panel")
# Issue #82 Wave 4: eval_panel orchestration helpers (streamlit-free).
_eval_panel = _load_component("eval_panel")
# Issue #82 Wave 5: project wizard helpers (YAML safe_load + best-practice merge).
_wizard = _load_component("wizard")

STATE_FILE = PROJECT_ROOT / ".cache" / "photon_app_state.json"

SYNC_INTERVAL_SECONDS = 30
_LOG_TAIL_BYTES = 65536

# Allowlist for repo_id / job_id path segments. Reject `/`, `\`, `..`,
# spaces, and any other metacharacter so path composition stays safe.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_logger = logging.getLogger(__name__)


def _safe_id(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.match(value):
        raise ValueError(f"Invalid {label}: {value!r}. Only [A-Za-z0-9_-]+ is allowed.")
    return value


def _filter_known_fields(dc_cls, raw: dict) -> dict:
    """Return only keys that match dataclass fields of ``dc_cls``."""
    known = {f.name for f in fields(dc_cls)}
    unknown = set(raw) - known
    if unknown:
        _logger.warning(
            "Ignoring unknown keys for %s: %s",
            dc_cls.__name__,
            sorted(unknown),
        )
    return {k: v for k, v in raw.items() if k in known}


def _atomic_write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


# 除外パターン: backup / training-only / eval matrix
# (UI から ingest 用に選んでも意味の無い config をフィルタする)
_USER_CONFIG_EXCLUDE_SUFFIXES = (".wave6_backup",)
_USER_CONFIG_EXCLUDE_PREFIXES = ("eval_",)
_USER_CONFIG_EXCLUDE_STEMS = (
    # training-only configs (ingest 用途では使わない)
    "institutional_docs_photon_retrain",
)


def _discover_user_configs(configs_dir: Path | None = None) -> list[str]:
    """Discover ``configs/*.yaml`` files suitable for ingest UI selection.

    Returns project-relative paths (e.g. ``configs/baseline.yaml``) sorted
    alphabetically. Excludes:

    - ``*.wave6_backup`` (historical backups)
    - ``eval_*.yaml`` (eval-matrix configs not meant for ingest)
    - ``*_retrain.yaml`` (training-only configs)
    """
    base = configs_dir if configs_dir is not None else PROJECT_ROOT / "configs"
    if not base.is_dir():
        return []
    out: list[str] = []
    for path in sorted(base.glob("*.yaml")):
        name = path.name
        if any(name.endswith(suffix) for suffix in _USER_CONFIG_EXCLUDE_SUFFIXES):
            continue
        if any(name.startswith(prefix) for prefix in _USER_CONFIG_EXCLUDE_PREFIXES):
            continue
        if path.stem in _USER_CONFIG_EXCLUDE_STEMS:
            continue
        try:
            rel = path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = path
        out.append(str(rel))
    return out


# Driver executed in a child ``python -c`` to chain the 4 index-build
# phases with argv-list subprocess.run calls (shell=False). No user input
# is interpolated into this string — every user value arrives via sys.argv.
_INDEX_PIPELINE_DRIVER = """
import subprocess, sys

repo_dir, repo_id, embed_model, config_path = sys.argv[1:5]

def run(argv, phase):
    print(f'>>> {phase}: ' + ' '.join(argv), flush=True)
    subprocess.run(argv, check=True)

# Phase 0: resolve commit (git SHA for git repos, synthesized
# 'manual-<mtime>' id for non-git directories like 制度文書 corpora).
# Shared with scripts/ingest_repo.py so all 3 phases agree on the same id.
from scripts.ingest_repo import resolve_commit
commit = resolve_commit(repo_dir, 'HEAD')
print(f'Phase 1: Ingest (commit={commit})', flush=True)
run(
    [
        sys.executable, '-m', 'scripts.ingest_repo',
        '--repo', repo_dir, '--repo-id', repo_id,
        '--commit', commit, '--config', config_path,
    ],
    'ingest',
)
print(f'Phase 2: BM25 + Embedding ({embed_model})', flush=True)
run(
    [
        sys.executable, '-m', 'scripts.build_indexes',
        '--repo-id', repo_id, '--commit', commit,
        '--embedding-model', embed_model, '--config', config_path,
    ],
    'build_indexes',
)
print('Phase 3: Symbol Graph', flush=True)
run(
    [
        sys.executable, '-m', 'scripts.build_symbol_graph',
        '--repo-id', repo_id, '--commit', commit, '--config', config_path,
    ],
    'build_symbol_graph',
)
print('Phase 3.5: Heading Graph', flush=True)
run(
    [
        sys.executable, '-m', 'scripts.build_heading_graph',
        '--repo-id', repo_id, '--commit', commit, '--config', config_path,
    ],
    'build_heading_graph',
)
print('DONE', flush=True)
"""


def _discover_checkpoints(ckpt_dir: Path) -> list[tuple[int, str, str]]:
    """Discover checkpoint directories under ``ckpt_dir``.

    Returns a list of ``(priority, path, label)`` tuples, sorted so that
    ``best/`` entries come first, then ``final/``, then ``step_XXXXXX/``.
    ``.tmp`` directories (e.g. the transient ``best.tmp/`` created during
    atomic checkpoint replacement) are excluded so the UI never surfaces a
    path that may vanish moments later.
    """
    entries: list[tuple[int, str, str]] = []
    if not ckpt_dir.exists():
        return entries
    for weights in ckpt_dir.rglob("weights.npz"):
        ck_path = weights.parent
        name = ck_path.name
        # Skip atomic-write scratch directories such as best.tmp.
        if name.endswith(".tmp"):
            continue
        if name == "best":
            priority = 0
            label = f"[best] {ck_path}"
        elif name == "final":
            priority = 1
            label = f"[final] {ck_path}"
        elif name.startswith("step_"):
            priority = 2
            label = f"[{name}] {ck_path}"
        else:
            priority = 3
            label = str(ck_path)
        entries.append((priority, str(ck_path), label))
    entries.sort(key=lambda t: (t[0], -len(t[1]), t[1]))
    return entries


# ================================================================
# Data models
# ================================================================


@dataclass
class TrainingJob:
    job_id: str
    repo_dir: str
    config_path: str
    pid: int | None = None
    started_at: str = ""
    status: str = "pending"  # pending | running | completed | failed
    log_file: str = ""
    last_step: int = 0
    max_steps: int = 0
    val_loss: float = 0.0
    # Issue #60: per-job progress log (`<log_dir>/train_log.jsonl`).
    # Empty string means the job predates the run-namespaced layout; the
    # UI will render "ログ未リンク" in that case.
    progress_log_file: str = ""


@dataclass
class IndexJob:
    job_id: str
    repo_dir: str
    repo_id: str
    config_path: str
    pid: int | None = None
    started_at: str = ""
    status: str = "pending"
    log_file: str = ""
    phase: str = ""  # ingest | bm25_embed | symbol_graph | heading_graph | completed
    embedding_model: str = ""


@dataclass
class EvalJob:
    """Evaluation job record persisted in AppState (Issue #82 Wave 1)."""

    job_id: str = ""
    project_name: str = ""
    eval_type: str = ""  # "static" | "multi_turn" | "baseline_compare"
    status: str = "pending"  # "pending" | "running" | "succeeded" | "failed"
    started_at: str = ""
    started_at_epoch: float = 0.0
    finished_at: str = ""
    pid: int | None = None
    log_file: str = ""
    result_json: str = ""
    marker_file: str = ""
    done_q: int = 0
    total_q: int = 0
    p50_latency_ms: float = 0.0
    nc_rate: float = 0.0
    error_message: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in ("succeeded", "failed")


@dataclass
class Project:
    name: str
    repo_id: str
    index_dir: str
    config_path: str
    photon_config_path: str = ""
    checkpoint_dir: str = ""
    use_photon: bool = False
    created_at: str = ""


@dataclass
class AppState:
    training_jobs: dict[str, TrainingJob] = field(default_factory=dict)
    index_jobs: dict[str, IndexJob] = field(default_factory=dict)
    projects: dict[str, Project] = field(default_factory=dict)
    chat_histories: dict[str, list[dict]] = field(default_factory=dict)
    eval_jobs: dict[str, EvalJob] = field(default_factory=dict)


# ================================================================
# State persistence
# ================================================================


def _load_state() -> AppState:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            state = AppState()
            for k, v in data.get("training_jobs", {}).items():
                state.training_jobs[k] = TrainingJob(
                    **_filter_known_fields(TrainingJob, v)
                )
            for k, v in data.get("index_jobs", {}).items():
                state.index_jobs[k] = IndexJob(**_filter_known_fields(IndexJob, v))
            for k, v in data.get("projects", {}).items():
                state.projects[k] = Project(**_filter_known_fields(Project, v))
            state.chat_histories = data.get("chat_histories", {})
            for k, v in data.get("eval_jobs", {}).items():
                state.eval_jobs[k] = EvalJob(**_filter_known_fields(EvalJob, v))
            # Issue #82 Wave 2 (W2-T4, D4-004): validate integrity of every
            # eval_job entry loaded from disk. A tampered state file could
            # otherwise feed arbitrary pid types, paths outside PROJECT_ROOT
            # or unknown status values into the sync/UI layer.
            _allowed_eval_status = {"pending", "running", "succeeded", "failed"}
            _project_root_abs = PROJECT_ROOT.resolve()
            for _k, _job in list(state.eval_jobs.items()):
                # Normalize pid: must be ``int`` or ``None``. Anything else
                # (e.g. a string via JSON tampering) is coerced to ``None``
                # so ``os.kill(pid, 0)`` can never see garbage.
                if not (_job.pid is None or isinstance(_job.pid, int)):
                    _logger.warning(
                        "eval_job %s has non-int pid %r; resetting to None",
                        _k,
                        _job.pid,
                    )
                    _job.pid = None
                # Validate path-like fields: they must resolve inside
                # PROJECT_ROOT when non-empty. Empty strings are allowed
                # (they indicate "not yet assigned").
                for _attr in ("log_file", "result_json", "marker_file"):
                    _value = getattr(_job, _attr, "")
                    if not _value:
                        continue
                    try:
                        _p = Path(_value).resolve()
                        _p.relative_to(_project_root_abs)
                    except (ValueError, OSError):
                        _job.status = "failed"
                        _job.error_message = "state tampering detected"
                        _logger.warning(
                            "eval_job %s has tampered %s: %r",
                            _k,
                            _attr,
                            _value,
                        )
                        break
                # Constrain status to the documented set; anything unknown
                # is funnelled into ``failed`` so the UI stops polling it.
                if _job.status not in _allowed_eval_status:
                    _logger.warning(
                        "eval_job %s has unknown status %r; forcing to failed",
                        _k,
                        _job.status,
                    )
                    _job.status = "failed"
            return state
        except Exception as exc:
            _logger.warning("Failed to load app state: %s", exc)
    return AppState()


def _save_state(state: AppState) -> None:
    data = {
        "training_jobs": {k: asdict(v) for k, v in state.training_jobs.items()},
        "index_jobs": {k: asdict(v) for k, v in state.index_jobs.items()},
        "projects": {k: asdict(v) for k, v in state.projects.items()},
        "chat_histories": state.chat_histories,
        "eval_jobs": {k: asdict(v) for k, v in state.eval_jobs.items()},
    }
    _atomic_write_text(STATE_FILE, json.dumps(data, ensure_ascii=False, indent=2))


def get_state() -> AppState:
    if "app_state" not in st.session_state:
        st.session_state.app_state = _load_state()
    return st.session_state.app_state


def save():
    _save_state(get_state())


# ================================================================
# Process helpers
# ================================================================


def _is_process_running(pid: int | None) -> bool:
    """Return True only if pid corresponds to a *live* process.

    ``os.kill(pid, 0)`` alone is not enough on macOS / Linux: a zombie
    (defunct) process whose parent has not yet ``wait()``-ed for it still
    exists in the process table and ``os.kill(pid, 0)`` returns success.
    The Streamlit driver is the parent of the index/eval subprocesses, so
    finished children remain zombies until Streamlit shuts down — which
    keeps ``_sync_index_job`` stuck on ``running`` (observed twice with
    Issue #170 / non-git ingest).

    The fix: after the kill-0 probe we ask ``ps -o stat=`` for the state
    code; any state starting with ``Z`` (e.g. ``Z+``) means the kernel
    has reaped the process body and only the exit-status entry remains
    — semantically equivalent to "not running" for our scheduler.
    """
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False

    # ``ps`` is part of POSIX and ships on every macOS / Linux box this
    # Streamlit app runs on, so we don't add ``psutil`` just for this.
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        # ``ps`` itself failing should not flip a real running process to
        # "dead"; preserve the legacy os.kill-only behaviour as a fallback.
        return True

    if result.returncode != 0:
        # pid disappeared between os.kill and ps — definitely not running.
        return False

    state = result.stdout.strip()
    if state.startswith("Z"):
        return False
    return True


def _read_training_progress(log_file: str) -> dict[str, Any]:
    """Read latest progress from a per-job training log.

    Returns a default result dict with ``last_step``, ``val_loss``,
    ``max_steps`` and Issue #60 early-stopping fields.  If ``log_file``
    is empty or missing, the default (all zeros / False) is returned so
    the UI can render "ログ未リンク" without crashing.
    """
    result: dict[str, Any] = {
        "last_step": 0,
        "val_loss": 0.0,
        "max_steps": 0,
        "best_step": 0,
        "best_val_loss": 0.0,
        "patience_counter": 0,
        "early_stopped": False,
    }
    if not log_file:
        return result
    log_path = Path(log_file)
    if not log_path.exists():
        return result
    try:
        lines = log_path.read_text().strip().split("\n")
        seen_eval = False
        for line in reversed(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in rec:
                result["last_step"] = max(result["last_step"], rec["step"])
            # eval entries: grab all known fields from the most recent one
            if "val_loss" in rec and not seen_eval:
                result["val_loss"] = rec.get("val_loss", result["val_loss"])
                result["best_step"] = rec.get("best_step", result["best_step"])
                result["best_val_loss"] = rec.get(
                    "best_val_loss", result["best_val_loss"]
                )
                result["patience_counter"] = rec.get(
                    "patience_counter", result["patience_counter"]
                )
                result["early_stopped"] = rec.get(
                    "early_stopped", result["early_stopped"]
                )
                seen_eval = True
    except Exception as exc:
        _logger.warning("progress read failed (%s): %s", log_file, exc)
    return result


def _read_log_tail(log_file: str, max_bytes: int = _LOG_TAIL_BYTES) -> str:
    if not log_file:
        return ""
    path = Path(log_file)
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


# ================================================================
# Job status reconciliation (pure helpers — no Streamlit access)
# ================================================================


def _sync_training_job(job: TrainingJob, progress: dict[str, Any] | None) -> bool:
    """Reconcile a training job against live process state and logs.

    Returns True if any field on `job` was mutated.
    """
    changed = False

    if progress:
        step = int(progress.get("last_step", 0) or 0)
        if step > job.last_step:
            job.last_step = step
            changed = True
        val_loss = float(progress.get("val_loss", 0.0) or 0.0)
        if val_loss > 0.0 and val_loss != job.val_loss:
            job.val_loss = val_loss
            changed = True

    if job.status == "running" and not _is_process_running(job.pid):
        tail = _read_log_tail(job.log_file)
        if "Training complete" in tail or job.last_step > 0:
            job.status = "completed"
        else:
            job.status = "failed"
        changed = True

    return changed


def _sync_index_job(job: IndexJob) -> bool:
    """Reconcile an index job against live process state and logs."""
    changed = False

    log_content = ""
    if job.log_file:
        path = Path(job.log_file)
        if path.exists():
            try:
                log_content = path.read_text(errors="replace")
            except OSError:
                log_content = ""

    if log_content and job.status == "running":
        if "Phase 3.5" in log_content:
            new_phase = "heading_graph"
        elif "Phase 3" in log_content:
            new_phase = "symbol_graph"
        elif "Phase 2" in log_content:
            new_phase = "bm25_embed"
        elif "Phase 1" in log_content:
            new_phase = "ingest"
        else:
            new_phase = job.phase
        if new_phase != job.phase:
            job.phase = new_phase
            changed = True

    if job.status == "running" and not _is_process_running(job.pid):
        if "DONE" in log_content:
            job.status = "completed"
            job.phase = "completed"
        else:
            job.status = "failed"
        changed = True

    return changed


def _sync_eval_job(job: EvalJob, now_epoch: float | None = None) -> bool:
    """Reconcile an eval job against live process state / marker / logs.

    Issue #82 Wave 4 (W4-T2, D3-004 / D4-003):

    * ``marker_file`` exists → ``status="succeeded"``; progress fields are
      pulled from ``result_json`` (falling back to the current values if
      the JSON is missing or malformed).
    * ``started_at_epoch`` is older than ``EVAL_WALL_CLOCK_TIMEOUT_SEC`` →
      ``status="failed"``, ``error_message="wall-clock timeout"``.
    * PID not alive AND no marker → ``status="failed"``, ``error_message``
      is the trailing 2KB of the log so the UI can surface crash context.
    * PID alive AND running: refresh progress from the log's latest
      ``PROGRESS`` line; only mutate if ``done_q`` advanced.

    Terminal jobs (``succeeded``/``failed``) are skipped — this keeps
    ``_sync_all_jobs`` cheap when the state file already reflects the
    final outcome.

    Returns ``True`` iff any field on ``job`` was mutated.
    """

    if job.is_terminal:
        return False

    now = now_epoch if now_epoch is not None else time.time()

    marker_path = Path(job.marker_file) if job.marker_file else None
    if marker_path is not None and marker_path.exists():
        # D3-004: marker_file is the authoritative success signal.
        result: dict[str, Any] = {}
        if job.result_json:
            try:
                result = json.loads(Path(job.result_json).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result = {}
        job.done_q = int(result.get("done_q", job.done_q))
        job.total_q = int(result.get("total_q", job.total_q))
        job.p50_latency_ms = float(result.get("p50_latency_ms", job.p50_latency_ms))
        job.nc_rate = float(result.get("nc_rate", job.nc_rate))
        job.status = "succeeded"
        job.finished_at = datetime.now().isoformat(timespec="seconds")
        return True

    started = job.started_at_epoch or now
    elapsed = now - started
    if elapsed > _eval_panel.EVAL_WALL_CLOCK_TIMEOUT_SEC:
        job.status = "failed"
        job.error_message = "wall-clock timeout"
        job.finished_at = datetime.now().isoformat(timespec="seconds")
        return True

    pid_alive = _is_process_running(job.pid)
    if pid_alive:
        changed = False
        if job.log_file:
            progress = _eval_panel.parse_eval_progress(Path(job.log_file))
            if progress:
                new_done = int(progress.get("done_q", job.done_q))
                new_total = int(progress.get("total_q", job.total_q))
                new_p50 = float(progress.get("p50_latency_ms", job.p50_latency_ms))
                new_nc = float(progress.get("nc_rate", job.nc_rate))
                if (
                    new_done != job.done_q
                    or new_total != job.total_q
                    or new_p50 != job.p50_latency_ms
                    or new_nc != job.nc_rate
                ):
                    job.done_q = new_done
                    job.total_q = new_total
                    job.p50_latency_ms = new_p50
                    job.nc_rate = new_nc
                    changed = True
        return changed

    # PID not alive AND no marker file → the child exited without
    # signalling success. Mark as failed and surface the log tail.
    job.status = "failed"
    tail = _eval_panel.tail_log_bytes(Path(job.log_file), 2048) if job.log_file else ""
    job.error_message = tail or "process died without marker"
    job.finished_at = datetime.now().isoformat(timespec="seconds")
    return True


def _sync_all_jobs(state: AppState) -> bool:
    """Reconcile every training/index/eval job in `state`. Returns True if mutated."""
    changed = False
    progress = _read_training_progress(str(PROJECT_ROOT / "logs" / "train_log.jsonl"))
    for job in state.training_jobs.values():
        if _sync_training_job(job, progress):
            changed = True
    for job in state.index_jobs.values():
        if _sync_index_job(job):
            changed = True
    # Issue #82 Wave 4 (W4-T2): daemon thread also reconciles eval jobs.
    for eval_job in state.eval_jobs.values():
        if _sync_eval_job(eval_job):
            changed = True
    return changed


_state_file_lock = threading.Lock()


def _sync_state_file() -> bool:
    """Load state file, reconcile job statuses, write back if anything changed."""
    with _state_file_lock:
        state = _load_state()
        if _sync_all_jobs(state):
            _save_state(state)
            return True
        return False


_sync_thread_started = False
_sync_thread_lock = threading.Lock()


def _start_background_sync(interval_s: float = SYNC_INTERVAL_SECONDS) -> None:
    """Start a daemon thread that keeps the state file authoritative.

    Runs once per Streamlit server process: the thread survives browser
    disconnects and script reruns, so job status keeps advancing even when
    no UI is mounted. Re-runs are cheap — we no-op after the first call.
    """
    global _sync_thread_started
    with _sync_thread_lock:
        if _sync_thread_started:
            return
        _sync_thread_started = True

    def _worker() -> None:
        while True:
            try:
                _sync_state_file()
            except Exception:
                pass
            time.sleep(interval_s)

    threading.Thread(target=_worker, daemon=True, name="photon_app_state_sync").start()


# ================================================================
# Page: Training
# ================================================================


def page_training():
    st.header("PHOTON モデル学習")
    state = get_state()

    st.subheader("新規学習ジョブ")

    repo_dir = st.text_input(
        "対象リポジトリのディレクトリ",
        placeholder="/path/to/your/repo",
        help="PHOTON モデルの学習に使うリポジトリのパス",
    )

    col1, col2 = st.columns(2)
    with col1:
        max_steps = st.number_input(
            "最大ステップ数", value=1000, min_value=100, step=100
        )
        batch_size = st.number_input("バッチサイズ", value=2, min_value=1, max_value=8)
    with col2:
        learning_rate = st.number_input(
            "学習率", value=0.00015, format="%.5f", step=0.00001
        )
        eval_every = st.number_input(
            "評価間隔 (steps)", value=100, min_value=50, step=50
        )

    if st.button("学習開始", type="primary", disabled=not repo_dir):
        if not Path(repo_dir).is_dir():
            st.error(f"ディレクトリが見つかりません: {repo_dir}")
        else:
            job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            repo_name = Path(repo_dir).name
            repo_id = repo_name.replace("-", "_").replace(" ", "_")

            try:
                repo_id = _safe_id(repo_id, label="repo_id")
                job_id = _safe_id(job_id, label="job_id")
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            # Run-specific checkpoint/log directories (Issue #60).
            run_ckpt_dir = PROJECT_ROOT / "checkpoints" / repo_id / job_id
            run_log_dir = PROJECT_ROOT / "logs" / job_id
            run_ckpt_dir.mkdir(parents=True, exist_ok=True)
            run_log_dir.mkdir(parents=True, exist_ok=True)
            progress_log_file = str(run_log_dir / "train_log.jsonl")

            # Generate config
            config_path = str(PROJECT_ROOT / "configs" / f"photon_{repo_id}.yaml")
            _generate_photon_config(
                config_path,
                repo_dir,
                repo_id,
                max_steps,
                batch_size,
                learning_rate,
                eval_every,
            )

            # Step 1: Generate corpus
            st.info("コーパス生成中...")

            # Ingest first
            ingest_cmd = [
                "python",
                "-m",
                "scripts.ingest_repo",
                "--repo",
                repo_dir,
                "--repo-id",
                repo_id,
                "--commit",
                "HEAD",
                "--config",
                "configs/baseline.yaml",
            ]
            subprocess.run(ingest_cmd, cwd=str(PROJECT_ROOT), capture_output=True)

            # Generate corpus
            corpus_cmd = [
                "python",
                "-m",
                "scripts.generate_training_corpus",
                "--repo-id",
                repo_id,
                "--config",
                "configs/baseline.yaml",
                "--photon-config",
                config_path,
                "--output-dir",
                str(PROJECT_ROOT / "data" / "processed"),
                "--commit",
                "HEAD",
            ]
            subprocess.run(corpus_cmd, cwd=str(PROJECT_ROOT), capture_output=True)

            # Step 2: Start training in background (argv list + shell=False).
            train_log = str(PROJECT_ROOT / "logs" / f"{job_id}.log")
            train_cmd = [
                "python",
                "-u",
                "-m",
                "scripts.train_photon",
                "--config",
                config_path,
                "--checkpoint-dir",
                str(run_ckpt_dir),
                "--log-dir",
                str(run_log_dir),
            ]
            train_log_fp = open(train_log, "w", encoding="utf-8")
            proc = subprocess.Popen(
                train_cmd,
                shell=False,
                cwd=str(PROJECT_ROOT),
                stdout=train_log_fp,
                stderr=subprocess.STDOUT,
            )

            job = TrainingJob(
                job_id=job_id,
                repo_dir=repo_dir,
                config_path=config_path,
                pid=proc.pid,
                started_at=datetime.now().isoformat(),
                status="running",
                log_file=train_log,
                max_steps=max_steps,
                progress_log_file=progress_log_file,
            )
            state.training_jobs[job_id] = job
            save()
            st.success(f"学習開始 (PID: {proc.pid})")
            st.rerun()

    # --- Status ---
    st.subheader("学習ステータス")

    if not state.training_jobs:
        st.info("学習ジョブはありません")
        return

    for job_id, job in sorted(state.training_jobs.items(), reverse=True):
        # Prefer the per-job progress log (Issue #60). Legacy jobs without
        # progress_log_file fall back to the global train_log.jsonl.
        job_log = job.progress_log_file or str(
            PROJECT_ROOT / "logs" / "train_log.jsonl"
        )
        progress = _read_training_progress(job_log)
        if _sync_training_job(job, progress):
            save()

        icon = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
        }.get(job.status, "?")

        with st.expander(
            f"{icon} {job_id} — {job.status}", expanded=(job.status == "running")
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("ステータス", job.status)
            col2.metric("開始時刻", job.started_at[:19] if job.started_at else "—")
            col3.metric("PID", str(job.pid or "—"))

            if job.max_steps > 0:
                pct = min(job.last_step / job.max_steps, 1.0)
                st.progress(pct, text=f"Step {job.last_step}/{job.max_steps}")

            if job.val_loss > 0:
                st.metric("最新 val_loss", f"{job.val_loss:.4f}")

            # Early stopping status (Issue #60)
            if progress.get("best_step", 0) > 0:
                es_col1, es_col2, es_col3 = st.columns(3)
                es_col1.metric("best_step", str(progress.get("best_step", 0)))
                bvl = progress.get("best_val_loss", 0.0)
                es_col2.metric(
                    "best_val_loss",
                    f"{bvl:.4f}" if bvl else "—",
                )
                es_col3.metric(
                    "patience",
                    str(progress.get("patience_counter", 0)),
                )
            if progress.get("early_stopped"):
                st.warning("Early stopping が発動しました")

            if not job.progress_log_file:
                st.caption("ログ未リンク (旧ジョブのため run 別パスなし)")

            st.text(f"Config: {job.config_path}")
            st.text(f"リポジトリ: {job.repo_dir}")

            # Issue #82 Wave 4 (W4-T3): eval runner section per training job.
            _render_eval_runner_section(state, job_id, job)


def _render_eval_runner_section(state: AppState, job_id: str, job: TrainingJob) -> None:
    """Render the [Run Static Eval] / [Run Multi-Turn Eval] controls + status.

    Issue #82 Wave 4 (W4-T3): async eval runner in the training page.
    ``MAX_CONCURRENT_EVAL=1`` is enforced by disabling both buttons when
    any eval job is currently in the ``running`` state (D4-003).
    """

    st.markdown("---")
    st.caption("評価ジョブ (Issue #82 Wave 4)")

    # Build a project selector: eval needs ``repo_id`` + ``config_path`` and
    # both live on ``Project``, not ``TrainingJob``.
    if not state.projects:
        st.info("先にプロジェクトを登録すると評価を実行できます。")
        return

    project_name = st.selectbox(
        "評価対象プロジェクト",
        options=list(state.projects.keys()),
        key=f"eval_proj_{job_id}",
    )
    proj = state.projects[project_name]

    running_evals = [ej for ej in state.eval_jobs.values() if ej.status == "running"]
    disable_buttons = len(running_evals) >= _eval_panel.MAX_CONCURRENT_EVAL
    if disable_buttons:
        st.caption("⏳ 他の評価ジョブが実行中のため新規実行は無効です。")

    col_s, col_m = st.columns(2)
    with col_s:
        static_clicked = st.button(
            "Run Static Eval",
            key=f"run_static_{job_id}",
            disabled=disable_buttons,
        )
    with col_m:
        mt_clicked = st.button(
            "Run Multi-Turn Eval",
            key=f"run_mt_{job_id}",
            disabled=disable_buttons,
        )

    if static_clicked:
        _launch_eval_job(state, proj, eval_type="static")
        st.rerun()
    if mt_clicked:
        _launch_eval_job(state, proj, eval_type="multi_turn")
        st.rerun()

    # Running / recent eval jobs for this project.
    related = [ej for ej in state.eval_jobs.values() if ej.project_name == proj.name]
    if not related:
        return
    for ej in sorted(related, key=lambda e: e.started_at, reverse=True)[:5]:
        icon = {
            "pending": "⏳",
            "running": "🔄",
            "succeeded": "✅",
            "failed": "❌",
        }.get(ej.status, "?")
        st.markdown(
            f"**{icon} {ej.eval_type}** · `{ej.job_id[:8]}…` · "
            f"{ej.status} · started {ej.started_at[:19]}"
        )
        if ej.total_q > 0:
            pct = min(ej.done_q / ej.total_q, 1.0)
            st.progress(
                pct,
                text=(
                    f"{ej.done_q}/{ej.total_q} Q · "
                    f"p50 {ej.p50_latency_ms:.0f}ms · "
                    f"NC {ej.nc_rate:.1%}"
                ),
            )
        if ej.status == "succeeded":
            st.caption(
                f"result: {ej.result_json}" if ej.result_json else "(no result_json)"
            )
        if ej.status == "failed" and ej.error_message:
            # Keep the error short in the UI; full tail lives on disk.
            snippet = ej.error_message.strip().splitlines()[-1][:200]
            st.warning(f"失敗: {snippet}")


def _launch_eval_job(
    state: AppState,
    proj: Project,
    eval_type: str,
) -> None:
    """Spawn an eval subprocess + persist a new EvalJob in ``state``.

    Issue #82 Wave 4 (W4-T3).  Path composition goes through
    ``eval_panel.make_eval_paths`` so result_json/log_file/marker_file are
    all confined to ``reports/eval_runs/`` and ``logs/eval/``.  The active
    YAML is resolved via :func:`_resolve_active_config_path` so chat and
    eval honour the same wizard-generated PHOTON config when present.
    """

    job_id = _eval_panel.sanitize_job_id()
    result_json, log_file, marker_file = _eval_panel.make_eval_paths(
        job_id, PROJECT_ROOT
    )
    config_path = _resolve_active_config_path(proj)
    try:
        cmd = _eval_panel.build_eval_job_cmd(
            eval_type=eval_type,
            project_name=proj.name,
            repo_id=proj.repo_id,
            config_path=config_path,
            output_json=result_json,
            marker_file=marker_file,
        )
    except ValueError as exc:
        st.error(f"評価コマンドの構築に失敗: {exc}")
        return
    try:
        proc = _eval_panel.start_eval_job(cmd, log_file)
    except OSError as exc:
        st.error(f"評価プロセスの起動に失敗: {exc}")
        return

    started_epoch = time.time()
    job = EvalJob(
        job_id=job_id,
        project_name=proj.name,
        eval_type=eval_type,
        status="running",
        started_at=datetime.now().isoformat(timespec="seconds"),
        started_at_epoch=started_epoch,
        pid=proc.pid,
        log_file=str(log_file),
        result_json=str(result_json),
        marker_file=str(marker_file),
    )
    state.eval_jobs[job_id] = job
    save()
    st.success(f"評価開始 (PID: {proc.pid}, job_id: {job_id[:8]}…)")


def _generate_photon_config(
    path: str,
    repo_dir: str,
    repo_id: str,
    max_steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
):
    """Generate a PHOTON config YAML from template."""
    template = (PROJECT_ROOT / "configs" / "photon_small.yaml").read_text()

    # Override key values
    import yaml

    cfg = yaml.safe_load(template)
    cfg["repo"]["repo_id"] = repo_id
    cfg["repo"]["repo_path"] = repo_dir
    cfg["repo"]["repo_commit"] = "HEAD"
    cfg["training"]["max_steps"] = max_steps
    cfg["training"]["micro_batch_size"] = batch_size
    cfg["training"]["learning_rate"] = lr
    cfg["training"]["eval_every_steps"] = eval_every
    cfg["training"]["train_corpus"] = "./data/processed/train_tiny.jsonl"
    cfg["training"]["val_corpus"] = "./data/processed/val_tiny.jsonl"

    # Atomic write: serialize to a string and hand off to the shared
    # temp-file + os.replace helper so a crash mid-write cannot leave a
    # partial YAML behind for scripts.train_photon to load.
    payload = yaml.dump(cfg, default_flow_style=False, allow_unicode=True)
    _atomic_write_text(Path(path), payload)


# ================================================================
# Page: Vector DB (Index)
# ================================================================


def page_index():
    st.header("ベクトルデータベース作成")
    state = get_state()

    st.subheader("新規作成")

    repo_dir = st.text_input(
        "対象リポジトリのディレクトリ",
        placeholder="/path/to/your/repo",
        key="idx_repo_dir",
    )
    repo_id = st.text_input(
        "リポジトリ ID (英数字)",
        placeholder="my_project",
        key="idx_repo_id",
        help="インデックスの識別子。英数字とアンダースコアのみ",
    )

    config_choices = _discover_user_configs()
    if not config_choices:
        st.error("configs/*.yaml が見つかりません")
        return
    default_config = "configs/baseline.yaml"
    default_index = (
        config_choices.index(default_config) if default_config in config_choices else 0
    )
    config_path = st.selectbox(
        "Config",
        options=config_choices,
        index=default_index,
        help=(
            "ingestion / retrieval / generation の設定。制度文書 (markdown 中心) は "
            "configs/institutional_docs.yaml、コードベース (Python など) は "
            "configs/baseline.yaml が無難。Embedding モデルは下の選択で override されます。"
        ),
    )

    embedding_models = {
        "all-MiniLM-L6-v2 (軽量・英語向け)": "sentence-transformers/all-MiniLM-L6-v2",
        "multilingual-e5-small (多言語対応)": "intfloat/multilingual-e5-small",
        "multilingual-e5-base (多言語・高精度)": "intfloat/multilingual-e5-base",
        "all-MiniLM-L12-v2 (英語・高精度)": "sentence-transformers/all-MiniLM-L12-v2",
    }
    embedding_label = st.selectbox(
        "Embedding モデル (config の値を override)",
        options=list(embedding_models.keys()),
        index=1,  # multilingual-e5-small をデフォルト
        help="ベクトル検索に使う embedding モデル。日本語を含む場合は multilingual 推奨",
    )
    embedding_model_id = embedding_models[embedding_label]

    if st.button("作成開始", type="primary", disabled=not (repo_dir and repo_id)):
        if not Path(repo_dir).is_dir():
            st.error(f"ディレクトリが見つかりません: {repo_dir}")
        else:
            # Validate repo_id with allowlist before passing to argv.
            try:
                repo_id = _safe_id(repo_id, label="repo_id")
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            job_id = f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                job_id = _safe_id(job_id, label="job_id")
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            log_file = str(PROJECT_ROOT / "logs" / f"{job_id}.log")
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

            # Chain the 4 phases (git rev-parse, ingest, build_indexes,
            # build_symbol_graph) in a Python driver that spawns each phase
            # with ``subprocess.run([...], shell=False)``. We pass the driver
            # to a background ``python -c`` so the Streamlit UI returns
            # immediately while the pipeline runs in the background. All
            # user-controlled values (repo_dir, repo_id, embedding_model_id,
            # config_path) arrive as argv and are never interpolated into a
            # shell string.
            driver = _INDEX_PIPELINE_DRIVER
            cmd_argv = [
                sys.executable,
                "-u",
                "-c",
                driver,
                repo_dir,
                repo_id,
                embedding_model_id,
                config_path,
            ]
            log_fp = open(log_file, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd_argv,
                shell=False,
                cwd=str(PROJECT_ROOT),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
            )

            job = IndexJob(
                job_id=job_id,
                repo_dir=repo_dir,
                repo_id=repo_id,
                config_path=config_path,
                pid=proc.pid,
                started_at=datetime.now().isoformat(),
                status="running",
                log_file=log_file,
                phase="ingest",
                embedding_model=embedding_model_id,
            )
            state.index_jobs[job_id] = job
            save()
            st.success(f"作成開始 (PID: {proc.pid})")
            st.rerun()

    # --- Status ---
    st.subheader("作成ステータス")

    if not state.index_jobs:
        st.info("作成ジョブはありません")
        return

    for job_id, job in sorted(state.index_jobs.items(), reverse=True):
        if _sync_index_job(job):
            save()

        icon = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
        }.get(job.status, "?")

        with st.expander(
            f"{icon} {job.repo_id} — {job.status}", expanded=(job.status == "running")
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("ステータス", job.status)
            col2.metric("フェーズ", job.phase)
            col3.metric("開始時刻", job.started_at[:19] if job.started_at else "—")

            if job.embedding_model:
                st.text(f"Embedding モデル: {job.embedding_model}")

            idx_dir = PROJECT_ROOT / "data" / "indexes" / job.repo_id
            if idx_dir.exists():
                files = list(idx_dir.iterdir())
                st.text(f"Index dir: {idx_dir} ({len(files)} files)")


# ================================================================
# Page: Projects
# ================================================================


def page_projects():
    st.header("RAG プロジェクト登録")
    state = get_state()

    st.subheader("新規プロジェクト")

    name = st.text_input("プロジェクト名", placeholder="my_project")

    # Available indexes
    idx_dir = PROJECT_ROOT / "data" / "indexes"
    available_indexes = []
    if idx_dir.exists():
        available_indexes = [d.name for d in idx_dir.iterdir() if d.is_dir()]

    repo_id = st.selectbox(
        "ベクトルデータベース (repo_id)",
        options=available_indexes if available_indexes else ["(なし — 先にDB作成)"],
    )

    # Available checkpoints: recursively discover any dir containing weights.npz
    # (supports the new layout checkpoints/<repo>/<job>/{best,final,step_XXXXXX}).
    # ``.tmp`` scratch dirs created during atomic replacement are excluded.
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    none_label = "(なし — baseline のみ)"
    ckpt_entries = _discover_checkpoints(ckpt_dir)
    available_ckpts = [none_label] + [t[2] for t in ckpt_entries]
    # Map display label back to path for selection.
    label_to_path = {t[2]: t[1] for t in ckpt_entries}

    selected_label = st.selectbox("PHOTON モデル (checkpoint)", options=available_ckpts)
    checkpoint = label_to_path.get(selected_label, selected_label)
    use_photon = selected_label != none_label

    # Config selection
    available_configs = sorted(
        str(p) for p in (PROJECT_ROOT / "configs").glob("*.yaml")
    )
    config_path = st.selectbox("Config ファイル", options=available_configs)

    # Issue #82 Wave 5 (W5-T2): opt-in PHOTON wizard. The expander keeps
    # the default UX minimal — users who only care about config_path /
    # checkpoint leave the panel collapsed and the wizard is a no-op.
    # When the user opens the expander AND PHOTON is enabled, submit
    # writes ``projects/<safe_id(name)>/photon.yaml`` via
    # ``wizard.generate_yaml_from_wizard`` (+ optional best-practice
    # merge) and overrides ``photon_config_path`` accordingly.
    with st.expander("PHOTON settings (Wave 2-4 toggles)", expanded=False):
        use_wizard = st.checkbox(
            "この form で PHOTON YAML を生成して保存",
            value=False,
            key="wizard_enable",
            help=(
                "オンにすると下記トグルから projects/<name>/photon.yaml を "
                "生成し、photon_config_path に自動設定します。"
            ),
        )
        # Domain templates are appended automatically so adding a new
        # entry to ``_DOMAIN_TEMPLATES`` propagates to the UI for free.
        _BASE_PROFILE_OPTIONS = [
            "photon_small",
            "photon_tiny",
            "photon_long_context",
        ]
        wiz_base_profile = st.selectbox(
            "Config template",
            options=_BASE_PROFILE_OPTIONS + list(_wizard._DOMAIN_TEMPLATES.keys()),
            key="wizard_base_profile",
        )
        wiz_recgen = st.checkbox(
            "RecGen enabled (inference.photon_generation_enabled)",
            value=False,
            key="wizard_recgen",
        )
        wiz_fallback: str | None = None
        if wiz_recgen:
            wiz_fallback = st.radio(
                "Fallback policy (inference.generation_fallback_policy)",
                options=list(_wizard.ALLOWED_FALLBACK_POLICIES),
                index=0,
                key="wizard_fallback",
                horizontal=True,
            )
        wiz_two_pass = st.checkbox(
            "2-pass search enabled (retrieval.two_pass_search.enabled)",
            value=False,
            key="wizard_two_pass",
        )
        wiz_pass1 = st.number_input(
            "pass1_top_k",
            min_value=1,
            value=64,
            step=1,
            key="wizard_pass1_top_k",
            disabled=not wiz_two_pass,
        )
        wiz_pass2 = st.number_input(
            "pass2_top_k",
            min_value=1,
            value=16,
            step=1,
            key="wizard_pass2_top_k",
            disabled=not wiz_two_pass,
        )
        wiz_wm = st.checkbox(
            "Working memory enabled (session_memory.working_memory.enabled)",
            value=True,
            key="wizard_wm",
        )
        wiz_wm_max_turns = st.number_input(
            "max_turns",
            min_value=1,
            value=8,
            step=1,
            key="wizard_wm_max_turns",
            disabled=not wiz_wm,
        )
        wiz_wm_agg = st.selectbox(
            "aggregation",
            options=["weighted", "attention", "last"],
            index=0,
            key="wizard_wm_agg",
            disabled=not wiz_wm,
        )
        wiz_wm_storage = st.selectbox(
            "storage_mode",
            options=["full", "top_level_only"],
            index=0,
            key="wizard_wm_storage",
            disabled=not wiz_wm,
        )
        wiz_pinning = st.checkbox(
            "past_turn_pinning_enabled",
            value=False,
            key="wizard_pinning",
            disabled=not wiz_wm,
        )
        wiz_apply_best = st.checkbox(
            "Apply best-practice when saving",
            value=False,
            key="wizard_apply_best",
            help=(
                "5 キー（safe_recgen / evidence_pruning / working_memory / "
                "photon_generation=false / two_pass_search=false）を選択 "
                "template にマージします。intentional conflict の profile "
                "では警告として表示されます。"
            ),
        )

    if st.button(
        "登録",
        type="primary",
        disabled=not (name and repo_id and repo_id != "(なし — 先にDB作成)"),
    ):
        # Issue #82 Wave 2 (W2-T1): validate project_name before any path
        # composition. Reject metacharacters / traversal via _safe_id.
        try:
            safe_name = _safe_id(name, label="project_name")
        except ValueError as exc:
            st.error(f"プロジェクト名が不正です: {exc}")
            return
        # Path-containment assertion for defense-in-depth: the project dir
        # (when saved) MUST resolve inside PROJECT_ROOT / "projects".
        projects_root = (PROJECT_ROOT / "projects").resolve()
        save_dir = (PROJECT_ROOT / "projects" / safe_name).resolve()
        assert save_dir.is_relative_to(projects_root), (
            f"project save dir escaped projects root: {save_dir}"
        )

        # Issue #82 Wave 5 (W5-T2): if the wizard panel opted in AND the
        # selected checkpoint enables PHOTON, generate a fresh YAML from
        # the chosen template + wizard toggles and (optionally) merge
        # best-practice keys. The resulting file lives inside the
        # validated ``save_dir`` so photon_config_path is contained.
        photon_config_for_project = config_path if use_photon else ""
        if use_photon and use_wizard:
            user_toggles: dict[str, Any] = {
                "recgen_enabled": bool(wiz_recgen),
                "two_pass_search_enabled": bool(wiz_two_pass),
                "two_pass_pass1_top_k": int(wiz_pass1),
                "two_pass_pass2_top_k": int(wiz_pass2),
                "working_memory_enabled": bool(wiz_wm),
                "working_memory_max_turns": int(wiz_wm_max_turns),
                "working_memory_aggregation": str(wiz_wm_agg),
                "working_memory_storage_mode": str(wiz_wm_storage),
                "past_turn_pinning_enabled": bool(wiz_pinning),
            }
            if wiz_recgen and wiz_fallback is not None:
                user_toggles["fallback_policy"] = wiz_fallback

            try:
                generated_yaml = _wizard.generate_yaml_from_wizard(
                    wiz_base_profile,
                    user_toggles,
                )
                if wiz_apply_best:
                    generated_yaml, warnings = _wizard.apply_best_practice(
                        generated_yaml,
                        wiz_base_profile,
                    )
                    for w in warnings:
                        st.warning(w)
            except ValueError as exc:
                st.error(f"wizard YAML 生成に失敗しました: {exc}")
                return

            repo_id_error = _wizard.validate_generated_repo_id(generated_yaml, repo_id)
            if repo_id_error is not None:
                st.error(repo_id_error)
                return

            save_dir.mkdir(parents=True, exist_ok=True)
            photon_yaml_path = (save_dir / "photon.yaml").resolve()
            # Defense-in-depth: ensure the final written path is still
            # inside projects_root after resolve().
            assert photon_yaml_path.is_relative_to(projects_root), (
                f"photon.yaml escaped projects root: {photon_yaml_path}"
            )
            _atomic_write_text(photon_yaml_path, generated_yaml)
            photon_config_for_project = str(photon_yaml_path)
            st.success(f"wizard YAML を保存しました: {photon_yaml_path}")

        project = Project(
            name=safe_name,
            repo_id=repo_id,
            index_dir=str(idx_dir / repo_id),
            config_path=config_path,
            photon_config_path=photon_config_for_project,
            checkpoint_dir=checkpoint if use_photon else "",
            use_photon=use_photon,
            created_at=datetime.now().isoformat(),
        )
        state.projects[safe_name] = project
        save()
        st.success(f"プロジェクト '{safe_name}' を登録しました")
        st.rerun()

    # --- List ---
    st.subheader("登録済みプロジェクト")

    if not state.projects:
        st.info("プロジェクトはありません")
        return

    for pname, proj in state.projects.items():
        with st.expander(f"{'🔬' if proj.use_photon else '📦'} {pname}"):
            st.text(f"repo_id:    {proj.repo_id}")
            st.text(f"config:     {proj.config_path}")
            st.text(f"PHOTON:     {'有効' if proj.use_photon else '無効 (baseline)'}")
            if proj.use_photon:
                st.text(f"checkpoint: {proj.checkpoint_dir}")
            st.text(f"作成日:     {proj.created_at[:19]}")

            if st.button("削除", key=f"del_{pname}"):
                del state.projects[pname]
                save()
                st.rerun()


# ================================================================
# Page: Chat
# ================================================================


def page_chat():
    st.header("チャット")
    state = get_state()

    if not state.projects:
        st.warning("先にプロジェクトを登録してください")
        return

    project_name = st.selectbox(
        "プロジェクト",
        options=list(state.projects.keys()),
    )
    proj = state.projects[project_name]

    st.caption(
        f"repo: {proj.repo_id} | "
        f"{'PHOTON' if proj.use_photon else 'baseline'} | "
        f"config: {Path(proj.config_path).name}"
    )

    # Issue #82 Wave 1 (W1-T1): block retries when MLX is unavailable for
    # a photon-provider project. ``_run_query`` sets this flag on the first
    # ImportError so we don't keep re-trying the heavy import every turn.
    photon_unavailable_key = f"photon_unavailable_{project_name}"
    photon_unavailable = st.session_state.get(photon_unavailable_key)
    if photon_unavailable:
        st.error(
            "PHOTON パイプラインを初期化できません "
            f"({photon_unavailable})。MLX がインストールされているか確認してください。"
        )

    # Session management
    session_key = f"chat_{project_name}"
    if session_key not in state.chat_histories:
        state.chat_histories[session_key] = []

    history = state.chat_histories[session_key]

    # Display history
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input — use text_input + button to avoid IME enter-to-send issue
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question_input = st.text_input(
            "質問",
            key=f"q_{session_key}",
            label_visibility="collapsed",
            placeholder="質問を入力して送信ボタンを押してください",
        )
    with col_btn:
        send_clicked = st.button(
            "送信",
            type="primary",
            key=f"send_{session_key}",
            disabled=bool(photon_unavailable),
        )

    question = question_input if send_clicked and question_input else None

    if question:
        # Add user message
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("回答生成中..."):
                answer, metadata = _run_query(proj, question, session_key)

            st.markdown(answer)

            if metadata:
                # Issue #177: refusal_score badge
                rs = metadata.get("refusal_score")
                if rs is not None:
                    if rs >= 0.7:
                        st.markdown("🔘 **拒絶** — refusal_score: 1.0")
                    else:
                        st.markdown("🟢 **回答** — refusal_score: 0.0")

                with st.expander("メトリクス"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Latency", f"{metadata.get('latency_ms', 0):.0f} ms")
                    col2.metric("Citations", str(metadata.get("cited_count", 0)))
                    col3.metric("Chunks", str(metadata.get("pack_size", 0)))

                # Issue #82 Wave 3: drift + turn_history panels.
                # Isolated so rendering failures do not break chat flow.
                try:
                    cfg_for_panels = load_config(proj.config_path)
                except Exception:
                    cfg_for_panels = None

                try:
                    dm_raw = metadata.get("drift_metrics")
                    dm_dict = None
                    if dm_raw is not None:
                        dm_dict = (
                            dm_raw.as_dict() if hasattr(dm_raw, "as_dict") else dm_raw
                        )
                    thresholds = _build_drift_thresholds(cfg_for_panels)
                    panel = _drift_panel.format_drift_panel(dm_dict, thresholds)
                    with st.expander("Drift metrics"):
                        if not panel["available"]:
                            st.info(panel["reason"])
                        else:
                            for row in panel["rows"]:
                                label = (
                                    f"{row['badge']} {row['name']}".strip()
                                    if row["badge"]
                                    else row["name"]
                                )
                                st.metric(label, row["value_str"])
                            fired = "Yes" if panel["safe_recgen_fired"] else "No"
                            st.caption(f"Safe RecGen fired: {fired}")
                except Exception as exc:
                    st.warning(f"drift panel render failed: {exc}")

                try:
                    wm_enabled = _working_memory_enabled(cfg_for_panels)
                    max_turns = 8
                    if cfg_for_panels is not None:
                        sm = getattr(cfg_for_panels, "session_memory", None)
                        wm = getattr(sm, "working_memory", None) if sm else None
                        max_turns = int(getattr(wm, "max_turns", 8) or 8) if wm else 8
                    hist_panel = _turn_history_panel.format_turn_history_panel(
                        metadata.get("photon_turn_history"),
                        metadata.get("session_turns"),
                        working_memory_enabled=wm_enabled,
                        max_turns=max_turns,
                    )
                    with st.expander("Turn history"):
                        if not hist_panel["available"]:
                            st.info(hist_panel["reason"])
                        elif not hist_panel["rows"]:
                            st.caption("(no turns recorded yet)")
                        else:
                            for row in hist_panel["rows"]:
                                st.markdown(
                                    f"- **turn {row.turn_id}** · "
                                    f"`{row.timestamp}` — "
                                    f"{row.question_text}"
                                )
                                if row.cited_chunk_ids:
                                    st.caption(
                                        "cited: " + ", ".join(row.cited_chunk_ids)
                                    )
                except Exception as exc:
                    st.warning(f"turn history render failed: {exc}")

                try:
                    rm = metadata.get("refusal_matches")
                    if rs is not None:
                        with st.expander("🚦 Refusal score detail", expanded=False):
                            st.write(f"**refusal_score**: {rs}")
                            if rm:
                                st.write("**検出フレーズ**:")
                                for phrase in rm:
                                    st.write(f"- `{phrase}`")
                            else:
                                st.write("検出フレーズなし（回答として判定）")
                except Exception as exc:
                    st.warning(f"refusal score detail render failed: {exc}")

                try:
                    debug_data = metadata.get("retrieval_debug")
                    if debug_data:
                        with st.expander("🔍 Retrieval debug", expanded=False):
                            import pandas as pd

                            df = pd.DataFrame(
                                [
                                    {
                                        "chunk_id": r.chunk_id,
                                        "rel_path": r.rel_path,
                                        "section": r.section or "",
                                        "source": r.source,
                                        "BM25 (norm)": r.bm25_score,
                                        "Embedding (norm)": r.embedding_score,
                                        "Rerank score": r.reranker_score,
                                        "Used": "✓" if r.used else "",
                                        "Citation": (
                                            f"[C:{r.citation_index}]"
                                            if r.citation_index is not None
                                            else ""
                                        ),
                                    }
                                    for r in debug_data
                                ]
                            )
                            st.dataframe(df, use_container_width=True)
                except Exception as exc:
                    st.warning(f"retrieval debug render failed: {exc}")

        history.append({"role": "assistant", "content": answer})
        save()

    # Clear button
    if history and st.button("会話をクリア"):
        state.chat_histories[session_key] = []
        save()
        st.rerun()


def _resolve_active_config_path(proj: Project) -> str:
    """Return the YAML config path that the chat / eval paths should load.

    ``proj.photon_config_path`` (the wizard-generated PHOTON YAML — domain
    template + best-practice merge + repo_id validation) takes priority
    over ``proj.config_path`` so chat and eval stay consistent. Empty /
    missing ``photon_config_path`` falls back to ``config_path``.
    """
    return proj.photon_config_path or proj.config_path


def _pipeline_cache_key(project_name: str, config_path: str) -> str:
    """Compose a session-state cache key that invalidates on config change.

    Including the resolved config path in the key means that if the wizard
    later regenerates ``photon_config_path`` (or the user toggles between
    PHOTON / baseline YAMLs for the same project), the next ``_run_query``
    call will not reuse a stale pipeline that was built against the previous
    config.
    """
    return f"pipeline_{project_name}_{config_path}"


def _run_query(proj: Project, question: str, session_key: str) -> tuple[str, dict]:
    """Run a query through the pipeline via ``build_pipeline(cfg)``.

    Issue #82 Wave 1 (W1-T1): route through the provider-routing factory so
    ``cfg.model.provider == "photon"`` reaches ``PhotonRAGPipeline`` and the
    resulting ``QueryResult.drift_metrics`` / ``turn_id`` become available
    to the UI. MLX import errors are caught and surfaced via the
    ``photon_unavailable_{proj.name}`` session-state flag so the chat page
    can block retries and display a clear error.
    """
    # Default metadata keeps the UI contract stable even on error paths.
    metadata_default: dict[str, Any] = {
        "latency_ms": 0,
        "cited_count": 0,
        "pack_size": 0,
        "no_citation": False,
        "drift_metrics": None,
        "turn_id": 0,
        "refusal_score": None,
        "refusal_matches": None,
    }

    # The wizard-generated PHOTON YAML (proj.photon_config_path) takes
    # priority over the bare config_path; cache key embeds the resolved
    # path so swapping configs invalidates the cached pipeline.
    config_path = _resolve_active_config_path(proj)
    pipeline_key = _pipeline_cache_key(proj.name, config_path)

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        _logger.exception("Failed to load config for %s", proj.name)
        return f"エラー: config load failed ({type(exc).__name__}: {exc})", dict(
            metadata_default
        )

    # UI で選択された ``proj.repo_id`` を真実とし、config 側の hardcoded な
    # ``repo.repo_id`` / ``repo.repo_commit`` を上書きする (build_pipeline は
    # ``data/indexes/{cfg.repo.repo_id}`` から index を読むため、ここで揃え
    # ないと別 repo の index がロードされる)。``repo_commit`` は chunks.db
    # から実際の値を解決し、graph_expansion の SQL filter にも整合させる。
    override_repo_for_pipeline(cfg, proj.repo_id)

    if pipeline_key not in st.session_state:
        try:
            pipeline = build_pipeline(cfg)
        except (ImportError, ModuleNotFoundError) as exc:
            st.session_state[f"photon_unavailable_{proj.name}"] = str(exc)
            _logger.warning("PHOTON pipeline unavailable for %s: %s", proj.name, exc)
            return (
                f"エラー: PHOTON pipeline unavailable ({exc})",
                dict(metadata_default),
            )
        except Exception as exc:
            _logger.exception("Failed to build pipeline for %s", proj.name)
            return (
                f"エラー: pipeline build failed ({type(exc).__name__}: {exc})",
                dict(metadata_default),
            )
        # CB-002: evict stale cached pipelines for the same project so a
        # config-path swap does not leak the previous pipeline (each
        # PhotonRAGPipeline pins MLX weights in memory).
        prefix = f"pipeline_{proj.name}_"
        stale_keys = [
            k
            for k in list(st.session_state.keys())
            if isinstance(k, str) and k.startswith(prefix) and k != pipeline_key
        ]
        for k in stale_keys:
            del st.session_state[k]
        st.session_state[pipeline_key] = pipeline

    pipeline = st.session_state[pipeline_key]

    try:
        result = pipeline.query(
            question=question,
            session_id=session_key,
            repo_id=proj.repo_id,
        )
    except Exception as exc:
        # Keep traceback in logs; keep UI message short.
        _logger.exception("pipeline.query failed for %s", proj.name)
        return (
            f"エラー: {type(exc).__name__}: {exc}",
            dict(metadata_default),
        )

    metadata = {
        "latency_ms": result.latency.total_ms,
        "cited_count": len(result.cited_chunk_ids),
        "pack_size": len(result.cited_chunk_ids),
        "no_citation": result.no_citation,
        "drift_metrics": getattr(result, "drift_metrics", None),
        "turn_id": getattr(result, "turn_id", 0),
        "retrieval_debug": getattr(result, "retrieval_debug", None),
        "refusal_score": getattr(result, "refusal_score", None),
        "refusal_matches": getattr(result, "refusal_matches", None),
    }

    # Issue #82 Wave 3 (W3-T3): surface turn-history for the chat panel.
    # PhotonRAGPipeline keeps PHOTON sessions in ``photon_inference._sessions``
    # and the baseline SessionManager in ``baseline.sessions``. When the
    # pipeline is a plain baseline_rag RepoRAGPipeline, ``photon_inference``
    # is absent and ``photon_turn_history`` stays ``None`` — the UI then
    # renders an "N/A (baseline_rag)" panel.
    photon_turn_history: list[Any] | None = None
    session_turns: list[Any] | None = None
    try:
        photon_inference = getattr(pipeline, "photon_inference", None)
        if photon_inference is not None:
            photon_session = getattr(photon_inference, "_sessions", {}).get(session_key)
            if photon_session is not None:
                photon_turn_history = list(
                    getattr(photon_session, "turn_history", []) or []
                )
            else:
                # Photon pipeline exists but this session has not reached a
                # state that records turn_history yet (e.g. first-turn
                # fail-closed) — render an empty panel rather than N/A.
                photon_turn_history = []
        sessions_mgr = getattr(getattr(pipeline, "baseline", None), "sessions", None)
        if sessions_mgr is None:
            sessions_mgr = getattr(pipeline, "sessions", None)
        if sessions_mgr is not None:
            internal = getattr(sessions_mgr, "_sessions", {}) or {}
            sess = internal.get(session_key)
            if sess is not None:
                session_turns = list(getattr(sess, "turns", []) or [])
    except Exception:
        _logger.exception("Failed to extract turn_history for project %s", proj.name)

    metadata["photon_turn_history"] = photon_turn_history
    metadata["session_turns"] = session_turns

    return result.answer, metadata


def _build_drift_thresholds(cfg: Any) -> dict[str, float | None]:
    """Map ``cfg.safe_recgen.thresholds`` to the 4 UI indicator slots.

    Issue #82 Wave 3 (W3-T3): ``configs/photon_small.yaml:262-266`` exposes
    only the ``latent_cosine_drift`` and ``topic_shift_score`` thresholds;
    the token/mid levels have no configured threshold so those slots are
    ``None`` (classify_drift returns "ok" when the threshold is None).
    """
    sr = getattr(cfg, "safe_recgen", None)
    if sr is None:
        return {
            "token_level": None,
            "mid_level": None,
            "top_level": None,
            "topic_shift": None,
        }
    thr = getattr(sr, "thresholds", None)
    if thr is None:
        return {
            "token_level": None,
            "mid_level": None,
            "top_level": None,
            "topic_shift": None,
        }
    # ``thr`` is a ``Config`` (dot-access wrapper) when loaded from YAML,
    # but a plain dict in tests. Both support ``.get(...)``.
    getter = thr.get if hasattr(thr, "get") else (lambda k, d=None: d)
    return {
        "token_level": None,
        "mid_level": None,
        "top_level": getter("latent_cosine_drift", None),
        "topic_shift": getter("topic_shift_score", None),
    }


def _working_memory_enabled(cfg: Any) -> bool:
    """Return ``True`` iff ``cfg.session_memory.working_memory.enabled``."""
    sm = getattr(cfg, "session_memory", None)
    if sm is None:
        return False
    wm = getattr(sm, "working_memory", None)
    if wm is None:
        return False
    return bool(getattr(wm, "enabled", False))


# ================================================================
# Main
# ================================================================


def main():
    st.set_page_config(
        page_title="PHOTON-RepoRAG",
        page_icon="🔬",
        layout="wide",
    )

    # Background sync keeps the state file correct even when no browser is
    # attached. On first script run we also do a synchronous pass so the UI
    # never shows stale `running` rows for jobs that already finished.
    if "initial_sync_done" not in st.session_state:
        try:
            _sync_state_file()
        except Exception:
            pass
        st.session_state.initial_sync_done = True
    _start_background_sync()

    # Reload from disk on each rerun so background-thread updates are visible
    # and any external edits are picked up.
    st.session_state.app_state = _load_state()

    st.sidebar.title("🔬 PHOTON-RepoRAG")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "メニュー",
        options=[
            "💬 チャット",
            "📦 ベクトルDB作成",
            "🧠 PHOTON学習",
            "📋 プロジェクト登録",
        ],
    )

    if page == "💬 チャット":
        page_chat()
    elif page == "📦 ベクトルDB作成":
        page_index()
    elif page == "🧠 PHOTON学習":
        page_training()
    elif page == "📋 プロジェクト登録":
        page_projects()


if __name__ == "__main__":
    main()
