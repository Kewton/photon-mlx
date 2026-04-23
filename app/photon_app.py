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
from baseline_reporag.pipeline_factory import build_pipeline  # noqa: E402

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


# Driver executed in a child ``python -c`` to chain the 4 index-build
# phases with argv-list subprocess.run calls (shell=False). No user input
# is interpolated into this string — every user value arrives via sys.argv.
_INDEX_PIPELINE_DRIVER = """
import subprocess, sys
repo_dir, repo_id, embed_model, config_path = sys.argv[1:5]

def run(argv, phase):
    print(f'>>> {phase}: ' + ' '.join(argv), flush=True)
    subprocess.run(argv, check=True)

# Phase 0: resolve commit SHA
out = subprocess.run(
    ['git', '-C', repo_dir, 'rev-parse', 'HEAD'],
    check=True, capture_output=True, text=True,
)
commit = out.stdout.strip()
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
    phase: str = ""  # ingest | bm25_embed | symbol_graph | completed
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
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


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
        if "Phase 3" in log_content:
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


def _sync_all_jobs(state: AppState) -> bool:
    """Reconcile every training/index job in `state`. Returns True if mutated."""
    changed = False
    progress = _read_training_progress(str(PROJECT_ROOT / "logs" / "train_log.jsonl"))
    for job in state.training_jobs.values():
        if _sync_training_job(job, progress):
            changed = True
    for job in state.index_jobs.values():
        if _sync_index_job(job):
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

    embedding_models = {
        "all-MiniLM-L6-v2 (軽量・英語向け)": "sentence-transformers/all-MiniLM-L6-v2",
        "multilingual-e5-small (多言語対応)": "intfloat/multilingual-e5-small",
        "multilingual-e5-base (多言語・高精度)": "intfloat/multilingual-e5-base",
        "all-MiniLM-L12-v2 (英語・高精度)": "sentence-transformers/all-MiniLM-L12-v2",
    }
    embedding_label = st.selectbox(
        "Embedding モデル",
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
            config_path = "configs/baseline.yaml"
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

    if st.button(
        "登録",
        type="primary",
        disabled=not (name and repo_id and repo_id != "(なし — 先にDB作成)"),
    ):
        # Issue #82 Wave 2 (W2-T1): validate project_name before any path
        # composition. Reject metacharacters / traversal via _safe_id.
        try:
            _safe_id(name, label="project_name")
        except ValueError as exc:
            st.error(f"プロジェクト名が不正です: {exc}")
            return
        # Path-containment assertion for defense-in-depth: the project dir
        # (when saved) MUST resolve inside PROJECT_ROOT / "projects".
        projects_root = (PROJECT_ROOT / "projects").resolve()
        save_dir = (PROJECT_ROOT / "projects" / name).resolve()
        assert save_dir.is_relative_to(projects_root), (
            f"project save dir escaped projects root: {save_dir}"
        )
        project = Project(
            name=name,
            repo_id=repo_id,
            index_dir=str(idx_dir / repo_id),
            config_path=config_path,
            photon_config_path=config_path if use_photon else "",
            checkpoint_dir=checkpoint if use_photon else "",
            use_photon=use_photon,
            created_at=datetime.now().isoformat(),
        )
        state.projects[name] = project
        save()
        st.success(f"プロジェクト '{name}' を登録しました")
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
                with st.expander("メトリクス"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Latency", f"{metadata.get('latency_ms', 0):.0f} ms")
                    col2.metric("Citations", str(metadata.get("cited_count", 0)))
                    col3.metric("Chunks", str(metadata.get("pack_size", 0)))

        history.append({"role": "assistant", "content": answer})
        save()

    # Clear button
    if history and st.button("会話をクリア"):
        state.chat_histories[session_key] = []
        save()
        st.rerun()


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
    }

    pipeline_key = f"pipeline_{proj.name}"

    try:
        cfg = load_config(proj.config_path)
    except Exception as exc:
        _logger.exception("Failed to load config for %s", proj.name)
        return f"エラー: config load failed ({type(exc).__name__}: {exc})", dict(
            metadata_default
        )

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
    }

    return result.answer, metadata


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
