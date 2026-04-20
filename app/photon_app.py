"""PHOTON-RepoRAG Management App (Streamlit)

Launch:
    streamlit run app/photon_app.py --server.port 3012 --server.baseUrlPath /proxy/photon
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
STATE_FILE = PROJECT_ROOT / ".cache" / "photon_app_state.json"


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


# ================================================================
# State persistence
# ================================================================


def _load_state() -> AppState:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            state = AppState()
            for k, v in data.get("training_jobs", {}).items():
                state.training_jobs[k] = TrainingJob(**v)
            for k, v in data.get("index_jobs", {}).items():
                state.index_jobs[k] = IndexJob(**v)
            for k, v in data.get("projects", {}).items():
                state.projects[k] = Project(**v)
            state.chat_histories = data.get("chat_histories", {})
            return state
        except Exception:
            pass
    return AppState()


def _save_state(state: AppState) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "training_jobs": {k: asdict(v) for k, v in state.training_jobs.items()},
        "index_jobs": {k: asdict(v) for k, v in state.index_jobs.items()},
        "projects": {k: asdict(v) for k, v in state.projects.items()},
        "chat_histories": state.chat_histories,
    }
    STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


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
    """Read last step and val_loss from training log."""
    result: dict[str, Any] = {"last_step": 0, "val_loss": 0.0, "max_steps": 0}
    log_path = Path(log_file)
    if not log_path.exists():
        return result
    try:
        lines = log_path.read_text().strip().split("\n")
        for line in reversed(lines):
            try:
                rec = json.loads(line)
                if "step" in rec:
                    result["last_step"] = max(result["last_step"], rec["step"])
                if "val_loss" in rec:
                    result["val_loss"] = rec["val_loss"]
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    return result


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

            # Step 2: Start training in background
            train_log = str(PROJECT_ROOT / "logs" / f"{job_id}.log")
            train_cmd = f"python -u -m scripts.train_photon --config {config_path} > {train_log} 2>&1"
            proc = subprocess.Popen(
                train_cmd,
                shell=True,
                cwd=str(PROJECT_ROOT),
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
        running = _is_process_running(job.pid)
        if job.status == "running" and not running:
            job.status = "completed"
            save()

        progress = _read_training_progress(
            str(PROJECT_ROOT / "logs" / "train_log.jsonl")
        )
        if progress["last_step"] > 0:
            job.last_step = progress["last_step"]
            job.val_loss = progress["val_loss"]

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

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


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

    if st.button("作成開始", type="primary", disabled=not (repo_dir and repo_id)):
        if not Path(repo_dir).is_dir():
            st.error(f"ディレクトリが見つかりません: {repo_dir}")
        else:
            job_id = f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_file = str(PROJECT_ROOT / "logs" / f"{job_id}.log")

            # Run all 3 steps in background
            # Get actual commit SHA so build_indexes/symbol_graph use it
            cmd = (
                f"REPO_COMMIT=$(git -C {repo_dir} rev-parse HEAD) && "
                f'echo "Phase 1: Ingest (commit=$REPO_COMMIT)" && '
                f"python -m scripts.ingest_repo --repo {repo_dir} --repo-id {repo_id} --commit $REPO_COMMIT --config configs/baseline.yaml && "
                f"echo 'Phase 2: BM25 + Embedding' && "
                f"python -m scripts.build_indexes --repo-id {repo_id} --commit $REPO_COMMIT --config configs/baseline.yaml && "
                f"echo 'Phase 3: Symbol Graph' && "
                f"python -m scripts.build_symbol_graph --repo-id {repo_id} --commit $REPO_COMMIT --config configs/baseline.yaml && "
                f"echo 'DONE'"
            )
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(PROJECT_ROOT),
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
            )

            job = IndexJob(
                job_id=job_id,
                repo_dir=repo_dir,
                repo_id=repo_id,
                config_path="configs/baseline.yaml",
                pid=proc.pid,
                started_at=datetime.now().isoformat(),
                status="running",
                log_file=log_file,
                phase="ingest",
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
        running = _is_process_running(job.pid)
        if job.status == "running":
            if not running:
                # Check if DONE
                log_path = Path(job.log_file)
                if log_path.exists() and "DONE" in log_path.read_text():
                    job.status = "completed"
                    job.phase = "completed"
                else:
                    job.status = "failed"
                save()
            else:
                # Update phase
                log_path = Path(job.log_file)
                if log_path.exists():
                    content = log_path.read_text()
                    if "Phase 3" in content:
                        job.phase = "symbol_graph"
                    elif "Phase 2" in content:
                        job.phase = "bm25_embed"
                    else:
                        job.phase = "ingest"

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

    # Available checkpoints
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    available_ckpts = ["(なし — baseline のみ)"]
    if ckpt_dir.exists():
        for d in sorted(ckpt_dir.iterdir()):
            if d.is_dir() and (d / "weights.npz").exists():
                available_ckpts.append(str(d))

    checkpoint = st.selectbox("PHOTON モデル (checkpoint)", options=available_ckpts)
    use_photon = checkpoint != "(なし — baseline のみ)"

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

    # Session management
    session_key = f"chat_{project_name}"
    if session_key not in state.chat_histories:
        state.chat_histories[session_key] = []

    history = state.chat_histories[session_key]

    # Display history
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if question := st.chat_input("質問を入力..."):
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
    """Run a query through the pipeline."""
    try:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT))

        from baseline_reporag.config import load_config
        from baseline_reporag.generation.generator import Generator
        from baseline_reporag.indexing.embedding import EmbeddingIndex
        from baseline_reporag.indexing.lexical import LexicalIndex
        from baseline_reporag.indexing.symbol_graph import SymbolGraph
        from baseline_reporag.ingestion.store import ChunkStore
        from baseline_reporag.logger import RunLogger
        from baseline_reporag.memory.session import SessionManager
        from baseline_reporag.pipeline import RepoRAGPipeline
        from baseline_reporag.retrieval.reranker import CrossEncoderReranker

        # Cache pipeline in session state
        pipeline_key = f"pipeline_{proj.name}"
        if pipeline_key not in st.session_state:
            cfg = load_config(proj.config_path)
            idx_dir = Path(cfg.paths.data_root) / "indexes" / proj.repo_id
            run_id = f"app_{proj.repo_id}_{int(time.time())}"

            reranker_cfg = cfg.retrieval.reranker
            reranker = (
                CrossEncoderReranker(
                    model_id=reranker_cfg.get(
                        "model_id", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    )
                )
                if reranker_cfg.get("enabled", False)
                else None
            )

            st.session_state[pipeline_key] = RepoRAGPipeline(
                config=cfg,
                store=ChunkStore(idx_dir / "chunks.db"),
                lexical=LexicalIndex.load(idx_dir / "lexical.pkl"),
                embedding=EmbeddingIndex.load(idx_dir / "embedding"),
                graph=SymbolGraph.load(idx_dir / "symbol_graph.json"),
                sessions=SessionManager(log_dir=Path(cfg.paths.log_root) / "sessions"),
                generator=Generator(
                    model_id=cfg.model.model_id,
                    max_new_tokens=cfg.generation.max_new_tokens,
                    temperature=cfg.generation.temperature,
                    top_p=cfg.generation.top_p,
                ),
                logger=RunLogger(cfg.paths.log_root, run_id),
                reranker=reranker,
            )

        pipeline = st.session_state[pipeline_key]
        result = pipeline.query(
            question=question,
            session_id=session_key,
            repo_id=proj.repo_id,
        )

        metadata = {
            "latency_ms": result.latency.total_ms,
            "cited_count": len(result.cited_chunk_ids),
            "pack_size": len(result.cited_chunk_ids),
            "no_citation": result.no_citation,
        }

        return result.answer, metadata

    except Exception as e:
        return f"エラーが発生しました: {e}", {}


# ================================================================
# Main
# ================================================================


def main():
    st.set_page_config(
        page_title="PHOTON-RepoRAG",
        page_icon="🔬",
        layout="wide",
    )

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
