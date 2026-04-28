"""Lightweight provider-routing factory for RepoRAG pipelines.

CB-004 / CB2-001 (codex-fix): ``build_pipeline`` used to live in
``baseline_reporag.photon_pipeline``, which eagerly imports ``mlx.core``.
Baseline-only environments (no MLX installed) therefore broke at import
time even when ``model.provider`` was ``"baseline"``.

This module keeps the factory surface MLX-free at import time by
**lazy-importing both the baseline and PHOTON pipeline modules**:

- ``baseline_reporag.pipeline`` transitively imports
  ``baseline_reporag.generation.generator`` which in turn performs
  ``import mlx_lm`` at module top (guarded by try/except, but still
  transitively pulls ``mlx.core`` when MLX is installed).
- ``baseline_reporag.photon_pipeline`` performs
  ``import mlx.core as mx`` unconditionally.

By deferring **both** imports to inside ``build_pipeline``, a plain
``import baseline_reporag.pipeline_factory`` never touches MLX and can
be used from pure-baseline entry points (``baseline_reporag.cli``,
``baseline_reporag.server`` and the ``scripts/run_*_eval.py`` set) even
on machines without MLX installed.

Only ``.contracts`` (MLX-free dataclasses) and ``.config`` are imported
at module load.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Config, is_symbol_graph_enabled, validate_repo_id
from .contracts import QueryResult  # re-exported; MLX-free

if TYPE_CHECKING:
    # Hint types without importing the heavy modules at runtime.
    from .photon_pipeline import PhotonRAGPipeline
    from .pipeline import RepoRAGPipeline


__all__ = ["QueryResult", "build_pipeline", "override_repo_for_pipeline"]


def override_repo_for_pipeline(
    cfg: Config,
    repo_id: str | None,
    *,
    data_root: str | None = None,
) -> None:
    """Mutate ``cfg`` in-place so ``build_pipeline()`` loads the right corpus.

    ``build_pipeline()`` resolves the index directory from
    ``cfg.repo.repo_id`` (``data/indexes/{cfg.repo.repo_id}``) — when a
    caller's ``repo_id`` differs from the config's hardcoded value (e.g.
    Streamlit project selection, ``--repo-id`` CLI override), the indexes
    loaded would be the wrong corpus.  Existing eval scripts
    (``scripts/run_baseline_eval.py``, ``scripts/run_multi_turn_eval.py``)
    document this and mutate ``cfg.repo.repo_id`` manually before
    ``build_pipeline``; this helper centralises that pattern and additionally
    resolves ``repo_commit`` from ``chunks.db`` so graph_expansion's
    ``iter_repo(repo_id, repo_commit)`` SQL filter sees real data.

    No-op when ``repo_id`` is falsy.

    Note: the override is destructive — the caller's ``cfg`` instance has
    its ``repo.repo_id`` and ``repo.repo_commit`` overwritten. Callers that
    reuse a single ``cfg`` for multiple repos must reload via
    ``load_config`` between switches.
    """
    if not repo_id:
        return
    cfg.repo.repo_id = repo_id
    actual_commit = _lookup_repo_commit_from_db(
        repo_id, data_root or cfg.paths.data_root
    )
    if actual_commit is not None:
        cfg.repo.repo_commit = actual_commit


def _lookup_repo_commit_from_db(repo_id: str, data_root: str) -> str | None:
    """Read one ``repo_commit`` value stored in ``chunks.db`` for ``repo_id``.

    Returns ``None`` when the index dir / DB does not exist or the table is
    empty — the caller keeps the config's existing value in that case (so
    fresh ingest is detectable as an error rather than silently replaced
    with the empty string).
    """
    db_path = Path(data_root) / "indexes" / repo_id / "chunks.db"
    if not db_path.is_file():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.DatabaseError:
        return None
    try:
        row = conn.execute(
            "SELECT repo_commit FROM chunks WHERE repo_id = ? LIMIT 1",
            (repo_id,),
        ).fetchone()
    except sqlite3.DatabaseError:
        row = None
    finally:
        conn.close()
    return row[0] if row else None


def build_pipeline(cfg: Config) -> "RepoRAGPipeline | PhotonRAGPipeline":
    """Factory: create the right pipeline based on ``cfg.model.provider``.

    ``provider == "photon"`` lazy-imports the PHOTON pipeline module (and
    therefore MLX); all other providers stay on the pure-baseline path.
    The baseline path still lazy-imports ``RepoRAGPipeline`` here so the
    factory module's own import remains MLX-free.
    """
    provider = getattr(cfg.model, "provider", None) or "baseline"

    if provider == "photon":
        # Lazy import: only pull in MLX when the config actually requests
        # the PHOTON pipeline.  Importing ``baseline_reporag.photon_pipeline``
        # transitively imports ``mlx.core`` (and is expensive), so baseline
        # environments must not hit this branch.
        from .photon_pipeline import (
            PhotonRAGPipeline,
            _build_baseline_deps,
            _build_photon_deps,
        )

        deps = _build_baseline_deps(cfg)
        photon_deps = _build_photon_deps(cfg)
        return PhotonRAGPipeline(cfg=cfg, baseline_deps=deps, photon_deps=photon_deps)

    # Baseline path: still lazy so the factory import stays MLX-free.
    # ``baseline_reporag.pipeline`` → ``.generation.generator`` → ``mlx_lm``.
    from .pipeline import RepoRAGPipeline

    deps = _build_baseline_deps_no_mlx(cfg)
    return RepoRAGPipeline(
        config=cfg,
        store=deps["store"],
        lexical=deps["lexical"],
        embedding=deps["embedding"],
        graph=deps["graph"],
        sessions=deps["sessions"],
        generator=deps["generator"],
        logger=deps["logger"],
        reranker=deps["reranker"],
    )


def _build_baseline_deps_no_mlx(cfg: Config) -> dict:
    """Canonical baseline dependency factory (Issue #62 refactor R-1).

    Single source of truth for building the baseline pipeline deps dict;
    ``baseline_reporag.photon_pipeline._build_baseline_deps`` is a thin
    wrapper that delegates here. The ``_no_mlx`` suffix is retained for
    historical clarity: the key invariant is that all heavy imports are
    performed lazily inside the function body so the module-level import
    of ``pipeline_factory`` remains MLX-free (CB-004 / CB2-001).
    """
    import uuid
    from pathlib import Path

    from .generation.generator import Generator
    from .indexing.embedding import EmbeddingIndex
    from .indexing.lexical import LexicalIndex
    from .indexing.symbol_graph import SymbolGraph
    from .ingestion.store import ChunkStore
    from .logger import RunLogger
    from .memory.session import SessionManager
    from .retrieval.reranker import CrossEncoderReranker

    # CB-004: refuse ``../outside`` / ``/tmp/x`` / etc. before they reach
    # ``Path`` concatenation. ``scripts/build_symbol_graph.py`` already
    # ran the same check; now every index-loading entry point does.
    repo_id = validate_repo_id(cfg.repo.repo_id)
    idx_dir = Path(cfg.paths.data_root) / "indexes" / repo_id
    store = ChunkStore(idx_dir / "chunks.db")
    lexical = LexicalIndex.load(idx_dir / "lexical.pkl")
    embedding = EmbeddingIndex.load(idx_dir / "embedding")
    # Issue #109: ``indexing.symbol_graph.enabled=false`` skips graph load.
    # The ``graph_expansion`` helper accepts ``graph=None`` and falls back
    # to file-neighbors only (see ``retrieval/graph_expansion.py``).
    graph: SymbolGraph | None = (
        SymbolGraph.load(idx_dir / "symbol_graph.json")
        if is_symbol_graph_enabled(cfg)
        else None
    )
    sessions = SessionManager(log_dir=Path(cfg.paths.log_root) / "sessions")
    generator = Generator(
        model_id=cfg.model.model_id,
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
    )
    run_id = f"bench_variant_{uuid.uuid4().hex[:8]}"
    logger = RunLogger(cfg.paths.log_root, run_id)

    reranker_cfg = cfg.retrieval.reranker
    reranker = (
        CrossEncoderReranker(
            model_id=reranker_cfg.get(
                "model_id", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            noise_patterns=reranker_cfg.get("noise_patterns"),
        )
        if reranker_cfg.get("enabled", False)
        else None
    )

    return {
        "store": store,
        "lexical": lexical,
        "embedding": embedding,
        "graph": graph,
        "sessions": sessions,
        "generator": generator,
        "logger": logger,
        "reranker": reranker,
    }
