"""PHOTON-RAG pipeline integration (Issue #3).

Provides:
- build_pipeline(cfg) — factory that routes to RepoRAGPipeline or PhotonRAGPipeline
- PhotonRAGPipeline — PHOTON-enhanced RAG with drift tracking and fallback
- tokenize_evidence_pack() — encode evidence text for PHOTON prefill
- compute_confidence() — extract confidence from PHOTON logits
"""

from __future__ import annotations

from math import prod
from typing import Any

import mlx.core as mx

from .config import Config
from .pipeline import QueryResult, RepoRAGPipeline


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def tokenize_evidence_pack(
    text: str,
    tokenizer: Any,
    cfg: Any,
    max_tokens: int = 2048,
) -> mx.array:
    """Tokenize evidence text with chunk-aligned padding.

    Args:
        text: raw evidence text.
        tokenizer: tokenizer with encode() and pad_token_id.
        cfg: config with hierarchy.chunk_sizes.
        max_tokens: hard cap on token count.

    Returns:
        mx.array of token ids, length is a multiple of prod(chunk_sizes).
    """
    ids = tokenizer.encode(text)
    if not ids:
        return mx.array([], dtype=mx.int32)

    if len(ids) > max_tokens:
        ids = ids[:max_tokens]

    padding_multiple = prod(cfg.hierarchy.chunk_sizes)
    remainder = len(ids) % padding_multiple
    if remainder != 0:
        pad_count = padding_multiple - remainder
        ids = ids + [tokenizer.pad_token_id] * pad_count

    return mx.array(ids, dtype=mx.int32)


def compute_confidence(logits: mx.array) -> float:
    """Compute mean max-softmax confidence from logits.

    Args:
        logits: (B, seq_len, vocab_size) tensor.

    Returns:
        float in [0, 1].
    """
    probs = mx.softmax(logits, axis=-1)
    max_probs = mx.max(probs, axis=-1)
    return float(mx.mean(max_probs).item())


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def _build_baseline_deps(cfg: Config) -> dict[str, Any]:
    """Construct real baseline pipeline dependencies from config."""
    from .generation.generator import Generator
    from .indexing.embedding import EmbeddingIndex
    from .indexing.lexical import LexicalIndex
    from .indexing.symbol_graph import SymbolGraph
    from .ingestion.store import ChunkStore
    from .logger import RunLogger
    from .memory.session import SessionManager

    store = ChunkStore(cfg.repo.repo_id, cfg.repo.repo_commit)
    lexical = LexicalIndex(store)
    embedding = EmbeddingIndex(store)
    graph = SymbolGraph(store)
    sessions = SessionManager(log_dir=cfg.memory.log_dir)
    generator = Generator(cfg.model.model_id)
    logger = RunLogger(cfg)

    return {
        "store": store,
        "lexical": lexical,
        "embedding": embedding,
        "graph": graph,
        "sessions": sessions,
        "generator": generator,
        "logger": logger,
    }


def _build_photon_deps(cfg: Config) -> dict[str, Any]:
    """Construct PHOTON-specific dependencies from config."""
    raise NotImplementedError("Wire _build_photon_deps to real PHOTON components.")


def build_pipeline(cfg: Config) -> RepoRAGPipeline | PhotonRAGPipeline:
    """Factory: create the right pipeline based on cfg.model.provider."""
    provider = getattr(cfg.model, "provider", None) or "baseline"
    deps = _build_baseline_deps(cfg)

    if provider == "photon":
        photon_deps = _build_photon_deps(cfg)
        return PhotonRAGPipeline(cfg=cfg, baseline_deps=deps, photon_deps=photon_deps)

    return RepoRAGPipeline(
        config=cfg,
        store=deps["store"],
        lexical=deps["lexical"],
        embedding=deps["embedding"],
        graph=deps["graph"],
        sessions=deps["sessions"],
        generator=deps["generator"],
        logger=deps["logger"],
    )


# ---------------------------------------------------------------------------
# PhotonRAGPipeline
# ---------------------------------------------------------------------------


class PhotonRAGPipeline:
    """PHOTON-enhanced RepoRAG pipeline with drift tracking and fallback."""

    def __init__(
        self,
        cfg: Config,
        baseline_deps: dict[str, Any],
        photon_deps: dict[str, Any],
    ) -> None:
        self.cfg = cfg
        self.baseline = RepoRAGPipeline(
            config=cfg,
            store=baseline_deps["store"],
            lexical=baseline_deps["lexical"],
            embedding=baseline_deps["embedding"],
            graph=baseline_deps["graph"],
            sessions=baseline_deps["sessions"],
            generator=baseline_deps["generator"],
            logger=baseline_deps["logger"],
        )
        self.photon_inference = photon_deps["photon_inference"]
        self.safe_recgen = photon_deps["safe_recgen"]
        self.photon_cfg = photon_deps["photon_cfg"]
        self.tokenizer = photon_deps["tokenizer"]

    def query(
        self,
        question: str,
        session_id: str = "",
        repo_id: str = "",
    ) -> QueryResult:
        """Run PHOTON-enhanced query with drift tracking and fallback."""
        # Delegate to baseline for now (full integration in later cycles)
        return self.baseline.query(question, session_id=session_id, repo_id=repo_id)
