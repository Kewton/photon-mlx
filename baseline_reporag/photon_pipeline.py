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
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from torch_ref.config import PhotonConfig

    from photon_mlx.inference import PhotonInference
    from photon_mlx.model import PhotonModel
    from photon_mlx.safe_recgen import SafeRecGenConfig, SafeRecGenController

    from torch_ref.config import (
        HierarchyConfig,
        ModelConfig,
        TokenizerConfig,
    )

    model_cfg = ModelConfig(
        architecture=cfg.model.get("architecture", "photon_decoder"),
        base_embed_dim=cfg.model.base_embed_dim,
        hidden_size=cfg.model.hidden_size,
        intermediate_size=cfg.model.intermediate_size,
        num_attention_heads=cfg.model.get("num_heads", 4),
        num_key_value_heads=cfg.model.get("num_heads", 4),
    )
    hierarchy_cfg = HierarchyConfig(
        levels=cfg.hierarchy.levels,
        chunk_sizes=cfg.hierarchy.chunk_sizes,
        encoder_layers_per_level=cfg.hierarchy.encoder_layers_per_level,
        decoder_layers_per_level=cfg.hierarchy.decoder_layers_per_level,
    )
    tok_cfg = TokenizerConfig(
        vocab_size=cfg.model.get("vocab_size", 1000),
    )
    photon_cfg = PhotonConfig(
        model=model_cfg,
        hierarchy=hierarchy_cfg,
        tokenizer=tok_cfg,
    )

    model = PhotonModel(photon_cfg)
    photon_inference = PhotonInference(model, photon_cfg)

    safe_recgen_enabled = getattr(cfg.get("inference"), "safe_recgen_enabled", True)
    if safe_recgen_enabled:
        sr_cfg_data = cfg.get("safe_recgen")
        if sr_cfg_data is not None:
            thresholds = sr_cfg_data.get("thresholds")
            sr_config = SafeRecGenConfig(
                enabled=True,
                confidence_floor=(
                    thresholds.confidence_floor
                    if thresholds and hasattr(thresholds, "confidence_floor")
                    else 0.40
                ),
            )
        else:
            sr_config = SafeRecGenConfig(enabled=True)
        safe_recgen = SafeRecGenController(sr_config)
    else:
        safe_recgen = None

    tokenizer = _get_stub_tokenizer(photon_cfg.tokenizer.vocab_size)

    return {
        "photon_inference": photon_inference,
        "safe_recgen": safe_recgen,
        "photon_cfg": photon_cfg,
        "tokenizer": tokenizer,
    }


class _StubTokenizer:
    """Minimal tokenizer for PHOTON prefill (encode/decode via utf-8 byte ids)."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")


def _get_stub_tokenizer(vocab_size: int) -> _StubTokenizer:
    return _StubTokenizer(vocab_size)


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
        # 1) Run PHOTON prefill for drift / confidence estimation
        evidence_tokens = tokenize_evidence_pack(question, self.tokenizer, self.cfg)
        if evidence_tokens.size > 0:
            input_ids = evidence_tokens.reshape(1, -1)
            logits, drift = self.photon_inference.session_forward(
                input_ids,
                session_id=session_id or "default",
                repo_id=repo_id or "unknown",
                repo_commit="HEAD",
            )
            confidence = compute_confidence(logits)
            drift_dict = drift.as_dict() if drift else None
        else:
            confidence = 1.0
            drift = None
            drift_dict = None

        # 2) Safe RecGen evaluation
        fallback_dict = None
        if self.safe_recgen is not None and drift is not None:
            decision = self.safe_recgen.evaluate(
                question, drift=drift, confidence=confidence
            )
            fallback_dict = decision.as_dict()

        # 3) Delegate generation to baseline pipeline
        result = self.baseline.query(question, session_id=session_id, repo_id=repo_id)

        # 4) Attach PHOTON metadata
        result.drift_metrics = drift_dict
        result.confidence = confidence
        result.fallback_decision = fallback_dict

        return result
