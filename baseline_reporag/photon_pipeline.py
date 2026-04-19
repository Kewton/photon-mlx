"""PHOTON-RAG pipeline integration (Issue #3).

Provides:
- build_pipeline(cfg) — factory that routes to RepoRAGPipeline or PhotonRAGPipeline
- PhotonRAGPipeline — PHOTON-enhanced RAG with drift tracking and fallback
- tokenize_evidence_pack() — encode evidence text for PHOTON prefill
- compute_confidence() — extract confidence from PHOTON logits
"""

from __future__ import annotations

import uuid
from math import prod
from typing import Any

import mlx.core as mx

from .citation import resolve_citations
from .config import Config
from .generation.evidence_pack import build_evidence_pack
from .generation.prompt import _EVIDENCE_HEADER, build_messages
from .pipeline import QueryResult, RepoRAGPipeline, apply_citation_postprocess
from .profiler import TurnProfiler
from .retrieval.graph_expansion import expand_with_graph
from .retrieval.hybrid import apply_file_type_boost, hybrid_search
from .retrieval.query_expansion import expand_query


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
    from pathlib import Path
    import uuid

    from .generation.generator import Generator
    from .indexing.embedding import EmbeddingIndex
    from .indexing.lexical import LexicalIndex
    from .indexing.symbol_graph import SymbolGraph
    from .ingestion.store import ChunkStore
    from .logger import RunLogger
    from .memory.session import SessionManager
    from .retrieval.reranker import CrossEncoderReranker

    idx_dir = Path(cfg.paths.data_root) / "indexes" / cfg.repo.repo_id
    store = ChunkStore(idx_dir / "chunks.db")
    lexical = LexicalIndex.load(idx_dir / "lexical.pkl")
    embedding = EmbeddingIndex.load(idx_dir / "embedding")
    graph = SymbolGraph.load(idx_dir / "symbol_graph.json")
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
            )
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
        reranker=deps["reranker"],
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
            reranker=baseline_deps["reranker"],
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
        """Run PHOTON-enhanced query with drift tracking and evidence pruning.

        On follow-up turns (turn 2+), PHOTON coarse state is used to prune
        the evidence pack from max_chunks down to pruned_max_chunks, halving
        LLM prefill time while retaining the most session-relevant chunks.
        """
        cfg = self.cfg
        bl = self.baseline  # access baseline components without calling query()
        prof = TurnProfiler()
        prof.start()

        session_id = session_id or str(uuid.uuid4())
        repo_id = repo_id or cfg.repo.repo_id
        session = bl.sessions.get_or_create(
            session_id,
            repo_id,
            cfg.repo.repo_commit,
        )

        # --- PHOTON prefill for drift / confidence ---
        photon_session_id = session_id or "default"
        evidence_tokens = tokenize_evidence_pack(question, self.tokenizer, cfg)
        if evidence_tokens.size > 0:
            input_ids = evidence_tokens.reshape(1, -1)
            logits, drift = self.photon_inference.session_forward(
                input_ids,
                session_id=photon_session_id,
                repo_id=repo_id or "unknown",
                repo_commit="HEAD",
            )
            confidence = compute_confidence(logits)
            drift_dict = drift.as_dict() if drift else None
        else:
            confidence = 1.0
            drift = None
            drift_dict = None

        # --- Safe RecGen evaluation ---
        fallback_dict = None
        if self.safe_recgen is not None and drift is not None:
            decision = self.safe_recgen.evaluate(
                question, drift=drift, confidence=confidence
            )
            fallback_dict = decision.as_dict()

        # --- Query expansion ---
        qe_cfg = cfg.retrieval.query_expansion
        if qe_cfg.get("enabled", False):
            _queries = expand_query(question)
            expansion_terms: str | None = _queries[1] if len(_queries) > 1 else None
        else:
            expansion_terms = None

        # --- Retrieval ---
        with prof.phase("retrieval"):
            raw = hybrid_search(
                query=question,
                lexical_index=bl.lexical,
                embedding_index=bl.embedding,
                lexical_top_k=cfg.retrieval.lexical_top_k,
                embedding_top_k=cfg.retrieval.embedding_top_k,
                fused_top_k=cfg.retrieval.fused_top_k,
                lexical_weight=cfg.retrieval.weights.lexical,
                embedding_weight=cfg.retrieval.weights.embedding,
                expanded_queries=[expansion_terms] if expansion_terms else [],
            )

        is_follow_up = len(session.turns) > 0

        # --- Reranking (skip on follow-up: PHOTON pruning handles selection) ---
        with prof.phase("reranking"):
            if bl.reranker is not None and not is_follow_up:
                raw = bl.reranker.rerank(
                    query=question,
                    results=raw,
                    store=bl.store,
                    top_k=cfg.retrieval.rerank_top_k,
                    rerank_query=expansion_terms,
                )

        # --- File-type boost ---
        file_type_boost = cfg.retrieval.get("file_type_boost", 0.0)
        if file_type_boost:
            raw = apply_file_type_boost(raw, boost=file_type_boost)

        # --- Graph expansion ---
        with prof.phase("graph_expansion"):
            expanded_ids = expand_with_graph(
                results=raw,
                store=bl.store,
                graph=bl.graph,
                repo_id=repo_id,
                repo_commit=cfg.repo.repo_commit,
                max_hops=cfg.retrieval.graph_expansion.max_hops,
                max_nodes=cfg.retrieval.graph_expansion.max_nodes,
                neighborhood_before=cfg.retrieval.neighborhood_expansion.before,
                neighborhood_after=cfg.retrieval.neighborhood_expansion.after,
            )

        # --- Evidence pruning (PHOTON-guided, follow-up turns only) ---
        inference_cfg = cfg.get("inference")
        pruning_enabled = (
            getattr(inference_cfg, "evidence_pruning_enabled", False)
            if inference_cfg is not None
            else False
        )
        pruned_max_chunks = (
            getattr(inference_cfg, "pruned_max_chunks", 8)
            if inference_cfg is not None
            else 8
        )
        effective_max_chunks = cfg.evidence_pack.max_chunks
        if pruning_enabled and is_follow_up:
            # Fetch chunk texts for scoring
            chunks_for_scoring = bl.store.get_many(expanded_ids)
            chunk_texts = [c.content for c in chunks_for_scoring]
            chunk_ids_for_scoring = [c.chunk_id for c in chunks_for_scoring]

            selected_indices = self.photon_inference.prune_evidence(
                chunk_texts=chunk_texts,
                chunk_ids=chunk_ids_for_scoring,
                session_id=photon_session_id,
                max_chunks=pruned_max_chunks,
            )
            # Filter expanded_ids to only selected chunks
            expanded_ids = [chunk_ids_for_scoring[i] for i in selected_indices]
            effective_max_chunks = pruned_max_chunks

        # --- Evidence pack ---
        with prof.phase("evidence_pack"):
            pack = build_evidence_pack(
                chunk_ids=expanded_ids,
                store=bl.store,
                session=session,
                max_chunks=effective_max_chunks,
                max_tokens=cfg.evidence_pack.max_tokens,
            )

        # --- Generation ---
        with prof.phase("generation"):
            evidence_text = pack.format_for_prompt()
            is_first_turn = len(session.turns) == 0
            if is_first_turn:
                evidence_text = f"{_EVIDENCE_HEADER}\n\n{evidence_text}"
            messages = build_messages(
                question=question,
                evidence_text=evidence_text,
                history_text=session.history_text(max_turns=4),
                include_few_shot=is_first_turn,
            )
            followup_tokens = 512 if not is_first_turn else None
            answer = bl.generator.generate(messages, max_new_tokens=followup_tokens)

        # --- Citation ---
        with prof.phase("citation"):
            citation = resolve_citations(answer, pack)
            answering_cfg = getattr(cfg, "answering", None)
            if answering_cfg is not None:
                postprocess_enabled = answering_cfg.get(
                    "citation_postprocess_enabled", True
                )
            else:
                postprocess_enabled = True
            if not isinstance(postprocess_enabled, bool):
                raise RuntimeError(
                    "answering.citation_postprocess_enabled must be bool, "
                    f"got {type(postprocess_enabled)}"
                )
            answer, citation, citation_postprocessed = apply_citation_postprocess(
                answer, pack, citation, enabled=postprocess_enabled
            )

        latency, memory = prof.finish()

        # --- Session update ---
        session_cited_ids = [] if citation_postprocessed else citation.cited_chunk_ids
        turn = session.add_turn(question, answer, session_cited_ids)
        bl.sessions.save(session)

        # --- Log ---
        bl.logger.log_turn(
            {
                "session_id": session_id,
                "turn_id": turn.turn_id,
                "repo_id": repo_id,
                "repo_commit": cfg.repo.repo_commit,
                "model_id": cfg.model.model_id,
                "question": question,
                "answer": answer,
                "retrieval_chunk_ids": [r.chunk_id for r in raw],
                "evidence_pack_ids": [c.chunk_id for c in pack.chunks],
                "cited_chunk_ids": citation.cited_chunk_ids,
                "wrong_citation_indices": citation.wrong_citation_indices,
                "no_citation": citation.no_citation,
                "citation_postprocessed": citation_postprocessed,
                "latency": latency.as_dict(),
                "memory": memory.as_dict(),
                "fallback_flag": bool(
                    fallback_dict and fallback_dict.get("should_fallback")
                ),
                "fallback_reason": (
                    fallback_dict.get("reasons") if fallback_dict else None
                ),
                "evidence_pruning_applied": pruning_enabled and is_follow_up,
            }
        )

        result = QueryResult(
            answer=answer,
            session_id=session_id,
            turn_id=turn.turn_id,
            cited_chunk_ids=citation.cited_chunk_ids,
            wrong_citation_indices=citation.wrong_citation_indices,
            no_citation=citation.no_citation,
            latency=latency,
            memory=memory,
            citation_postprocessed=citation_postprocessed,
        )

        # Attach PHOTON metadata
        result.drift_metrics = drift_dict
        result.confidence = confidence
        result.fallback_decision = fallback_dict

        return result
