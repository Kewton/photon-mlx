"""
Core query pipeline shared by server.py and cli.py.

Wires together retrieval → graph expansion → evidence pack →
generation → citation with profiling and logging.
"""

from __future__ import annotations

import uuid

from .citation import (
    CitationResult,
    apply_claim_support_guard,
    compute_refusal_score,
    normalise_citation_markers,
    resolve_citations,
)
from .citation_eligibility import apply_citation_budget_rerank
from .config import Config

# CB2-001 (codex-fix): ``QueryResult`` moved to the MLX-free
# ``baseline_reporag.contracts`` module so ``pipeline_factory`` and other
# baseline-only callers can reference it without transitively pulling
# in ``mlx_lm`` via ``.generation.generator``.  Re-exported here for
# backward compatibility — existing imports (``from baseline_reporag.
# pipeline import QueryResult``) keep working.
from .contracts import QueryResult
from .generation.evidence_pack import EvidencePack, build_evidence_pack
from .generation.generator import Generator
from .generation.prompt import ABSTAIN_MARKER, _EVIDENCE_HEADER, build_messages
from .indexing.embedding import EmbeddingIndex
from .indexing.lexical import LexicalIndex
from .indexing.graph_protocol import GraphLike
from .ingestion.store import ChunkStore
from .logger import RunLogger
from .memory.session import SessionManager
from .profiler import TurnProfiler
from .retrieval.debug_builder import (
    build_retrieval_debug_rows,
    finalise_retrieval_debug,
)
from .retrieval.graph_expansion import expand_with_graph
from .retrieval.hybrid import apply_file_type_boost, hybrid_search
from .retrieval.query_expansion import expand_query
from .retrieval.reranker import CrossEncoderReranker

__all__ = [
    "QueryResult",
    "RepoRAGPipeline",
    "apply_citation_postprocess",
]


def apply_citation_postprocess(
    answer: str,
    pack: EvidencePack,
    citation: CitationResult,
    enabled: bool = True,
) -> tuple[str, CitationResult, bool]:
    """Append [C:1] to no-citation answers when appropriate.

    When the LLM omits citation markers, auto-attach ``[C:1]`` so downstream
    metrics and UI can surface the top-ranked evidence chunk. Skipped when:

    - ``enabled`` is False
    - the answer is empty or whitespace-only
    - the answer already contains a valid citation (``no_citation=False``)
    - ``pack.chunks`` is empty (retrieval failure)
    - the answer starts with ``ABSTAIN_MARKER`` (rule 4 legitimate abstain)

    Returns the (possibly modified) answer, citation result, and a flag
    indicating whether post-processing was applied.

    Invariant: ``pack.chunks[0]`` must map to citation index 1
    (guaranteed by ``build_evidence_pack``). A ``RuntimeError`` is raised
    on violation so we fail-closed instead of silently mis-attributing.
    """
    if not isinstance(enabled, bool):
        raise TypeError(f"enabled must be bool, got {type(enabled)}")
    if not enabled:
        return answer, citation, False
    if not answer.strip():
        return answer, citation, False
    if not (
        citation.no_citation and pack.chunks and not answer.startswith(ABSTAIN_MARKER)
    ):
        return answer, citation, False
    target_index = pack.chunk_indices.get(pack.chunks[0].chunk_id)
    if target_index != 1:
        raise RuntimeError(
            f"Invariant violation: pack.chunks[0] must map to [C:1], got {target_index}"
        )
    answer = answer.rstrip() + " [C:1]"
    citation = resolve_citations(answer, pack)
    return answer, citation, True


class RepoRAGPipeline:
    """End-to-end baseline RepoRAG query pipeline with profiling."""

    def __init__(
        self,
        config: Config,
        store: ChunkStore,
        lexical: LexicalIndex,
        embedding: EmbeddingIndex,
        graph: GraphLike | None,
        sessions: SessionManager,
        generator: Generator,
        logger: RunLogger,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.cfg = config
        self.store = store
        self.lexical = lexical
        self.embedding = embedding
        self.graph = graph
        self.sessions = sessions
        self.generator = generator
        self.logger = logger
        self.reranker = reranker

    def query(
        self,
        question: str,
        session_id: str = "",
        repo_id: str = "",
        *,
        seed: int | None = None,
    ) -> QueryResult:
        cfg = self.cfg
        prof = TurnProfiler()
        prof.start()

        session_id = session_id or str(uuid.uuid4())
        repo_id = repo_id or cfg.repo.repo_id
        session = self.sessions.get_or_create(
            session_id,
            repo_id,
            cfg.repo.repo_commit,
        )

        # --- Query expansion (computed once, shared by retrieval + reranker) ---
        qe_cfg = cfg.retrieval.query_expansion
        if qe_cfg.get("enabled", False):
            _queries = expand_query(question, mapping=qe_cfg.get("domain_map"))
            expansion_terms: str | None = _queries[1] if len(_queries) > 1 else None
        else:
            expansion_terms = None

        # --- Retrieval ---
        with prof.phase("retrieval"):
            raw = hybrid_search(
                query=question,
                lexical_index=self.lexical,
                embedding_index=self.embedding,
                lexical_top_k=cfg.retrieval.lexical_top_k,
                embedding_top_k=cfg.retrieval.embedding_top_k,
                fused_top_k=cfg.retrieval.fused_top_k,
                lexical_weight=cfg.retrieval.weights.lexical,
                embedding_weight=cfg.retrieval.weights.embedding,
                expanded_queries=[expansion_terms] if expansion_terms else [],
                repo_id=repo_id,
            )

        # Snapshot pre-rerank scores for debug rows (Issue #176)
        raw_snapshot = list(raw)

        # --- Reranking (noise filter + optional cross-encoder) ---
        reranked_top: list = []
        rejected_debug: list = []
        with prof.phase("reranking"):
            if self.reranker is not None:
                reranked_top, rejected_debug = self.reranker.rerank_with_debug(
                    query=question,
                    results=raw,
                    store=self.store,
                    top_k=cfg.retrieval.rerank_top_k,
                    rerank_query=expansion_terms,
                    rejected_debug_top_n=10,
                )
                raw = reranked_top

        # --- File-type boost (post-reranking) ---
        file_type_boost = cfg.retrieval.get("file_type_boost", 0.0)
        if file_type_boost:
            raw = apply_file_type_boost(raw, boost=file_type_boost)

        # --- Graph expansion ---
        with prof.phase("graph_expansion"):
            expanded_refs = expand_with_graph(
                results=raw,
                store=self.store,
                graph=self.graph,
                repo_id=repo_id,
                repo_commit=cfg.repo.repo_commit,
                max_hops=cfg.retrieval.graph_expansion.max_hops,
                max_nodes=cfg.retrieval.graph_expansion.max_nodes,
                neighborhood_before=cfg.retrieval.neighborhood_expansion.before,
                neighborhood_after=cfg.retrieval.neighborhood_expansion.after,
            )

        # Build skeleton debug rows before evidence pack trims the set (Issue #176)
        debug_rows = build_retrieval_debug_rows(
            raw_snapshot=raw_snapshot,
            reranked_top=reranked_top if self.reranker is not None else list(raw),
            rejected=rejected_debug,
            expanded_refs=expanded_refs,
            store=self.store,
        )

        # --- Evidence pack ---
        chunk_ids = [ref.chunk_id for ref in expanded_refs]
        with prof.phase("evidence_pack"):
            pack = build_evidence_pack(
                chunk_ids=chunk_ids,
                store=self.store,
                session=session,
                max_chunks=cfg.evidence_pack.max_chunks,
                max_tokens=cfg.evidence_pack.max_tokens,
            )

        # --- Generation ---
        with prof.phase("generation"):
            evidence_text = pack.format_for_prompt()
            # INVARIANT: session.add_turn() is called AFTER this point (line ~132),
            # so len(session.turns) == 0 correctly identifies the first turn.
            is_first_turn = len(session.turns) == 0
            if is_first_turn:
                evidence_text = f"{_EVIDENCE_HEADER}\n\n{evidence_text}"
            messages = build_messages(
                question=question,
                evidence_text=evidence_text,
                history_text=session.history_text(max_turns=4),
            )
            # Issue #143: only forward ``seed`` when an eval explicitly
            # asked for it.  Interactive/CLI/server callers leave it at
            # ``None`` and we MUST keep the legacy single-positional-arg
            # call shape so the 17+ existing MagicMock tests keep passing
            # without spurious ``seed=None`` kwargs leaking through.
            # DR3-002: ``if seed is not None`` (NOT ``if seed:``); seed=0
            # is a valid deterministic seed and must propagate.
            if seed is not None:
                answer = self.generator.generate(messages, seed=seed)
            else:
                answer = self.generator.generate(messages)

        # --- Citation ---
        with prof.phase("citation"):
            answer = normalise_citation_markers(answer)
            citation = resolve_citations(answer, pack)
            answer, citation, claim_guard = apply_claim_support_guard(
                question=question,
                answer=answer,
                pack=pack,
                citation=citation,
            )
            # post-processing: auto-attach [C:1] to no-citation answers.
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
            citation_budget = apply_citation_budget_rerank(
                question=question,
                answer=answer,
                pack=pack,
                citation=citation,
                context_text=session.history_text(max_turns=4),
                retrieval_scores={r.chunk_id: r.score for r in raw},
            )
            answer = citation_budget.answer
            citation = citation_budget.citation

        # Finalise debug rows: set used/citation_index now that pack + citations are known
        pack_chunk_ids = [c.chunk_id for c in pack.chunks]
        debug_rows = finalise_retrieval_debug(
            rows=debug_rows,
            pack_chunk_ids=pack_chunk_ids,
            cited_chunk_ids=citation.cited_chunk_ids,
        )

        latency, memory = prof.finish()

        # --- Session update ---
        # Do not pollute session memory with machine-attached citations:
        # the auto-attached [C:1] is a display/observability aid, not a
        # user-intended reference, and would bias MT retrieval priority.
        session_cited_ids = [] if citation_postprocessed else citation.cited_chunk_ids
        turn = session.add_turn(question, answer, session_cited_ids)
        self.sessions.save(session)

        # --- Log ---
        self.logger.log_turn(
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
                "claim_support_guard_applied": claim_guard.applied,
                "claim_support_guard_reason": claim_guard.reason,
                "claim_support_guard_terms": claim_guard.unsupported_terms or [],
                "citation_budget_reranked": citation_budget.changed,
                "citation_budget_removed_indices": citation_budget.removed_indices,
                "citation_budget_replaced_indices": citation_budget.replaced_indices,
                "citation_eligibility_scores": [
                    vars(score) for score in citation_budget.scores
                ],
                "latency": latency.as_dict(),
                "memory": memory.as_dict(),
                "fallback_flag": False,
                "fallback_reason": None,
            }
        )

        r_score, r_matches = compute_refusal_score(answer)
        return QueryResult(
            answer=answer,
            session_id=session_id,
            turn_id=turn.turn_id,
            cited_chunk_ids=citation.cited_chunk_ids,
            wrong_citation_indices=citation.wrong_citation_indices,
            no_citation=citation.no_citation,
            latency=latency,
            memory=memory,
            citation_postprocessed=citation_postprocessed,
            # CB-003 (codex-fix): RepoRAGPipeline always uses the Qwen
            # generator; PhotonRAGPipeline overrides this when PHOTON
            # produces the answer and propagates the closed-enum fallback
            # reason when it falls back.
            generator_used="qwen",
            generator_fallback_reason=None,
            retrieval_debug=debug_rows,
            claim_support_guard_applied=claim_guard.applied,
            claim_support_guard_reason=claim_guard.reason,
            claim_support_guard_terms=claim_guard.unsupported_terms or [],
            citation_budget_reranked=citation_budget.changed,
            citation_budget_removed_indices=citation_budget.removed_indices,
            citation_budget_replaced_indices=citation_budget.replaced_indices,
            citation_eligibility_scores=[vars(score) for score in citation_budget.scores],
            refusal_score=r_score,
            refusal_matches=r_matches,
        )
