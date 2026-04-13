"""
Core query pipeline shared by server.py and cli.py.

Wires together retrieval → graph expansion → evidence pack →
generation → citation with profiling and logging.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from .citation import resolve_citations
from .config import Config
from .generation.evidence_pack import build_evidence_pack
from .generation.generator import Generator
from .generation.prompt import _EVIDENCE_HEADER, build_messages
from .indexing.embedding import EmbeddingIndex
from .indexing.lexical import LexicalIndex
from .indexing.symbol_graph import SymbolGraph
from .ingestion.store import ChunkStore
from .logger import RunLogger
from .memory.session import SessionManager
from .profiler import LatencyBreakdown, MemorySnapshot, TurnProfiler
from .retrieval.graph_expansion import expand_with_graph
from .retrieval.hybrid import hybrid_search


@dataclass
class QueryResult:
    answer: str
    session_id: str
    turn_id: int
    cited_chunk_ids: list[str]
    wrong_citation_indices: list[int]
    no_citation: bool
    latency: LatencyBreakdown
    memory: MemorySnapshot


class RepoRAGPipeline:
    """End-to-end baseline RepoRAG query pipeline with profiling."""

    def __init__(
        self,
        config: Config,
        store: ChunkStore,
        lexical: LexicalIndex,
        embedding: EmbeddingIndex,
        graph: SymbolGraph,
        sessions: SessionManager,
        generator: Generator,
        logger: RunLogger,
    ) -> None:
        self.cfg = config
        self.store = store
        self.lexical = lexical
        self.embedding = embedding
        self.graph = graph
        self.sessions = sessions
        self.generator = generator
        self.logger = logger

    def query(
        self,
        question: str,
        session_id: str = "",
        repo_id: str = "",
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
            )

        # --- Graph expansion ---
        with prof.phase("graph_expansion"):
            expanded_ids = expand_with_graph(
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

        # --- Evidence pack ---
        with prof.phase("evidence_pack"):
            pack = build_evidence_pack(
                chunk_ids=expanded_ids,
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
            answer = self.generator.generate(messages)

        # --- Citation ---
        with prof.phase("citation"):
            citation = resolve_citations(answer, pack)

        latency, memory = prof.finish()

        # --- Session update ---
        turn = session.add_turn(question, answer, citation.cited_chunk_ids)
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
                "latency": latency.as_dict(),
                "memory": memory.as_dict(),
                "fallback_flag": False,
                "fallback_reason": None,
            }
        )

        return QueryResult(
            answer=answer,
            session_id=session_id,
            turn_id=turn.turn_id,
            cited_chunk_ids=citation.cited_chunk_ids,
            wrong_citation_indices=citation.wrong_citation_indices,
            no_citation=citation.no_citation,
            latency=latency,
            memory=memory,
        )
