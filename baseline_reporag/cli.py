"""
Baseline RepoRAG – CLI.

Usage (single question):
    python -m baseline_reporag.cli \
        --repo-id fastapi_fastapi \
        --question "認証処理の入口はどこですか？"

Usage (interactive):
    python -m baseline_reporag.cli --repo-id fastapi_fastapi
"""
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

from .citation import resolve_citations
from .config import load_config
from .generation.evidence_pack import build_evidence_pack
from .generation.generator import Generator
from .generation.prompt import build_messages
from .indexing.embedding import EmbeddingIndex
from .indexing.lexical import LexicalIndex
from .indexing.symbol_graph import SymbolGraph
from .ingestion.store import ChunkStore
from .logger import RunLogger
from .memory.session import SessionManager
from .retrieval.graph_expansion import expand_with_graph
from .retrieval.hybrid import hybrid_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline RepoRAG CLI")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--question", default="")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = args.repo_id or cfg.repo.repo_id
    repo_commit = cfg.repo.repo_commit
    idx_dir = Path(cfg.paths.data_root) / "indexes" / repo_id

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
    run_id = (
        f"baseline_{repo_id}"
        f"_{time.strftime('%Y%m%d')}"
        f"_{repo_commit[:7]}"
    )
    logger = RunLogger(cfg.paths.log_root, run_id)

    session_id = args.session_id or str(uuid.uuid4())
    session = sessions.get_or_create(session_id, repo_id, repo_commit)
    print(f"session: {session_id}\n")

    def run_query(question: str) -> None:
        t0 = time.perf_counter()

        raw = hybrid_search(
            query=question,
            lexical_index=lexical,
            embedding_index=embedding,
            lexical_top_k=cfg.retrieval.lexical_top_k,
            embedding_top_k=cfg.retrieval.embedding_top_k,
            fused_top_k=cfg.retrieval.fused_top_k,
            lexical_weight=cfg.retrieval.weights.lexical,
            embedding_weight=cfg.retrieval.weights.embedding,
        )
        expanded_ids = expand_with_graph(
            results=raw,
            store=store,
            graph=graph,
            repo_id=repo_id,
            repo_commit=repo_commit,
            max_hops=cfg.retrieval.graph_expansion.max_hops,
            max_nodes=cfg.retrieval.graph_expansion.max_nodes,
            neighborhood_before=cfg.retrieval.neighborhood_expansion.before,
            neighborhood_after=cfg.retrieval.neighborhood_expansion.after,
        )
        pack = build_evidence_pack(
            chunk_ids=expanded_ids,
            store=store,
            session=session,
            max_chunks=cfg.evidence_pack.max_chunks,
            max_tokens=cfg.evidence_pack.max_tokens,
        )
        messages = build_messages(
            question=question,
            evidence_text=pack.format_for_prompt(),
            history_text=session.history_text(max_turns=4),
        )
        answer = generator.generate(messages)
        citation = resolve_citations(answer, pack)

        turn = session.add_turn(question, answer, citation.cited_chunk_ids)
        sessions.save(session)

        latency_ms = (time.perf_counter() - t0) * 1000

        logger.log_turn({
            "session_id": session_id,
            "turn_id": turn.turn_id,
            "repo_id": repo_id,
            "repo_commit": repo_commit,
            "model_id": cfg.model.model_id,
            "question": question,
            "answer": answer,
            "retrieval_chunk_ids": [r.chunk_id for r in raw],
            "evidence_pack_ids": [c.chunk_id for c in pack.chunks],
            "cited_chunk_ids": citation.cited_chunk_ids,
            "wrong_citation_indices": citation.wrong_citation_indices,
            "no_citation": citation.no_citation,
            "latency_ms": latency_ms,
            "fallback_flag": False,
            "fallback_reason": None,
        })

        print(f"\n[Turn {turn.turn_id}]  {latency_ms:.0f} ms\n")
        print(answer)
        if citation.no_citation:
            print("\n[WARNING] No citations in this answer.")
        if citation.wrong_citation_indices:
            print(f"[WARNING] Unknown citation indices: {citation.wrong_citation_indices}")
        print(f"\nCited chunks: {citation.cited_chunk_ids}")

    if args.question:
        run_query(args.question)
    else:
        print("Type your question (empty line to quit):\n")
        while True:
            try:
                q = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            run_query(q)


if __name__ == "__main__":
    main()
