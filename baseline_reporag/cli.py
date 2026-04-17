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
from pathlib import Path

from .config import load_config
from .generation.generator import Generator
from .indexing.embedding import EmbeddingIndex
from .indexing.lexical import LexicalIndex
from .indexing.symbol_graph import SymbolGraph
from .ingestion.store import ChunkStore
from .logger import RunLogger
from .memory.session import SessionManager
from .pipeline import RepoRAGPipeline
from .retrieval.reranker import CrossEncoderReranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline RepoRAG CLI")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--question", default="")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = args.repo_id or cfg.repo.repo_id
    idx_dir = Path(cfg.paths.data_root) / "indexes" / repo_id
    run_id = f"baseline_{repo_id}_{time.strftime('%Y%m%d')}_{cfg.repo.repo_commit[:7]}"

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
    pipeline = RepoRAGPipeline(
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

    def run_query(question: str) -> None:
        result = pipeline.query(
            question=question,
            session_id=args.session_id,
            repo_id=repo_id,
        )
        print(
            f"\n[Turn {result.turn_id}]  {result.latency.total_ms:.0f} ms"
            f"  (retrieval {result.latency.retrieval_ms:.0f}"
            f" | gen {result.latency.generation_ms:.0f})"
            f"  mem {result.memory.peak_mb:.1f} MB\n"
        )
        print(result.answer)
        if result.no_citation:
            print("\n[WARNING] No citations in this answer.")
        if result.wrong_citation_indices:
            print(
                f"[WARNING] Unknown citation indices: {result.wrong_citation_indices}"
            )
        print(f"\nCited: {result.cited_chunk_ids}")
        print(f"Session: {result.session_id}")

    if args.question:
        run_query(args.question)
    else:
        print(f"session: {args.session_id or '(auto)'}")
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
