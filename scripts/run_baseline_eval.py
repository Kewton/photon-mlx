"""
run_baseline_eval.py  –  Run baseline RepoRAG against the static eval set.

Usage:
    python scripts/run_baseline_eval.py --max-questions 10
    python scripts/run_baseline_eval.py  # all 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline eval")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--eval-set", default="data/eval_sets/static_eval.jsonl")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = cfg.repo.repo_id
    idx_dir = Path(cfg.paths.data_root) / "indexes" / repo_id
    run_id = f"baseline_eval_{repo_id}_{time.strftime('%Y%m%d_%H%M%S')}"

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

    # Load eval questions
    questions = []
    for line in Path(args.eval_set).read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line))
    if args.max_questions > 0:
        questions = questions[: args.max_questions]

    print(f"run_id: {run_id}")
    print(f"questions: {len(questions)}")
    print()

    output_path = (
        Path(args.output) if args.output else Path(f"logs/{run_id}_predictions.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            print(
                f"[{i}/{len(questions)}] {q['id']} ({q['category']}) ...",
                end="",
                flush=True,
            )
            result = pipeline.query(
                question=q["question"],
                session_id=f"eval-{q['id']}",
                repo_id=repo_id,
            )
            pred = {
                "eval_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": result.answer,
                "cited_chunk_ids": result.cited_chunk_ids,
                "no_citation": result.no_citation,
                "latency_ms": result.latency.total_ms,
                "retrieval_ms": result.latency.retrieval_ms,
                "generation_ms": result.latency.generation_ms,
                "memory_peak_mb": result.memory.peak_mb,
            }
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            print(
                f" {result.latency.total_ms:.0f}ms  cites={len(result.cited_chunk_ids)}"
            )

    print(f"\nPredictions saved -> {output_path}")
    print(f"Run log -> logs/{run_id}.jsonl")


if __name__ == "__main__":
    main()
