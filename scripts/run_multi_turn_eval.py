"""
run_multi_turn_eval.py  –  Run multi-turn session eval against baseline.

Usage:
    python scripts/run_multi_turn_eval.py --max-sessions 5
    python scripts/run_multi_turn_eval.py  # all 30
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-turn eval")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--eval-set", default="data/eval_sets/multi_turn_eval.jsonl")
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = cfg.repo.repo_id
    idx_dir = Path(cfg.paths.data_root) / "indexes" / repo_id
    run_id = f"mt_eval_{repo_id}_{time.strftime('%Y%m%d_%H%M%S')}"

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
    )

    # Load sessions
    sessions = []
    for line in Path(args.eval_set).read_text(encoding="utf-8").splitlines():
        if line.strip():
            sessions.append(json.loads(line))
    if args.max_sessions > 0:
        sessions = sessions[: args.max_sessions]

    print(f"run_id: {run_id}")
    print(f"sessions: {len(sessions)}")
    print()

    output_path = (
        Path(args.output) if args.output else Path(f"logs/{run_id}_predictions.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session_stats: list[dict] = []

    with open(output_path, "w", encoding="utf-8") as f:
        for si, session_data in enumerate(sessions, 1):
            sid = session_data["session_id"]
            turns = session_data["turns"]
            print(f"[Session {si}/{len(sessions)}] {sid} ({len(turns)} turns)")

            turn_latencies: list[float] = []
            turn_citations: list[int] = []

            for turn in turns:
                result = pipeline.query(
                    question=turn["question"],
                    session_id=f"eval-{sid}",
                    repo_id=repo_id,
                )
                pred = {
                    "session_id": sid,
                    "turn_id": turn["turn_id"],
                    "question": turn["question"],
                    "answer": result.answer,
                    "cited_chunk_ids": result.cited_chunk_ids,
                    "no_citation": result.no_citation,
                    "latency_ms": result.latency.total_ms,
                    "retrieval_ms": result.latency.retrieval_ms,
                    "generation_ms": result.latency.generation_ms,
                    "memory_peak_mb": result.memory.peak_mb,
                }
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

                turn_latencies.append(result.latency.total_ms)
                turn_citations.append(len(result.cited_chunk_ids))

                print(
                    f"  T{turn['turn_id']}: {result.latency.total_ms:.0f}ms "
                    f"cites={len(result.cited_chunk_ids)}"
                )

            # Session summary
            avg_lat = sum(turn_latencies) / len(turn_latencies)
            followup_lat = sum(turn_latencies[1:]) / max(len(turn_latencies) - 1, 1)
            session_stats.append(
                {
                    "session_id": sid,
                    "avg_latency_ms": avg_lat,
                    "followup_avg_latency_ms": followup_lat,
                    "first_turn_latency_ms": turn_latencies[0],
                    "total_citations": sum(turn_citations),
                    "no_citation_turns": sum(1 for c in turn_citations if c == 0),
                }
            )
            print(f"  → avg {avg_lat:.0f}ms, follow-up avg {followup_lat:.0f}ms\n")

    # Summary
    print(f"\n{'=' * 50}")
    print("Session summary:")
    for s in session_stats:
        print(
            f"  {s['session_id']}: avg={s['avg_latency_ms']:.0f}ms "
            f"followup={s['followup_avg_latency_ms']:.0f}ms "
            f"cites={s['total_citations']} "
            f"no_cite_turns={s['no_citation_turns']}"
        )

    # Save summary
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(session_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nPredictions -> {output_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
