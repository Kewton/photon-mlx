"""
run_stress_eval.py  –  Run stress eval for concurrent session testing.

Usage:
    python scripts/run_stress_eval.py --dry-run
    python scripts/run_stress_eval.py --max-sessions 3
    python scripts/run_stress_eval.py --config configs/baseline.yaml --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _get_rss_mb() -> float:
    """Return current process RSS in MB."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback: use os on macOS / Linux
        import resource

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in bytes on macOS, kilobytes on Linux
        if sys.platform == "darwin":
            return rusage.ru_maxrss / (1024 * 1024)
        return rusage.ru_maxrss / 1024


def _percentile(data: list[float], pct: int) -> float:
    """Compute the *pct*-th percentile of *data* (0-100)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def load_sessions(path: str | Path, max_sessions: int = 0) -> list[dict]:
    """Load stress eval sessions from a JSONL file."""
    sessions: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            sessions.append(json.loads(line))
    if max_sessions > 0:
        sessions = sessions[:max_sessions]
    return sessions


def run_dry(sessions: list[dict], concurrency: int) -> None:
    """Dry-run mode: print what would be executed without LLM inference."""
    total_turns = sum(len(s["turns"]) for s in sessions)
    print("=" * 60)
    print("DRY-RUN MODE — no LLM inference will be performed")
    print("=" * 60)
    print(f"  Eval set sessions : {len(sessions)}")
    print(f"  Total turns       : {total_turns}")
    print(f"  Concurrency       : {concurrency}")
    print()
    for i, session in enumerate(sessions, 1):
        sid = session["session_id"]
        turns = session["turns"]
        group = session.get("concurrency_group", "-")
        print(f"  [{i}] session={sid}  turns={len(turns)}  group={group}")
        for t in turns:
            q_preview = t["question"][:60]
            print(f"       T{t['turn_id']}: {q_preview}...")
    print()
    print("Done (dry-run). No resources consumed.")


def run_real(
    sessions: list[dict],
    config_path: str,
    concurrency: int,
) -> None:
    """Run sessions sequentially (concurrent threading deferred)."""
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

    cfg = load_config(config_path)
    repo_id = cfg.repo.repo_id
    idx_dir = Path(cfg.paths.data_root) / "indexes" / repo_id
    run_id = f"stress_eval_{repo_id}_{time.strftime('%Y%m%d_%H%M%S')}"

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

    total_turns = sum(len(s["turns"]) for s in sessions)
    print(f"run_id      : {run_id}")
    print(f"sessions    : {len(sessions)}")
    print(f"total turns : {total_turns}")
    print(f"concurrency : {concurrency} (sequential execution)")
    print()

    output_path = Path(f"logs/{run_id}_predictions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_latencies: list[float] = []
    peak_rss_mb: float = _get_rss_mb()
    session_summaries: list[dict] = []

    with open(output_path, "w", encoding="utf-8") as f:
        for si, session_data in enumerate(sessions, 1):
            sid = session_data["session_id"]
            turns = session_data["turns"]
            print(f"[Session {si}/{len(sessions)}] {sid} ({len(turns)} turns)")

            turn_latencies: list[float] = []

            for turn in turns:
                t0 = time.perf_counter()
                result = pipeline.query(
                    question=turn["question"],
                    session_id=f"stress-{sid}",
                    repo_id=repo_id,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

                pred = {
                    "session_id": sid,
                    "turn_id": turn["turn_id"],
                    "question": turn["question"],
                    "answer": result.answer,
                    "cited_chunk_ids": result.cited_chunk_ids,
                    "no_citation": result.no_citation,
                    "latency_ms": elapsed_ms,
                    "retrieval_ms": result.latency.retrieval_ms,
                    "generation_ms": result.latency.generation_ms,
                    "memory_peak_mb": result.memory.peak_mb,
                }
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

                turn_latencies.append(elapsed_ms)
                all_latencies.append(elapsed_ms)

                current_rss = _get_rss_mb()
                if current_rss > peak_rss_mb:
                    peak_rss_mb = current_rss

                print(
                    f"  T{turn['turn_id']}: {elapsed_ms:.0f}ms "
                    f"cites={len(result.cited_chunk_ids)}"
                )

            avg_lat = statistics.mean(turn_latencies) if turn_latencies else 0.0
            session_summaries.append(
                {
                    "session_id": sid,
                    "turns": len(turns),
                    "avg_latency_ms": avg_lat,
                    "p50_latency_ms": _percentile(turn_latencies, 50),
                    "p90_latency_ms": _percentile(turn_latencies, 90),
                }
            )
            print(f"  -> avg {avg_lat:.0f}ms\n")

    # ---- Summary ----
    print("=" * 60)
    print("STRESS EVAL SUMMARY")
    print("=" * 60)
    print(f"  Total sessions    : {len(sessions)}")
    print(f"  Total turns       : {len(all_latencies)}")
    print(f"  Latency P50       : {_percentile(all_latencies, 50):.0f} ms")
    print(f"  Latency P90       : {_percentile(all_latencies, 90):.0f} ms")
    print(f"  Latency mean      : {statistics.mean(all_latencies):.0f} ms")
    print(f"  Peak RSS          : {peak_rss_mb:.1f} MB")
    print()

    summary = {
        "run_id": run_id,
        "total_sessions": len(sessions),
        "total_turns": len(all_latencies),
        "concurrency": concurrency,
        "latency_p50_ms": _percentile(all_latencies, 50),
        "latency_p90_ms": _percentile(all_latencies, 90),
        "latency_mean_ms": statistics.mean(all_latencies),
        "peak_rss_mb": peak_rss_mb,
        "sessions": session_summaries,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Predictions -> {output_path}")
    print(f"Summary     -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stress eval for concurrent session testing"
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--eval-set", default="data/eval_sets/stress_eval.jsonl")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrency level (threading deferred; currently sequential)",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Limit number of sessions to run (0 = all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without LLM inference",
    )
    args = parser.parse_args()

    sessions = load_sessions(args.eval_set, args.max_sessions)

    if not sessions:
        print(f"No sessions found in {args.eval_set}")
        sys.exit(1)

    if args.dry_run:
        run_dry(sessions, args.concurrency)
    else:
        run_real(sessions, args.config, args.concurrency)


if __name__ == "__main__":
    main()
