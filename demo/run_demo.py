"""
run_demo.py  –  Run a demo scenario against the baseline RepoRAG.

Usage:
    python demo/run_demo.py --scenario demo-01 --config configs/baseline.yaml
    python demo/run_demo.py --list
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.scenarios import SCENARIOS, print_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a demo scenario")
    parser.add_argument("--scenario", default="")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--list", action="store_true", help="List all scenarios")
    args = parser.parse_args()

    if args.list or not args.scenario:
        print_scenarios()
        return

    scenario = next((s for s in SCENARIOS if s.id == args.scenario), None)
    if not scenario:
        print(f"Unknown scenario: {args.scenario}")
        print(f"Available: {[s.id for s in SCENARIOS]}")
        return

    from baseline_reporag.config import (
        is_symbol_graph_enabled,
        load_config,
        validate_repo_id,
    )
    from baseline_reporag.generation.generator import Generator
    from baseline_reporag.indexing.embedding import EmbeddingIndex
    from baseline_reporag.indexing.lexical import LexicalIndex
    from baseline_reporag.indexing.symbol_graph import SymbolGraph
    from baseline_reporag.ingestion.store import ChunkStore
    from baseline_reporag.logger import RunLogger
    from baseline_reporag.memory.session import SessionManager
    from baseline_reporag.pipeline import RepoRAGPipeline

    cfg = load_config(args.config)
    # CB-005: same allowlist as the CLI / pipeline factory. Fail-fast if
    # the config carries ``../outside`` or an absolute path.
    repo_id = validate_repo_id(cfg.repo.repo_id)
    idx_dir = Path(cfg.paths.data_root) / "indexes" / repo_id
    run_id = f"demo_{scenario.id}_{time.strftime('%Y%m%d')}"

    # Issue #109: skip SymbolGraph.load when the feature is disabled
    # (non-Python repositories).  graph=None is safe: expand_with_graph
    # falls back to file-neighbors only.
    graph: SymbolGraph | None = (
        SymbolGraph.load(idx_dir / "symbol_graph.json")
        if is_symbol_graph_enabled(cfg)
        else None
    )

    pipeline = RepoRAGPipeline(
        config=cfg,
        store=ChunkStore(idx_dir / "chunks.db"),
        lexical=LexicalIndex.load(idx_dir / "lexical.pkl"),
        embedding=EmbeddingIndex.load(idx_dir / "embedding"),
        graph=graph,
        sessions=SessionManager(log_dir=Path(cfg.paths.log_root) / "sessions"),
        generator=Generator(
            model_id=cfg.model.model_id,
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            top_p=cfg.generation.top_p,
        ),
        logger=RunLogger(cfg.paths.log_root, run_id),
    )

    session_id = f"demo-{scenario.id}"

    print(f"\n{'=' * 60}")
    print(f"Demo: [{scenario.id}] {scenario.title}")
    print(f"Axis: {scenario.axis}")
    print(f"Session: {session_id}")
    print(f"{'=' * 60}\n")

    for i, turn in enumerate(scenario.turns, 1):
        print(f"--- Turn {i} ---")
        print(f"Q: {turn.question}")
        if turn.notes:
            print(f"[expect: {turn.notes}]")
        print()

        result = pipeline.query(
            question=turn.question,
            session_id=session_id,
            repo_id=repo_id,
        )

        print(f"A: {result.answer}\n")
        print(f"  Cited: {result.cited_chunk_ids}")
        print(
            f"  Latency: {result.latency.total_ms:.0f} ms"
            f"  (retrieval {result.latency.retrieval_ms:.0f}"
            f" | gen {result.latency.generation_ms:.0f})"
        )
        print(f"  Memory: {result.memory.peak_mb:.1f} MB")
        if result.no_citation:
            print("  [WARNING] No citations")
        print()

    print(f"{'=' * 60}")
    print(f"Demo complete. Log: logs/{run_id}.jsonl")


if __name__ == "__main__":
    main()
