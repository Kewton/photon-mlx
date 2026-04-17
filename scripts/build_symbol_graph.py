"""
build_symbol_graph.py  –  Build the symbol and dependency graph.

Usage:
    python scripts/build_symbol_graph.py --repo-id fastapi_fastapi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_reporag.config import load_config
from baseline_reporag.indexing.symbol_graph import SymbolGraph
from baseline_reporag.ingestion.store import ChunkStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the symbol graph")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_commit = cfg.repo.repo_commit
    idx_dir = Path(cfg.paths.data_root) / "indexes" / args.repo_id

    store = ChunkStore(idx_dir / "chunks.db")
    print(f"Building symbol graph for {args.repo_id}@{repo_commit} ...")

    graph = SymbolGraph()
    graph.build(store, args.repo_id, repo_commit)
    graph.save(idx_dir / "symbol_graph.json")

    store.close()
    print(f"Done -> {idx_dir}/symbol_graph.json")


if __name__ == "__main__":
    main()
