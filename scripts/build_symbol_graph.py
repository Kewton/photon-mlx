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

from baseline_reporag.config import (
    is_symbol_graph_enabled,
    load_config,
    validate_repo_id,
)
from baseline_reporag.indexing.symbol_graph import SymbolGraph
from baseline_reporag.ingestion.store import ChunkStore


def _validate_repo_id(repo_id: str) -> None:
    # CB-004: delegate to the shared helper so CLI / factory / demo all
    # enforce the same allowlist. The wrapper exists so existing test
    # patches on ``_validate_repo_id`` (and the ``main`` call site) keep
    # the same import-level shape.
    validate_repo_id(repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the symbol graph")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--commit", default=None, help="Override repo_commit from config"
    )
    args = parser.parse_args()

    _validate_repo_id(args.repo_id)

    cfg = load_config(args.config)

    # Issue #109: ``indexing.symbol_graph.enabled=false`` skips build/save
    # entirely (intended for non-Python repositories). The wrapper
    # (app/photon_app.py) owns the ``DONE`` contract — this script only
    # prints the skip reason.
    if not is_symbol_graph_enabled(cfg):
        print("Skipped: indexing.symbol_graph.enabled=false")
        return

    repo_commit = args.commit if args.commit else cfg.repo.repo_commit
    idx_dir = Path(cfg.paths.data_root) / "indexes" / args.repo_id

    store = ChunkStore(idx_dir / "chunks.db")
    print(f"Building symbol graph for {args.repo_id}@{repo_commit} ...")

    # CB-006: ensure the sqlite handle is closed even if build/save raise.
    # Previously a failure in the middle left the handle open; subsequent
    # retries could hit WAL/lock contention or descriptor leaks.
    try:
        graph = SymbolGraph()
        graph.build(store, args.repo_id, repo_commit)
        graph.save(idx_dir / "symbol_graph.json")
    finally:
        store.close()
    print(f"Done -> {idx_dir}/symbol_graph.json")


if __name__ == "__main__":
    main()
