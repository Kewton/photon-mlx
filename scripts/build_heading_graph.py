"""build_heading_graph.py — Build the markdown heading hierarchy graph.

Usage:
    python -m scripts.build_heading_graph --repo-id institutional_documents
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_reporag.config import (
    is_heading_graph_enabled,
    load_config,
    validate_repo_id,
)
from baseline_reporag.indexing.heading_graph import HeadingGraph
from baseline_reporag.ingestion.store import ChunkStore


def _validate_repo_id(repo_id: str) -> None:
    # CB-004: delegate to the shared helper so CLI / factory / demo all
    # enforce the same allowlist.
    validate_repo_id(repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the heading graph")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--commit", default=None, help="Override repo_commit from config"
    )
    args = parser.parse_args()

    _validate_repo_id(args.repo_id)

    cfg = load_config(args.config)

    # heading_graph.enabled=false → skip (default is OFF, DR2-002).
    # The caller (app/photon_app.py) owns the DONE contract.
    if not is_heading_graph_enabled(cfg):
        print("Skipped: indexing.heading_graph.enabled=false (or not set)")
        return

    repo_commit = args.commit if args.commit else cfg.repo.repo_commit
    idx_dir = Path(cfg.paths.data_root) / "indexes" / args.repo_id

    store = ChunkStore(idx_dir / "chunks.db")
    print(f"Building heading graph for {args.repo_id}@{repo_commit} ...")

    # CB-006: ensure store is closed even if build/save raise (DR2-014).
    try:
        graph = HeadingGraph()
        graph.build(store, args.repo_id, repo_commit)
        graph.save(idx_dir / "heading_graph.json")
    finally:
        store.close()
    print(f"Done -> {idx_dir}/heading_graph.json")


if __name__ == "__main__":
    main()
