"""
build_indexes.py  –  Build lexical and embedding indexes.

Usage:
    python scripts/build_indexes.py --repo-id fastapi_fastapi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_reporag.config import load_config
from baseline_reporag.indexing.embedding import EmbeddingIndex
from baseline_reporag.indexing.lexical import LexicalIndex
from baseline_reporag.ingestion.store import ChunkStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lexical and embedding indexes")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_commit = cfg.repo.repo_commit
    idx_dir = Path(cfg.paths.data_root) / "indexes" / args.repo_id

    store = ChunkStore(idx_dir / "chunks.db")
    n = store.count(args.repo_id, repo_commit)
    print(f"Building indexes for {args.repo_id}@{repo_commit}  ({n} chunks)")

    print("  [1/2] BM25 lexical index ...")
    lexical = LexicalIndex()
    lexical.build(store, args.repo_id, repo_commit)
    lexical.save(idx_dir / "lexical.pkl")
    print(f"        saved -> {idx_dir}/lexical.pkl")

    print("  [2/2] Embedding index ...")
    embedding = EmbeddingIndex(
        model_id=cfg.indexing.embedding.model_id,
    )
    embedding.build(
        store,
        args.repo_id,
        repo_commit,
        batch_size=cfg.indexing.embedding.batch_size,
    )
    embedding.save(idx_dir / "embedding")
    print(f"        saved -> {idx_dir}/embedding/")

    store.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
