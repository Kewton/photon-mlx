"""
ingest_repo.py  –  Ingest a repository into the chunk store.

Usage:
    python scripts/ingest_repo.py \
        --repo /path/to/fastapi \
        --repo-id fastapi_fastapi \
        --commit <SHA>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_reporag.config import load_config
from baseline_reporag.ingestion.chunker import chunk_file
from baseline_reporag.ingestion.extractor import extract_files
from baseline_reporag.ingestion.store import ChunkStore


def resolve_commit(repo_path: str, ref: str) -> str:
    """Resolve ``ref`` to a 40-char commit-id string used in chunks.db.

    For git repositories, returns the actual commit SHA (legacy behaviour).
    For non-git directories (PDF/markdown corpora 等の制度文書 use case)
    falls back to a deterministic id derived from the maximum file mtime
    under ``repo_path``: re-ingesting the same snapshot returns the same
    id, while edits/additions produce a new id. The 40-char length matches
    git SHA conventions so downstream tooling that assumes SHA-shape ids
    keeps working.

    Use a 40-char SHA explicitly (``--commit <SHA>``) to bypass both paths.
    """
    if ref == "HEAD" or len(ref) < 40:
        try:
            result = subprocess.run(
                ["git", "-C", repo_path, "rev-parse", ref],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return _synthesize_non_git_commit(repo_path)
    return ref


def _synthesize_non_git_commit(repo_path: str) -> str:
    """Make a stable 40-char ``manual-<mtime>`` id for non-git directories.

    Falls back to top-level mtime when ``rglob`` finds no files (empty dir).
    Per-file mtimes are walked so editing any file inside produces a fresh
    id even when the parent directory's mtime would not change.
    """
    repo = Path(repo_path)
    if not repo.exists():
        raise FileNotFoundError(f"repo_path not found: {repo_path}")

    max_mtime = repo.stat().st_mtime
    for path in repo.rglob("*"):
        try:
            mtime = path.stat().st_mtime
        except (OSError, FileNotFoundError):
            # Broken symlinks etc — skip silently; they wouldn't affect
            # the chunk content anyway.
            continue
        if mtime > max_mtime:
            max_mtime = mtime

    mtime_ms = int(max_mtime * 1000)
    # ``manual-`` (7 chars) + 33 zero-padded digits = 40 chars total.
    synthesized = f"manual-{mtime_ms:033d}"
    print(
        f"[warn] {repo_path} is not a git repository; "
        f"using synthesized commit id {synthesized!r} "
        f"(based on max file mtime).",
        file=sys.stderr,
    )
    return synthesized


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a repository")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_commit = resolve_commit(args.repo, args.commit)
    print(f"repo_id:     {args.repo_id}")
    print(f"repo_commit: {repo_commit}")

    idx_dir = Path(cfg.paths.data_root) / "indexes" / args.repo_id
    store = ChunkStore(idx_dir / "chunks.db")

    chunking = cfg.ingestion.chunking
    total_files = 0
    total_chunks = 0

    for file_rec in extract_files(
        repo_path=args.repo,
        include=cfg.repo.include,
        exclude=cfg.repo.exclude,
    ):
        chunks = chunk_file(
            content=file_rec.content,
            rel_path=file_rec.rel_path,
            language=file_rec.language,
            repo_id=args.repo_id,
            repo_commit=repo_commit,
            max_chars=chunking.max_chars,
            overlap_chars=chunking.overlap_chars,
        )
        for chunk in chunks:
            store.upsert(chunk)
        total_files += 1
        total_chunks += len(chunks)
        if total_files % 100 == 0:
            print(f"  {total_files} files  {total_chunks} chunks ...", flush=True)

    store.commit()
    store.close()
    print(f"\nDone: {total_files} files, {total_chunks} chunks -> {idx_dir}/chunks.db")


if __name__ == "__main__":
    main()
