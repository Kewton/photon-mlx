"""
generate_training_corpus.py  –  Generate training corpus for PHOTON from ingested repo.

Tokenizes chunk contents and writes JSONL with {"tokens": [...]} per document.
Creates train/val splits.

Usage:
    python scripts/generate_training_corpus.py \
        --repo-id fastapi_fastapi \
        --output-dir data/processed \
        --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from baseline_reporag.config import load_config
from baseline_reporag.ingestion.store import ChunkStore
from torch_ref.config import TokenizerConfig, load_photon_config

HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def validate_tokenizer_id(tokenizer_id: str) -> str:
    """Validate that *tokenizer_id* looks like a HuggingFace repo id.

    Raises ``ValueError`` for local paths, URLs, or traversal patterns.
    """
    if "://" in tokenizer_id or tokenizer_id.startswith(("/", ".", "~")):
        raise ValueError(
            f"tokenizer_id must be a HuggingFace repo id, got: {tokenizer_id!r}"
        )
    if ".." in tokenizer_id or "\\" in tokenizer_id:
        raise ValueError(f"tokenizer_id contains unsafe characters: {tokenizer_id!r}")
    if not HF_REPO_ID_RE.fullmatch(tokenizer_id):
        raise ValueError(
            f"tokenizer_id does not match HuggingFace repo id pattern: {tokenizer_id!r}"
        )
    return tokenizer_id


def resolve_tokenizer_id(
    cli_tokenizer_id: str | None,
    config_tokenizer_id: str | None,
) -> str:
    """Resolve tokenizer id with priority: CLI > config > default."""
    if cli_tokenizer_id:
        return validate_tokenizer_id(cli_tokenizer_id)
    if config_tokenizer_id:
        return validate_tokenizer_id(config_tokenizer_id)
    return validate_tokenizer_id(TokenizerConfig().tokenizer_id)


class Tokenizer(Protocol):
    """Protocol for tokenizer objects (e.g. HuggingFace AutoTokenizer)."""

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...


def tokenize_text(text: str, tokenizer: Tokenizer) -> list[int]:
    """Tokenize text using the given tokenizer without special tokens."""
    return tokenizer.encode(text, add_special_tokens=False)


def simple_tokenize(text: str, vocab_size: int = 32000) -> list[int]:
    """
    Byte-level tokenization fallback.

    For proper training, replace with Llama tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        return tok.encode(text)
    """
    return [b % vocab_size for b in text.encode("utf-8")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training corpus")
    parser.add_argument("--repo-id", default="fastapi_fastapi")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--photon-config",
        default=None,
        help="Path to PHOTON config YAML (for tokenizer_id)",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="HuggingFace tokenizer repo id (overrides config)",
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-chunks", type=int, default=0, help="Limit chunks (0 = all)"
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Override repo_commit from config (use for non-default repos)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_commit = args.commit if args.commit else cfg.repo.repo_commit
    idx_dir = Path(cfg.paths.data_root) / "indexes" / args.repo_id
    store = ChunkStore(idx_dir / "chunks.db")

    # Resolve tokenizer
    config_tokenizer_id: str | None = None
    if args.photon_config:
        photon_cfg = load_photon_config(args.photon_config)
        config_tokenizer_id = photon_cfg.tokenizer.tokenizer_id
    tokenizer_id = resolve_tokenizer_id(args.tokenizer_id, config_tokenizer_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)
    print(f"Using tokenizer: {tokenizer_id}")

    # Collect all chunks
    print(f"Loading chunks for {args.repo_id}@{repo_commit[:7]}...")
    docs: list[dict] = []
    try:
        for chunk in store.iter_repo(args.repo_id, repo_commit):
            tokens = tokenize_text(chunk.content, tokenizer)
            if len(tokens) < 16:
                continue
            docs.append(
                {
                    "tokens": tokens,
                    "chunk_id": chunk.chunk_id,
                    "rel_path": chunk.rel_path,
                }
            )
            if args.max_chunks and len(docs) >= args.max_chunks:
                break
    finally:
        store.close()
    print(f"  {len(docs)} documents (chunks with >= 16 tokens)")

    if not docs:
        raise ValueError(
            "No documents with >= 16 tokens. Check corpus size or adjust --max-chunks."
        )

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    val_size = max(1, int(len(docs) * args.val_ratio))
    val_docs = docs[:val_size]
    train_docs = docs[val_size:]

    # Write
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_docs in [("train_tiny", train_docs), ("val_tiny", val_docs)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for doc in split_docs:
                f.write(json.dumps({"tokens": doc["tokens"]}) + "\n")
        total_tokens = sum(len(d["tokens"]) for d in split_docs)
        print(
            f"  {split_name}: {len(split_docs)} docs, {total_tokens:,} tokens -> {path}"
        )

    # Also create small split for training
    small_train = train_docs[: min(len(train_docs), 5000)]
    small_val = val_docs[: min(len(val_docs), 500)]
    for split_name, split_docs in [
        ("train_small", small_train),
        ("val_small", small_val),
    ]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for doc in split_docs:
                f.write(json.dumps({"tokens": doc["tokens"]}) + "\n")
        total_tokens = sum(len(d["tokens"]) for d in split_docs)
        print(
            f"  {split_name}: {len(split_docs)} docs, {total_tokens:,} tokens -> {path}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
