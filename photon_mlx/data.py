"""Training data pipeline for PHOTON models."""

from __future__ import annotations

import json
import random
from pathlib import Path

import mlx.core as mx


def load_jsonl(path: str | Path) -> list[list[int]]:
    """Load a JSONL corpus where each line is {"tokens": [int, ...]}."""
    docs: list[list[int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        docs.append(obj["tokens"])
    return docs


def pack_sequences(
    docs: list[list[int]],
    context_length: int,
    pad_id: int = 0,
) -> list[list[int]]:
    """Pack documents into fixed-length sequences (greedy bin-packing)."""
    packed: list[list[int]] = []
    buf: list[int] = []
    for doc in docs:
        for tok in doc:
            buf.append(tok)
            if len(buf) == context_length:
                packed.append(buf)
                buf = []
    # Drop remainder shorter than context_length
    return packed


def create_batches(
    sequences: list[list[int]],
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> list[mx.array]:
    """Return list of (batch_size, context_length) arrays."""
    if shuffle:
        rng = random.Random(seed)
        sequences = list(sequences)
        rng.shuffle(sequences)
    batches: list[mx.array] = []
    for i in range(0, len(sequences) - batch_size + 1, batch_size):
        batch = sequences[i : i + batch_size]
        batches.append(mx.array(batch))
    return batches


def iterate_batches(
    corpus_path: str | Path,
    context_length: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> list[mx.array]:
    """End-to-end: load → pack → batch."""
    docs = load_jsonl(corpus_path)
    seqs = pack_sequences(docs, context_length)
    return create_batches(seqs, batch_size, shuffle=shuffle, seed=seed)
