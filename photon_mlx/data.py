"""Training data pipeline for PHOTON models."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import mlx.core as mx

# Issue #135 / DR4-002: maximum bytes per JSONL line. Caps memory DoS
# from a malicious corpus while comfortably accommodating
# context_length=2048 + small metadata.
_MAX_LINE_BYTES = 2 * 1024 * 1024  # 2 MiB


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


# ---------------------------------------------------------------------------
# Issue #135 / Phase 3: mixed-corpus loader
# ---------------------------------------------------------------------------


def _validate_corpus_paths(corpus_paths: dict[str, float]) -> None:
    """DR1-003 strict validation: empty / non-numeric / out-of-range / sum-off."""
    if not corpus_paths:
        raise ValueError("corpus_paths must not be empty (DR1-003)")
    total = 0.0
    for path, weight in corpus_paths.items():
        # ``bool`` is a subclass of ``int`` — reject explicitly.
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(
                f"corpus weight must be a real number, got {type(weight).__name__} "
                f"for {path!r}"
            )
        if not math.isfinite(float(weight)):
            raise ValueError(
                f"corpus weight must be finite, got {weight!r} for {path!r}"
            )
        if weight <= 0.0:
            raise ValueError(f"corpus weight must be > 0, got {weight} for {path!r}")
        total += float(weight)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"corpus weights must sum to 1.0 (±1e-6), got {total} (DR1-003)"
        )


def _resolve_approved_roots(
    approved_roots: Iterable[str | Path] | None,
) -> list[Path]:
    """Default to production roots when caller does not pass any."""
    if approved_roots is None:
        return [
            Path("data/training").resolve(),
            Path("data/processed").resolve(),
        ]
    return [Path(r).resolve() for r in approved_roots]


def _validate_corpus_path_security(raw_path: str, approved_roots: list[Path]) -> Path:
    """DR4-002: reject missing files, escape via symlink, and non-files."""
    p = Path(raw_path)
    try:
        resolved = p.resolve(strict=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"corpus path does not exist: {raw_path}") from e
    if not resolved.is_file():
        raise ValueError(f"corpus path is not a regular file: {resolved}")
    for root in approved_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"corpus path is outside approved roots {[str(r) for r in approved_roots]}: "
        f"{resolved} (DR4-002)"
    )


def _load_validated_jsonl(
    path: Path,
    *,
    vocab_size: int,
    context_length: int,
) -> list[list[int]]:
    """DR4-002: load JSONL with per-line size cap and integer-token validation."""
    docs: list[list[int]] = []
    max_tokens = context_length * 4
    with path.open("rb") as fh:
        for raw in fh:
            if len(raw) > _MAX_LINE_BYTES:
                raise ValueError(
                    f"JSONL line in {path} exceeds {_MAX_LINE_BYTES} bytes (DR4-002)"
                )
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            if "tokens" not in obj:
                raise ValueError(f"JSONL line missing 'tokens': {path}")
            tokens = obj["tokens"]
            if not isinstance(tokens, list) or not tokens:
                raise ValueError(
                    f"'tokens' must be a non-empty list, got {type(tokens).__name__}"
                )
            if len(tokens) > max_tokens:
                raise ValueError(
                    f"'tokens' length {len(tokens)} exceeds context_length*4 "
                    f"({max_tokens}) in {path}"
                )
            for tok in tokens:
                # JSON has no int/float distinction — reject anything that
                # isn't an exact int (booleans are int subclasses, reject too).
                if isinstance(tok, bool) or not isinstance(tok, int):
                    raise TypeError(
                        f"token must be int, got {type(tok).__name__} in {path}"
                    )
                if tok < 0 or tok >= vocab_size:
                    raise ValueError(
                        f"token {tok} out of range [0, {vocab_size}) in {path}"
                    )
            docs.append(tokens)
    return docs


def iterate_mixed_batches(
    corpus_paths: dict[str, float],
    *,
    context_length: int,
    batch_size: int,
    vocab_size: int,
    seed: int = 42,
    shuffle: bool = True,
    val_split: float = 0.0,
    approved_roots: Iterable[str | Path] | None = None,
) -> list[mx.array] | tuple[list[mx.array], list[mx.array]]:
    """Mixed-corpus end-to-end loader (Issue #135 / DR1-003 / DR1-005 / DR4-002).

    Each corpus is loaded and packed into its OWN sequence pool — packing
    never crosses a corpus boundary (S5-005). The final batch list is built
    by sampling sequences from those pools at the requested ratio
    (sequence-level mixing per DR1-007). The token-level ratio is a
    measured by-product, not a control target.

    Returns
    -------
    list[mx.array] when ``val_split == 0.0``, else
    ``tuple[list[mx.array], list[mx.array]]`` = (train_batches, val_batches).
    val_batches is held out from the SAME pool as train_batches with the
    same train_corpora_mix ratio (DR1-005 simplification: no separate
    val_corpora_mix dict).

    Validation
    ----------
    Strict per DR1-003: empty corpus_paths, non-numeric weight, weight
    <= 0 or non-finite, and sum(weights) outside ±1e-6 of 1.0 all raise
    ValueError. Per DR4-002: paths must resolve under one of
    ``approved_roots`` (default = ``data/training/`` + ``data/processed/``);
    JSONL tokens must be a non-empty list of ints in ``[0, vocab_size)``.
    """
    _validate_corpus_paths(corpus_paths)
    if not isinstance(val_split, (int, float)) or val_split < 0 or val_split >= 1:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")

    roots = _resolve_approved_roots(approved_roots)

    rng = random.Random(seed)

    pools: dict[str, list[list[int]]] = {}
    for raw_path, weight in corpus_paths.items():
        resolved = _validate_corpus_path_security(raw_path, roots)
        docs = _load_validated_jsonl(
            resolved,
            vocab_size=vocab_size,
            context_length=context_length,
        )
        seqs = pack_sequences(docs, context_length)
        if shuffle:
            rng.shuffle(seqs)
        pools[raw_path] = seqs
        # Discard ``weight`` reference here — we keep a parallel weight
        # list below for the sampling loop.

    weights = [corpus_paths[p] for p in pools]
    paths = list(pools.keys())

    # Compute total target size from the available pools so we never
    # over-sample a small corpus. The bottleneck is the smallest scaled
    # pool: floor(min(len(pool_i) / weight_i)).
    if any(len(pools[p]) == 0 for p in paths):
        # At least one pool produced zero packed sequences (corpus too
        # short for context_length). Fall through with empty result.
        return [] if val_split == 0.0 else ([], [])

    max_total = min(int(len(pools[p]) / w) for p, w in zip(paths, weights))
    if max_total <= 0:
        return [] if val_split == 0.0 else ([], [])

    # Build the mixed sequence list by repeatedly drawing one corpus
    # according to ``weights`` and popping its next sequence. Cursor per
    # corpus tracks how many sequences we've consumed from its pool.
    cursors = {p: 0 for p in paths}
    target_per_corpus = {p: int(max_total * w) for p, w in zip(paths, weights)}

    # Round-robin schedule: produce a list whose i-th element is the
    # corpus to sample from. We seed the RNG so the schedule is
    # reproducible.
    schedule_rng = random.Random(seed ^ 0x9E3779B9)
    schedule: list[str] = []
    for path in paths:
        schedule.extend([path] * target_per_corpus[path])
    schedule_rng.shuffle(schedule)

    mixed: list[list[int]] = []
    for path in schedule:
        if cursors[path] < len(pools[path]):
            mixed.append(pools[path][cursors[path]])
            cursors[path] += 1

    if val_split > 0.0:
        # Hold out the LAST ``val_split`` fraction so train/val are drawn
        # from the same shuffled mixture (preserves the corpora ratio in
        # both halves, DR1-005).
        n_val_seqs = max(int(len(mixed) * val_split), batch_size)
        if n_val_seqs >= len(mixed):
            # Not enough material for both splits — return empty val so
            # callers can decide whether to error.
            train_seqs = mixed
            val_seqs: list[list[int]] = []
        else:
            train_seqs = mixed[: len(mixed) - n_val_seqs]
            val_seqs = mixed[len(mixed) - n_val_seqs :]
        train_batches = create_batches(train_seqs, batch_size, shuffle=False, seed=seed)
        val_batches = create_batches(val_seqs, batch_size, shuffle=False, seed=seed)
        return train_batches, val_batches

    return create_batches(mixed, batch_size, shuffle=False, seed=seed)
