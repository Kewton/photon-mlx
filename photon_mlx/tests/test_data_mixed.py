"""Tests for ``photon_mlx.data.iterate_mixed_batches`` (Issue #135 / Phase 3).

Pins the Phase 3 contract from the design policy:

- Strict validation (DR1-003): empty corpus_paths, weight <= 0 or non-finite,
  sum(weights) not within 1e-6 of 1.0 all raise ``ValueError``.
- Sequence-level weighted sampling (DR1-007 / S5-005): each corpus is
  pack_sequenced into its OWN pool (no cross-corpus boundary leak); the
  final batch list samples sequences in the requested ratio.
- Backwards-compat with ``iterate_batches`` (S3-001): ``val_split == 0``
  returns ``list[mx.array]`` so the trainer can swap it in transparently.
- ``val_split > 0`` (DR1-005 / DR2-008): returns
  ``(train_batches, val_batches)`` tuple drawn from the SAME pool, so the
  train/val mix preserves the train_corpora_mix ratio (no separate
  ``val_corpora_mix`` dict — that schema was simplified out per DR1-005).
- Reproducibility (DR1-007): same ``seed`` produces the same sample order.
- Path security (DR4-002): non-existent paths and symlinks pointing
  outside approved roots are rejected.
- JSONL token validation (DR4-002): tokens must be non-empty list[int]
  with 0 <= token < vocab_size; bool / float / negative / oversized
  values raise ValueError.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx


def _write_corpus(path: Path, sequences: list[list[int]]) -> None:
    """Write a JSONL corpus where each line is {"tokens": [...]}.."""
    path.write_text(
        "\n".join(json.dumps({"tokens": s}) for s in sequences) + "\n",
        encoding="utf-8",
    )


def _approve(*paths: Path) -> list[Path]:
    """Helper: build approved_roots list from tmp paths so tests bypass
    the production guard (``data/training/``, ``data/processed/``)."""
    return list(paths)


class TestStrictValidation:
    """DR1-003: all boundary violations raise ValueError, never warn-and-fallback."""

    def test_empty_corpus_paths_raises(self):
        from photon_mlx.data import iterate_mixed_batches

        try:
            iterate_mixed_batches({}, context_length=8, batch_size=2, vocab_size=256)
        except ValueError:
            return
        raise AssertionError("empty corpus_paths must raise ValueError")

    def test_zero_weight_raises(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        _write_corpus(a, [[1, 2, 3, 4, 5, 6, 7, 8]])
        _write_corpus(b, [[9, 10, 11, 12, 13, 14, 15, 16]])

        try:
            iterate_mixed_batches(
                {str(a): 1.0, str(b): 0.0},
                context_length=8,
                batch_size=1,
                vocab_size=256,
                approved_roots=_approve(tmp_path),
            )
        except ValueError as e:
            assert "weight" in str(e).lower()
            return
        raise AssertionError("zero weight must raise ValueError")

    def test_negative_weight_raises(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        _write_corpus(a, [[1, 2, 3, 4]])

        try:
            iterate_mixed_batches(
                {str(a): -0.5},
                context_length=4,
                batch_size=1,
                vocab_size=256,
                approved_roots=_approve(tmp_path),
            )
        except ValueError:
            return
        raise AssertionError("negative weight must raise ValueError")

    def test_non_numeric_weight_raises(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        _write_corpus(a, [[1, 2, 3, 4]])

        try:
            iterate_mixed_batches(
                {str(a): "0.5"},  # type: ignore[dict-item]
                context_length=4,
                batch_size=1,
                vocab_size=256,
                approved_roots=_approve(tmp_path),
            )
        except (ValueError, TypeError):
            return
        raise AssertionError("non-numeric weight must raise")

    def test_sum_weights_off_target_raises(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        _write_corpus(a, [[1, 2, 3, 4]])
        _write_corpus(b, [[5, 6, 7, 8]])

        try:
            iterate_mixed_batches(
                {str(a): 0.4, str(b): 0.5},  # sum = 0.9
                context_length=4,
                batch_size=1,
                vocab_size=256,
                approved_roots=_approve(tmp_path),
            )
        except ValueError as e:
            assert "1.0" in str(e) or "sum" in str(e).lower()
            return
        raise AssertionError("sum(weights) != 1.0 must raise ValueError")

    def test_sum_weights_within_tolerance_passes(self, tmp_path):
        """sum(weights) within 1e-6 of 1.0 is accepted (float tolerance)."""
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        # Each pool needs at least batch_size sequences after packing.
        _write_corpus(a, [[1, 2, 3, 4] * 8])  # 32 tokens → 4 sequences of 8
        _write_corpus(b, [[5, 6, 7, 8] * 8])

        # 0.5 + 0.4999999999 = 0.9999999999 — within 1e-6 of 1.0
        result = iterate_mixed_batches(
            {str(a): 0.5, str(b): 0.4999999999},
            context_length=8,
            batch_size=2,
            vocab_size=256,
            approved_roots=_approve(tmp_path),
        )
        assert isinstance(result, list)
        assert len(result) > 0


class TestReturnType:
    """S3-001 / DR1-006 / DR2-008: list when val_split=0, tuple otherwise."""

    def test_no_val_split_returns_list(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        _write_corpus(a, [[1, 2, 3, 4] * 8])
        _write_corpus(b, [[5, 6, 7, 8] * 8])

        result = iterate_mixed_batches(
            {str(a): 0.5, str(b): 0.5},
            context_length=8,
            batch_size=2,
            vocab_size=256,
            approved_roots=_approve(tmp_path),
        )
        assert isinstance(result, list)
        assert all(isinstance(b, mx.array) for b in result)
        assert all(b.shape == (2, 8) for b in result)

    def test_val_split_returns_tuple(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        # Need enough sequences so val_split=0.2 yields >= 1 val batch.
        # Use many short docs (each <= context_length*4) instead of one
        # long doc, so the per-line byte/length cap is respected.
        _write_corpus(a, [[1, 2, 3, 4] * 8 for _ in range(8)])  # 8 docs × 32 tokens
        _write_corpus(b, [[5, 6, 7, 8] * 8 for _ in range(8)])

        result = iterate_mixed_batches(
            {str(a): 0.5, str(b): 0.5},
            context_length=8,
            batch_size=2,
            seed=42,
            val_split=0.2,
            vocab_size=256,
            approved_roots=_approve(tmp_path),
        )
        assert isinstance(result, tuple)
        train_batches, val_batches = result
        assert isinstance(train_batches, list)
        assert isinstance(val_batches, list)
        assert len(train_batches) > 0
        assert len(val_batches) > 0


class TestSequenceMixing:
    """DR1-007 / S5-005: sequence-level mixing within target ratio."""

    def test_pool_independence_across_corpora(self, tmp_path):
        """pack_sequences must NOT bridge documents across corpora."""
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        # Each corpus has ONE doc with exactly 4 tokens; context_length=8
        # would force a cross-corpus pack if pools weren't independent.
        # Independent pools => each corpus produces 0 packed sequences,
        # leading to 0 batches (no cross-corpus accidental sequence).
        _write_corpus(a, [[1, 2, 3, 4]])
        _write_corpus(b, [[5, 6, 7, 8]])

        result = iterate_mixed_batches(
            {str(a): 0.5, str(b): 0.5},
            context_length=8,
            batch_size=1,
            vocab_size=256,
            approved_roots=_approve(tmp_path),
        )
        # No corpus produced a complete sequence, so no batches.
        assert result == []

    def test_sequence_ratio_approximates_target(self, tmp_path):
        """Sequence-level sampling preserves the requested ratio.

        With 100 train sequences pulled from a 30/70 mix and a fixed seed,
        the JP fraction should land within ±5% of 0.3 (small-sample bound).
        """
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"  # represents 30% pool
        b = tmp_path / "b.jsonl"  # represents 70% pool

        # 200 sequences each so the sampler has plenty of headroom.
        _write_corpus(a, [[1] * 8 for _ in range(200)])
        _write_corpus(b, [[2] * 8 for _ in range(200)])

        result = iterate_mixed_batches(
            {str(a): 0.3, str(b): 0.7},
            context_length=8,
            batch_size=1,
            seed=12345,
            vocab_size=256,
            approved_roots=_approve(tmp_path),
        )

        # Identify which corpus each batch came from by the unique fill token.
        a_count = sum(1 for batch in result if int(batch[0, 0].item()) == 1)
        b_count = sum(1 for batch in result if int(batch[0, 0].item()) == 2)
        total = a_count + b_count
        assert total > 0
        a_ratio = a_count / total
        assert 0.25 <= a_ratio <= 0.35, (
            f"sequence ratio drifted: a={a_ratio:.3f} (target 0.30 ±0.05)"
        )

    def test_seed_reproducibility(self, tmp_path):
        """Same seed → same batch order."""
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        _write_corpus(a, [[i] * 8 for i in range(50)])
        _write_corpus(b, [[100 + i] * 8 for i in range(50)])

        kwargs = dict(
            corpus_paths={str(a): 0.5, str(b): 0.5},
            context_length=8,
            batch_size=2,
            seed=999,
            vocab_size=256,
            approved_roots=_approve(tmp_path),
        )
        r1 = iterate_mixed_batches(**kwargs)
        r2 = iterate_mixed_batches(**kwargs)

        assert len(r1) == len(r2)
        for a_batch, b_batch in zip(r1, r2):
            diff = mx.max(mx.abs(a_batch - b_batch)).item()
            assert diff == 0.0, "same seed must produce identical sample order"


class TestPathSecurity:
    """DR4-002: file path validation rejects symlink escape and missing files."""

    def test_missing_file_raises(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        bogus = tmp_path / "nope.jsonl"

        try:
            iterate_mixed_batches(
                {str(bogus): 1.0},
                context_length=8,
                batch_size=1,
                vocab_size=256,
                approved_roots=_approve(tmp_path),
            )
        except (FileNotFoundError, ValueError):
            return
        raise AssertionError("missing file must raise FileNotFoundError or ValueError")

    def test_path_outside_approved_roots_raises(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        outside = tmp_path / "outside.jsonl"
        _write_corpus(outside, [[1, 2, 3, 4] * 8])

        # Approved root is a sibling dir — outside.jsonl is NOT under it.
        sibling = tmp_path / "approved"
        sibling.mkdir()

        try:
            iterate_mixed_batches(
                {str(outside): 1.0},
                context_length=8,
                batch_size=1,
                vocab_size=256,
                approved_roots=_approve(sibling),
            )
        except ValueError as e:
            assert "approved" in str(e).lower() or "root" in str(e).lower()
            return
        raise AssertionError("path outside approved roots must raise ValueError")


class TestTokenValidation:
    """DR4-002: JSONL tokens must be valid integer ids."""

    def test_oversized_token_raises(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        _write_corpus(a, [[1, 2, 999_999, 4]])

        try:
            iterate_mixed_batches(
                {str(a): 1.0},
                context_length=4,
                batch_size=1,
                vocab_size=256,
                approved_roots=_approve(tmp_path),
            )
        except ValueError as e:
            assert "vocab" in str(e).lower() or "token" in str(e).lower()
            return
        raise AssertionError("token >= vocab_size must raise ValueError")

    def test_negative_token_raises(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        _write_corpus(a, [[1, -2, 3, 4]])

        try:
            iterate_mixed_batches(
                {str(a): 1.0},
                context_length=4,
                batch_size=1,
                vocab_size=256,
                approved_roots=_approve(tmp_path),
            )
        except ValueError:
            return
        raise AssertionError("negative token must raise ValueError")

    def test_non_int_token_raises(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        # JSON has no "int" type — write a float and ensure it's rejected.
        a.write_text(
            json.dumps({"tokens": [1, 2.5, 3, 4]}) + "\n",
            encoding="utf-8",
        )

        try:
            iterate_mixed_batches(
                {str(a): 1.0},
                context_length=4,
                batch_size=1,
                vocab_size=256,
                approved_roots=_approve(tmp_path),
            )
        except (TypeError, ValueError):
            return
        raise AssertionError("non-int token must raise")

    def test_empty_tokens_raises(self, tmp_path):
        from photon_mlx.data import iterate_mixed_batches

        a = tmp_path / "a.jsonl"
        a.write_text(json.dumps({"tokens": []}) + "\n", encoding="utf-8")

        try:
            iterate_mixed_batches(
                {str(a): 1.0},
                context_length=4,
                batch_size=1,
                vocab_size=256,
                approved_roots=_approve(tmp_path),
            )
        except ValueError:
            return
        raise AssertionError("empty tokens list must raise ValueError")
