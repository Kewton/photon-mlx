"""Unit tests for baseline_reporag.retrieval.reranker noise filter.

Covers Issue #111: domain-agnostic generalization of `noise_patterns`.

We avoid loading the sentence-transformers CrossEncoder model by constructing
the reranker via ``__new__`` and manually setting ``self._noise_patterns``.
"""

from __future__ import annotations

import pytest

from baseline_reporag.retrieval.reranker import (
    _NOISE_PATTERNS,
    _format_passage_for_rerank,
    CrossEncoderReranker,
)


def _make_reranker(noise_patterns) -> CrossEncoderReranker:
    """Instantiate CrossEncoderReranker without loading the ML model."""
    obj = CrossEncoderReranker.__new__(CrossEncoderReranker)
    effective = _NOISE_PATTERNS if noise_patterns is None else tuple(noise_patterns)
    object.__setattr__(obj, "_noise_patterns", effective)
    return obj


def _chunk_id(path: str) -> str:
    """Construct a chunk_id in the 'repo::path' format used by ChunkStore."""
    return f"fastapi::{path}"


class _Chunk:
    def __init__(
        self,
        *,
        rel_path: str = "",
        section_header: str = "",
        content: str = "",
    ) -> None:
        self.rel_path = rel_path
        self.section_header = section_header
        self.content = content


# --------------------------------------------------------------------------- #
# Case 8: noise_patterns=None → backward compat with _NOISE_PATTERNS
# --------------------------------------------------------------------------- #


def test_case_8_default_noise_patterns_filters_builtin():
    reranker = _make_reranker(None)
    assert reranker._is_noise(_chunk_id("docs/llm-prompt.md")) is True
    assert reranker._is_noise(_chunk_id(".github/sponsors.yml")) is True
    assert reranker._is_noise(_chunk_id("scripts/language_names.yml")) is True
    assert (
        reranker._is_noise(_chunk_id(".github/DISCUSSION_TEMPLATE/general.yml")) is True
    )
    assert reranker._is_noise(_chunk_id("docs/general-llm-prompt.md")) is True
    # Non-noise paths pass through
    assert reranker._is_noise(_chunk_id("fastapi/routing.py")) is False
    assert reranker._is_noise(_chunk_id("fastapi/applications.py")) is False


# --------------------------------------------------------------------------- #
# Case 9: noise_patterns=[] → noise filter fully disabled (pass-through)
# --------------------------------------------------------------------------- #


def test_case_9_empty_noise_patterns_disables_filter():
    reranker = _make_reranker([])
    # Even previously-noise paths now pass through
    assert reranker._is_noise(_chunk_id("docs/llm-prompt.md")) is False
    assert reranker._is_noise(_chunk_id(".github/sponsors.yml")) is False
    assert reranker._is_noise(_chunk_id("fastapi/routing.py")) is False


# --------------------------------------------------------------------------- #
# Case 10: noise_patterns=[...] → filter only by specified patterns
# --------------------------------------------------------------------------- #


def test_case_10_custom_noise_patterns_filter():
    reranker = _make_reranker(["custom_pattern"])
    assert reranker._is_noise(_chunk_id("foo/custom_pattern_impl.py")) is True
    assert reranker._is_noise(_chunk_id("docs/llm-prompt.md")) is False
    assert reranker._is_noise(_chunk_id("fastapi/routing.py")) is False


def test_case_10_full_builtin_pass_through_snapshot():
    """Passing _NOISE_PATTERNS explicitly → matches default (None) behavior."""
    r_explicit = _make_reranker(list(_NOISE_PATTERNS))
    r_default = _make_reranker(None)
    test_ids = [
        _chunk_id("docs/llm-prompt.md"),
        _chunk_id(".github/sponsors.yml"),
        _chunk_id("scripts/language_names.yml"),
        _chunk_id(".github/DISCUSSION_TEMPLATE/general.yml"),
        _chunk_id("docs/general-llm-prompt.md"),
        _chunk_id("fastapi/routing.py"),
        _chunk_id("fastapi/applications.py"),
        _chunk_id("tests/test_routing.py"),
    ]
    for cid in test_ids:
        assert r_explicit._is_noise(cid) == r_default._is_noise(cid), (
            f"Snapshot mismatch on {cid}"
        )


def test_format_passage_for_rerank_includes_metadata_before_content():
    passage = _format_passage_for_rerank(
        _Chunk(
            rel_path="docs/safety-net-guarantee-1.md",
            section_header="必要書類",
            content="認定書、登記簿謄本、決算書類を提出します。",
        )
    )

    assert "Document path: docs/safety-net-guarantee-1.md" in passage
    assert "Section: 必要書類" in passage
    assert passage.index("Document path:") < passage.index("Content:")


def test_format_passage_for_rerank_truncates_content_not_metadata():
    passage = _format_passage_for_rerank(
        _Chunk(
            rel_path="docs/form.md",
            section_header="Overview",
            content="x" * 700,
        ),
        max_chars=12,
    )

    assert "Document path: docs/form.md" in passage
    assert "Section: Overview" in passage
    assert "x" * 12 in passage
    assert "x" * 13 not in passage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
