"""Tests for hybrid retrieval helpers: _extract_extension and apply_file_type_boost."""

from __future__ import annotations

import pytest

from baseline_reporag.retrieval.hybrid import (
    RetrievalResult,
    _extract_extension,
    apply_file_type_boost,
)


# ---------------------------------------------------------------------------
# _extract_extension
# ---------------------------------------------------------------------------


class TestExtractExtension:
    def test_python_file(self) -> None:
        assert _extract_extension("repo::src/app.py::0-120") == ".py"

    def test_markdown_file(self) -> None:
        assert _extract_extension("repo::docs/README.md::0-50") == ".md"

    def test_yaml_file(self) -> None:
        assert _extract_extension("repo::configs/base.yaml::0-30") == ".yaml"

    def test_nested_path(self) -> None:
        assert _extract_extension("repo::a/b/c/d.ts::10-20") == ".ts"

    def test_no_extension(self) -> None:
        assert _extract_extension("repo::Makefile::0-10") == ""

    def test_dotfile(self) -> None:
        # e.g. .gitignore -> ".gitignore"
        assert _extract_extension("repo::.gitignore::0-5") == ".gitignore"

    def test_missing_parts(self) -> None:
        assert _extract_extension("single_part") == ""

    def test_empty_string(self) -> None:
        assert _extract_extension("") == ""

    def test_two_parts_no_span(self) -> None:
        assert _extract_extension("repo::main.go") == ".go"


# ---------------------------------------------------------------------------
# apply_file_type_boost
# ---------------------------------------------------------------------------


def _make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        lexical_score=0.0,
        embedding_score=0.0,
    )


class TestApplyFileTypeBoost:
    def test_zero_boost_returns_same_list(self) -> None:
        results = [
            _make_result("r::a.md::0-10", 1.0),
            _make_result("r::b.py::0-10", 0.5),
        ]
        out = apply_file_type_boost(results, boost=0.0)
        assert out is results  # identity — no copy

    def test_boost_lifts_py_above_md(self) -> None:
        results = [
            _make_result("r::docs/guide.md::0-10", 0.80),
            _make_result("r::src/main.py::0-10", 0.70),
        ]
        out = apply_file_type_boost(results, boost=0.15)
        # .py score becomes 0.85, .md stays 0.80 -> .py should be first
        assert out[0].chunk_id == "r::src/main.py::0-10"
        assert out[0].score == pytest.approx(0.85)
        assert out[1].chunk_id == "r::docs/guide.md::0-10"
        assert out[1].score == pytest.approx(0.80)

    def test_non_py_files_unchanged(self) -> None:
        results = [
            _make_result("r::a.yaml::0-10", 0.9),
            _make_result("r::b.md::0-10", 0.8),
        ]
        out = apply_file_type_boost(results, boost=0.15)
        assert out[0].score == pytest.approx(0.9)
        assert out[1].score == pytest.approx(0.8)

    def test_preserves_original_fields(self) -> None:
        r = RetrievalResult(
            chunk_id="r::x.py::0-5",
            score=1.0,
            lexical_score=0.6,
            embedding_score=0.4,
        )
        out = apply_file_type_boost([r], boost=0.1)
        assert out[0].lexical_score == pytest.approx(0.6)
        assert out[0].embedding_score == pytest.approx(0.4)

    def test_empty_list(self) -> None:
        assert apply_file_type_boost([], boost=0.15) == []

    def test_sorting_after_boost(self) -> None:
        results = [
            _make_result("r::a.md::0-10", 1.0),
            _make_result("r::b.py::0-10", 0.90),
            _make_result("r::c.md::0-10", 0.80),
            _make_result("r::d.py::0-10", 0.60),
        ]
        out = apply_file_type_boost(results, boost=0.15)
        scores = [r.score for r in out]
        assert scores == sorted(scores, reverse=True)
        # b.py (0.90+0.15=1.05) > a.md (1.0) > c.md (0.80) > d.py (0.60+0.15=0.75)
        assert out[0].chunk_id == "r::b.py::0-10"
        assert out[1].chunk_id == "r::a.md::0-10"
