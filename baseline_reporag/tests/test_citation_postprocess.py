"""Tests for citation post-processing (Issue #14)."""

from __future__ import annotations

# Guard for MLX absence (same pattern as test_pipeline_integration.py)
import importlib.util
import sys
from types import ModuleType

if importlib.util.find_spec("mlx") is None:
    for _mod in ("mlx", "mlx.core", "mlx_lm", "mlx_lm.sample_utils"):
        if _mod not in sys.modules:
            _stub = ModuleType(_mod)
            _stub.make_sampler = lambda **kw: None  # type: ignore[attr-defined]
            sys.modules[_mod] = _stub

import pytest  # noqa: E402

from baseline_reporag.citation import CitationResult  # noqa: E402
from baseline_reporag.generation.evidence_pack import EvidencePack  # noqa: E402
from baseline_reporag.generation.prompt import ABSTAIN_MARKER  # noqa: E402
from baseline_reporag.ingestion.chunker import Chunk  # noqa: E402
from baseline_reporag.pipeline import apply_citation_postprocess  # noqa: E402


def _make_chunk(chunk_id: str = "c1", content: str = "sample code") -> Chunk:
    """Helper to create a minimal Chunk."""
    return Chunk(
        chunk_id=chunk_id,
        repo_id="test_repo",
        repo_commit="abc",
        rel_path="test.py",
        language="python",
        start_line=1,
        end_line=5,
        content=content,
        symbols=[],
        section_header="",
        file_header="# test.py",
    )


def _make_pack(chunks: list[Chunk] | None = None) -> EvidencePack:
    """Helper: EvidencePack with chunk_indices[chunks[0]] = 1."""
    if chunks is None:
        chunks = [_make_chunk()]
    chunk_indices = {c.chunk_id: i + 1 for i, c in enumerate(chunks)}
    return EvidencePack(chunks=chunks, chunk_indices=chunk_indices)


def _no_citation() -> CitationResult:
    return CitationResult(
        cited_chunk_ids=[],
        cited_chunks=[],
        wrong_citation_indices=[],
        no_citation=True,
    )


def _with_citation() -> CitationResult:
    return CitationResult(
        cited_chunk_ids=["c1"],
        cited_chunks=[],
        wrong_citation_indices=[],
        no_citation=False,
    )


# T-001: no_citation=True かつ pack 非空で [C:1] が付与される
class TestPostprocessAddsCitation:
    def test_adds_c1_when_no_citation(self):
        answer = "This is the answer."
        pack = _make_pack()
        citation = _no_citation()
        new_answer, new_citation, postprocessed = apply_citation_postprocess(
            answer, pack, citation
        )
        assert postprocessed is True
        assert "[C:1]" in new_answer
        assert new_citation.no_citation is False


# T-002: pack.chunks=[] の場合はスキップ
class TestPostprocessSkipsEmptyPack:
    def test_skips_when_pack_empty(self):
        answer = "No citations."
        pack = EvidencePack(chunks=[], chunk_indices={})
        citation = _no_citation()
        new_answer, new_citation, postprocessed = apply_citation_postprocess(
            answer, pack, citation
        )
        assert postprocessed is False
        assert new_answer == answer


# T-003: ABSTAIN_MARKER を含む回答はスキップ
class TestPostprocessSkipsAbstainMarker:
    def test_skips_abstain_marker_answer(self):
        answer = f"{ABSTAIN_MARKER}。詳細は不明です。"
        pack = _make_pack()
        citation = _no_citation()
        new_answer, new_citation, postprocessed = apply_citation_postprocess(
            answer, pack, citation
        )
        assert postprocessed is False
        assert new_answer == answer


# T-004: no_citation=False はスキップ
class TestPostprocessSkipsWhenAlreadyCited:
    def test_skips_when_already_cited(self):
        answer = "The code is in [C:1]."
        pack = _make_pack()
        citation = _with_citation()
        new_answer, new_citation, postprocessed = apply_citation_postprocess(
            answer, pack, citation
        )
        assert postprocessed is False


# T-005: enabled=False で無効化（純粋関数テスト）
class TestPostprocessFlagDisabled:
    def test_disabled_when_enabled_false(self):
        answer = "No citation here."
        pack = _make_pack()
        citation = _no_citation()
        new_answer, new_citation, postprocessed = apply_citation_postprocess(
            answer, pack, citation, enabled=False
        )
        assert postprocessed is False
        assert new_answer == answer
        assert new_citation.no_citation is True


# T-006: citation_postprocessed フラグが True に設定
class TestPostprocessedFlag:
    def test_returns_postprocessed_true(self):
        answer = "The answer."
        pack = _make_pack()
        citation = _no_citation()
        _, _, postprocessed = apply_citation_postprocess(answer, pack, citation)
        assert postprocessed is True


# T-008: 冪等性（既に [C:1] がある場合）
class TestPostprocessIdempotent:
    def test_idempotent_when_c1_exists(self):
        answer = "See [C:1] for details."
        pack = _make_pack()
        # no_citation=False なので適用されない
        citation = _with_citation()
        new_answer, _, postprocessed = apply_citation_postprocess(
            answer, pack, citation
        )
        assert postprocessed is False
        assert new_answer.count("[C:1]") == 1


# T-009: pack.chunks が1件のみ
class TestPostprocessSingleChunk:
    def test_single_chunk_pack(self):
        answer = "Single chunk answer."
        pack = _make_pack([_make_chunk("only_chunk")])
        citation = _no_citation()
        new_answer, new_citation, postprocessed = apply_citation_postprocess(
            answer, pack, citation
        )
        assert postprocessed is True
        assert "[C:1]" in new_answer


# T-010: ABSTAIN_MARKER 定数が prompt.py rule 4 と一致
class TestAbstainMarkerConstant:
    def test_abstain_marker_matches_rule4(self):
        from baseline_reporag.generation.prompt import ABSTAIN_MARKER, _SYSTEM

        assert ABSTAIN_MARKER in _SYSTEM


# CR-001: ABSTAIN_MARKER を引用した場合はスキップされないこと
class TestAbstainMarkerQuoted:
    def test_quoted_abstain_marker_is_not_skipped(self):
        """回答内にマーカーを引用しても先頭でなければスキップしない。"""
        # no_citation=True の状態（[C:N] なし）で ABSTAIN_MARKER が先頭でない場合
        answer_no_cite = (
            f"このシステムは「{ABSTAIN_MARKER}」という応答を返す場合がある。"
        )
        pack = _make_pack()
        citation = _no_citation()
        new_answer, new_citation, postprocessed = apply_citation_postprocess(
            answer_no_cite, pack, citation
        )
        assert postprocessed is True  # 先頭でないのでスキップされない
        assert "[C:1]" in new_answer
        assert new_citation.no_citation is False


# CR-002: 空回答テスト
class TestEmptyAnswer:
    def test_empty_answer_skipped(self):
        pack = _make_pack()
        citation = _no_citation()
        new_answer, _, postprocessed = apply_citation_postprocess("", pack, citation)
        assert postprocessed is False
        assert new_answer == ""

    def test_whitespace_only_answer_skipped(self):
        pack = _make_pack()
        citation = _no_citation()
        new_answer, _, postprocessed = apply_citation_postprocess(
            "   \n", pack, citation
        )
        assert postprocessed is False
        assert new_answer == "   \n"


# CR-003: enabled 型検証テスト
class TestEnabledTypeCheck:
    def test_enabled_string_raises_type_error(self):
        pack = _make_pack()
        citation = _no_citation()
        with pytest.raises(TypeError):
            apply_citation_postprocess("answer", pack, citation, enabled="false")

    def test_enabled_int_raises_type_error(self):
        pack = _make_pack()
        citation = _no_citation()
        with pytest.raises(TypeError):
            apply_citation_postprocess("answer", pack, citation, enabled=1)
