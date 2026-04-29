from __future__ import annotations

from baseline_reporag.citation import is_refusal_answer, resolve_citations
from baseline_reporag.generation.evidence_pack import EvidencePack
from baseline_reporag.ingestion.chunker import Chunk


def _make_chunk(chunk_id: str, rel_path: str = "test.py") -> Chunk:
    """テスト用の最小限の Chunk を作成する。"""
    return Chunk(
        chunk_id=chunk_id,
        repo_id="test_repo",
        repo_commit="abc123",
        rel_path=rel_path,
        language="python",
        start_line=1,
        end_line=10,
        content="def hello(): pass",
        symbols=["hello"],
        section_header="",
        file_header="# test.py",
    )


def _make_pack(n: int = 3) -> EvidencePack:
    """n 個のチャンクを持つ EvidencePack を作成する。"""
    chunks = [_make_chunk(f"chunk_{i}") for i in range(n)]
    chunk_indices = {c.chunk_id: i + 1 for i, c in enumerate(chunks)}
    return EvidencePack(chunks=chunks, chunk_indices=chunk_indices)


class TestResolveCitations:
    """resolve_citations() の基本動作を検証する。"""

    def test_resolve_citations_with_valid_citations(self) -> None:
        """正常な [C:N] が検出されること。"""
        pack = _make_pack(3)
        answer = "The function is in [C:1] and used by [C:3]."
        result = resolve_citations(answer, pack)
        assert set(result.cited_chunk_ids) == {"chunk_0", "chunk_2"}
        assert result.no_citation is False
        assert result.wrong_citation_indices == []

    def test_resolve_citations_no_citation(self) -> None:
        """citation なしの回答で no_citation=True。"""
        pack = _make_pack(3)
        answer = "The function exists in the codebase."
        result = resolve_citations(answer, pack)
        assert result.cited_chunk_ids == []
        assert result.no_citation is True

    def test_resolve_citations_wrong_citation(self) -> None:
        """evidence pack に存在しないインデックスが wrong_citation_indices に入ること。"""
        pack = _make_pack(2)  # indices 1, 2 only
        answer = "See [C:1] and [C:5]."
        result = resolve_citations(answer, pack)
        assert "chunk_0" in result.cited_chunk_ids  # [C:1] is valid
        assert 5 in result.wrong_citation_indices  # [C:5] is invalid
        assert result.no_citation is False


class TestEchoBackResistance:
    """_EVIDENCE_HEADER の echo-back 耐性を検証する。"""

    def test_echoback_cn_not_matched(self) -> None:
        """[C:N]（文字 N）が citation として検出されないこと。"""
        pack = _make_pack(3)
        answer = "You MUST cite using [C:N] notation. The router is in [C:1]."
        result = resolve_citations(answer, pack)
        assert result.cited_chunk_ids == ["chunk_0"]  # [C:1] のみ
        assert result.no_citation is False

    def test_echoback_header_only(self) -> None:
        """ヘッダーの echo-back のみで実 citation がない場合 no_citation=True。"""
        pack = _make_pack(3)
        answer = "IMPORTANT: You MUST cite every factual claim using [C:N] notation."
        result = resolve_citations(answer, pack)
        assert result.no_citation is True

    def test_echoback_mixed(self) -> None:
        """[C:N] + 実 citation [C:1] 混在時に [C:1] のみ検出。"""
        pack = _make_pack(3)
        answer = "Use [C:N] notation. The implementation is in [C:1] and [C:2]."
        result = resolve_citations(answer, pack)
        assert set(result.cited_chunk_ids) == {"chunk_0", "chunk_1"}
        assert result.no_citation is False


# Issue #154 Bug 2: refusal detection -------------------------------------


class TestIsRefusalAnswer:
    """is_refusal_answer は形式的な [C:N] と無関係に「根拠なし」回答を検出する。"""

    def test_detects_abstain_marker(self) -> None:
        assert is_refusal_answer("根拠が不足しています。詳細不明。") is True

    def test_detects_marker_followed_by_citation(self) -> None:
        # The bug: baseline writes refusal + [C:1] and used to be counted as cited
        assert (
            is_refusal_answer(
                "根拠が不足しています。提供されたドキュメントには情報がありません [C:1]"
            )
            is True
        )

    def test_detects_short_refusal(self) -> None:
        assert is_refusal_answer("根拠不足。") is True

    def test_detects_jouhou_ga_arimasen(self) -> None:
        assert is_refusal_answer("該当する情報がありません。") is True

    def test_normal_answer_is_not_refusal(self) -> None:
        assert (
            is_refusal_answer("The router is registered in fastapi/cli.py [C:1]")
            is False
        )

    def test_empty_string_is_not_refusal(self) -> None:
        assert is_refusal_answer("") is False


class TestResolveCitationsRefusalFlag:
    """resolve_citations は CitationResult.is_refusal を埋める。"""

    def test_refusal_with_citation_marks_is_refusal(self) -> None:
        pack = _make_pack(3)
        answer = "根拠が不足しています。提供されたドキュメントからは特定不能 [C:1]"
        result = resolve_citations(answer, pack)
        assert result.is_refusal is True
        # 既存挙動の互換性: 形式的な [C:1] は no_citation=False のまま
        assert result.no_citation is False

    def test_refusal_without_citation(self) -> None:
        pack = _make_pack(3)
        answer = "根拠が不足しています。"
        result = resolve_citations(answer, pack)
        assert result.is_refusal is True
        assert result.no_citation is True

    def test_normal_answer_not_refusal(self) -> None:
        pack = _make_pack(3)
        answer = "The function is in [C:1]."
        result = resolve_citations(answer, pack)
        assert result.is_refusal is False
