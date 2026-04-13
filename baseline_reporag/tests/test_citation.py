from __future__ import annotations

from baseline_reporag.citation import resolve_citations
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
