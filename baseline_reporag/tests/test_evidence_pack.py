from __future__ import annotations

from baseline_reporag.generation.evidence_pack import EvidencePack
from baseline_reporag.ingestion.chunker import Chunk


def _make_chunk(
    chunk_id: str,
    rel_path: str = "app/main.py",
    content: str = "def hello(): pass",
    start_line: int = 1,
    end_line: int = 10,
    section_header: str = "",
) -> Chunk:
    """テスト用の Chunk を作成する。"""
    return Chunk(
        chunk_id=chunk_id,
        repo_id="test_repo",
        repo_commit="abc123",
        rel_path=rel_path,
        language="python",
        start_line=start_line,
        end_line=end_line,
        content=content,
        symbols=["hello"],
        section_header=section_header,
        file_header="# app/main.py",
    )


class TestFormatForPrompt:
    """EvidencePack.format_for_prompt() のフォーマット検証。"""

    def test_format_for_prompt_basic(self) -> None:
        """チャンクが正しくフォーマットされること。"""
        chunk = _make_chunk("c1", content="def main(): pass")
        pack = EvidencePack(chunks=[chunk], chunk_indices={"c1": 1})
        result = pack.format_for_prompt()
        assert "[C:1]" in result
        assert "app/main.py" in result
        assert "def main(): pass" in result

    def test_format_for_prompt_indices(self) -> None:
        """インデックスが 1-based で正確であること。"""
        chunks = [
            _make_chunk("c1", content="first"),
            _make_chunk("c2", content="second"),
            _make_chunk("c3", content="third"),
        ]
        indices = {"c1": 1, "c2": 2, "c3": 3}
        pack = EvidencePack(chunks=chunks, chunk_indices=indices)
        result = pack.format_for_prompt()
        assert "[C:1]" in result
        assert "[C:2]" in result
        assert "[C:3]" in result
        # Verify order: C:1 appears before C:2 before C:3
        assert result.index("[C:1]") < result.index("[C:2]")
        assert result.index("[C:2]") < result.index("[C:3]")

    def test_format_for_prompt_with_section_header(self) -> None:
        """section_header 付きチャンクが正しくフォーマットされること。"""
        chunk = _make_chunk(
            "c1",
            content="class MyRouter: pass",
            section_header="MyRouter",
        )
        pack = EvidencePack(chunks=[chunk], chunk_indices={"c1": 1})
        result = pack.format_for_prompt()
        assert "[MyRouter]" in result

    def test_format_for_prompt_empty_chunks(self) -> None:
        """空チャンクリストでエラーにならないこと。"""
        pack = EvidencePack(chunks=[], chunk_indices={})
        result = pack.format_for_prompt()
        assert result == ""
