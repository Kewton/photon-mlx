"""Tests for baseline_reporag.citation — resolve_citations."""

from __future__ import annotations

from baseline_reporag.citation import resolve_citations
from baseline_reporag.generation.evidence_pack import EvidencePack
from baseline_reporag.ingestion.store import Chunk


def _make_chunk(chunk_id: str, content: str = "x = 1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        repo_id="repo",
        repo_commit="abc",
        rel_path="src/main.py",
        language="python",
        start_line=1,
        end_line=5,
        content=content,
        symbols=[],
        section_header="",
        file_header="",
    )


def _make_pack(n: int = 2) -> EvidencePack:
    chunks = [_make_chunk(f"c{i}") for i in range(1, n + 1)]
    indices = {c.chunk_id: i + 1 for i, c in enumerate(chunks)}
    return EvidencePack(chunks=chunks, chunk_indices=indices)


# ------------------------------------------------------------------


def test_resolve_single_citation() -> None:
    pack = _make_pack(2)
    result = resolve_citations("See [C:1] for details.", pack)
    assert result.cited_chunk_ids == ["c1"]
    assert result.no_citation is False


def test_resolve_multiple_citations() -> None:
    pack = _make_pack(3)
    result = resolve_citations("Found in [C:1] and [C:2].", pack)
    assert set(result.cited_chunk_ids) == {"c1", "c2"}
    assert result.no_citation is False


def test_no_citation_detected() -> None:
    pack = _make_pack(2)
    result = resolve_citations("No evidence here.", pack)
    assert result.no_citation is True
    assert result.cited_chunk_ids == []


def test_wrong_citation_detected() -> None:
    pack = _make_pack(2)
    result = resolve_citations("See [C:99] for info.", pack)
    assert result.wrong_citation_indices == [99]
    assert result.cited_chunk_ids == []


def test_duplicate_citations_deduplicated() -> None:
    pack = _make_pack(2)
    result = resolve_citations("[C:1] and again [C:1].", pack)
    assert result.cited_chunk_ids == ["c1"]
    assert len(result.cited_chunk_ids) == 1


def test_header_echoback_not_detected() -> None:
    """Literal '[C:N]' (letter N, not a digit) must not be detected."""
    pack = _make_pack(2)
    result = resolve_citations("Use [C:N] notation to cite.", pack)
    assert result.no_citation is True
