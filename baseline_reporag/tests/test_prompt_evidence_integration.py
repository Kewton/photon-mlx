"""Integration tests: prompt + evidence_pack interplay."""

from __future__ import annotations

from baseline_reporag.generation.evidence_pack import EvidencePack, _EVIDENCE_HEADER
from baseline_reporag.generation.prompt import build_messages
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


def test_build_messages_preserves_headered_evidence_text() -> None:
    """Evidence text with header must pass through to the user message."""
    pack = _make_pack(2)
    evidence_text = pack.format_for_prompt()
    msgs = build_messages("What?", evidence_text)
    user_content = msgs[1]["content"]
    assert _EVIDENCE_HEADER in user_content
    assert "[C:1]" in user_content
    assert "[C:2]" in user_content


def test_placeholder_echoback_and_valid_citation_can_coexist() -> None:
    """An LLM answer echoing the instruction '[C:N]' alongside a real
    citation [C:1] must still credit the real citation."""
    from baseline_reporag.citation import resolve_citations

    pack = _make_pack(2)
    answer = "As per [C:N] notation, see [C:1] for the entry point."
    result = resolve_citations(answer, pack)
    assert result.cited_chunk_ids == ["c1"]
    assert result.no_citation is False
