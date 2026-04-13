"""Tests for baseline_reporag.generation.evidence_pack — EvidencePack.format_for_prompt."""

from __future__ import annotations

import re

from baseline_reporag.generation.evidence_pack import EvidencePack, _EVIDENCE_HEADER
from baseline_reporag.ingestion.store import Chunk


def _make_chunk(
    chunk_id: str,
    content: str = "x = 1",
    section_header: str = "",
) -> Chunk:
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
        section_header=section_header,
        file_header="",
    )


def _make_pack(chunks: list[Chunk]) -> EvidencePack:
    indices = {c.chunk_id: i + 1 for i, c in enumerate(chunks)}
    return EvidencePack(chunks=chunks, chunk_indices=indices)


# ------------------------------------------------------------------


def test_format_for_prompt_contains_header() -> None:
    """Output must start with _EVIDENCE_HEADER."""
    pack = _make_pack([_make_chunk("c1")])
    text = pack.format_for_prompt()
    assert text.startswith(_EVIDENCE_HEADER)


def test_format_for_prompt_header_no_digit_citation() -> None:
    """The header itself must not contain [C:<digit>] patterns."""
    assert not re.search(r"\[C:\d+\]", _EVIDENCE_HEADER)


def test_format_for_prompt_chunk_indices() -> None:
    pack = _make_pack([_make_chunk("c1"), _make_chunk("c2")])
    text = pack.format_for_prompt()
    assert "[C:1]" in text
    assert "[C:2]" in text


def test_format_for_prompt_separator() -> None:
    pack = _make_pack([_make_chunk("c1"), _make_chunk("c2")])
    text = pack.format_for_prompt()
    assert "---" in text


def test_format_for_prompt_empty_chunks() -> None:
    pack = _make_pack([])
    text = pack.format_for_prompt()
    assert text == ""


def test_format_for_prompt_section_header() -> None:
    chunk = _make_chunk("c1", section_header="MyClass.method")
    pack = _make_pack([chunk])
    text = pack.format_for_prompt()
    assert "MyClass.method" in text
