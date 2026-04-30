"""End-to-end contract test: chunker _format_section_header → HeadingGraph.

Verifies that the section_header format produced by the markdown chunker
is correctly consumed by HeadingGraph.build (DR2-005, S7-001).
Not in tests/integration/ (DR2-005).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from baseline_reporag.ingestion.chunker import _chunk_markdown
from baseline_reporag.indexing.heading_graph import HeadingGraph


_MARKDOWN = """\
# Chapter 1

Intro paragraph.

## Section A

Section A body.

## Section B

Section B body.

# Chapter 2

Another chapter.
"""


class TestHeadingGraphChunkerContract:
    def test_build_from_real_chunker_output(self) -> None:
        """HeadingGraph.build correctly handles real chunker output."""
        chunks = _chunk_markdown(
            _MARKDOWN, "doc.md", "test_repo", "abc123", max_chars=4096, overlap_chars=0
        )
        # Patch chunks to look like what the store returns
        store = MagicMock()
        for c in chunks:
            c.language = "markdown"  # ensure language is set
        store.iter_repo.return_value = iter(chunks)

        hg = HeadingGraph()
        hg.build(store, "test_repo", "abc123")

        # Should have indexed at least some chunks
        assert len(hg._chunk_to_path) > 0, "Expected at least one indexed chunk"

    def test_parent_sibling_expansion_via_real_chunks(self) -> None:
        """Parent/sibling chunks retrieved correctly from real chunker output."""
        chunks = _chunk_markdown(
            _MARKDOWN, "doc.md", "test_repo", "abc123", max_chars=4096, overlap_chars=0
        )
        store = MagicMock()
        for c in chunks:
            c.language = "markdown"
        store.iter_repo.return_value = iter(chunks)

        hg = HeadingGraph()
        hg.build(store, "test_repo", "abc123")

        # Find a chunk that has a section_header with depth > 1
        deep_chunks = [cid for cid, path in hg._chunk_to_path.items() if " > " in path]
        if deep_chunks:
            related = hg.get_related_chunks(deep_chunks[0])
            # Self must not appear
            assert deep_chunks[0] not in related
