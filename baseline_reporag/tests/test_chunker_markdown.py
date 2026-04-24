"""Markdown chunker tests (Issue #109).

Covers:
- Heading boundary detection (H1-H3 only; H4+ not a boundary nor path element)
- Article boundary regex (``第N条`` / ``第N節`` with half-width / full-width /
  kanji digits; indented lines excluded)
- Code fence handling (backtick fences only; tildes / indents are body text)
- Empty-section skip
- Section-header path construction (``"H1 > H3"`` when intermediate absent)
- Size limit fallback (paragraph split; single-paragraph overflow via
  ``_slide_char_window``)
- Chunk field contract (``language``, ``symbols``, ``section_header``,
  ``chunk_id``, ``file_header``)
- ``chunk_file`` dispatch routing ``language="markdown"`` into
  ``_chunk_markdown``
"""

from __future__ import annotations

from baseline_reporag.ingestion.chunker import (
    _chunk_markdown,
    _slide_char_window,
    chunk_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _md_chunks(content: str, *, max_chars: int = 2400, overlap_chars: int = 300):
    return _chunk_markdown(
        content=content,
        rel_path="doc.md",
        repo_id="repo",
        repo_commit="head",
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


# ---------------------------------------------------------------------------
# 1. Heading boundary detection
# ---------------------------------------------------------------------------


class TestHeadingBoundaries:
    def test_h1_split_produces_two_chunks(self):
        content = "# Top\n\nbody-a\n\n# Next\n\nbody-b\n"
        chunks = _md_chunks(content)
        assert len(chunks) == 2
        assert chunks[0].section_header == "Top"
        assert chunks[1].section_header == "Next"

    def test_h1_h2_h3_path_composition(self):
        content = (
            "# Parent\n\nintro\n\n## Child\n\nchild-body\n\n### Grand\n\ngrand-body\n"
        )
        chunks = _md_chunks(content)
        headers = [c.section_header for c in chunks]
        assert "Parent" in headers
        assert "Parent > Child" in headers
        assert "Parent > Child > Grand" in headers

    def test_h4_plus_is_not_a_boundary_and_excluded_from_path(self):
        content = (
            "# Top\n\n"
            "body-a\n\n"
            "#### Detail\n\n"
            "detail-body text that stays in the Top section.\n"
        )
        chunks = _md_chunks(content)
        assert len(chunks) == 1
        assert chunks[0].section_header == "Top"
        # H4 heading text must NOT appear in section_header
        assert "Detail" not in chunks[0].section_header

    def test_parent_skip_only_lists_existing_levels(self):
        # H1 then directly H3 (no H2)
        content = "# Root\n\nbody-root\n\n### Leaf\n\nbody-leaf\n"
        chunks = _md_chunks(content)
        headers = [c.section_header for c in chunks]
        # No '> (empty)' or double '>' artefact
        assert "Root > Leaf" in headers
        for h in headers:
            assert "> >" not in h
            assert "(" not in h


# ---------------------------------------------------------------------------
# 2. Article boundary detection
# ---------------------------------------------------------------------------


class TestArticleBoundaries:
    def test_kanji_numeral_article_boundary(self):
        content = "# Law\n\npreamble\n\n第十条 text of article ten.\n\n"
        chunks = _md_chunks(content)
        # Preamble + article chunk
        joined = "\n".join(c.content for c in chunks)
        assert "article ten" in joined
        # An article-body chunk exists
        assert any("第十条" in c.content for c in chunks)

    def test_halfwidth_and_fullwidth_digit_article(self):
        content = (
            "# Law\n\n"
            "第1条 halfwidth one\n\n"
            "第１０条 fullwidth ten\n\n"
            "第2節 section two\n"
        )
        chunks = _md_chunks(content)
        # Each article starts a new section boundary
        assert any("halfwidth one" in c.content for c in chunks)
        assert any("fullwidth ten" in c.content for c in chunks)
        assert any("section two" in c.content for c in chunks)
        # We should have more than a single chunk.
        assert len(chunks) >= 3

    def test_indented_article_line_is_not_a_boundary(self):
        content = (
            "# Law\n\n"
            "See the example below:\n\n"
            "    第1条 (indented, inside a block quote)\n\n"
            "continuing prose about article one.\n"
        )
        chunks = _md_chunks(content)
        # Indented line must not create a new article section
        # So only the H1 heading forms a boundary
        assert len(chunks) == 1
        assert chunks[0].section_header == "Law"


# ---------------------------------------------------------------------------
# 3. Code fence handling
# ---------------------------------------------------------------------------


class TestCodeFenceHandling:
    def test_hash_inside_backtick_fence_is_not_heading(self):
        content = (
            "# Real\n\n"
            "Before fence.\n\n"
            "```\n"
            "# not a heading\n"
            "## also not\n"
            "```\n\n"
            "After fence.\n"
        )
        chunks = _md_chunks(content)
        # Only one heading-driven section
        assert len(chunks) == 1
        assert chunks[0].section_header == "Real"
        # Fence content survives in body
        assert "# not a heading" in chunks[0].content

    def test_longer_backtick_fence_not_closed_by_shorter_inner_line(self):
        """CB-002: 4-backtick fence must not close on a 3-backtick inner line.

        Markdown spec allows any run of 3+ backticks as an opener, but
        the closing fence must be at least as long as the opener. If the
        chunker toggles on ``^```` the 3-backtick inner line closes the
        fence and the subsequent ``# inside`` registers as a heading.
        """
        content = (
            "# Top\n\n"
            "Before fence.\n\n"
            "````\n"
            "nested ```\n"
            "# inside fenced block\n"
            "````\n\n"
            "After fence.\n"
        )
        chunks = _md_chunks(content)
        # Only one heading boundary (``# Top``) — ``# inside`` is still
        # inside the 4-backtick fence.
        assert len(chunks) == 1
        assert chunks[0].section_header == "Top"
        assert "# inside fenced block" in chunks[0].content

    def test_tilde_lines_are_body_not_fence(self):
        """Tilde fences (``~~~``) are plain body text per design #6.

        The chunker does NOT treat ``~~~`` as a code fence, so it does
        not toggle fence state — meaning tilde lines themselves are
        normal body lines and are NOT heading boundaries. Headings
        adjacent to tilde lines still work as usual.
        """
        content = (
            "# Top\n\n"
            "Body before.\n\n"
            "~~~\n"
            "tilde line without leading hash\n"
            "~~~\n\n"
            "After.\n\n"
            "## Sub\n\nsub-body\n"
        )
        chunks = _md_chunks(content)
        headers = [c.section_header for c in chunks]
        assert "Top" in headers
        assert "Top > Sub" in headers


# ---------------------------------------------------------------------------
# 4. Empty section skipping
# ---------------------------------------------------------------------------


class TestEmptySections:
    def test_empty_section_skipped(self):
        content = "# Header1\n\n\n# Header2\n\nbody here\n"
        chunks = _md_chunks(content)
        # First H1 has empty body → skip
        assert len(chunks) == 1
        assert chunks[0].section_header == "Header2"


# ---------------------------------------------------------------------------
# 5. Size limit / paragraph split / single-paragraph overflow
# ---------------------------------------------------------------------------


class TestSizeLimit:
    def test_paragraph_split_when_section_exceeds_max_chars(self):
        para1 = "p1 " * 200  # ~600 chars
        para2 = "p2 " * 200
        para3 = "p3 " * 200
        content = f"# Big\n\n{para1}\n\n{para2}\n\n{para3}\n"
        chunks = _md_chunks(content, max_chars=500, overlap_chars=50)
        assert len(chunks) >= 2
        assert all(c.section_header == "Big" for c in chunks)

    def test_single_paragraph_overflow_falls_back_to_slide_window(self):
        long_para = "x" * 3000
        content = f"# Over\n\n{long_para}\n"
        chunks = _md_chunks(content, max_chars=500, overlap_chars=50)
        assert len(chunks) >= 2
        assert all(c.section_header == "Over" for c in chunks)

    def test_paragraph_split_chunk_ids_are_unique(self):
        """CB-001: paragraph-split oversize sections must have unique chunk_ids.

        Previously every split piece reused ``sec.start_line``/
        ``sec.end_line`` so ``repo::doc.md::2-5`` was emitted multiple
        times, which collides in ``ChunkStore``. The fix differentiates
        each slice (distinct line range or suffix).
        """
        para1 = "p1 " * 200
        para2 = "p2 " * 200
        para3 = "p3 " * 200
        content = f"# Big\n\n{para1}\n\n{para2}\n\n{para3}\n"
        chunks = _md_chunks(content, max_chars=500, overlap_chars=50)
        assert len(chunks) >= 2
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), f"duplicate chunk_ids: {ids}"

    def test_single_paragraph_overflow_chunk_ids_are_unique(self):
        """CB-001: slide-window fallback must also emit unique chunk_ids."""
        long_para = "x" * 3000
        content = f"# Over\n\n{long_para}\n"
        chunks = _md_chunks(content, max_chars=500, overlap_chars=50)
        assert len(chunks) >= 2
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), f"duplicate chunk_ids: {ids}"

    def test_slide_char_window_single_short_text(self):
        out = _slide_char_window("hello", max_chars=10, overlap_chars=2)
        assert out == [(0, 5, "hello")]

    def test_slide_char_window_advances_by_step(self):
        text = "a" * 25
        out = _slide_char_window(text, max_chars=10, overlap_chars=2)
        # step = 10 - 2 = 8; windows start at 0, 8, 16 (last extends to 25)
        assert out[0][0] == 0
        assert out[1][0] == 8
        # every returned window's body matches the slice
        for start, end, body in out:
            assert body == text[start:end]
        # last window ends at text length
        assert out[-1][1] == len(text)


# ---------------------------------------------------------------------------
# 6. Chunk field contract
# ---------------------------------------------------------------------------


class TestChunkFields:
    def test_markdown_chunks_populate_all_fields(self):
        content = "# Heading\n\n## Sub\n\nbody content for the subsection\n"
        chunks = _md_chunks(content)
        c = chunks[-1]
        assert c.language == "markdown"
        assert c.symbols == []
        assert c.section_header == "Heading > Sub"
        # chunk_id contract: repo_id::rel_path::start-end
        assert c.chunk_id.startswith("repo::doc.md::")
        # file_header captures the head of content
        assert c.file_header.startswith("# Heading")


# ---------------------------------------------------------------------------
# 7. chunk_file dispatch
# ---------------------------------------------------------------------------


class TestChunkFileDispatch:
    def test_markdown_language_dispatches_to_markdown_chunker(self):
        content = "# Doc\n\nbody content\n"
        chunks = chunk_file(
            content=content,
            rel_path="README.md",
            language="markdown",
            repo_id="repo",
            repo_commit="head",
        )
        assert len(chunks) == 1
        assert chunks[0].language == "markdown"
        assert chunks[0].section_header == "Doc"

    def test_non_markdown_falls_back_to_existing_behaviour(self):
        content = "some plain text body\n"
        chunks = chunk_file(
            content=content,
            rel_path="notes.txt",
            language="plaintext",
            repo_id="repo",
            repo_commit="head",
        )
        assert chunks[0].language == "plaintext"
        assert chunks[0].section_header == ""
