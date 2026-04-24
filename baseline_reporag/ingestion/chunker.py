from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str  # {repo_id}::{rel_path}::{start_line}-{end_line}
    repo_id: str
    repo_commit: str
    rel_path: str
    language: str
    start_line: int  # 1-based
    end_line: int  # 1-based, inclusive
    content: str
    symbols: list[str]  # function/class names defined in this chunk
    section_header: str  # nearest enclosing symbol name
    file_header: str  # first ~300 chars of the file


def _file_header(content: str, max_chars: int = 300) -> str:
    return content[:max_chars]


def _make_id(repo_id: str, rel_path: str, start: int, end: int) -> str:
    return f"{repo_id}::{rel_path}::{start}-{end}"


# ---------------------------------------------------------------------------
# Python-aware chunker
# ---------------------------------------------------------------------------


def _chunk_python(
    content: str,
    rel_path: str,
    repo_id: str,
    repo_commit: str,
    max_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    lines = content.splitlines(keepends=True)
    header = _file_header(content)

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_plain(
            content, rel_path, repo_id, repo_commit, "python", max_chars, overlap_chars
        )

    # Collect top-level defs (FunctionDef, AsyncFunctionDef, ClassDef)
    boundaries: list[tuple[int, int, str]] = sorted(
        (node.lineno, node.end_lineno, node.name)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.col_offset == 0  # top-level only
    )

    if not boundaries:
        return _chunk_plain(
            content, rel_path, repo_id, repo_commit, "python", max_chars, overlap_chars
        )

    chunks: list[Chunk] = []

    # Preamble (module docstring, imports, constants before first def)
    first_def_line = boundaries[0][0]
    if first_def_line > 1:
        preamble = "".join(lines[: first_def_line - 1])
        if preamble.strip():
            chunks.append(
                Chunk(
                    chunk_id=_make_id(repo_id, rel_path, 1, first_def_line - 1),
                    repo_id=repo_id,
                    repo_commit=repo_commit,
                    rel_path=rel_path,
                    language="python",
                    start_line=1,
                    end_line=first_def_line - 1,
                    content=preamble,
                    symbols=[],
                    section_header="",
                    file_header=header,
                )
            )

    # Merge consecutive defs into chunks respecting max_chars
    cur_start: int | None = None
    cur_end: int | None = None
    cur_syms: list[str] = []

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_syms
        if cur_start is None:
            return
        body = "".join(lines[cur_start - 1 : cur_end])
        chunks.append(
            Chunk(
                chunk_id=_make_id(repo_id, rel_path, cur_start, cur_end),
                repo_id=repo_id,
                repo_commit=repo_commit,
                rel_path=rel_path,
                language="python",
                start_line=cur_start,
                end_line=cur_end,
                content=body,
                symbols=list(cur_syms),
                section_header=cur_syms[0] if cur_syms else "",
                file_header=header,
            )
        )
        cur_start = cur_end = None
        cur_syms = []

    for s, e, name in boundaries:
        if cur_start is None:
            cur_start, cur_end, cur_syms = s, e, [name]
        else:
            merged = "".join(lines[cur_start - 1 : e])
            if len(merged) > max_chars:
                flush()
                cur_start, cur_end, cur_syms = s, e, [name]
            else:
                cur_end = e
                cur_syms.append(name)

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Shared sliding-window helper (Issue #109 DR2-004)
# ---------------------------------------------------------------------------


def _slide_char_window(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[int, int, str]]:
    """Slide a character window over ``text``.

    Returns ``(start_char, end_char, body)`` tuples. Pure function — no line
    indexing, no ``Chunk`` construction. Used by ``_chunk_plain`` (wrapped
    back into line indices) and by the markdown paragraph overflow fallback
    (consumed directly so offsets land inside the original paragraph).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    step = max(1, max_chars - max(0, overlap_chars))
    if len(text) <= max_chars:
        return [(0, len(text), text)]
    out: list[tuple[int, int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        out.append((start, end, text[start:end]))
        if end >= n:
            break
        start += step
    return out


# ---------------------------------------------------------------------------
# Plain text / line-based chunker (used for non-Python files and fallback)
# ---------------------------------------------------------------------------


def _chunk_plain(
    content: str,
    rel_path: str,
    repo_id: str,
    repo_commit: str,
    language: str,
    max_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    lines = content.splitlines(keepends=True)
    header = _file_header(content)
    chunks: list[Chunk] = []
    start_idx = 0  # 0-based line index

    while start_idx < len(lines):
        end_idx = start_idx
        char_count = 0
        while end_idx < len(lines) and char_count < max_chars:
            char_count += len(lines[end_idx])
            end_idx += 1

        body = "".join(lines[start_idx:end_idx])
        chunks.append(
            Chunk(
                chunk_id=_make_id(repo_id, rel_path, start_idx + 1, end_idx),
                repo_id=repo_id,
                repo_commit=repo_commit,
                rel_path=rel_path,
                language=language,
                start_line=start_idx + 1,
                end_line=end_idx,
                content=body,
                symbols=[],
                section_header="",
                file_header=header,
            )
        )

        # Overlap: step back by overlap_chars
        overlap_seen = 0
        back = end_idx
        while back > start_idx and overlap_seen < overlap_chars:
            back -= 1
            overlap_seen += len(lines[back])
        start_idx = back if back > start_idx else end_idx

    return chunks


# ---------------------------------------------------------------------------
# Markdown chunker (Issue #109)
# ---------------------------------------------------------------------------

# Heading regex: captures only H1-H3 (^#{1,3} ) per design decision #6.
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
# Article/section boundary: 第N条 / 第N節 with half-width, full-width, or
# kanji digits. Leading whitespace disqualifies (indented lines are body).
_ARTICLE_RE = re.compile(r"^第[0-9０-９〇一二三四五六七八九十百千]+[条節](?:\s|$)")
# Backtick code fence. Captures the full backtick run so we can compare
# opener vs. closer length (CB-002: 4-backtick opener must not close on a
# 3-backtick inner line). Tildes are intentionally excluded per design #6.
_BACKTICK_FENCE_RE = re.compile(r"^(`{3,})")


@dataclass
class _Boundary:
    line_idx: int  # 0-based index into the `lines` list
    kind: str  # "heading" | "article"
    level: int  # 1..3 for headings, 0 for articles
    title: str


@dataclass
class _Section:
    header_path: list[str]
    body: str
    start_line: int  # 1-based
    end_line: int  # 1-based inclusive
    article_lead: bool = field(default=False)


def _detect_markdown_boundaries(lines: list[str]) -> list[_Boundary]:
    """Scan ``lines`` and return boundary positions.

    Headings (H1-H3) and article/section openers (``第N条``/``第N節`` at
    column 0) act as boundaries. Backtick fences toggle a guard so that
    ``#`` characters inside fenced code do not register as headings.
    Tilde-fenced and indented blocks are treated as normal body.

    CB-002: the length of the opening backtick run is kept in
    ``open_fence_len``. A line only closes the fence if its leading
    backtick run is at least that long — otherwise shorter inner
    ``````` lines would close a 4-backtick opener and leak ``#`` lines
    back into the heading scanner.
    """
    boundaries: list[_Boundary] = []
    open_fence_len: int | None = None
    for i, raw in enumerate(lines):
        stripped = raw.rstrip("\n\r")
        fence_match = _BACKTICK_FENCE_RE.match(stripped)
        if fence_match:
            run_len = len(fence_match.group(1))
            if open_fence_len is None:
                open_fence_len = run_len
            elif run_len >= open_fence_len:
                open_fence_len = None
            # A shorter inner run is body, not a fence toggle — keep
            # scanning, but the line stays suppressed from heading
            # detection because we're still inside the fence.
            continue
        if open_fence_len is not None:
            continue
        m = _HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            boundaries.append(
                _Boundary(line_idx=i, kind="heading", level=level, title=title)
            )
            continue
        if _ARTICLE_RE.match(stripped):
            title = stripped.strip()
            boundaries.append(
                _Boundary(line_idx=i, kind="article", level=0, title=title)
            )
    return boundaries


def _format_section_header(path: list[str]) -> str:
    return " > ".join(p for p in path if p)


def _build_sections(lines: list[str], boundaries: list[_Boundary]) -> list[_Section]:
    """Convert boundary positions into concrete sections with bodies.

    ``path`` only carries H1-H3 titles (H4+ aren't detected as boundaries
    in the first place). Article openers don't push a new path level —
    the article body is attached to the current heading path. Each
    article opener produces its own section so the article line itself
    becomes part of that section's body.
    """
    if not lines:
        return []

    if not boundaries:
        body = "".join(lines)
        if not body.strip():
            return []
        return [_Section(header_path=[], body=body, start_line=1, end_line=len(lines))]

    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)

    # Optional preamble (lines before the first boundary)
    first_idx = boundaries[0].line_idx
    if first_idx > 0:
        body = "".join(lines[:first_idx])
        if body.strip():
            sections.append(
                _Section(
                    header_path=[],
                    body=body,
                    start_line=1,
                    end_line=first_idx,
                )
            )

    for i, b in enumerate(boundaries):
        start = b.line_idx
        end = boundaries[i + 1].line_idx if i + 1 < len(boundaries) else len(lines)
        if b.kind == "heading":
            # Trim stack to strictly shallower levels, then push.
            while heading_stack and heading_stack[-1][0] >= b.level:
                heading_stack.pop()
            heading_stack.append((b.level, b.title))
            header_path = [title for _, title in heading_stack]
            article_lead = False
        else:
            header_path = [title for _, title in heading_stack]
            article_lead = True

        # Body for this section: lines[start:end]. For a heading we skip
        # the heading line itself so the header isn't duplicated inside
        # the body content.
        body_start = start + 1 if b.kind == "heading" else start
        body = "".join(lines[body_start:end])
        if not body.strip():
            continue
        sections.append(
            _Section(
                header_path=header_path,
                body=body,
                start_line=body_start + 1,
                end_line=end,
                article_lead=article_lead,
            )
        )

    return sections


_PARA_SEP_RE = re.compile(r"\n[ \t]*\n+")


def _split_paragraphs(text: str) -> list[str]:
    """Split on one-or-more blank lines. Empty fragments are dropped."""
    parts = _PARA_SEP_RE.split(text)
    return [p for p in parts if p.strip()]


def _apply_size_limit(
    sections: list[_Section],
    *,
    rel_path: str,
    repo_id: str,
    repo_commit: str,
    max_chars: int,
    overlap_chars: int,
    file_header: str,
) -> list[Chunk]:
    """Emit ``Chunk`` objects, splitting oversize sections on paragraphs.

    Paragraphs are joined greedily up to ``max_chars``. A single paragraph
    bigger than ``max_chars`` falls back to :func:`_slide_char_window`
    (see design decision #6).

    CB-001: when a section is split into multiple pieces, every piece
    reused ``sec.start_line`` / ``sec.end_line`` — which collides in
    ``chunk_id`` and overwrites entries in ``ChunkStore``. We now buffer
    the pieces per section and, if there are 2+, append a ``#partN``
    suffix to the chunk_id. Single-chunk sections keep the original
    ``{repo}::{path}::{start}-{end}`` shape so existing citation /
    retrieval tests are untouched.
    """
    chunks: list[Chunk] = []
    for sec in sections:
        header = _format_section_header(sec.header_path)
        body = sec.body
        if len(body) <= max_chars:
            chunks.append(
                _make_md_chunk(
                    body=body,
                    rel_path=rel_path,
                    repo_id=repo_id,
                    repo_commit=repo_commit,
                    start_line=sec.start_line,
                    end_line=sec.end_line,
                    section_header=header,
                    file_header=file_header,
                )
            )
            continue

        # Oversize section: collect pieces first, then stamp unique ids.
        pieces: list[str] = []
        paragraphs = _split_paragraphs(body)
        current_parts: list[str] = []
        current_len = 0
        for para in paragraphs:
            if len(para) > max_chars:
                if current_parts:
                    pieces.append("\n\n".join(current_parts))
                    current_parts = []
                    current_len = 0
                for _s, _e, piece in _slide_char_window(
                    para, max_chars=max_chars, overlap_chars=overlap_chars
                ):
                    pieces.append(piece)
                continue
            if current_len and current_len + len(para) + 2 > max_chars:
                pieces.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            current_parts.append(para)
            current_len += len(para) + (2 if current_len else 0)
        if current_parts:
            pieces.append("\n\n".join(current_parts))

        # Emit with a stable ``#partN`` suffix so chunk_ids stay unique
        # even when line numbers cannot be recovered reliably.
        total = len(pieces)
        for idx, piece in enumerate(pieces, start=1):
            suffix = f"#part{idx}of{total}" if total > 1 else ""
            chunks.append(
                _make_md_chunk(
                    body=piece,
                    rel_path=rel_path,
                    repo_id=repo_id,
                    repo_commit=repo_commit,
                    start_line=sec.start_line,
                    end_line=sec.end_line,
                    section_header=header,
                    file_header=file_header,
                    id_suffix=suffix,
                )
            )
    return chunks


def _make_md_chunk(
    *,
    body: str,
    rel_path: str,
    repo_id: str,
    repo_commit: str,
    start_line: int,
    end_line: int,
    section_header: str,
    file_header: str,
    id_suffix: str = "",
) -> Chunk:
    # CB-001: ``id_suffix`` is empty for single-chunk sections (unchanged
    # id shape) and ``#partNofM`` when a section was split. The suffix is
    # appended to ``chunk_id`` only; ``start_line`` / ``end_line`` still
    # point at the enclosing section so citation ranges remain valid.
    return Chunk(
        chunk_id=_make_id(repo_id, rel_path, start_line, end_line) + id_suffix,
        repo_id=repo_id,
        repo_commit=repo_commit,
        rel_path=rel_path,
        language="markdown",
        start_line=start_line,
        end_line=end_line,
        content=body,
        symbols=[],
        section_header=section_header,
        file_header=file_header,
    )


def _chunk_markdown(
    content: str,
    rel_path: str,
    repo_id: str,
    repo_commit: str,
    max_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """Chunk a Markdown document into heading/article-aware sections.

    Honours H1-H3 headings, ``第N条`` / ``第N節`` article/section openers,
    and backtick code fences. H4+ headings are not boundaries and are not
    reflected in ``section_header``. Tilde fences and indented blocks are
    plain body. Oversize sections split on paragraphs; single paragraphs
    exceeding ``max_chars`` fall back to a character-level sliding window.
    """
    lines = content.splitlines(keepends=True)
    header = _file_header(content)

    if not lines or not content.strip():
        return []

    boundaries = _detect_markdown_boundaries(lines)
    sections = _build_sections(lines, boundaries)
    return _apply_size_limit(
        sections,
        rel_path=rel_path,
        repo_id=repo_id,
        repo_commit=repo_commit,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        file_header=header,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chunk_file(
    content: str,
    rel_path: str,
    language: str,
    repo_id: str,
    repo_commit: str,
    max_chars: int = 2400,
    overlap_chars: int = 300,
) -> list[Chunk]:
    if language == "python":
        return _chunk_python(
            content, rel_path, repo_id, repo_commit, max_chars, overlap_chars
        )
    if language == "markdown":
        return _chunk_markdown(
            content, rel_path, repo_id, repo_commit, max_chars, overlap_chars
        )
    return _chunk_plain(
        content, rel_path, repo_id, repo_commit, language, max_chars, overlap_chars
    )
