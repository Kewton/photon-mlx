from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str         # {repo_id}::{rel_path}::{start_line}-{end_line}
    repo_id: str
    repo_commit: str
    rel_path: str
    language: str
    start_line: int       # 1-based
    end_line: int         # 1-based, inclusive
    content: str
    symbols: list[str]    # function/class names defined in this chunk
    section_header: str   # nearest enclosing symbol name
    file_header: str      # first ~300 chars of the file


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
        return _chunk_plain(content, rel_path, repo_id, repo_commit, "python",
                            max_chars, overlap_chars)

    # Collect top-level defs (FunctionDef, AsyncFunctionDef, ClassDef)
    boundaries: list[tuple[int, int, str]] = sorted(
        (node.lineno, node.end_lineno, node.name)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.col_offset == 0  # top-level only
    )

    if not boundaries:
        return _chunk_plain(content, rel_path, repo_id, repo_commit, "python",
                            max_chars, overlap_chars)

    chunks: list[Chunk] = []

    # Preamble (module docstring, imports, constants before first def)
    first_def_line = boundaries[0][0]
    if first_def_line > 1:
        preamble = "".join(lines[: first_def_line - 1])
        if preamble.strip():
            chunks.append(Chunk(
                chunk_id=_make_id(repo_id, rel_path, 1, first_def_line - 1),
                repo_id=repo_id, repo_commit=repo_commit, rel_path=rel_path,
                language="python", start_line=1, end_line=first_def_line - 1,
                content=preamble, symbols=[], section_header="", file_header=header,
            ))

    # Merge consecutive defs into chunks respecting max_chars
    cur_start: int | None = None
    cur_end: int | None = None
    cur_syms: list[str] = []

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_syms
        if cur_start is None:
            return
        body = "".join(lines[cur_start - 1: cur_end])
        chunks.append(Chunk(
            chunk_id=_make_id(repo_id, rel_path, cur_start, cur_end),
            repo_id=repo_id, repo_commit=repo_commit, rel_path=rel_path,
            language="python", start_line=cur_start, end_line=cur_end,
            content=body, symbols=list(cur_syms),
            section_header=cur_syms[0] if cur_syms else "",
            file_header=header,
        ))
        cur_start = cur_end = None
        cur_syms = []

    for s, e, name in boundaries:
        "".join(lines[s - 1: e])
        if cur_start is None:
            cur_start, cur_end, cur_syms = s, e, [name]
        else:
            merged = "".join(lines[cur_start - 1: e])
            if len(merged) > max_chars:
                flush()
                cur_start, cur_end, cur_syms = s, e, [name]
            else:
                cur_end = e
                cur_syms.append(name)

    flush()
    return chunks


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
    start_idx = 0   # 0-based line index

    while start_idx < len(lines):
        end_idx = start_idx
        char_count = 0
        while end_idx < len(lines) and char_count < max_chars:
            char_count += len(lines[end_idx])
            end_idx += 1

        body = "".join(lines[start_idx:end_idx])
        chunks.append(Chunk(
            chunk_id=_make_id(repo_id, rel_path, start_idx + 1, end_idx),
            repo_id=repo_id, repo_commit=repo_commit, rel_path=rel_path,
            language=language, start_line=start_idx + 1, end_line=end_idx,
            content=body, symbols=[], section_header="", file_header=header,
        ))

        # Overlap: step back by overlap_chars
        overlap_seen = 0
        back = end_idx
        while back > start_idx and overlap_seen < overlap_chars:
            back -= 1
            overlap_seen += len(lines[back])
        start_idx = back if back > start_idx else end_idx

    return chunks


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
        return _chunk_python(content, rel_path, repo_id, repo_commit,
                             max_chars, overlap_chars)
    return _chunk_plain(content, rel_path, repo_id, repo_commit, language,
                        max_chars, overlap_chars)
