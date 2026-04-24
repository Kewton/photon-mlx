"""Corpus walk + DocIndex construction for institutional eval-set generation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_ARTICLE_RE = re.compile(r"第\d+条")
_PENALTY_RE = re.compile(r"(罰則|罰金|懲役|過料)")
_EXCEPTION_RE = re.compile(r"(但し書き|ただし書|但書|例外|経過措置)")


@dataclass(frozen=True)
class DocIndex:
    """Lightweight per-document flags used by the generator and sampler."""

    doc_id: str
    path: Path
    has_articles: bool
    has_penalty: bool
    has_exception: bool
    metadata: dict


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _scan_document(document_path: Path) -> tuple[bool, bool, bool]:
    text = _read_text(document_path)
    if not text:
        return False, False, False
    return (
        bool(_ARTICLE_RE.search(text)),
        bool(_PENALTY_RE.search(text)),
        bool(_EXCEPTION_RE.search(text)),
    )


def build_doc_index(root: Path) -> list[DocIndex]:
    """Walk ``root`` once and build a list of DocIndex entries.

    Each immediate subdirectory of ``root`` that contains ``document.md``
    becomes one DocIndex. ``metadata.json`` is loaded when present.
    """
    if not root.exists() or not root.is_dir():
        return []

    entries: list[DocIndex] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        document = child / "document.md"
        if not document.exists():
            continue
        has_art, has_pen, has_exc = _scan_document(document)
        metadata = _load_metadata(child / "metadata.json")
        entries.append(
            DocIndex(
                doc_id=child.name,
                path=document,
                has_articles=has_art,
                has_penalty=has_pen,
                has_exception=has_exc,
                metadata=metadata,
            )
        )
    return entries


def iter_article_docs(root: Path) -> Iterator[DocIndex]:
    """Yield DocIndex entries whose document.md contains 第N条 markers."""
    for doc in build_doc_index(root):
        if doc.has_articles:
            yield doc


def iter_section_docs(root: Path, keyword: str) -> Iterator[DocIndex]:
    """Yield DocIndex entries whose document.md contains ``keyword``."""
    for doc in build_doc_index(root):
        text = _read_text(doc.path)
        if keyword in text:
            yield doc


def build_context(
    metadata: dict,
    document_text: str,
    *,
    max_chars: int = 8000,
) -> str:
    """Format metadata + document body into a prompt context snippet."""
    title = metadata.get("title", "")
    preamble = metadata.get("preamble", "")
    effective_date = metadata.get("effective_date", "")

    header_lines: list[str] = []
    if title:
        header_lines.append(f"タイトル: {title}")
    if effective_date:
        header_lines.append(f"施行日: {effective_date}")
    if preamble:
        header_lines.append(f"前文: {preamble}")
    header = "\n".join(header_lines)

    body = document_text[:max_chars]
    if header:
        return f"{header}\n---\n{body}"
    return body
