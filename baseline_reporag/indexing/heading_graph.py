"""HeadingGraph — markdown heading hierarchy graph for institutional doc retrieval.

Scope (Phase 1, DR1-007): parent + sibling chunk expansion only.
anchor_ref cross-section traversal is deferred to a later phase.

Storage invariant (DR4-001): load validates JSON structure and rejects files
exceeding _MAX_FILE_BYTES, malformed structures, and unknown top-level keys.

Atomic write (DR2-009): save uses tmp + os.replace for crash-safe updates.

Chunker contract: section_header format is "H1 > H2 > H3" produced by
baseline_reporag.ingestion.chunker._format_section_header.

LSP contract (DR1-002):
1. Unknown chunk_id → empty list, never raises.
2. Self is never in the returned list.
3. max_hops=1 returns direct neighbours (parent + sibling sections).
4. Return order is insertion-order deterministic.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ingestion.store import ChunkStore

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB (DR4-001)
_SEP = " > "
_MAX_HEADER_DEPTH = 10
_MAX_SEGMENT_LEN = 500


def _is_valid_section_header(path: str) -> bool:
    """Validate section_header format (DR4-002)."""
    if not path:
        return False
    segments = path.split(_SEP)
    if len(segments) > _MAX_HEADER_DEPTH:
        return False
    for seg in segments:
        if not seg or len(seg) > _MAX_SEGMENT_LEN:
            return False
    return True


class HeadingGraph:
    """2-dict heading graph: _sections (path→chunk_ids) + _chunk_to_path."""

    def __init__(self) -> None:
        # section_header string → ordered list of chunk_ids in that section
        self._sections: dict[str, list[str]] = defaultdict(list)
        # chunk_id → its section_header string
        self._chunk_to_path: dict[str, str] = {}

    def build(self, store: ChunkStore, repo_id: str, repo_commit: str) -> None:
        """Single-pass build from the chunk store (DR2-008: markdown only)."""
        self._sections = defaultdict(list)
        self._chunk_to_path = {}
        for chunk in store.iter_repo(repo_id, repo_commit):
            if chunk.language != "markdown":
                continue
            if not chunk.section_header:
                continue
            if not _is_valid_section_header(chunk.section_header):
                # Log chunk_id only — never log raw section_header content (DR4-002)
                logger.debug(
                    "Skipping chunk %s: invalid section_header", chunk.chunk_id
                )
                continue
            self._sections[chunk.section_header].append(chunk.chunk_id)
            self._chunk_to_path[chunk.chunk_id] = chunk.section_header

    def get_related_chunks(self, chunk_id: str, max_hops: int = 1) -> list[str]:
        """Return parent-section + sibling chunks, excluding *chunk_id* itself.

        max_hops is accepted for LSP contract compliance but the current
        Phase 1 implementation treats any value ≥1 as one-hop (parent/sibling).
        """
        path = self._chunk_to_path.get(chunk_id)
        if path is None:
            return []

        seen: set[str] = {chunk_id}
        result: list[str] = []

        # Parent section chunks
        if _SEP in path:
            parent_path = path.rsplit(_SEP, 1)[0]
            for cid in self._sections.get(parent_path, []):
                if cid not in seen:
                    result.append(cid)
                    seen.add(cid)

        # Sibling chunks (same section, self excluded)
        for cid in self._sections.get(path, []):
            if cid not in seen:
                result.append(cid)
                seen.add(cid)

        return result

    def save(self, path: str | Path) -> None:
        """Atomic save via tmp + os.replace (stronger than SymbolGraph, DR2-009)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "sections": {k: v for k, v in self._sections.items()},
                "chunk_to_path": self._chunk_to_path,
            },
            ensure_ascii=False,
        )
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path) -> HeadingGraph:
        """Load from JSON with structure validation (DR4-001)."""
        path = Path(path)
        file_size = path.stat().st_size
        if file_size > _MAX_FILE_BYTES:
            raise ValueError(
                f"heading_graph.json exceeds size limit ({file_size} > {_MAX_FILE_BYTES})"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("heading_graph.json must be a JSON object")
        required = {"sections", "chunk_to_path"}
        if set(data.keys()) != required:
            raise ValueError(
                f"heading_graph.json must have exactly keys {required}, got {set(data.keys())}"
            )
        sections = data["sections"]
        chunk_to_path = data["chunk_to_path"]
        if not isinstance(sections, dict):
            raise ValueError("sections must be a JSON object")
        if not isinstance(chunk_to_path, dict):
            raise ValueError("chunk_to_path must be a JSON object")
        for k, v in sections.items():
            if not isinstance(v, list):
                raise ValueError(
                    f"sections[{k!r}] must be a list, got {type(v).__name__}"
                )
            for i, item in enumerate(v):
                if not isinstance(item, str):
                    raise ValueError(
                        f"sections[{k!r}][{i}] must be a str, got {type(item).__name__}"
                    )
        for k, v in chunk_to_path.items():
            if not isinstance(v, str):
                raise ValueError(
                    f"chunk_to_path[{k!r}] must be a str, got {type(v).__name__}"
                )
        g = cls()
        g._sections = defaultdict(list, {k: list(v) for k, v in sections.items()})
        g._chunk_to_path = dict(chunk_to_path)
        return g
