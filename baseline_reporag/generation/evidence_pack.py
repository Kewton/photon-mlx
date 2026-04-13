from __future__ import annotations

from dataclasses import dataclass

from ..ingestion.store import ChunkStore, Chunk
from ..memory.session import SessionState

# Citation instruction header — keep in sync with _SYSTEM (prompt.py) and
# _FORMAT_HINT (prompt.py).  See also: design-policy §5-2 DR1-001.
_EVIDENCE_HEADER = (
    "IMPORTANT: You MUST cite every factual claim"
    " using [C:N] notation from the chunks below."
)


@dataclass
class EvidencePack:
    chunks: list[Chunk]
    chunk_indices: dict[str, int]  # chunk_id -> 1-based citation index

    def format_for_prompt(self) -> str:
        if not self.chunks:
            return ""
        parts: list[str] = []
        for chunk in self.chunks:
            idx = self.chunk_indices[chunk.chunk_id]
            header = (
                f"[C:{idx}] {chunk.rel_path}"
                f"  (lines {chunk.start_line}–{chunk.end_line})"
            )
            if chunk.section_header:
                header += f"  [{chunk.section_header}]"
            parts.append(f"{header}\n{chunk.content.strip()}")
        body = "\n\n---\n\n".join(parts)
        return f"{_EVIDENCE_HEADER}\n\n{body}"


def build_evidence_pack(
    chunk_ids: list[str],
    store: ChunkStore,
    session: SessionState,
    max_chunks: int = 16,
    max_tokens: int = 16000,
) -> EvidencePack:
    cited_set = set(session.cited_chunk_ids)
    pinned_set = set(session.pinned_chunk_ids)

    def priority(cid: str) -> int:
        if cid in pinned_set:
            return 0
        if cid in cited_set:
            return 1
        return 2

    ordered_ids = sorted(set(chunk_ids), key=priority)[:max_chunks]
    chunks = store.get_many(ordered_ids)

    # Approximate token budget: 1 token ≈ 4 chars
    char_budget = max_tokens * 4
    selected: list[Chunk] = []
    total_chars = 0
    for chunk in chunks:
        if total_chars + len(chunk.content) > char_budget:
            break
        selected.append(chunk)
        total_chars += len(chunk.content)

    chunk_indices = {c.chunk_id: i + 1 for i, c in enumerate(selected)}
    return EvidencePack(chunks=selected, chunk_indices=chunk_indices)
