from __future__ import annotations

from dataclasses import dataclass

from ..ingestion.store import ChunkStore, Chunk
from ..memory.session import SessionState


@dataclass
class EvidencePack:
    chunks: list[Chunk]
    chunk_indices: dict[str, int]  # chunk_id -> 1-based citation index

    def format_for_prompt(self) -> str:
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
        return "\n\n---\n\n".join(parts)


def build_evidence_pack(
    chunk_ids: list[str],
    store: ChunkStore,
    session: SessionState,
    max_chunks: int = 16,
    max_tokens: int = 16000,
    recent_citation_turns: int = 2,
) -> EvidencePack:
    # Use only recently cited chunks for citation bias to avoid crowding out
    # fresh retrieval in later turns.  Cumulative cited_chunk_ids grows every
    # turn and causes T5-T6 degradation where stale citations dominate the pack.
    recent_turns = session.recent_history(max_turns=recent_citation_turns)
    recent_cited: set[str] = set()
    for t in recent_turns:
        recent_cited.update(t.cited_chunk_ids)

    pinned_set = set(session.pinned_chunk_ids)

    def priority(cid: str) -> int:
        if cid in pinned_set:
            return 0
        if cid in recent_cited:
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
