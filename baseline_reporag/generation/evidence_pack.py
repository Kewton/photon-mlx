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


def _merge_pinned_sets(
    session: SessionState,
    additional_pinned_ids: list[str] | None,
) -> set[str]:
    """Merge ``session.pinned_chunk_ids`` with optional caller-supplied pins.

    Issue #103 / DR1-010: extracted so :func:`build_evidence_pack` keeps a
    single, shallow concern (pin-set composition) and to give the test
    suite a focused unit for the merge contract.

    DR2-003: ``session`` is non-``None`` per the existing
    :func:`build_evidence_pack` contract — both
    :class:`baseline_reporag.pipeline.RepoRAGPipeline` and
    :class:`baseline_reporag.photon_pipeline.PhotonRAGPipeline` obtain the
    session via ``SessionManager.get_or_create`` which never returns
    ``None``.
    """
    base = set(session.pinned_chunk_ids)
    if additional_pinned_ids:
        base = base | set(additional_pinned_ids)
    return base


def build_evidence_pack(
    chunk_ids: list[str],
    store: ChunkStore,
    session: SessionState,
    max_chunks: int = 16,
    max_tokens: int = 16000,
    recent_citation_turns: int = 2,
    *,
    additional_pinned_ids: list[str] | None = None,
) -> EvidencePack:
    # Use only recently cited chunks for citation bias to avoid crowding out
    # fresh retrieval in later turns.  Cumulative cited_chunk_ids grows every
    # turn and causes T5-T6 degradation where stale citations dominate the pack.
    recent_turns = session.recent_history(max_turns=recent_citation_turns)
    recent_cited: set[str] = set()
    for t in recent_turns:
        recent_cited.update(t.cited_chunk_ids)

    # Issue #103: ``additional_pinned_ids`` (kw-only) lets PHOTON pipeline
    # promote past-turn cited chunks into the next pack at priority=0
    # without mutating ``session.pinned_chunk_ids`` (DR2-003 keeps the
    # existing signature ordering — ``recent_citation_turns`` default 2 is
    # preserved). Candidate de-dup keeps first-seen order so retrieval/rerank
    # rank is not lost among chunks with the same priority.
    pinned_set = _merge_pinned_sets(session, additional_pinned_ids)

    def priority(cid: str) -> int:
        if cid in pinned_set:
            return 0
        if cid in recent_cited:
            return 1
        return 2

    # Codex CB-001 fix: union ``additional_pinned_ids`` into the candidate
    # set BEFORE ``sorted(...)`` so pinned chunks that are NOT already in
    # the current retrieval result are still surfaced. This is the entire
    # purpose of the Issue #103 pinning channel (rescuing past-turn
    # chunks that retrieval missed). NOTE: ``session.pinned_chunk_ids``
    # is intentionally NOT unioned here — its semantics (existing
    # ``priority()`` re-order only) stay untouched, only the new
    # transient ``additional_pinned_ids`` may inject new IDs.
    candidate_ids: list[str] = []
    seen_candidates: set[str] = set()
    if additional_pinned_ids:
        for cid in additional_pinned_ids:
            if cid in seen_candidates:
                continue
            seen_candidates.add(cid)
            candidate_ids.append(cid)
    for cid in chunk_ids:
        if cid in seen_candidates:
            continue
        seen_candidates.add(cid)
        candidate_ids.append(cid)

    original_order = {cid: index for index, cid in enumerate(candidate_ids)}
    if additional_pinned_ids:
        ordered_pins = []
        seen_pins: set[str] = set()
        for cid in additional_pinned_ids:
            if cid in seen_pins or cid not in seen_candidates:
                continue
            seen_pins.add(cid)
            ordered_pins.append(cid)
        ordered_ids = (
            ordered_pins
            + [
                cid
                for cid in sorted(
                    candidate_ids,
                    key=lambda candidate_id: (
                        priority(candidate_id),
                        original_order[candidate_id],
                    ),
                )
                if cid not in seen_pins
            ]
        )[:max_chunks]
    else:
        ordered_ids = sorted(
            candidate_ids,
            key=lambda candidate_id: (
                priority(candidate_id),
                original_order[candidate_id],
            ),
        )[:max_chunks]
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
