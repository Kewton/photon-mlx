from __future__ import annotations

import re
from dataclasses import dataclass

from .generation.evidence_pack import EvidencePack
from .ingestion.store import Chunk

# Issue #154 Bug 2: phrases that mark an answer as a legitimate refusal
# ("we don't have evidence to answer"). When present, the answer must be
# graded as no-citation regardless of any formal ``[C:N]`` markers — baseline
# was producing refusals like ``根拠が不足しています ... [C:1]`` and the
# regex-only check was treating them as cited, unfairly penalising PHOTON
# which honestly omits the marker on refusals.
REFUSAL_PATTERNS: tuple[str, ...] = (
    "根拠が不足しています",
    "根拠不足",
    "情報がありません",
    "情報はありません",
    "わかりません",
    "判断できません",
    "特定できません",
)


def is_refusal_answer(answer: str) -> bool:
    """True if *answer* contains any known refusal / abstain phrase."""
    if not answer:
        return False
    return any(p in answer for p in REFUSAL_PATTERNS)


@dataclass
class CitationResult:
    cited_chunk_ids: list[str]
    cited_chunks: list[Chunk]
    wrong_citation_indices: list[int]  # [C:N] indices absent from the pack
    no_citation: bool
    is_refusal: bool = False


def resolve_citations(answer: str, pack: EvidencePack) -> CitationResult:
    """Parse [C:N] references in answer and map them to chunk IDs."""
    indices = [int(m.group(1)) for m in re.finditer(r"\[C:(\d+)\]", answer)]

    # Build reverse map: 1-based index -> Chunk
    index_to_chunk: dict[int, Chunk] = {}
    for chunk, idx in zip(pack.chunks, pack.chunk_indices.values()):
        index_to_chunk[idx] = chunk

    cited_chunk_ids: list[str] = []
    wrong_indices: list[int] = []
    seen: set[str] = set()

    for idx in set(indices):
        chunk = index_to_chunk.get(idx)
        if chunk:
            if chunk.chunk_id not in seen:
                cited_chunk_ids.append(chunk.chunk_id)
                seen.add(chunk.chunk_id)
        else:
            wrong_indices.append(idx)

    cited_chunks = [c for c in pack.chunks if c.chunk_id in seen]

    return CitationResult(
        cited_chunk_ids=cited_chunk_ids,
        cited_chunks=cited_chunks,
        wrong_citation_indices=wrong_indices,
        no_citation=len(indices) == 0,
        is_refusal=is_refusal_answer(answer),
    )
