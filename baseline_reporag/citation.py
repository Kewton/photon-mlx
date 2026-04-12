from __future__ import annotations

import re
from dataclasses import dataclass

from .generation.evidence_pack import EvidencePack
from .ingestion.store import Chunk


@dataclass
class CitationResult:
    cited_chunk_ids: list[str]
    cited_chunks: list[Chunk]
    wrong_citation_indices: list[int]  # [C:N] indices absent from the pack
    no_citation: bool


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
    )
