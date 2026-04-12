from __future__ import annotations

from ..ingestion.store import ChunkStore
from ..indexing.symbol_graph import SymbolGraph
from .hybrid import RetrievalResult


def expand_with_graph(
    results: list[RetrievalResult],
    store: ChunkStore,
    graph: SymbolGraph,
    repo_id: str,
    repo_commit: str,
    max_hops: int = 1,
    max_nodes: int = 24,
    neighborhood_before: int = 1,
    neighborhood_after: int = 1,
) -> list[str]:
    """Return deduplicated chunk IDs: original + graph neighbors + file neighbors."""
    seen: set[str] = set()
    ordered: list[str] = []

    def add(cid: str) -> None:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    for r in results:
        add(r.chunk_id)

    # Graph-based neighbors
    for r in results:
        if len(ordered) >= max_nodes:
            break
        for neighbor_id in graph.get_related_chunks(r.chunk_id, max_hops=max_hops):
            if len(ordered) >= max_nodes:
                break
            add(neighbor_id)

    # File-level neighbors (before/after in the same file)
    primary_chunks = store.get_many([r.chunk_id for r in results])
    for chunk in primary_chunks:
        if len(ordered) >= max_nodes:
            break
        for neighbor in store.get_neighbors(
            chunk, before=neighborhood_before, after=neighborhood_after
        ):
            if len(ordered) >= max_nodes:
                break
            add(neighbor.chunk_id)

    return ordered
