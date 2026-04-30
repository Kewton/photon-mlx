from __future__ import annotations

from dataclasses import dataclass

from ..ingestion.store import ChunkStore
from ..indexing.graph_protocol import GraphLike
from .hybrid import RetrievalResult


@dataclass(frozen=True)
class ExpandedChunkRef:
    """Chunk reference with retrieval-source annotation (Issue #176).

    source values:
      "retrieval" — survived hybrid search + rerank
      "graph"     — added by symbol call-graph expansion
      "neighbor"  — added by same-file neighbor expansion
    """

    chunk_id: str
    source: str


def expand_with_graph(
    results: list[RetrievalResult],
    store: ChunkStore,
    graph: GraphLike | None,
    repo_id: str,
    repo_commit: str,
    max_hops: int = 1,
    max_nodes: int = 24,
    neighborhood_before: int = 1,
    neighborhood_after: int = 1,
) -> list[ExpandedChunkRef]:
    """Return deduplicated ExpandedChunkRef list: original + graph + file neighbors.

    When ``graph is None`` (Issue #109: ``indexing.symbol_graph.enabled=false``
    for non-Python repositories, or heading_graph disabled), the graph-neighbor
    expansion is skipped but file-neighbor expansion via ``store.get_neighbors``
    still runs. Accepts any :class:`GraphLike` implementor (LSP contract).

    Each ref carries a ``source`` tag ("retrieval" | "graph" | "neighbor") so
    callers can distinguish how each chunk entered the evidence pool (Issue #176).
    """
    seen: set[str] = set()
    ordered: list[ExpandedChunkRef] = []

    def add(cid: str, source: str) -> None:
        if cid not in seen:
            seen.add(cid)
            ordered.append(ExpandedChunkRef(chunk_id=cid, source=source))

    for r in results:
        add(r.chunk_id, "retrieval")

    # Graph-based neighbors (skipped when SymbolGraph is disabled)
    if graph is not None:
        for r in results:
            if len(ordered) >= max_nodes:
                break
            for neighbor_id in graph.get_related_chunks(r.chunk_id, max_hops=max_hops):
                if len(ordered) >= max_nodes:
                    break
                add(neighbor_id, "graph")

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
            add(neighbor.chunk_id, "neighbor")

    return ordered
