from __future__ import annotations

from typing import Any, Literal

from ..indexing.symbol_graph import EdgeKind, SymbolGraph
from ..ingestion.chunker import Chunk
from ..ingestion.store import ChunkStore
from .hybrid import RetrievalResult


def _chunk_kind(chunk: Chunk) -> Literal["class", "function", "module"]:
    """Classify a chunk for adaptive neighborhood expansion (Issue #91).

    - Non-python chunks always fall back to ``"function"`` (safe default).
    - ``"module"`` when we have no symbol anchor (module preamble / top-level).
    - ``"class"`` when the first symbol's leading char is uppercase.
    - ``"function"`` otherwise.
    """
    if chunk.language != "python":
        return "function"
    if not chunk.symbols or not chunk.section_header:
        return "module"
    first = chunk.symbols[0]
    if first and first[0].isupper():
        return "class"
    return "function"


def _normalize_edge_weights(
    edge_weights: Any,
) -> dict[EdgeKind, float] | None:
    """Convert an ``edge_weights`` value (None / dict / Config-like) to the
    ``dict[EdgeKind, float]`` shape ``SymbolGraph`` expects, or None.
    """
    if edge_weights is None:
        return None
    if hasattr(edge_weights, "to_dict"):
        edge_weights = edge_weights.to_dict()
    if not isinstance(edge_weights, dict):
        return None

    out: dict[EdgeKind, float] = {}
    for k, v in edge_weights.items():
        try:
            kind = EdgeKind(k) if not isinstance(k, EdgeKind) else k
        except ValueError:
            continue
        try:
            out[kind] = float(v)
        except (TypeError, ValueError):
            continue
    return out or None


def _normalize_by_kind(
    neighborhood_by_kind: Any,
) -> dict[str, tuple[int, int]] | None:
    if neighborhood_by_kind is None:
        return None
    if hasattr(neighborhood_by_kind, "to_dict"):
        neighborhood_by_kind = neighborhood_by_kind.to_dict()
    if not isinstance(neighborhood_by_kind, dict):
        return None

    out: dict[str, tuple[int, int]] = {}
    for k, v in neighborhood_by_kind.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                out[str(k)] = (int(v[0]), int(v[1]))
            except (TypeError, ValueError):
                continue
    return out or None


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
    *,
    edge_weights: Any = None,
    adaptive_neighborhood: bool = False,
    neighborhood_by_kind: Any = None,
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

    weights = _normalize_edge_weights(edge_weights)
    by_kind = _normalize_by_kind(neighborhood_by_kind)

    # Graph-based neighbors
    for r in results:
        if len(ordered) >= max_nodes:
            break
        if weights is not None:
            neighbors = graph.get_related_chunks(
                r.chunk_id, max_hops=max_hops, weights=weights
            )
        else:
            neighbors = graph.get_related_chunks(r.chunk_id, max_hops=max_hops)
        for neighbor_id in neighbors:
            if len(ordered) >= max_nodes:
                break
            add(neighbor_id)

    # File-level neighbors (before/after in the same file)
    primary_chunks = store.get_many([r.chunk_id for r in results])
    for chunk in primary_chunks:
        if len(ordered) >= max_nodes:
            break
        before, after = neighborhood_before, neighborhood_after
        if adaptive_neighborhood and by_kind is not None:
            kind = _chunk_kind(chunk)
            before, after = by_kind.get(kind, (before, after))
        for neighbor in store.get_neighbors(chunk, before=before, after=after):
            if len(ordered) >= max_nodes:
                break
            add(neighbor.chunk_id)

    return ordered
