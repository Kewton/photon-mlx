"""GraphLike Protocol — shared interface for graph-based chunk neighbor providers.

Scope (ISP, DR1-009): query-time operations only. build/save/load are excluded
because callers never need them through this interface; keeping them out avoids
forcing implementors to expose storage details they may not have.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GraphLike(Protocol):
    """Structural protocol for graph-based chunk neighbor lookup.

    Behaviour contract (DR1-002, LSP):
    1. Unknown chunk_id returns an empty list — never raises KeyError.
    2. The caller's chunk_id is never included in the returned list.
    3. max_hops controls traversal depth; max_hops=1 returns direct neighbours.
    4. Return order is deterministic for a given graph state (insertion-order
       stable or sorted), allowing callers to rely on reproducible results.
    """

    def get_related_chunks(self, chunk_id: str, max_hops: int = 1) -> list[str]:
        """Return chunk IDs related to *chunk_id* within *max_hops* steps."""
        ...
