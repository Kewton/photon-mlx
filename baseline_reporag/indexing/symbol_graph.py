from __future__ import annotations

import ast
import json
from collections import defaultdict
from enum import Enum
from pathlib import Path

from ..ingestion.store import ChunkStore


class EdgeKind(str, Enum):
    """Kind of edge in the symbol graph."""

    CALL = "call"
    INHERIT = "inherit"
    IMPORT = "import"


# Default BFS edge weights. Hop decay (0.5 per hop) is applied on top of this.
_DEFAULT_HOP_DECAY = 0.5


class SymbolGraph:
    """Maps Python symbols to chunk IDs and tracks typed edges (call / inherit / import)."""

    def __init__(self) -> None:
        # symbol_name -> [chunk_ids where it is defined]
        self._definitions: dict[str, list[str]] = defaultdict(list)
        # chunk_id -> [(neighbor_id, edge_kind), ...]
        self._edges: dict[str, list[tuple[str, EdgeKind]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, store: ChunkStore, repo_id: str, repo_commit: str) -> None:
        self._definitions = defaultdict(list)
        self._edges = defaultdict(list)

        # Pass 1: collect symbol definitions and record (chunk_id -> rel_path)
        chunk_rel_paths: dict[str, str] = {}
        for chunk in store.iter_repo(repo_id, repo_commit):
            chunk_rel_paths[chunk.chunk_id] = chunk.rel_path
            for sym in chunk.symbols:
                self._definitions[sym].append(chunk.chunk_id)

        # Pass 2: walk AST for call / inherit / import edges (Python only)
        for chunk in store.iter_repo(repo_id, repo_commit):
            if chunk.language != "python":
                continue
            try:
                tree = ast.parse(chunk.content)
            except SyntaxError:
                continue

            edges_with_kind: list[tuple[str, EdgeKind]] = []
            seen_pair: set[tuple[str, EdgeKind]] = set()

            def _add(target: str, kind: EdgeKind) -> None:
                if target == chunk.chunk_id:
                    return
                pair = (target, kind)
                if pair in seen_pair:
                    return
                seen_pair.add(pair)
                edges_with_kind.append(pair)

            # CALL edges: ast.Call -> func.id / func.attr
            called: set[str] = set()
            # INHERIT edges: ClassDef.bases
            inherited: set[str] = set()
            # IMPORT edges: Import / ImportFrom
            imported: list[str] = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        called.add(node.func.attr)
                elif isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            inherited.add(base.id)
                        elif isinstance(base, ast.Attribute):
                            inherited.add(base.attr)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        # "import foo.bar" -> try qualified name first, then the tail.
                        imported.append(alias.name)
                        tail = alias.name.rsplit(".", 1)[-1]
                        if tail != alias.name:
                            imported.append(tail)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported.append(alias.name)

            for sym in called:
                for target_id in self._definitions.get(sym, []):
                    _add(target_id, EdgeKind.CALL)

            for sym in inherited:
                for target_id in self._definitions.get(sym, []):
                    _add(target_id, EdgeKind.INHERIT)

            for sym in imported:
                candidates = self._definitions.get(sym, [])
                if not candidates:
                    continue
                resolved = self._resolve_import_target(
                    candidates,
                    importing_chunk_id=chunk.chunk_id,
                    importing_rel_path=chunk.rel_path,
                    chunk_rel_paths=chunk_rel_paths,
                )
                if resolved is not None:
                    _add(resolved, EdgeKind.IMPORT)

            if edges_with_kind:
                self._edges[chunk.chunk_id].extend(edges_with_kind)

    @staticmethod
    def _resolve_import_target(
        candidates: list[str],
        *,
        importing_chunk_id: str,
        importing_rel_path: str,
        chunk_rel_paths: dict[str, str],
    ) -> str | None:
        """Pick a single best target for an import symbol (DR4-002).

        Priority: intra-file -> same directory/package -> first candidate.
        Self-chunk is skipped at the caller via ``_add``.
        """
        intra_file = [
            c
            for c in candidates
            if chunk_rel_paths.get(c) == importing_rel_path and c != importing_chunk_id
        ]
        if intra_file:
            return intra_file[0]

        importing_dir = (
            importing_rel_path.rsplit("/", 1)[0] if "/" in importing_rel_path else ""
        )
        same_dir = [
            c
            for c in candidates
            if c != importing_chunk_id
            and (
                (
                    chunk_rel_paths.get(c, "").rsplit("/", 1)[0]
                    if "/" in chunk_rel_paths.get(c, "")
                    else ""
                )
                == importing_dir
            )
        ]
        if same_dir:
            return same_dir[0]

        for c in candidates:
            if c != importing_chunk_id:
                return c
        return None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_related_chunks(
        self,
        chunk_id: str,
        max_hops: int = 1,
        weights: dict[EdgeKind, float] | None = None,
    ) -> list[str]:
        """Return neighbors up to ``max_hops`` away.

        - ``weights=None`` preserves the original insertion-order semantics.
        - When ``weights`` is provided, BFS accumulates ``edge_weight * hop_decay**hop``
          per reachable neighbor and the result is sorted by score descending.
        """
        if weights is None:
            return self._get_related_unweighted(chunk_id, max_hops)
        return self._get_related_weighted(chunk_id, max_hops, weights)

    def _get_related_unweighted(self, chunk_id: str, max_hops: int) -> list[str]:
        visited: set[str] = set()
        ordered: list[str] = []
        frontier = {chunk_id}
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for cid in frontier:
                for neighbor, _kind in self._edges.get(cid, []):
                    if neighbor not in visited and neighbor != chunk_id:
                        if neighbor not in ordered:
                            ordered.append(neighbor)
                        next_frontier.add(neighbor)
            visited |= frontier
            frontier = next_frontier - visited
        return ordered

    def _get_related_weighted(
        self,
        chunk_id: str,
        max_hops: int,
        weights: dict[EdgeKind, float],
    ) -> list[str]:
        scores: dict[str, float] = {}
        # BFS: track best score discovered per neighbor at each hop.
        current: dict[str, float] = {chunk_id: 1.0}
        for hop in range(max_hops):
            nxt: dict[str, float] = {}
            decay = _DEFAULT_HOP_DECAY**hop
            for cid, base in current.items():
                for neighbor, kind in self._edges.get(cid, []):
                    if neighbor == chunk_id:
                        continue
                    w = weights.get(kind, 0.0)
                    inc = base * w * decay
                    if inc <= 0:
                        continue
                    scores[neighbor] = scores.get(neighbor, 0.0) + inc
                    # Propagate the current cumulative score for the next hop.
                    nxt[neighbor] = max(nxt.get(neighbor, 0.0), scores[neighbor])
            current = nxt
        return [
            cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized_edges: dict[str, list[dict[str, str]]] = {
            cid: [{"to": t, "kind": k.value} for (t, k) in edges]
            for cid, edges in self._edges.items()
        }
        path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "definitions": dict(self._definitions),
                    "edges": serialized_edges,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> SymbolGraph:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        g = cls()
        g._definitions = defaultdict(list, data.get("definitions", {}))

        raw_edges = data.get("edges", {}) or {}
        version = data.get("version")
        is_v2 = version == 2 or _looks_like_v2_edges(raw_edges)

        parsed: dict[str, list[tuple[str, EdgeKind]]] = defaultdict(list)
        if is_v2:
            for cid, edges in raw_edges.items():
                for e in edges:
                    if isinstance(e, dict):
                        parsed[cid].append(
                            (e["to"], EdgeKind(e.get("kind", EdgeKind.CALL.value)))
                        )
                    else:
                        # Mixed or legacy-looking entry: fall back to CALL.
                        parsed[cid].append((e, EdgeKind.CALL))
        else:
            # v1: edges are list[str]; treat all as CALL.
            for cid, neighbors in raw_edges.items():
                for n in neighbors:
                    parsed[cid].append((n, EdgeKind.CALL))
        g._edges = parsed
        return g


def _looks_like_v2_edges(raw_edges: dict) -> bool:
    """True iff any edge value is a dict (v2 schema marker)."""
    for edges in raw_edges.values():
        for e in edges:
            return isinstance(e, dict)
    return False
