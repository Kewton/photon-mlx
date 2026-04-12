from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

from ..ingestion.store import ChunkStore


class SymbolGraph:
    """Maps Python symbols to chunk IDs and tracks call-level edges."""

    def __init__(self) -> None:
        # symbol_name -> [chunk_ids where it is defined]
        self._definitions: dict[str, list[str]] = defaultdict(list)
        # chunk_id -> [chunk_ids it references]
        self._edges: dict[str, list[str]] = defaultdict(list)

    def build(self, store: ChunkStore, repo_id: str, repo_commit: str) -> None:
        self._definitions = defaultdict(list)
        self._edges = defaultdict(list)

        # Pass 1: collect symbol definitions
        for chunk in store.iter_repo(repo_id, repo_commit):
            for sym in chunk.symbols:
                self._definitions[sym].append(chunk.chunk_id)

        # Pass 2: find call edges in Python chunks
        for chunk in store.iter_repo(repo_id, repo_commit):
            if chunk.language != "python":
                continue
            try:
                tree = ast.parse(chunk.content)
            except SyntaxError:
                continue
            called: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        called.add(node.func.attr)
            for sym in called:
                for target_id in self._definitions.get(sym, []):
                    if target_id != chunk.chunk_id:
                        self._edges[chunk.chunk_id].append(target_id)

    def get_related_chunks(self, chunk_id: str, max_hops: int = 1) -> list[str]:
        visited: set[str] = set()
        frontier = {chunk_id}
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for cid in frontier:
                for neighbor in self._edges.get(cid, []):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            visited |= frontier
            frontier = next_frontier - visited
        visited.discard(chunk_id)
        return list(visited)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "definitions": dict(self._definitions),
                "edges": dict(self._edges),
            }),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> SymbolGraph:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        g = cls()
        g._definitions = defaultdict(list, data["definitions"])
        g._edges = defaultdict(list, data["edges"])
        return g
