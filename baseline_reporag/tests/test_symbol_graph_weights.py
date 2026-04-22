"""Tests for SymbolGraph edge-kind / weight extensions (Issue #91)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline_reporag.indexing.symbol_graph import EdgeKind, SymbolGraph
from baseline_reporag.ingestion.chunker import Chunk


class _FakeStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def iter_repo(self, repo_id: str, repo_commit: str):
        for c in self._chunks:
            if c.repo_id == repo_id and c.repo_commit == repo_commit:
                yield c


def _make_chunk(
    *,
    chunk_id: str,
    rel_path: str,
    content: str,
    symbols: list[str],
    repo_id: str = "r",
    repo_commit: str = "c",
    language: str = "python",
    section_header: str = "",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        repo_id=repo_id,
        repo_commit=repo_commit,
        rel_path=rel_path,
        language=language,
        start_line=1,
        end_line=10,
        content=content,
        symbols=symbols,
        section_header=section_header,
        file_header="",
    )


def _edge_kinds_from(
    edges: list,
) -> set[EdgeKind]:
    return {kind for (_, kind) in edges}


def _targets(edges: list) -> list[str]:
    return [t for (t, _) in edges]


def test_edge_kinds_collected() -> None:
    """build() should collect CALL, INHERIT, IMPORT edges on a synthetic repo."""
    chunk_a = _make_chunk(
        chunk_id="a",
        rel_path="pkg/a.py",
        content="class Base:\n    def hello(self):\n        pass\n",
        symbols=["Base", "hello"],
        section_header="Base",
    )
    chunk_b = _make_chunk(
        chunk_id="b",
        rel_path="pkg/b.py",
        content=(
            "from pkg.a import Base\n\n"
            "class Child(Base):\n"
            "    def greet(self):\n"
            "        return hello()\n"
        ),
        symbols=["Child", "greet"],
        section_header="Child",
    )
    store = _FakeStore([chunk_a, chunk_b])

    g = SymbolGraph()
    g.build(store, "r", "c")

    edges = g._edges.get("b", [])
    kinds = _edge_kinds_from(edges)
    # INHERIT Base, IMPORT Base, CALL hello
    assert EdgeKind.INHERIT in kinds
    assert EdgeKind.IMPORT in kinds
    assert EdgeKind.CALL in kinds

    # INHERIT edge should point to the chunk that defines Base
    inherit_targets = [t for (t, k) in edges if k is EdgeKind.INHERIT]
    assert inherit_targets == ["a"]


def test_get_related_chunks_weighted() -> None:
    """weights should reorder neighbors by cumulative score (descending)."""
    g = SymbolGraph()
    # hand-wire edges: chunk x -> [y (IMPORT), z (CALL), w (INHERIT)]
    g._edges["x"] = [
        ("y", EdgeKind.IMPORT),
        ("z", EdgeKind.CALL),
        ("w", EdgeKind.INHERIT),
    ]

    related = g.get_related_chunks(
        "x",
        max_hops=1,
        weights={
            EdgeKind.CALL: 1.0,
            EdgeKind.INHERIT: 0.8,
            EdgeKind.IMPORT: 0.5,
        },
    )
    # Expected order: CALL (1.0) > INHERIT (0.8) > IMPORT (0.5)
    assert related == ["z", "w", "y"]


def test_get_related_chunks_unweighted_back_compat() -> None:
    """weights=None preserves old behavior (insertion order, dedup)."""
    g = SymbolGraph()
    g._edges["x"] = [
        ("y", EdgeKind.IMPORT),
        ("z", EdgeKind.CALL),
        ("w", EdgeKind.INHERIT),
    ]
    related = g.get_related_chunks("x", max_hops=1)
    # must contain exactly {y, z, w}; order insertion-based is acceptable
    assert set(related) == {"y", "z", "w"}


def test_save_load_v2_roundtrip(tmp_path: Path) -> None:
    g = SymbolGraph()
    g._definitions["foo"] = ["chunk_foo"]
    g._edges["x"] = [("y", EdgeKind.CALL), ("z", EdgeKind.INHERIT)]
    path = tmp_path / "graph.json"
    g.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 2
    # edges must be list of {"to": ..., "kind": ...} dicts
    edges_x = data["edges"]["x"]
    assert edges_x[0] == {"to": "y", "kind": "call"}
    assert edges_x[1] == {"to": "z", "kind": "inherit"}

    g2 = SymbolGraph.load(path)
    assert g2._definitions["foo"] == ["chunk_foo"]
    assert g2._edges["x"] == [
        ("y", EdgeKind.CALL),
        ("z", EdgeKind.INHERIT),
    ]


def test_load_v1_back_compat(tmp_path: Path) -> None:
    """v1 JSON (edges are list[str], no version key) loads as all-CALL edges."""
    v1 = {
        "definitions": {"foo": ["chunk_foo"]},
        "edges": {"x": ["y", "z"]},
    }
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(v1), encoding="utf-8")

    g = SymbolGraph.load(path)
    assert g._definitions["foo"] == ["chunk_foo"]
    assert g._edges["x"] == [
        ("y", EdgeKind.CALL),
        ("z", EdgeKind.CALL),
    ]


def test_import_resolution_intra_file_priority() -> None:
    """When multiple chunks define a symbol with the same name, intra-file wins."""
    # Two chunks define `util`: one in the importing file (intra-file),
    # one elsewhere. Expect intra-file to be chosen for the IMPORT edge.
    chunk_caller = _make_chunk(
        chunk_id="caller",
        rel_path="pkg/caller.py",
        content="import util\n\ndef use():\n    return util.do()\n",
        symbols=["use"],
        section_header="use",
    )
    chunk_util_same_file = _make_chunk(
        chunk_id="caller_util",
        rel_path="pkg/caller.py",
        content="def util():\n    pass\n",
        symbols=["util"],
        section_header="util",
    )
    chunk_util_other = _make_chunk(
        chunk_id="other_util",
        rel_path="pkg_other/util.py",
        content="def util():\n    pass\n",
        symbols=["util"],
        section_header="util",
    )
    store = _FakeStore([chunk_caller, chunk_util_same_file, chunk_util_other])
    g = SymbolGraph()
    g.build(store, "r", "c")

    import_targets = [
        t for (t, k) in g._edges.get("caller", []) if k is EdgeKind.IMPORT
    ]
    # The IMPORT edge must point to the intra-file candidate only
    assert "caller_util" in import_targets
    assert "other_util" not in import_targets


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
