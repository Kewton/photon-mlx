"""Unit tests for expand_with_graph edge-weight / adaptive-neighborhood extensions (Issue #91)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from baseline_reporag.indexing.symbol_graph import EdgeKind, SymbolGraph
from baseline_reporag.ingestion.chunker import Chunk
from baseline_reporag.retrieval.graph_expansion import (
    _chunk_kind,
    expand_with_graph,
)
from baseline_reporag.retrieval.hybrid import RetrievalResult


def _mk_chunk(
    *,
    chunk_id: str,
    rel_path: str = "pkg/a.py",
    symbols: list[str] | None = None,
    section_header: str = "",
    language: str = "python",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        repo_id="r",
        repo_commit="c",
        rel_path=rel_path,
        language=language,
        start_line=1,
        end_line=10,
        content="",
        symbols=symbols if symbols is not None else [],
        section_header=section_header,
        file_header="",
    )


def _mk_retrieval(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, score=1.0, lexical_score=0.5, embedding_score=0.5
    )


def test_default_behavior_unchanged() -> None:
    """Without new kwargs, output must match the pre-change path."""
    # Primary result: chunk "a"; graph neighbor "b"; store returns no file neighbors.
    results = [_mk_retrieval("a")]
    graph = MagicMock(spec=SymbolGraph)
    graph.get_related_chunks.return_value = ["b"]

    store = MagicMock()
    store.get_many.return_value = [_mk_chunk(chunk_id="a")]
    store.get_neighbors.return_value = []

    ids = expand_with_graph(
        results=results,
        store=store,
        graph=graph,
        repo_id="r",
        repo_commit="c",
        max_hops=1,
        max_nodes=24,
        neighborhood_before=1,
        neighborhood_after=1,
    )
    assert ids == ["a", "b"]
    # weights not passed when kwargs omitted
    kwargs = graph.get_related_chunks.call_args.kwargs
    assert "weights" not in kwargs or kwargs["weights"] is None


def test_edge_weights_reorder() -> None:
    """edge_weights kwarg must be forwarded and change neighbor order."""
    # Real graph: a -> [(b, IMPORT), (c, CALL), (d, INHERIT)]
    graph = SymbolGraph()
    graph._edges["a"] = [
        ("b", EdgeKind.IMPORT),
        ("c", EdgeKind.CALL),
        ("d", EdgeKind.INHERIT),
    ]
    store = MagicMock()
    store.get_many.return_value = [_mk_chunk(chunk_id="a")]
    store.get_neighbors.return_value = []

    ids = expand_with_graph(
        results=[_mk_retrieval("a")],
        store=store,
        graph=graph,
        repo_id="r",
        repo_commit="c",
        max_hops=1,
        max_nodes=24,
        neighborhood_before=0,
        neighborhood_after=0,
        edge_weights={"call": 1.0, "inherit": 0.8, "import": 0.5},
    )
    # "a" first (primary), then graph neighbors by score: c > d > b
    assert ids == ["a", "c", "d", "b"]


def test_adaptive_neighborhood_class_vs_function() -> None:
    """class chunks use (2,2), function chunks use (1,1) when adaptive=True."""
    graph = MagicMock(spec=SymbolGraph)
    graph.get_related_chunks.return_value = []

    # chunk "cls" is a class, chunk "fn" is a function
    cls_chunk = _mk_chunk(chunk_id="cls", symbols=["MyClass"], section_header="MyClass")
    fn_chunk = _mk_chunk(chunk_id="fn", symbols=["do_it"], section_header="do_it")

    store = MagicMock()
    store.get_many.return_value = [cls_chunk, fn_chunk]
    store.get_neighbors.return_value = []

    expand_with_graph(
        results=[_mk_retrieval("cls"), _mk_retrieval("fn")],
        store=store,
        graph=graph,
        repo_id="r",
        repo_commit="c",
        neighborhood_before=1,
        neighborhood_after=1,
        adaptive_neighborhood=True,
        neighborhood_by_kind={
            "class": (2, 2),
            "function": (1, 1),
            "module": (0, 0),
        },
    )
    calls = store.get_neighbors.call_args_list
    assert len(calls) == 2
    # First call: cls_chunk, (2, 2)
    assert calls[0].kwargs["before"] == 2
    assert calls[0].kwargs["after"] == 2
    # Second call: fn_chunk, (1, 1)
    assert calls[1].kwargs["before"] == 1
    assert calls[1].kwargs["after"] == 1


def test_adaptive_neighborhood_fallback_non_python() -> None:
    """Non-python chunks are treated as function (safe fallback)."""
    graph = MagicMock(spec=SymbolGraph)
    graph.get_related_chunks.return_value = []

    rust_chunk = _mk_chunk(
        chunk_id="r1",
        language="rust",
        symbols=["SomeStruct"],
        section_header="SomeStruct",
    )
    store = MagicMock()
    store.get_many.return_value = [rust_chunk]
    store.get_neighbors.return_value = []

    expand_with_graph(
        results=[_mk_retrieval("r1")],
        store=store,
        graph=graph,
        repo_id="r",
        repo_commit="c",
        neighborhood_before=1,
        neighborhood_after=1,
        adaptive_neighborhood=True,
        neighborhood_by_kind={"class": (3, 3), "function": (1, 1), "module": (0, 0)},
    )
    call = store.get_neighbors.call_args_list[0]
    assert call.kwargs["before"] == 1
    assert call.kwargs["after"] == 1


def test_max_nodes_cap_still_enforced() -> None:
    """max_nodes cap must hold even with new kwargs active."""
    graph = SymbolGraph()
    graph._edges["a"] = [(f"n{i}", EdgeKind.CALL) for i in range(50)]
    store = MagicMock()
    store.get_many.return_value = [_mk_chunk(chunk_id="a")]
    store.get_neighbors.return_value = []

    ids = expand_with_graph(
        results=[_mk_retrieval("a")],
        store=store,
        graph=graph,
        repo_id="r",
        repo_commit="c",
        max_hops=1,
        max_nodes=5,
        edge_weights={"call": 1.0, "inherit": 0.8, "import": 0.5},
    )
    assert len(ids) == 5


def test_chunk_kind_module_when_no_symbols() -> None:
    c = _mk_chunk(chunk_id="c", symbols=[], section_header="")
    assert _chunk_kind(c) == "module"


def test_chunk_kind_class_when_capitalized() -> None:
    c = _mk_chunk(chunk_id="c", symbols=["Foo"], section_header="Foo")
    assert _chunk_kind(c) == "class"


def test_chunk_kind_function_when_lowercase() -> None:
    c = _mk_chunk(chunk_id="c", symbols=["foo"], section_header="foo")
    assert _chunk_kind(c) == "function"


def test_chunk_kind_non_python_fallback() -> None:
    c = _mk_chunk(chunk_id="c", symbols=["Foo"], section_header="Foo", language="rust")
    assert _chunk_kind(c) == "function"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
