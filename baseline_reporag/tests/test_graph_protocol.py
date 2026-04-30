"""Tests for GraphLike Protocol — structural (duck-typing) compatibility."""

from __future__ import annotations

from baseline_reporag.indexing.graph_protocol import GraphLike
from baseline_reporag.indexing.symbol_graph import SymbolGraph


class TestGraphLikeProtocol:
    def test_symbol_graph_is_graphlike(self) -> None:
        """SymbolGraph must structurally satisfy GraphLike without inheritance."""
        g = SymbolGraph()
        assert isinstance(g, GraphLike)

    def test_get_related_chunks_signature_present(self) -> None:
        """GraphLike must expose get_related_chunks(chunk_id, max_hops=1) -> list[str]."""
        import inspect

        sig = inspect.signature(GraphLike.get_related_chunks)
        params = list(sig.parameters.keys())
        assert "chunk_id" in params
        assert "max_hops" in params
        assert sig.parameters["max_hops"].default == 1

    def test_arbitrary_class_satisfies_protocol(self) -> None:
        """Any class with matching get_related_chunks satisfies GraphLike."""

        class FakeGraph:
            def get_related_chunks(self, chunk_id: str, max_hops: int = 1) -> list[str]:
                return []

        assert isinstance(FakeGraph(), GraphLike)

    def test_class_without_method_fails(self) -> None:
        """A class lacking get_related_chunks must NOT satisfy GraphLike."""

        class NoMethod:
            pass

        assert not isinstance(NoMethod(), GraphLike)
