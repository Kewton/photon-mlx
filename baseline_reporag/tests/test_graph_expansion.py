"""Tests for ``expand_with_graph`` including ``graph=None`` support (Issue #109)."""

from __future__ import annotations

from unittest.mock import MagicMock

from baseline_reporag.ingestion.chunker import Chunk
from baseline_reporag.retrieval.graph_expansion import expand_with_graph
from baseline_reporag.retrieval.hybrid import RetrievalResult


def _make_chunk(chunk_id: str, rel_path: str = "a.py") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        repo_id="test",
        repo_commit="abc",
        rel_path=rel_path,
        language="python",
        start_line=1,
        end_line=10,
        content="pass",
        symbols=[],
        section_header="",
        file_header="",
    )


def _make_result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, score=1.0, lexical_score=0.5, embedding_score=0.5
    )


class TestExpandWithGraph:
    def test_graph_provided_preserves_existing_behaviour(self):
        """``graph != None`` keeps graph-neighbor expansion + file-neighbors."""
        results = [_make_result("c0"), _make_result("c1")]

        store = MagicMock()
        store.get_many.return_value = [_make_chunk("c0"), _make_chunk("c1")]
        store.get_neighbors.return_value = [_make_chunk("c2")]

        graph = MagicMock()
        graph.get_related_chunks.return_value = ["c3"]

        out = expand_with_graph(
            results=results,
            store=store,
            graph=graph,
            repo_id="test",
            repo_commit="abc",
        )

        assert out[0] == "c0"
        assert out[1] == "c1"
        assert "c3" in out  # graph-neighbor
        assert "c2" in out  # file-neighbor
        graph.get_related_chunks.assert_called()

    def test_graph_none_skips_graph_neighbors_keeps_file_neighbors(self):
        """``graph=None`` must skip graph-neighbors yet preserve file-neighbors."""
        results = [_make_result("c0"), _make_result("c1")]

        store = MagicMock()
        store.get_many.return_value = [_make_chunk("c0"), _make_chunk("c1")]
        store.get_neighbors.return_value = [_make_chunk("c2")]

        out = expand_with_graph(
            results=results,
            store=store,
            graph=None,
            repo_id="test",
            repo_commit="abc",
        )

        # original candidates preserved
        assert out[0] == "c0"
        assert out[1] == "c1"
        # file-neighbors preserved
        assert "c2" in out
        # store continues to be consulted for file-neighbors
        store.get_many.assert_called_once()
        store.get_neighbors.assert_called()

    def test_empty_candidates_returns_empty(self):
        """No candidates → empty output, store should not error."""
        store = MagicMock()
        store.get_many.return_value = []

        out = expand_with_graph(
            results=[],
            store=store,
            graph=None,
            repo_id="test",
            repo_commit="abc",
        )
        assert out == []
