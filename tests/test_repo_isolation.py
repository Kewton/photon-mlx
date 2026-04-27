"""Cross-repo retrieval isolation tests (Issue #154 Bug 1).

If two repos' chunks ever share an index in memory, ``hybrid_search`` must drop
chunks whose ``chunk_id`` does not begin with ``{repo_id}::``. The bug report
showed a commandmate query returning ``fastapi_fastapi::...`` chunks; this
suite pins the contract so the regression cannot reappear.
"""

from __future__ import annotations

from baseline_reporag.indexing.embedding import EmbeddingResult
from baseline_reporag.indexing.lexical import LexicalResult
from baseline_reporag.retrieval.hybrid import hybrid_search


class _FakeLexicalIndex:
    def __init__(self, results: list[LexicalResult]) -> None:
        self._results = results

    def search(self, query: str, top_k: int = 20) -> list[LexicalResult]:
        return list(self._results[:top_k])


class _FakeEmbeddingIndex:
    def __init__(self, results: list[EmbeddingResult]) -> None:
        self._results = results

    def search(self, query: str, top_k: int = 20) -> list[EmbeddingResult]:
        return list(self._results[:top_k])


def _mixed_lexical() -> _FakeLexicalIndex:
    return _FakeLexicalIndex(
        [
            LexicalResult("fastapi_fastapi::fastapi/cli.py::1-7", 5.0),
            LexicalResult("commandmate::src/cli/main.ts::1-30", 4.0),
            LexicalResult("fastapi_fastapi::fastapi/openapi/docs.py::40-194", 3.0),
            LexicalResult("commandmate::src/web/router.ts::1-50", 2.0),
        ]
    )


def _mixed_embedding() -> _FakeEmbeddingIndex:
    return _FakeEmbeddingIndex(
        [
            EmbeddingResult("fastapi_fastapi::fastapi/openapi/docs.py::197-298", 0.9),
            EmbeddingResult("commandmate::src/lib/util.ts::10-40", 0.8),
            EmbeddingResult("fastapi_fastapi::fastapi/cli.py::1-7", 0.7),
            EmbeddingResult("commandmate::src/web/router.ts::1-50", 0.6),
        ]
    )


def test_hybrid_search_drops_other_repo_chunks() -> None:
    """When repo_id is set, results from other repos must be filtered out."""
    results = hybrid_search(
        query="cli web ui",
        lexical_index=_mixed_lexical(),
        embedding_index=_mixed_embedding(),
        lexical_top_k=10,
        embedding_top_k=10,
        fused_top_k=10,
        repo_id="commandmate",
    )

    assert results, "expected at least one commandmate chunk to survive the filter"
    for r in results:
        assert r.chunk_id.startswith("commandmate::"), (
            f"cross-repo leakage: {r.chunk_id} returned for repo_id=commandmate"
        )


def test_hybrid_search_repo_filter_with_expanded_queries() -> None:
    """Filter must also apply to expanded-query result merging."""
    results = hybrid_search(
        query="primary",
        lexical_index=_mixed_lexical(),
        embedding_index=_mixed_embedding(),
        lexical_top_k=10,
        embedding_top_k=10,
        fused_top_k=10,
        expanded_queries=["expansion-term"],
        repo_id="commandmate",
    )

    for r in results:
        assert r.chunk_id.startswith("commandmate::")


def test_hybrid_search_no_repo_id_keeps_legacy_behaviour() -> None:
    """Default (repo_id='') must not filter — preserves backward-compat."""
    results = hybrid_search(
        query="cli web ui",
        lexical_index=_mixed_lexical(),
        embedding_index=_mixed_embedding(),
        lexical_top_k=10,
        embedding_top_k=10,
        fused_top_k=10,
    )
    repos_seen = {r.chunk_id.split("::", 1)[0] for r in results}
    assert "fastapi_fastapi" in repos_seen and "commandmate" in repos_seen


def test_hybrid_search_unknown_repo_returns_empty() -> None:
    """Filtering for a repo with no matching chunks yields an empty result."""
    results = hybrid_search(
        query="anything",
        lexical_index=_mixed_lexical(),
        embedding_index=_mixed_embedding(),
        lexical_top_k=10,
        embedding_top_k=10,
        fused_top_k=10,
        repo_id="unknown_repo",
    )
    assert results == []
