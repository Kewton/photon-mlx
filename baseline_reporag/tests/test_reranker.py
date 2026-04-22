"""Unit tests for the CrossEncoderReranker (Issue #89).

Mock-based tests exercise the `rerank()` pipeline without loading a real
sentence-transformers model.  The optional smoke test at the end actually
loads ``BAAI/bge-reranker-base`` and is gated by the ``RUN_SLOW_TESTS=1``
environment variable to avoid forcing a ~550 MB download in CI.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from baseline_reporag.retrieval.hybrid import RetrievalResult
from baseline_reporag.retrieval.reranker import CrossEncoderReranker, _is_noise


class _FakeChunk:
    def __init__(self, chunk_id: str, content: str) -> None:
        self.chunk_id = chunk_id
        self.content = content


class _FakeStore:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._m = {c.chunk_id: c for c in chunks}

    def get_many(self, chunk_ids: list[str]) -> list[_FakeChunk]:
        return [self._m[cid] for cid in chunk_ids if cid in self._m]


def _rr(chunk_id: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        lexical_score=0.0,
        embedding_score=0.0,
    )


@pytest.fixture
def mock_model() -> MagicMock:
    m = MagicMock()
    m.predict = MagicMock(return_value=np.array([0.5]))
    return m


@pytest.fixture
def make_reranker(mock_model):
    def _make(predict_return=None):
        if predict_return is not None:
            mock_model.predict = MagicMock(return_value=np.asarray(predict_return))
        return CrossEncoderReranker(_model=mock_model)

    return _make


def test_is_noise_filters_known_patterns() -> None:
    assert _is_noise("REPO::docs/llm-prompt.md")
    assert _is_noise("REPO::.github/sponsors.yml")
    assert _is_noise("REPO::docs/general-llm-prompt.md")
    assert not _is_noise("REPO::app/main.py")
    assert not _is_noise("REPO::fastapi/routing.py")


def test_rerank_empty_input_returns_empty(make_reranker) -> None:
    reranker = make_reranker()
    assert reranker.rerank("q", [], _FakeStore([]), top_k=5) == []


def test_rerank_filters_noise_and_returns_top_k(make_reranker) -> None:
    reranker = make_reranker(predict_return=[0.9, 0.3])
    results = [
        _rr("R::app/a.py"),
        _rr("R::docs/llm-prompt.md"),  # noise, removed
        _rr("R::app/b.py"),
    ]
    store = _FakeStore([_FakeChunk("R::app/a.py", "x"), _FakeChunk("R::app/b.py", "y")])
    out = reranker.rerank("q", results, store, top_k=1)
    assert len(out) == 1
    assert out[0].chunk_id == "R::app/a.py"
    assert out[0].score == pytest.approx(0.9)


def test_rerank_uses_rerank_query_when_provided(make_reranker, mock_model) -> None:
    reranker = make_reranker(predict_return=[0.5])
    results = [_rr("R::app/a.py")]
    store = _FakeStore([_FakeChunk("R::app/a.py", "body")])

    reranker.rerank(
        "日本語クエリ", results, store, top_k=1, rerank_query="english terms"
    )

    pairs = mock_model.predict.call_args[0][0]
    assert pairs[0][0] == "english terms"  # rerank_query wins over raw query


def test_rerank_falls_back_to_all_if_noise_removes_everything(make_reranker) -> None:
    reranker = make_reranker(predict_return=[0.42])
    results = [_rr("R::docs/llm-prompt.md")]  # only noise entries
    store = _FakeStore([_FakeChunk("R::docs/llm-prompt.md", "content")])

    out = reranker.rerank("q", results, store, top_k=1)

    assert len(out) == 1
    assert out[0].chunk_id == "R::docs/llm-prompt.md"


def test_rerank_preserves_lexical_and_embedding_scores(make_reranker) -> None:
    reranker = make_reranker(predict_return=[0.7])
    original = RetrievalResult(
        chunk_id="R::app/a.py",
        score=0.1,
        lexical_score=0.25,
        embedding_score=0.9,
    )
    store = _FakeStore([_FakeChunk("R::app/a.py", "x")])

    out = reranker.rerank("q", [original], store, top_k=1)

    assert out[0].lexical_score == pytest.approx(0.25)
    assert out[0].embedding_score == pytest.approx(0.9)
    assert out[0].score == pytest.approx(0.7)  # rewritten with rerank score


def test_default_model_id_is_bge_reranker_base() -> None:
    """Issue #89: new default should point to BAAI/bge-reranker-base.

    We inspect the constructor signature rather than instantiating to avoid
    a real HuggingFace download in this quick test.
    """
    import inspect

    sig = inspect.signature(CrossEncoderReranker.__init__)
    default = sig.parameters["model_id"].default
    assert default == "BAAI/bge-reranker-base"


@pytest.mark.skipif(
    os.getenv("RUN_SLOW_TESTS") != "1",
    reason="slow real-model smoke test; set RUN_SLOW_TESTS=1 to run",
)
def test_bge_reranker_base_loads_and_predicts() -> None:
    """Smoke test: real BAAI/bge-reranker-base loads via sentence-transformers."""
    reranker = CrossEncoderReranker(model_id="BAAI/bge-reranker-base")
    scores = reranker._model.predict([("python function", "def foo(): pass")])
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
