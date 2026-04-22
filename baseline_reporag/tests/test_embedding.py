"""Unit tests for EmbeddingIndex (Issue #90).

Covers the bge-small-en-v1.5 migration: new default model_id, save/load
round-trip of the persisted model identifier, dimension-agnostic behaviour,
and the new ``expected_model_id`` compatibility guard on ``load``.

Tests T1-T7 mock the underlying ``SentenceTransformer`` so no HuggingFace
download is needed. T8 is a slow real-model smoke test gated by
``RUN_SLOW_TESTS=1``, matching the pattern used in ``test_reranker.py``.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from baseline_reporag.indexing.embedding import EmbeddingIndex
from baseline_reporag.ingestion.chunker import Chunk


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeStore:
    """Minimal ChunkStore-like iterable used by ``EmbeddingIndex.build``."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)

    def iter_repo(self, repo_id: str, repo_commit: str) -> Iterator[Chunk]:
        for c in self._chunks:
            if c.repo_id == repo_id and c.repo_commit == repo_commit:
                yield c


def _make_chunk(chunk_id: str, content: str = "hello") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        repo_id="R",
        repo_commit="abc",
        rel_path="a.py",
        language="python",
        start_line=1,
        end_line=2,
        content=content,
        symbols=[],
        section_header="",
        file_header="",
    )


def _mock_sentence_transformer(dim: int) -> MagicMock:
    """Return a MagicMock that behaves like a SentenceTransformer for a given dim."""
    st = MagicMock()

    def _encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        n = len(texts) if isinstance(texts, list) else 1
        arr = np.full((n, dim), 0.1, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    st.encode.side_effect = _encode
    return st


# ---------------------------------------------------------------------------
# T1: default model_id regression
# ---------------------------------------------------------------------------


def test_default_model_id_is_bge_small_en_v1_5() -> None:
    """Issue #90: new default should point to BAAI/bge-small-en-v1.5.

    We inspect the constructor signature rather than instantiating to avoid
    a real HuggingFace download in this quick test.
    """
    sig = inspect.signature(EmbeddingIndex.__init__)
    default = sig.parameters["model_id"].default
    assert default == "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# T2: save/load round-trip preserves model_id
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_model_id(tmp_path: Path) -> None:
    custom_id = "custom-org/custom-model"
    idx = EmbeddingIndex(model_id=custom_id)

    with patch(
        "baseline_reporag.indexing.embedding.SentenceTransformer",
        return_value=_mock_sentence_transformer(dim=384),
    ):
        store = _FakeStore([_make_chunk("R::a.py::1-2")])
        idx.build(store, repo_id="R", repo_commit="abc")
        save_dir = tmp_path / "embedding"
        idx.save(save_dir)

    loaded = EmbeddingIndex.load(save_dir)
    # The private _model_id attribute round-trips via model_id.txt
    assert loaded._model_id == custom_id


# ---------------------------------------------------------------------------
# T3: expected_model_id matching is accepted
# ---------------------------------------------------------------------------


def test_load_with_expected_model_id_matching(tmp_path: Path) -> None:
    idx = EmbeddingIndex(model_id="BAAI/bge-small-en-v1.5")
    with patch(
        "baseline_reporag.indexing.embedding.SentenceTransformer",
        return_value=_mock_sentence_transformer(dim=384),
    ):
        store = _FakeStore([_make_chunk("R::a.py::1-2")])
        idx.build(store, repo_id="R", repo_commit="abc")
        save_dir = tmp_path / "embedding"
        idx.save(save_dir)

    loaded = EmbeddingIndex.load(save_dir, expected_model_id="BAAI/bge-small-en-v1.5")
    assert loaded._model_id == "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# T4: expected_model_id mismatch raises with actionable message
# ---------------------------------------------------------------------------


def test_load_with_expected_model_id_mismatch_raises(tmp_path: Path) -> None:
    idx = EmbeddingIndex(model_id="sentence-transformers/all-MiniLM-L6-v2")
    with patch(
        "baseline_reporag.indexing.embedding.SentenceTransformer",
        return_value=_mock_sentence_transformer(dim=384),
    ):
        store = _FakeStore([_make_chunk("R::a.py::1-2")])
        idx.build(store, repo_id="R", repo_commit="abc")
        save_dir = tmp_path / "embedding"
        idx.save(save_dir)

    with pytest.raises(ValueError) as excinfo:
        EmbeddingIndex.load(save_dir, expected_model_id="BAAI/bge-small-en-v1.5")

    msg = str(excinfo.value)
    assert "sentence-transformers/all-MiniLM-L6-v2" in msg
    assert "BAAI/bge-small-en-v1.5" in msg
    # The message must include actionable rebuild guidance.
    assert "rm -rf" in msg
    assert "scripts/build_indexes.py" in msg


# ---------------------------------------------------------------------------
# T5: omitting expected_model_id is backward compatible
# ---------------------------------------------------------------------------


def test_load_without_expected_model_id_is_backward_compatible(tmp_path: Path) -> None:
    idx = EmbeddingIndex(model_id="sentence-transformers/all-MiniLM-L6-v2")
    with patch(
        "baseline_reporag.indexing.embedding.SentenceTransformer",
        return_value=_mock_sentence_transformer(dim=384),
    ):
        store = _FakeStore([_make_chunk("R::a.py::1-2")])
        idx.build(store, repo_id="R", repo_commit="abc")
        save_dir = tmp_path / "embedding"
        idx.save(save_dir)

    loaded = EmbeddingIndex.load(save_dir)
    assert loaded._model_id == "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# T6 / T7: dimension-agnostic build/search/save/load
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [384, 768])
def test_embeddings_dimension_agnostic(tmp_path: Path, dim: int) -> None:
    idx = EmbeddingIndex(model_id="fake-model")
    with patch(
        "baseline_reporag.indexing.embedding.SentenceTransformer",
        return_value=_mock_sentence_transformer(dim=dim),
    ):
        store = _FakeStore(
            [_make_chunk("R::a.py::1-2"), _make_chunk("R::b.py::1-2", content="world")]
        )
        idx.build(store, repo_id="R", repo_commit="abc")
        assert idx._embeddings is not None
        assert idx._embeddings.shape == (2, dim)

        save_dir = tmp_path / "embedding"
        idx.save(save_dir)

        loaded = EmbeddingIndex.load(save_dir)
        assert loaded._embeddings is not None
        assert loaded._embeddings.shape == (2, dim)

        results = loaded.search("query text", top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id in {"R::a.py::1-2", "R::b.py::1-2"}


# ---------------------------------------------------------------------------
# T8: slow real-model smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("RUN_SLOW_TESTS") != "1",
    reason="slow real-model smoke test; set RUN_SLOW_TESTS=1 to run",
)
def test_bge_small_en_v1_5_loads_and_encodes() -> None:
    """Smoke test: real BAAI/bge-small-en-v1.5 loads via sentence-transformers."""
    idx = EmbeddingIndex(model_id="BAAI/bge-small-en-v1.5")
    model = idx._model_()
    vecs = model.encode(
        ["hello world"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 1
    assert vecs.shape[1] == 384  # bge-small-en-v1.5 output dim
