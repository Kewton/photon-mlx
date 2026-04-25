from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from baseline_reporag.indexing.embedding import (
    EmbeddingIndex,
    _e5_passage_prefix,
    _e5_query_prefix,
)


class _FakeChunk(NamedTuple):
    chunk_id: str
    rel_path: str
    section_header: str
    content: str


class _FakeStore:
    def __init__(self, chunks: Iterable[_FakeChunk]) -> None:
        self._chunks = list(chunks)

    def iter_repo(self, repo_id: str, repo_commit: str) -> Iterator[_FakeChunk]:
        yield from self._chunks


def _make_fake_store(rows: Iterable[tuple[str, str, str, str]]) -> _FakeStore:
    return _FakeStore(_FakeChunk(*r) for r in rows)


class TestE5PrefixHelpers:
    def test_passage_prefix_applies_for_e5(self) -> None:
        result = _e5_passage_prefix("intfloat/multilingual-e5-small", ["hello"])
        assert result == ["passage: hello"]

    def test_passage_prefix_skips_for_non_e5(self) -> None:
        result = _e5_passage_prefix("sentence-transformers/all-MiniLM-L6-v2", ["hello"])
        assert result == ["hello"]

    def test_query_prefix_applies_for_e5_base(self) -> None:
        assert _e5_query_prefix("intfloat/multilingual-e5-base", "q") == "query: q"

    def test_query_prefix_skips_for_non_e5(self) -> None:
        assert _e5_query_prefix("BAAI/bge-small-en", "q") == "q"

    def test_passage_prefix_returns_unchanged_for_empty_model_id(self) -> None:
        assert _e5_passage_prefix("", ["hello"]) == ["hello"]

    def test_query_prefix_returns_unchanged_for_empty_model_id(self) -> None:
        assert _e5_query_prefix("", "hello") == "hello"

    # DR1-001 / DR2-001 / DR4-006: E5 instruct fail-fast かつ ValueError 本文に
    # model_id を含めない (private HF org 名 / R&D codename の leak 防止)
    def test_passage_prefix_raises_for_e5_instruct(self) -> None:
        with pytest.raises(ValueError, match="E5 instruct variants") as exc_info:
            _e5_passage_prefix("intfloat/multilingual-e5-large-instruct", ["hello"])
        # DR4-006: model_id (private HF org 名等) を error 本文へ含めない
        assert "intfloat" not in str(exc_info.value)
        assert "multilingual-e5-large-instruct" not in str(exc_info.value)

    def test_query_prefix_raises_for_e5_instruct(self) -> None:
        with pytest.raises(ValueError, match="E5 instruct variants") as exc_info:
            _e5_query_prefix("intfloat/multilingual-e5-large-instruct", "hello")
        # DR4-006: model_id (private HF org 名等) を error 本文へ含めない
        assert "intfloat" not in str(exc_info.value)
        assert "multilingual-e5-large-instruct" not in str(exc_info.value)


class TestEmbeddingIndexBuildE5Prefix:
    @patch("baseline_reporag.indexing.embedding.SentenceTransformer")
    def test_build_passes_passage_prefix_to_encode_for_e5(self, mock_st) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((2, 384), dtype=np.float32)
        mock_st.return_value = mock_model
        idx = EmbeddingIndex(model_id="intfloat/multilingual-e5-small")
        store = _make_fake_store(
            [
                ("c1", "doc1.md", "header", "body1"),
                ("c2", "doc2.md", "header", "body2"),
            ]
        )
        idx.build(store, repo_id="r", repo_commit="abc")
        encoded_texts = mock_model.encode.call_args[0][0]
        assert all(t.startswith("passage: ") for t in encoded_texts)

    @patch("baseline_reporag.indexing.embedding.SentenceTransformer")
    def test_build_passes_no_prefix_to_encode_for_non_e5(self, mock_st) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        mock_st.return_value = mock_model
        idx = EmbeddingIndex(model_id="sentence-transformers/all-MiniLM-L6-v2")
        store = _make_fake_store([("c1", "doc1.md", "header", "body1")])
        idx.build(store, repo_id="r", repo_commit="abc")
        encoded_texts = mock_model.encode.call_args[0][0]
        assert all(not t.startswith("passage: ") for t in encoded_texts)

    @patch("baseline_reporag.indexing.embedding.SentenceTransformer")
    def test_build_applies_prefix_after_truncate(self, mock_st) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        mock_st.return_value = mock_model
        long_content = "a" * 2050
        idx = EmbeddingIndex(model_id="intfloat/multilingual-e5-small")
        store = _make_fake_store([("c1", "", "", long_content)])
        idx.build(store, repo_id="r", repo_commit="abc")
        encoded_text = mock_model.encode.call_args[0][0][0]
        assert encoded_text.startswith("passage: ")
        # prefix(9) + truncated body(2048) = 2057 chars
        assert len(encoded_text) == 9 + 2048


class TestEmbeddingIndexSearchE5Prefix:
    @patch("baseline_reporag.indexing.embedding.SentenceTransformer")
    def test_search_passes_query_prefix_to_encode_for_e5(self, mock_st) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384], dtype=np.float32)
        mock_st.return_value = mock_model
        idx = EmbeddingIndex(model_id="intfloat/multilingual-e5-small")
        idx._embeddings = np.zeros((1, 384), dtype=np.float32)
        idx._chunk_ids = ["c1"]
        idx.search("hello")
        encoded_query = mock_model.encode.call_args[0][0]
        assert encoded_query == ["query: hello"]

    @patch("baseline_reporag.indexing.embedding.SentenceTransformer")
    def test_search_passes_no_prefix_to_encode_for_non_e5(self, mock_st) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384], dtype=np.float32)
        mock_st.return_value = mock_model
        idx = EmbeddingIndex(model_id="sentence-transformers/all-MiniLM-L6-v2")
        idx._embeddings = np.zeros((1, 384), dtype=np.float32)
        idx._chunk_ids = ["c1"]
        idx.search("hello")
        encoded_query = mock_model.encode.call_args[0][0]
        assert encoded_query == ["hello"]
