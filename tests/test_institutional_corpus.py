"""Tests for baseline_reporag.eval.institutional.corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline_reporag.eval.institutional.corpus import (
    DocIndex,
    build_context,
    build_doc_index,
    iter_article_docs,
    iter_section_docs,
)


def _write_doc(root: Path, name: str, body: str, metadata: dict | None = None) -> Path:
    doc_dir = root / name
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "document.md").write_text(body, encoding="utf-8")
    if metadata is not None:
        (doc_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
    return doc_dir


@pytest.fixture
def mock_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    _write_doc(
        root,
        "0001_rental",
        "第1条 この法律の目的は賃貸住宅の適正管理にある。\n第2条 定義\n罰則: 第30条",
        {
            "title": "賃貸住宅管理業法",
            "preamble": "国土交通省",
            "effective_date": "2021-06-15",
        },
    )
    _write_doc(
        root,
        "0002_overview",
        "本書は事業概観です。条文は含まれません。",
        {"title": "事業概観パンフレット"},
    )
    _write_doc(
        root,
        "0003_exception",
        "第1条の但書として、経過措置を第3条で定める。",
        {"title": "経過措置令"},
    )
    return root


def test_build_doc_index_returns_list_of_doc_index(mock_corpus: Path) -> None:
    index = build_doc_index(mock_corpus)
    assert len(index) == 3
    assert all(isinstance(d, DocIndex) for d in index)
    by_id = {d.doc_id: d for d in index}
    assert by_id["0001_rental"].has_articles is True
    assert by_id["0001_rental"].has_penalty is True
    assert by_id["0002_overview"].has_articles is False
    assert by_id["0003_exception"].has_exception is True


def test_build_doc_index_handles_missing_metadata(tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    _write_doc(root, "only", "第1条 テスト本文")
    index = build_doc_index(root)
    assert len(index) == 1
    assert index[0].metadata == {}


def test_build_doc_index_returns_empty_for_missing_root(tmp_path: Path) -> None:
    assert build_doc_index(tmp_path / "missing") == []


def test_iter_article_docs_filters_article_bearing(mock_corpus: Path) -> None:
    docs = list(iter_article_docs(mock_corpus))
    ids = {d.doc_id for d in docs}
    assert "0001_rental" in ids
    assert "0003_exception" in ids
    assert "0002_overview" not in ids


def test_iter_section_docs_keyword_filter(mock_corpus: Path) -> None:
    docs = list(iter_section_docs(mock_corpus, "罰則"))
    assert [d.doc_id for d in docs] == ["0001_rental"]


def test_build_context_truncates_and_injects_metadata() -> None:
    body = "X" * 10_000
    ctx = build_context(
        {"title": "Law A", "preamble": "Preamble", "effective_date": "2020-01-01"},
        body,
        max_chars=100,
    )
    assert "タイトル: Law A" in ctx
    assert "施行日: 2020-01-01" in ctx
    assert "前文: Preamble" in ctx
    assert ctx.count("X") == 100


def test_build_context_no_metadata_returns_body() -> None:
    ctx = build_context({}, "body only", max_chars=1000)
    assert ctx == "body only"
