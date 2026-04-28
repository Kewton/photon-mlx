"""Tests for baseline_reporag.eval.institutional.generator."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from baseline_reporag.eval.institutional.corpus import DocIndex
from baseline_reporag.eval.institutional.generator import (
    GenerationFailure,
    generate_question,
)


class _FakeLLMClient:
    name = "fake"
    model = "fake-model"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        seed: int | None = 42,
        response_format: str = "json_object",
    ) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise RuntimeError("no more fake responses")
        return self._responses.pop(0)


def _make_doc(tmp_path: Path) -> DocIndex:
    doc_dir = tmp_path / "0001_example"
    doc_dir.mkdir()
    doc_path = doc_dir / "document.md"
    doc_path.write_text("第1条 これは例です。\n第2条 定義。", encoding="utf-8")
    return DocIndex(
        doc_id="0001_example",
        path=doc_path,
        has_articles=True,
        has_penalty=False,
        has_exception=False,
        metadata={"title": "Example Law"},
    )


def test_generate_question_returns_valid_row(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    payload = json.dumps(
        {
            "question": "第1条は何を定めているか？",
            "reference_answer": "例の規定",
            "expected_citation_patterns": ["第1条"],
            "grading_notes": "第1条を引用していること",
        },
        ensure_ascii=False,
    )
    client = _FakeLLMClient([payload])
    row = generate_question(doc=doc, category="definition", seq=1, client=client)
    assert row["id"] == "INST-DEFINITION-001"
    assert row["category"] == "definition"
    assert row["source_document_id"] == "0001_example"
    assert row["expected_citation_patterns"] == ["第1条"]
    assert row["answerable"] is True
    assert row["generator_model"] == "fake-model"
    assert row["human_verified"] is False


def test_generate_question_id_format_article_lookup(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    payload = json.dumps(
        {
            "question": "Q",
            "reference_answer": "A",
            "expected_citation_patterns": [],
            "grading_notes": "",
        }
    )
    client = _FakeLLMClient([payload])
    row = generate_question(doc=doc, category="article_lookup", seq=7, client=client)
    assert row["id"] == "INST-ARTICLE-LOOKUP-007"


def test_generate_question_retries_on_missing_keys(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    good = json.dumps(
        {
            "question": "Q",
            "reference_answer": "A",
            "expected_citation_patterns": [],
            "grading_notes": "G",
        }
    )
    client = _FakeLLMClient(["not json", '{"question": "only"}', good])
    row = generate_question(
        doc=doc, category="overview", seq=1, client=client, sleep_fn=lambda _s: None
    )
    assert row["question"] == "Q"
    assert len(client.calls) == 3


def test_generate_question_fails_after_three_failures(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    client = _FakeLLMClient(["bad1", "bad2", "bad3"])
    with pytest.raises(GenerationFailure):
        generate_question(
            doc=doc, category="overview", seq=1, client=client, sleep_fn=lambda _s: None
        )


def test_generate_question_strips_code_fence(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    fenced = (
        "```json\n"
        + json.dumps(
            {
                "question": "Q",
                "reference_answer": "A",
                "expected_citation_patterns": [],
                "grading_notes": "G",
            }
        )
        + "\n```"
    )
    client = _FakeLLMClient([fenced])
    row = generate_question(
        doc=doc, category="scope", seq=1, client=client, sleep_fn=lambda _s: None
    )
    assert row["question"] == "Q"


def test_generate_question_handles_empty_document(tmp_path: Path) -> None:
    doc_dir = tmp_path / "empty"
    doc_dir.mkdir()
    doc_path = doc_dir / "document.md"
    doc_path.write_text("", encoding="utf-8")
    doc = DocIndex(
        doc_id="empty",
        path=doc_path,
        has_articles=False,
        has_penalty=False,
        has_exception=False,
        metadata={},
    )
    payload = json.dumps(
        {
            "question": "Q",
            "reference_answer": "A",
            "expected_citation_patterns": [],
            "grading_notes": "G",
        }
    )
    client = _FakeLLMClient([payload])
    row = generate_question(
        doc=doc, category="overview", seq=1, client=client, sleep_fn=lambda _s: None
    )
    assert row["source_document_id"] == "empty"


def test_doc_index_is_frozen(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    with pytest.raises(Exception):
        doc.doc_id = "mutated"  # type: ignore[misc]
    copy = replace(doc, doc_id="renamed")
    assert copy.doc_id == "renamed"
    assert doc.doc_id == "0001_example"
