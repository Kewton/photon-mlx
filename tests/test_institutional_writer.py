"""Tests for baseline_reporag.eval.institutional.writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline_reporag.eval.institutional.writer import (
    append_jsonl,
    read_existing_ids,
    validate_record,
    write_generation_summary,
    write_jsonl,
)


def _valid_record(row_id: str = "INST-DEFINITION-001") -> dict:
    return {
        "id": row_id,
        "category": "definition",
        "difficulty": "medium",
        "question": "Q?",
        "reference_answer": "A",
        "reference_chunk_ids": ["c1"],
        "grading_notes": "note",
        "rubric": {
            "correctness": {"max": 2, "notes": ""},
            "grounding": {"max": 2, "notes": ""},
            "usefulness": {"max": 1, "notes": ""},
        },
        "answerable": True,
    }


def test_validate_record_passes_on_minimal() -> None:
    validate_record(_valid_record())


def test_validate_record_raises_on_missing_required() -> None:
    rec = _valid_record()
    del rec["rubric"]
    with pytest.raises(AssertionError):
        validate_record(rec)


def test_validate_record_rejects_unknown_key() -> None:
    rec = _valid_record()
    rec["foo"] = "bar"
    with pytest.raises(AssertionError):
        validate_record(rec)


@pytest.mark.parametrize("key", ["question", "reference_answer"])
@pytest.mark.parametrize("empty_value", ["", "   ", "\t\n"])
def test_validate_record_rejects_empty_required_strings(
    key: str, empty_value: str
) -> None:
    rec = _valid_record()
    rec[key] = empty_value
    with pytest.raises(AssertionError):
        validate_record(rec)


@pytest.mark.parametrize(
    "pattern",
    ["第3条", "第3条第1項", "第3条第1項第2号", "第10条", "第100条第2項第99号"],
)
def test_citation_pattern_fullmatch_valid(pattern: str) -> None:
    rec = _valid_record()
    rec["expected_citation_patterns"] = [pattern]
    validate_record(rec)


@pytest.mark.parametrize(
    "pattern",
    ["第3条第1項第2号追加", "第3条の2", "Article 3", "", "第"],
)
def test_citation_pattern_fullmatch_invalid(pattern: str) -> None:
    rec = _valid_record()
    rec["expected_citation_patterns"] = [pattern]
    with pytest.raises(AssertionError):
        validate_record(rec)


def test_append_jsonl_writes_atomic(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    append_jsonl(path, _valid_record("INST-DEFINITION-001"))
    append_jsonl(path, _valid_record("INST-DEFINITION-002"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == ["INST-DEFINITION-001", "INST-DEFINITION-002"]


def test_write_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    write_jsonl(path, [_valid_record("A-001"), _valid_record("A-002")])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_append_jsonl_loadable_by_existing_consumer(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    rec = _valid_record("INST-X-001")
    rec["expected_citation_patterns"] = ["第3条"]
    rec["source_document_id"] = "doc_a"
    append_jsonl(path, rec)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["id"] == "INST-X-001"
    assert rows[0]["category"] == "definition"
    assert rows[0]["question"] == "Q?"


def test_read_existing_ids(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    write_jsonl(path, [_valid_record("A-001"), _valid_record("A-002")])
    assert read_existing_ids(path) == {"A-001", "A-002"}


def test_read_existing_ids_missing_returns_empty(tmp_path: Path) -> None:
    assert read_existing_ids(tmp_path / "nope.jsonl") == set()


def test_write_generation_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "reports" / "summary.json"
    write_generation_summary(summary_path, {"provider": "openai", "succeeded": 10})
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["succeeded"] == 10
