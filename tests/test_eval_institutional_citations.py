"""Tests for baseline_reporag.eval.institutional.citation_eval."""

from __future__ import annotations

import json
from pathlib import Path

from baseline_reporag.eval.institutional.citation_eval import (
    DictChunkLookup,
    Grade,
    grade_eval_set,
    grade_prediction,
)


def _lookup(data: dict[str, tuple[str, str]]) -> DictChunkLookup:
    return DictChunkLookup(data=data)


def test_layer1_chunk_id_match() -> None:
    lookup = _lookup({"c1": ("第3条 内容", "doc_a")})
    grade = grade_prediction(
        {"cited_chunk_ids": ["c1"], "no_citation": False},
        {"reference_chunk_ids": ["c1"], "source_document_id": "doc_a"},
        lookup,
    )
    assert grade is Grade.CORRECT_CHUNK


def test_layer2_pattern_regex_match_same_doc() -> None:
    lookup = _lookup({"c2": ("第3条第1項の規定による", "doc_a")})
    grade = grade_prediction(
        {"cited_chunk_ids": ["c2"], "no_citation": False},
        {
            "reference_chunk_ids": ["other"],
            "source_document_id": "doc_a",
            "expected_citation_patterns": ["第3条第1項"],
        },
        lookup,
    )
    assert grade is Grade.ARTICLE_LEVEL_CORRECT


def test_layer2_cross_doc_match_blocked() -> None:
    lookup = _lookup({"c3": ("第3条第1項の規定", "doc_b")})
    grade = grade_prediction(
        {"cited_chunk_ids": ["c3"], "no_citation": False},
        {
            "reference_chunk_ids": ["other"],
            "source_document_id": "doc_a",
            "expected_citation_patterns": ["第3条第1項"],
        },
        lookup,
    )
    assert grade is Grade.WRONG_CITATION


def test_article_regex_boundary_prefix_mismatch() -> None:
    lookup = _lookup({"c1": ("第10条 内容のみ", "doc_a")})
    grade = grade_prediction(
        {"cited_chunk_ids": ["c1"], "no_citation": False},
        {
            "reference_chunk_ids": ["other"],
            "source_document_id": "doc_a",
            "expected_citation_patterns": ["第1条"],
        },
        lookup,
    )
    assert grade is Grade.WRONG_CITATION


def test_wrong_citation_indices_pre_gate_overrides_layer1() -> None:
    lookup = _lookup({"c1": ("第3条", "doc_a")})
    grade = grade_prediction(
        {
            "cited_chunk_ids": ["c1"],
            "wrong_citation_indices": [99],
            "no_citation": False,
        },
        {"reference_chunk_ids": ["c1"], "source_document_id": "doc_a"},
        lookup,
    )
    assert grade is Grade.WRONG_CITATION


def test_no_citation_branch() -> None:
    lookup = _lookup({})
    grade = grade_prediction(
        {"cited_chunk_ids": [], "no_citation": True},
        {"reference_chunk_ids": ["c1"], "source_document_id": "doc_a"},
        lookup,
    )
    assert grade is Grade.NO_CITATION


def test_dict_chunk_lookup_defaults() -> None:
    lookup = _lookup({})
    assert lookup.get_chunk_text("missing") == ""
    assert lookup.get_doc_id("missing") == ""


def test_grade_eval_set_produces_expected_report(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "id": "INST-DEFINITION-001",
                "category": "definition",
                "reference_chunk_ids": ["c1"],
                "source_document_id": "doc_a",
                "expected_citation_patterns": ["第3条"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "INST-ARTICLE-LOOKUP-001",
                "category": "article_lookup",
                "reference_chunk_ids": ["cx"],
                "source_document_id": "doc_a",
                "expected_citation_patterns": ["第3条"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_log_path = tmp_path / "run.jsonl"
    run_log_path.write_text(
        json.dumps(
            {
                "eval_id": "INST-DEFINITION-001",
                "cited_chunk_ids": ["c1"],
                "wrong_citation_indices": [],
                "no_citation": False,
            }
        )
        + "\n"
        + json.dumps(
            {
                "eval_id": "INST-ARTICLE-LOOKUP-001",
                "cited_chunk_ids": ["c2"],
                "wrong_citation_indices": [],
                "no_citation": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    lookup = _lookup(
        {
            "c1": ("第3条 内容", "doc_a"),
            "c2": ("第3条 他の表現", "doc_a"),
        }
    )
    report = grade_eval_set(
        eval_set_path=eval_path, run_log_path=run_log_path, lookup=lookup
    )
    assert report["total"] == 2
    assert report["correct_chunk"] == 1
    assert report["article_level_correct"] == 1
    assert "per_category" in report
    assert "definition" in report["per_category"]
    assert report["input_mode"] == "run_log"


def test_grade_eval_set_predictions_fallback(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {"id": "Q1", "category": "definition", "reference_chunk_ids": ["c1"]}
        )
        + "\n",
        encoding="utf-8",
    )
    pred_path = tmp_path / "pred.jsonl"
    pred_path.write_text(
        json.dumps({"eval_id": "Q1", "cited_chunk_ids": ["c1"], "no_citation": False})
        + "\n",
        encoding="utf-8",
    )
    lookup = _lookup({"c1": ("body", "doc_a")})
    report = grade_eval_set(
        eval_set_path=eval_path, predictions_path=pred_path, lookup=lookup
    )
    assert report["input_mode"] == "predictions"
    assert report["total"] == 1
