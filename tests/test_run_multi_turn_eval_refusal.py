"""Refusal-aware output for ``scripts/run_multi_turn_eval.py`` (Issue #156).

The eval script previously emitted only ``no_citation`` per turn, which
forced raw NC to be misread as the true failure rate. These tests pin the
extended JSONL schema (``is_refusal`` / ``true_failure``) and the
refusal-aware run summary so downstream readers can separate legitimate
abstentions ("根拠が不足しています") from real hallucination misses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline_reporag.contracts import QueryResult  # noqa: E402
from baseline_reporag.profiler import LatencyBreakdown, MemorySnapshot  # noqa: E402
from scripts.regrade_eval_with_refusal import regrade_predictions  # noqa: E402
from scripts.run_multi_turn_eval import (  # noqa: E402
    build_prediction_record,
    summarize_run_predictions,
)


def _make_result(answer: str, cited: list[str], total_ms: float = 100.0) -> QueryResult:
    return QueryResult(
        answer=answer,
        session_id="s1",
        turn_id=1,
        cited_chunk_ids=list(cited),
        wrong_citation_indices=[],
        no_citation=not cited,
        latency=LatencyBreakdown(total_ms=total_ms),
        memory=MemorySnapshot(peak_mb=0.0, current_mb=0.0),
    )


# ---------------------------------------------------------------------------
# build_prediction_record
# ---------------------------------------------------------------------------


def test_build_prediction_record_marks_refusal_and_clears_true_failure() -> None:
    """A no-cite refusal answer must be flagged ``is_refusal`` only."""
    result = _make_result("根拠が不足しています。情報がありません。", cited=[])
    rec = build_prediction_record(
        session_id="s1", turn_id=2, question="Q?", result=result
    )
    assert rec["no_citation"] is True
    assert rec["is_refusal"] is True
    assert rec["true_failure"] is False


def test_build_prediction_record_marks_true_failure_when_no_cite_and_not_refusal() -> (
    None
):
    """A no-cite non-refusal answer is the only thing that should count as a true failure."""
    result = _make_result("Some answer that hallucinates without citation.", cited=[])
    rec = build_prediction_record(
        session_id="s1", turn_id=3, question="Q?", result=result
    )
    assert rec["no_citation"] is True
    assert rec["is_refusal"] is False
    assert rec["true_failure"] is True


def test_build_prediction_record_with_citations_is_neither_refusal_nor_failure() -> (
    None
):
    result = _make_result("Cited answer [C:1]", cited=["chunk-1"])
    rec = build_prediction_record(
        session_id="s1", turn_id=4, question="Q?", result=result
    )
    assert rec["no_citation"] is False
    assert rec["is_refusal"] is False
    assert rec["true_failure"] is False
    assert rec["cited_chunk_ids"] == ["chunk-1"]


# ---------------------------------------------------------------------------
# summarize_run_predictions
# ---------------------------------------------------------------------------


def test_summarize_run_predictions_separates_refusal_from_true_failure() -> None:
    rows = [
        # cited
        {
            "no_citation": False,
            "is_refusal": False,
            "true_failure": False,
            "latency_ms": 100.0,
        },
        # legitimate refusal
        {
            "no_citation": True,
            "is_refusal": True,
            "true_failure": False,
            "latency_ms": 200.0,
        },
        # true failure (no cite, not a refusal)
        {
            "no_citation": True,
            "is_refusal": False,
            "true_failure": True,
            "latency_ms": 300.0,
        },
    ]
    summary = summarize_run_predictions(rows)
    assert summary["total_turns"] == 3
    assert summary["no_citation_turns"] == 2
    assert summary["refusal_turns"] == 1
    assert summary["true_failure_turns"] == 1
    assert summary["raw_nc_rate"] == 2 / 3
    assert summary["true_failure_rate"] == 1 / 3


def test_summarize_run_predictions_recomputes_refusal_when_field_missing() -> None:
    """Older predictions JSONL (pre-Issue #156) lack ``is_refusal``; the
    refusal phrase must still be recovered from ``answer`` so a regrade
    over those logs reports the correct true-failure rate."""
    rows = [
        # legacy row: no is_refusal/true_failure, but answer is a refusal
        {"no_citation": True, "answer": "根拠が不足しています"},
        # legacy row: no_citation true and answer is *not* a refusal
        {"no_citation": True, "answer": "Hallucinated answer."},
        # legacy row: cited
        {"no_citation": False, "answer": "Cited [C:1]"},
    ]
    summary = summarize_run_predictions(rows)
    assert summary["total_turns"] == 3
    assert summary["no_citation_turns"] == 2
    assert summary["refusal_turns"] == 1
    assert summary["true_failure_turns"] == 1


def test_summarize_run_predictions_empty_returns_zero_rates() -> None:
    summary = summarize_run_predictions([])
    assert summary["total_turns"] == 0
    assert summary["raw_nc_rate"] == 0.0
    assert summary["true_failure_rate"] == 0.0
    assert summary["refusal_turns"] == 0


# ---------------------------------------------------------------------------
# regrade_eval_with_refusal
# ---------------------------------------------------------------------------


def test_regrade_predictions_recovers_refusal_from_legacy_answer(
    tmp_path: Path,
) -> None:
    """Legacy predictions JSONL (no is_refusal field) regrades to the
    same true-failure rate that #135 Phase 7 worker found by hand:
    raw NC counts every blank-cite turn, but refusal phrases drop out
    so true_failure_rate < raw_nc_rate."""
    pred_path = tmp_path / "legacy.predictions.jsonl"
    legacy_rows = [
        {"no_citation": True, "answer": "根拠が不足しています。", "latency_ms": 100.0},
        {"no_citation": True, "answer": "Hallucinated answer.", "latency_ms": 200.0},
        {"no_citation": False, "answer": "Cited [C:1]", "latency_ms": 150.0},
    ]
    pred_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in legacy_rows) + "\n",
        encoding="utf-8",
    )

    summary = regrade_predictions(pred_path)

    assert summary["total_turns"] == 3
    assert summary["no_citation_turns"] == 2
    assert summary["refusal_turns"] == 1
    assert summary["true_failure_turns"] == 1
    assert summary["source"] == "legacy.predictions.jsonl"
