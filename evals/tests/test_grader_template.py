from __future__ import annotations

import json
from unittest.mock import patch

from evals.grader_template import (
    build_grader_messages,
    compute_aggregate,
    grade_one,
)


# ------------------------------------------------------------------
# build_grader_messages
# ------------------------------------------------------------------


class TestBuildGraderMessages:
    def test_returns_system_and_user_messages(self):
        msgs = build_grader_messages("q", "ref", "notes", "ans", ["c1"])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_message_contains_question(self):
        msgs = build_grader_messages("my question?", "", "", "ans", [])
        assert "my question?" in msgs[1]["content"]


# ------------------------------------------------------------------
# grade_one (mock LLM)
# ------------------------------------------------------------------


class TestGradeOne:
    def test_returns_parsed_scores(self):
        mock_response = json.dumps(
            {
                "correctness": 2,
                "grounding": 1,
                "usefulness": 1,
                "reason": "Good answer",
            }
        )
        with patch("evals.grader_template._call_llm_judge", return_value=mock_response):
            result = grade_one(
                question="What is X?",
                reference_answer="X is Y.",
                grading_notes="",
                answer="X is Y.",
                cited_chunk_ids=["c1"],
            )
        assert result["correctness"] == 2
        assert result["grounding"] == 1
        assert result["usefulness"] == 1
        assert "reason" in result

    def test_handles_json_in_markdown_fence(self):
        mock_response = '```json\n{"correctness": 1, "grounding": 0, "usefulness": 0, "reason": "bad"}\n```'
        with patch("evals.grader_template._call_llm_judge", return_value=mock_response):
            result = grade_one("q", "", "", "a", [])
        assert result["correctness"] == 1

    def test_passes_model_id(self):
        mock_response = json.dumps(
            {"correctness": 0, "grounding": 0, "usefulness": 0, "reason": "n/a"}
        )
        with patch(
            "evals.grader_template._call_llm_judge", return_value=mock_response
        ) as mock:
            grade_one("q", "", "", "a", [], model_id="test-model")
            call_args = mock.call_args
            assert (
                call_args[1].get("model_id") == "test-model"
                or call_args[0][1] == "test-model"
            )


# ------------------------------------------------------------------
# compute_aggregate (dynamic rubric)
# ------------------------------------------------------------------


class TestComputeAggregate:
    def test_basic_3dim(self):
        scores = [
            {"correctness": 2, "grounding": 2, "usefulness": 1},
            {"correctness": 0, "grounding": 0, "usefulness": 0},
        ]
        agg = compute_aggregate(scores)
        assert agg["n"] == 2
        assert agg["correctness_mean"] == 1.0
        assert agg["grounding_mean"] == 1.0
        assert agg["usefulness_mean"] == 0.5

    def test_total_100_uses_dynamic_max(self):
        scores = [{"correctness": 2, "grounding": 2, "usefulness": 1}]
        rubric = {
            "correctness": {"min": 0, "max": 2},
            "grounding": {"min": 0, "max": 2},
            "usefulness": {"min": 0, "max": 1},
        }
        agg = compute_aggregate(scores, rubric=rubric)
        # total = 5, max = 5 → 100.0
        assert agg["total_100"] == 100.0

    def test_4dim_with_session_consistency(self):
        scores = [
            {
                "correctness": 2,
                "grounding": 2,
                "usefulness": 1,
                "session_consistency": 1.0,
            }
        ]
        rubric = {
            "correctness": {"min": 0, "max": 2},
            "grounding": {"min": 0, "max": 2},
            "usefulness": {"min": 0, "max": 1},
            "session_consistency": {"min": 0, "max": 1},
        }
        agg = compute_aggregate(scores, rubric=rubric)
        # total = 6, max = 6 → 100.0
        assert agg["total_100"] == 100.0
        assert "session_consistency_mean" in agg

    def test_empty_scores(self):
        agg = compute_aggregate([])
        assert agg == {}

    def test_rubric_none_falls_back_to_default(self):
        scores = [{"correctness": 2, "grounding": 2, "usefulness": 1}]
        agg = compute_aggregate(scores)
        # default rubric: max=2+2+1=5 → total_100 = 5/5*100 = 100
        assert agg["total_100"] == 100.0
