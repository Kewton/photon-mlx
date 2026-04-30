"""Integration tests for Issue #177: refusal_score surfaced via QueryResult.

Verifies that compute_refusal_score integrates correctly with contracts and
that the pipeline sets refusal_score / refusal_matches on QueryResult.
"""

from __future__ import annotations

from baseline_reporag.citation import compute_refusal_score
from baseline_reporag.contracts import QueryResult


def _make_minimal_query_result(answer: str) -> "QueryResult":
    """Build a minimal QueryResult with refusal_score populated."""
    from unittest.mock import MagicMock

    latency = MagicMock()
    latency.total_ms = 100.0
    latency.retrieval_ms = 50.0
    latency.generation_ms = 50.0
    memory = MagicMock()
    memory.peak_mb = 256.0

    r_score, r_matches = compute_refusal_score(answer)
    return QueryResult(
        answer=answer,
        session_id="test-session",
        turn_id=1,
        cited_chunk_ids=[],
        wrong_citation_indices=[],
        no_citation=True,
        latency=latency,
        memory=memory,
        refusal_score=r_score,
        refusal_matches=r_matches,
    )


class TestRefusalScoreOnQueryResult:
    """QueryResult.refusal_score は拒絶文字列で 1.0、通常回答で 0.0 になること。"""

    def test_refusal_answer_score_is_one(self) -> None:
        result = _make_minimal_query_result(
            "根拠が不足しています。該当情報はありません。"
        )
        assert result.refusal_score == 1.0

    def test_normal_answer_score_is_zero(self) -> None:
        result = _make_minimal_query_result(
            "The router is defined in fastapi/cli.py [C:1]."
        )
        assert result.refusal_score == 0.0

    def test_refusal_matches_populated(self) -> None:
        result = _make_minimal_query_result("根拠が不足しています。")
        assert result.refusal_matches is not None
        assert len(result.refusal_matches) > 0
        assert "根拠が不足しています" in result.refusal_matches

    def test_normal_answer_matches_empty(self) -> None:
        result = _make_minimal_query_result("Here is the answer [C:1].")
        assert result.refusal_matches == []

    def test_refusal_score_none_default(self) -> None:
        """デフォルト None: 既存コールは refusal_score を省略できる。"""
        from unittest.mock import MagicMock

        latency = MagicMock()
        latency.total_ms = 100.0
        memory = MagicMock()
        memory.peak_mb = 256.0
        result = QueryResult(
            answer="test",
            session_id="s",
            turn_id=1,
            cited_chunk_ids=[],
            wrong_citation_indices=[],
            no_citation=True,
            latency=latency,
            memory=memory,
        )
        assert result.refusal_score is None
        assert result.refusal_matches is None

    def test_new_pattern_gaitou_jouhou(self) -> None:
        """Issue #177: 「該当する情報は含まれていません」が refusal_score=1.0 になること。"""
        result = _make_minimal_query_result("該当する情報は含まれていません。")
        assert result.refusal_score == 1.0

    def test_new_pattern_miatarimasen(self) -> None:
        """Issue #177: 「見当たりません」が refusal_score=1.0 になること。"""
        result = _make_minimal_query_result("コードチャンクには見当たりません。")
        assert result.refusal_score == 1.0


class TestEvalFallback:
    """run_multi_turn_eval.build_prediction_record の refusal_score フォールバック。"""

    def test_uses_pipeline_score_when_available(self) -> None:
        """result.refusal_score が設定されている場合はそちらを使う。"""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_multi_turn_eval import build_prediction_record

        result = _make_minimal_query_result("根拠が不足しています。")
        row = build_prediction_record(
            session_id="s", turn_id=1, question="q?", result=result
        )
        assert row["is_refusal"] is True

    def test_fallback_when_score_is_none(self) -> None:
        """refusal_score=None のとき is_refusal_answer() フォールバックが動くこと。"""
        import sys
        from pathlib import Path
        from unittest.mock import MagicMock

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from run_multi_turn_eval import build_prediction_record

        latency = MagicMock()
        latency.total_ms = 100.0
        latency.retrieval_ms = 50.0
        latency.generation_ms = 50.0
        memory = MagicMock()
        memory.peak_mb = 256.0
        result = QueryResult(
            answer="根拠が不足しています。",
            session_id="s",
            turn_id=1,
            cited_chunk_ids=[],
            wrong_citation_indices=[],
            no_citation=True,
            latency=latency,
            memory=memory,
            refusal_score=None,
        )
        row = build_prediction_record(
            session_id="s", turn_id=1, question="q?", result=result
        )
        assert row["is_refusal"] is True
