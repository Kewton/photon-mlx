"""Tests for baseline_reporag/comparison.py (Issue #179)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from baseline_reporag.comparison import (
    ComparisonResult,
    DeltaResult,
    VariantResult,
    compare,
    compute_delta,
    run_variant_with_pipeline,
)


def _make_mock_pipeline(
    answer: str = "test answer",
    cited: list[str] | None = None,
    no_citation: bool = False,
    total_ms: float = 1000.0,
    retrieval_ms: float = 100.0,
    generation_ms: float = 900.0,
    peak_mb: float = 50.0,
) -> MagicMock:
    """Build a minimal mock pipeline whose .query() returns a QueryResult-like object."""
    from baseline_reporag.contracts import QueryResult
    from baseline_reporag.profiler import LatencyBreakdown, MemorySnapshot

    result = QueryResult(
        answer=answer,
        session_id="test-session",
        turn_id=1,
        cited_chunk_ids=["c1"] if cited is None else cited,
        wrong_citation_indices=[],
        no_citation=no_citation,
        latency=LatencyBreakdown(
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            citation_ms=0.0,
            total_ms=total_ms,
        ),
        memory=MemorySnapshot(peak_mb=peak_mb, current_mb=25.0),
        citation_postprocessed=False,
    )
    pipeline = MagicMock()
    pipeline.query.return_value = result
    return pipeline


class TestVariantResult:
    def test_dataclass_fields(self) -> None:
        vr = VariantResult(
            variant_id="baseline",
            config_path="configs/baseline.yaml",
            answer="hello",
            cited_chunk_ids=["c1"],
            no_citation=False,
            latency_total_ms=1000.0,
            latency_retrieval_ms=100.0,
            latency_generation_ms=900.0,
            memory_peak_mb=50.0,
        )
        assert vr.variant_id == "baseline"
        assert vr.answer == "hello"


class TestRunVariantWithPipeline:
    def test_maps_query_result_to_variant_result(self) -> None:
        pipeline = _make_mock_pipeline(
            answer="The answer.",
            cited=["c1", "c2"],
            total_ms=2000.0,
            retrieval_ms=200.0,
            generation_ms=1800.0,
            peak_mb=60.0,
        )
        result = run_variant_with_pipeline(
            pipeline=pipeline,
            question="What?",
            session_id="sess-1",
            repo_id="my_repo",
            variant_id="baseline",
            config_path="configs/baseline.yaml",
        )
        assert isinstance(result, VariantResult)
        assert result.variant_id == "baseline"
        assert result.answer == "The answer."
        assert result.cited_chunk_ids == ["c1", "c2"]
        assert result.no_citation is False
        assert result.latency_total_ms == 2000.0
        assert result.latency_retrieval_ms == 200.0
        assert result.latency_generation_ms == 1800.0
        assert result.memory_peak_mb == 60.0
        pipeline.query.assert_called_once_with(
            question="What?", session_id="sess-1", repo_id="my_repo"
        )

    def test_no_citation_flag_preserved(self) -> None:
        pipeline = _make_mock_pipeline(no_citation=True, cited=[])
        result = run_variant_with_pipeline(
            pipeline=pipeline,
            question="Q?",
            session_id="s",
            repo_id="r",
            variant_id="photon",
        )
        assert result.no_citation is True
        assert result.cited_chunk_ids == []

    def test_config_path_defaults_to_empty(self) -> None:
        pipeline = _make_mock_pipeline()
        result = run_variant_with_pipeline(
            pipeline=pipeline,
            question="Q?",
            session_id="s",
            repo_id="r",
            variant_id="baseline",
        )
        assert result.config_path == ""


class TestComputeDelta:
    def _make_variant(
        self,
        variant_id: str,
        latency_ms: float,
        cited: list[str],
    ) -> VariantResult:
        return VariantResult(
            variant_id=variant_id,
            config_path="",
            answer="",
            cited_chunk_ids=cited,
            no_citation=False,
            latency_total_ms=latency_ms,
            latency_retrieval_ms=0.0,
            latency_generation_ms=latency_ms,
            memory_peak_mb=0.0,
        )

    def test_latency_delta_and_pct(self) -> None:
        baseline = self._make_variant("baseline", 2000.0, ["c1"])
        photon = self._make_variant("photon", 1000.0, ["c1"])
        delta = compute_delta(baseline, photon)
        assert delta.latency_delta_ms == pytest.approx(-1000.0)
        assert delta.latency_delta_pct == pytest.approx(-50.0)

    def test_cited_overlap_jaccard_full(self) -> None:
        baseline = self._make_variant("baseline", 1000.0, ["c1", "c2"])
        photon = self._make_variant("photon", 800.0, ["c1", "c2"])
        delta = compute_delta(baseline, photon)
        assert delta.cited_overlap_jaccard == pytest.approx(1.0)

    def test_cited_overlap_jaccard_no_overlap(self) -> None:
        baseline = self._make_variant("baseline", 1000.0, ["c1"])
        photon = self._make_variant("photon", 800.0, ["c2"])
        delta = compute_delta(baseline, photon)
        assert delta.cited_overlap_jaccard == pytest.approx(0.0)

    def test_cited_overlap_jaccard_partial(self) -> None:
        baseline = self._make_variant("baseline", 1000.0, ["c1", "c2"])
        photon = self._make_variant("photon", 800.0, ["c2", "c3"])
        delta = compute_delta(baseline, photon)
        # intersection={c2}, union={c1,c2,c3} → 1/3
        assert delta.cited_overlap_jaccard == pytest.approx(1 / 3)

    def test_cited_overlap_both_empty(self) -> None:
        baseline = self._make_variant("baseline", 1000.0, [])
        photon = self._make_variant("photon", 800.0, [])
        delta = compute_delta(baseline, photon)
        assert delta.cited_overlap_jaccard == pytest.approx(0.0)

    def test_latency_delta_zero_baseline(self) -> None:
        baseline = self._make_variant("baseline", 0.0, [])
        photon = self._make_variant("photon", 500.0, [])
        delta = compute_delta(baseline, photon)
        assert delta.latency_delta_ms == pytest.approx(500.0)
        assert delta.latency_delta_pct == pytest.approx(0.0)


class TestCompare:
    def test_returns_comparison_result(self) -> None:
        baseline_pipeline = _make_mock_pipeline(
            answer="baseline answer", total_ms=2000.0
        )
        photon_pipeline = _make_mock_pipeline(answer="photon answer", total_ms=1000.0)
        result = compare(
            baseline_pipeline=baseline_pipeline,
            photon_pipeline=photon_pipeline,
            question="What?",
            repo_id="repo",
            baseline_session_id="sess-b",
            photon_session_id="sess-p",
        )
        assert isinstance(result, ComparisonResult)
        assert result.question == "What?"
        assert result.baseline.variant_id == "baseline"
        assert result.photon.variant_id == "photon"
        assert result.baseline.answer == "baseline answer"
        assert result.photon.answer == "photon answer"
        assert isinstance(result.delta, DeltaResult)
        assert result.delta.latency_delta_ms == pytest.approx(-1000.0)

    def test_both_pipelines_queried(self) -> None:
        bp = _make_mock_pipeline()
        pp = _make_mock_pipeline()
        compare(
            baseline_pipeline=bp,
            photon_pipeline=pp,
            question="Q?",
            repo_id="r",
            baseline_session_id="sb",
            photon_session_id="sp",
        )
        bp.query.assert_called_once()
        pp.query.assert_called_once()
