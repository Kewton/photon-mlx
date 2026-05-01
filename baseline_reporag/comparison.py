"""Comparison utilities for baseline vs PHOTON side-by-side evaluation (Issue #179).

Public API:
    run_variant_with_pipeline — low-level: execute one pipeline variant, return VariantResult
    compare                   — high-level: sequential baseline+PHOTON execution + delta

Private:
    _build_and_run — CLI/scripts wrapper: build pipeline from config_path, then delegate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class VariantResult:
    """Result of a single pipeline variant execution."""

    variant_id: str
    config_path: str
    answer: str
    cited_chunk_ids: list[str]
    no_citation: bool
    latency_total_ms: float
    latency_retrieval_ms: float
    latency_generation_ms: float
    memory_peak_mb: float
    generator_used: str | None = None
    generator_fallback_reason: str | None = None
    retrieval_debug: list[Any] | None = None
    model_id: str | None = None


@dataclass
class DeltaResult:
    """Delta metrics between baseline and PHOTON variants."""

    latency_delta_ms: float
    latency_delta_pct: float
    cited_overlap_jaccard: float


@dataclass
class ComparisonResult:
    """Side-by-side comparison of baseline and PHOTON."""

    question: str
    baseline: VariantResult
    photon: VariantResult
    delta: DeltaResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_variant_with_pipeline(
    pipeline: Any,
    question: str,
    session_id: str,
    repo_id: str,
    variant_id: str,
    config_path: str = "",
) -> VariantResult:
    """Execute *pipeline* for one question and return a VariantResult.

    This is the single source of query execution logic — both the UI (with
    cached pipeline) and the CLI wrapper (_build_and_run) delegate here.
    """
    result = pipeline.query(question=question, session_id=session_id, repo_id=repo_id)
    cfg = getattr(pipeline, "config", None) or getattr(pipeline, "cfg", None)
    model_id = getattr(getattr(cfg, "model", None), "model_id", None)
    return VariantResult(
        variant_id=variant_id,
        config_path=config_path,
        answer=result.answer,
        cited_chunk_ids=list(result.cited_chunk_ids),
        no_citation=bool(result.no_citation),
        latency_total_ms=float(result.latency.total_ms),
        latency_retrieval_ms=float(result.latency.retrieval_ms),
        latency_generation_ms=float(result.latency.generation_ms),
        memory_peak_mb=float(result.memory.peak_mb),
        generator_used=getattr(result, "generator_used", None),
        generator_fallback_reason=getattr(result, "generator_fallback_reason", None),
        retrieval_debug=getattr(result, "retrieval_debug", None),
        model_id=model_id,
    )


def compare(
    baseline_pipeline: Any,
    photon_pipeline: Any,
    question: str,
    repo_id: str,
    baseline_session_id: str,
    photon_session_id: str,
) -> ComparisonResult:
    """Run baseline and PHOTON and return a ComparisonResult with delta.

    The two variants intentionally run sequentially. Running multiple MLX /
    Metal-backed pipelines concurrently in the Streamlit process can abort
    the whole process at the native command-buffer layer.
    """
    b = run_variant_with_pipeline(
        baseline_pipeline,
        question,
        baseline_session_id,
        repo_id,
        "baseline",
    )
    p = run_variant_with_pipeline(
        photon_pipeline,
        question,
        photon_session_id,
        repo_id,
        "photon",
    )
    return ComparisonResult(
        question=question, baseline=b, photon=p, delta=compute_delta(b, p)
    )


def compute_delta(baseline: VariantResult, photon: VariantResult) -> DeltaResult:
    """Compute latency delta and Jaccard citation overlap between two variants."""
    latency_delta_ms = photon.latency_total_ms - baseline.latency_total_ms
    latency_delta_pct = (
        (latency_delta_ms / baseline.latency_total_ms * 100)
        if baseline.latency_total_ms
        else 0.0
    )
    b_cited = set(baseline.cited_chunk_ids)
    p_cited = set(photon.cited_chunk_ids)
    union = b_cited | p_cited
    cited_overlap_jaccard = len(b_cited & p_cited) / len(union) if union else 0.0
    return DeltaResult(
        latency_delta_ms=latency_delta_ms,
        latency_delta_pct=latency_delta_pct,
        cited_overlap_jaccard=cited_overlap_jaccard,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_and_run(
    variant_id: str,
    config_path: str,
    question: str,
    repo_id: str,
    session_id: str,
) -> VariantResult:
    """CLI/scripts helper: load config, build pipeline, delegate to run_variant_with_pipeline."""
    from baseline_reporag.config import load_config
    from baseline_reporag.pipeline_factory import (
        build_pipeline,
        override_repo_for_pipeline,
    )

    cfg = load_config(config_path)
    resolved_repo_id = repo_id or cfg.repo.repo_id
    override_repo_for_pipeline(cfg, resolved_repo_id)
    pipeline = build_pipeline(cfg)
    return run_variant_with_pipeline(
        pipeline,
        question,
        session_id,
        resolved_repo_id,
        variant_id,
        config_path,
    )
