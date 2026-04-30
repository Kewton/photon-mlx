"""MLX-free shared contracts for baseline and PHOTON pipelines.

CB2-001 (codex-fix): ``QueryResult`` used to live in
``baseline_reporag.pipeline``, which eagerly imports the MLX-backed
``Generator`` at module load.  That made any module referencing the
dataclass transitively depend on MLX — including
``baseline_reporag.pipeline_factory``, defeating the factory's entire
"MLX-free at import time" guarantee.

This module is deliberately lightweight: only ``dataclasses`` and
``typing`` from stdlib are imported at load.  It lets pure-baseline
callers (cli, server, eval scripts) reference ``QueryResult`` without
dragging in ``mlx_lm`` / ``mlx.core``.

``baseline_reporag.pipeline.QueryResult`` is re-exported from here for
backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Forward-reference only: the profiler module is MLX-free but we
    # still avoid an eager import so ``contracts`` stays minimal.
    from .profiler import LatencyBreakdown, MemorySnapshot


@dataclass
class RetrievalDebugRow:
    """Per-chunk retrieval debug info for Issue #176 UI panel.

    score fields are None for graph/neighbor/photon_pruned/working_memory sources.
    citation_index is None for non-adopted chunks; cited = (citation_index is not None).
    source values: "retrieval" | "graph" | "neighbor" | "photon_pruned" | "working_memory"
    """

    chunk_id: str
    rel_path: str
    section: str | None
    bm25_score: float | None
    embedding_score: float | None
    fused_score: float | None
    reranker_score: float | None
    used: bool
    citation_index: int | None
    source: str


@dataclass
class QueryResult:
    """Result of a single pipeline query turn.

    The fields mirror the historical ``baseline_reporag.pipeline.QueryResult``
    so existing callers keep working through the re-export.
    """

    answer: str
    session_id: str
    turn_id: int
    cited_chunk_ids: list[str]
    wrong_citation_indices: list[int]
    no_citation: bool
    latency: "LatencyBreakdown"
    memory: "MemorySnapshot"
    drift_metrics: dict[str, Any] | None = None
    confidence: float | None = None
    fallback_decision: dict[str, Any] | None = None
    citation_postprocessed: bool = False
    # Issue #62 Phase 1 (CB-003 codex-fix): expose the generator that
    # actually produced ``answer`` so side-by-side comparison tools can
    # distinguish a real PHOTON answer from a Qwen fallback.  None on
    # historical rows (pre-#62) and on paths that have not populated it.
    # Closed enum: None | "qwen" | "photon".
    generator_used: str | None = None
    # Closed enum (§7.2): None | "_TokenizerEncodeFailure" | "ValueError"
    #                    | "RuntimeError" | "empty_output".
    generator_fallback_reason: str | None = None
    # Issue #176: per-turn retrieval debug rows (normalized scores + source).
    # None when debug is disabled or on pre-#176 historical rows.
    retrieval_debug: list[RetrievalDebugRow] | None = None
    # Issue #177: refusal score (0.0 = answer, 1.0 = refusal) and matched phrases.
    # None on pre-#177 historical rows or when not yet computed.
    refusal_score: float | None = None
    refusal_matches: list[str] | None = None
