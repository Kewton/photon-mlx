"""Latency and memory profiling utilities for benchmark runs."""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class LatencyBreakdown:
    retrieval_ms: float = 0.0
    graph_expansion_ms: float = 0.0
    evidence_pack_ms: float = 0.0
    generation_ms: float = 0.0
    citation_ms: float = 0.0
    total_ms: float = 0.0
    photon_prefill_ms: float = 0.0
    drift_eval_ms: float = 0.0
    safe_recgen_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "retrieval_ms": round(self.retrieval_ms, 2),
            "graph_expansion_ms": round(self.graph_expansion_ms, 2),
            "evidence_pack_ms": round(self.evidence_pack_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "citation_ms": round(self.citation_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


@dataclass
class MemorySnapshot:
    peak_mb: float = 0.0
    current_mb: float = 0.0

    def as_dict(self) -> dict:
        return {
            "peak_mb": round(self.peak_mb, 2),
            "current_mb": round(self.current_mb, 2),
        }


class StopWatch:
    """Accumulates wall-clock time across multiple start/stop calls."""

    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is not None:
            self._elapsed += (time.perf_counter() - self._start) * 1000
            self._start = None
        return self._elapsed

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed


class TurnProfiler:
    """Profile a single turn's latency breakdown and peak memory."""

    def __init__(self) -> None:
        self._watches: dict[str, StopWatch] = {}
        self._overall = StopWatch()
        self._tracemalloc_started = False

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        w = self._watches.setdefault(name, StopWatch())
        w.start()
        try:
            yield
        finally:
            w.stop()

    def start(self) -> None:
        self._overall.start()
        tracemalloc.start()
        self._tracemalloc_started = True

    def finish(self) -> tuple[LatencyBreakdown, MemorySnapshot]:
        self._overall.stop()

        latency = LatencyBreakdown(
            retrieval_ms=self._watches.get("retrieval", StopWatch()).elapsed_ms,
            graph_expansion_ms=self._watches.get(
                "graph_expansion", StopWatch()
            ).elapsed_ms,
            evidence_pack_ms=self._watches.get("evidence_pack", StopWatch()).elapsed_ms,
            generation_ms=self._watches.get("generation", StopWatch()).elapsed_ms,
            citation_ms=self._watches.get("citation", StopWatch()).elapsed_ms,
            total_ms=self._overall.elapsed_ms,
        )

        if self._tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory = MemorySnapshot(
                peak_mb=peak / (1024 * 1024),
                current_mb=current / (1024 * 1024),
            )
        else:
            memory = MemorySnapshot()

        return latency, memory
