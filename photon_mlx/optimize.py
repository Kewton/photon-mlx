"""
Mac (Apple Silicon) optimization utilities for PHOTON inference.

- Fixed-shape decode step compilation
- Memory measurement stabilization
- Warmup routines
- Padding for compile-friendly shapes
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx


# ================================================================
# Shape utilities
# ================================================================


def pad_to_multiple(seq_len: int, multiple: int) -> int:
    """Return the smallest value >= seq_len that is a multiple of `multiple`."""
    if seq_len % multiple == 0:
        return seq_len
    return seq_len + (multiple - seq_len % multiple)


def pad_input_ids(input_ids: mx.array, target_len: int, pad_id: int = 0) -> mx.array:
    """Pad (B, T) to (B, target_len) on the right."""
    B, T = input_ids.shape
    if T >= target_len:
        return input_ids
    padding = mx.full((B, target_len - T), pad_id, dtype=input_ids.dtype)
    return mx.concatenate([input_ids, padding], axis=1)


# ================================================================
# Memory measurement
# ================================================================


@dataclass
class MemoryReport:
    active_bytes: int = 0
    peak_bytes: int = 0
    cache_bytes: int = 0

    @property
    def active_mb(self) -> float:
        return self.active_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    def as_dict(self) -> dict:
        return {
            "active_mb": round(self.active_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "cache_bytes": self.cache_bytes,
        }


def measure_memory() -> MemoryReport:
    """Snapshot current MLX metal memory usage."""
    try:
        active = mx.metal.get_active_memory()
        peak = mx.metal.get_peak_memory()
        cache = mx.metal.get_cache_memory()
    except AttributeError:
        return MemoryReport()
    return MemoryReport(active_bytes=active, peak_bytes=peak, cache_bytes=cache)


def reset_peak_memory() -> None:
    """Reset peak memory counter."""
    try:
        mx.metal.reset_peak_memory()
    except AttributeError:
        pass


# ================================================================
# Warmup
# ================================================================


def warmup_model(
    model_fn: callable,
    input_shape: tuple[int, int],
    vocab_size: int,
    n_warmup: int = 3,
) -> float:
    """
    Run `n_warmup` forward passes to warm up Metal compilation cache.
    Returns average warmup latency in ms.
    """
    times: list[float] = []
    for _ in range(n_warmup):
        ids = mx.random.randint(0, vocab_size, input_shape)
        t0 = time.perf_counter()
        out = model_fn(ids)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


# ================================================================
# Benchmark harness
# ================================================================


@dataclass
class LatencyResult:
    total_ms: float
    per_token_ms: float
    tokens_per_sec: float

    def as_dict(self) -> dict:
        return {
            "total_ms": round(self.total_ms, 2),
            "per_token_ms": round(self.per_token_ms, 4),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
        }


def benchmark_forward(
    model_fn: callable,
    input_ids: mx.array,
    n_runs: int = 10,
) -> LatencyResult:
    """Benchmark forward pass latency over n_runs."""
    B, T = input_ids.shape

    # Warmup
    out = model_fn(input_ids)
    mx.eval(out)

    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = model_fn(input_ids)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    total_tokens = B * T
    return LatencyResult(
        total_ms=avg_ms,
        per_token_ms=avg_ms / total_tokens,
        tokens_per_sec=total_tokens / (avg_ms / 1000),
    )


def benchmark_session(
    session_fn: callable,
    queries: list[mx.array],
    session_id: str = "bench",
) -> list[LatencyResult]:
    """Benchmark multi-turn session: returns per-turn latency."""
    results: list[LatencyResult] = []
    for ids in queries:
        B, T = ids.shape
        t0 = time.perf_counter()
        out = session_fn(ids, session_id)
        mx.eval(out)
        elapsed = (time.perf_counter() - t0) * 1000
        total_tokens = B * T
        results.append(
            LatencyResult(
                total_ms=elapsed,
                per_token_ms=elapsed / total_tokens,
                tokens_per_sec=total_tokens / (elapsed / 1000),
            )
        )
    return results
