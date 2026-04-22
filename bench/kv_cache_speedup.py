"""Benchmark: top-level KV cache speedup for PhotonModel.generate().

Issue #54 / Phase 1.

Measures cached vs non-cached ``generate()`` on a tiny model with randomly
initialized weights (shape-accurate — we only care about wall-clock, not
model quality). Outputs per-phase timings and memory footprint.

Per-phase breakdown (Issue #54 / design policy §9 / AC-04):
    - ``prefill`` — first-iteration full bottom-up encode + full top-down
      decode (``top_kv_cache is None``).
    - ``encoder_replay`` — per-step bottom-up replay
      (``_encode_bottom_up``) for iterations >= 1.
    - ``top_level_increment`` — per-step top-level decoder portion of
      ``_decode_from_enc_outputs`` (cache append/replace) for iterations
      >= 1.
    - ``local_tail_decode`` — per-step lower decoders + local decoder +
      lm_head (the "tail" after the top-level block loop) for iterations
      >= 1.

Instrumentation is bench-local: we wrap
``PhotonModel._encode_bottom_up``, ``_decode_from_enc_outputs``, and
``_decode_level`` with timing shims and derive the split by subtracting
the sum of ``_decode_level`` durations (the "tail" components) from the
outer ``_decode_from_enc_outputs`` duration. Model code is untouched.

Security notes (DR4-001 / §7):
    - No ``eval`` / ``exec`` / ``subprocess`` / ``os.system`` / ``shell=True``.
    - Output path is constrained to ``<repo>/bench/reports/`` (fixed directory).

Usage:
    python bench/kv_cache_speedup.py \
        --config configs/photon_tiny.yaml \
        --prompt-len 2048 \
        --gen 64 \
        --runs 3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from photon_mlx.model import PhotonModel  # noqa: E402
from photon_mlx.optimize import measure_memory, reset_peak_memory  # noqa: E402
from torch_ref.config import load_photon_config  # noqa: E402


# Phase keys used in the per-phase breakdown.
_PHASES = ("prefill", "encoder_replay", "top_level_increment", "local_tail_decode")


def _make_prompt(B: int, T: int, vocab_size: int, seed: int = 0) -> mx.array:
    mx.random.seed(seed)
    return mx.random.randint(1, vocab_size, (B, T), dtype=mx.int32)


class _PhaseRecorder:
    """Bench-local recorder for per-phase timings.

    Wraps ``PhotonModel._encode_bottom_up``, ``_decode_from_enc_outputs``,
    and ``_decode_level`` as a context manager and, for each
    ``generate()`` call, derives four phase timings:

    - ``prefill`` — step 0 total (encode + full decode).
    - ``encoder_replay`` — ``_encode_bottom_up`` for steps >= 1.
    - ``top_level_increment`` — ``_decode_from_enc_outputs`` minus its
      nested ``_decode_level`` calls, for steps >= 1.
    - ``local_tail_decode`` — sum of ``_decode_level`` durations, for
      steps >= 1.

    Timings are recorded in seconds via ``time.perf_counter()``.
    ``mx.eval`` is called on returned arrays before the stop-timer to
    force lazy kernels to complete (MLX is lazy by default).
    """

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {p: [] for p in _PHASES}
        # Per-call scratch state, reset at each generate() invocation.
        self._step: int = 0
        self._decode_level_sum: float = 0.0

    @contextlib.contextmanager
    def patch(self, model: PhotonModel) -> Iterator[None]:
        orig_encode = model._encode_bottom_up
        orig_decode = model._decode_from_enc_outputs
        orig_decode_level = model._decode_level

        def timed_encode(input_ids: mx.array):  # type: ignore[no-untyped-def]
            t0 = time.perf_counter()
            out = orig_encode(input_ids)
            # enc_outputs is a list of mx.arrays — force materialization.
            mx.eval(out)
            dt = time.perf_counter() - t0
            if self._step >= 1:
                self.samples["encoder_replay"].append(dt)
            # Stash on the recorder so timed_decode can bundle encode+decode
            # into the prefill bucket for step 0.
            self._last_encode_dt = dt
            return out

        def timed_decode(enc_outputs, top_kv_cache=None):  # type: ignore[no-untyped-def]
            # Reset the per-call sink before entering the decode call so
            # nested _decode_level invocations aggregate cleanly.
            self._decode_level_sum = 0.0
            t0 = time.perf_counter()
            logits, cache_out = orig_decode(enc_outputs, top_kv_cache=top_kv_cache)
            mx.eval(logits)
            dt = time.perf_counter() - t0
            if self._step == 0:
                # Prefill bundles encode + decode so downstream stats reflect
                # the full first-pass cost (design policy §9 "prefill").
                self.samples["prefill"].append(self._last_encode_dt + dt)
            else:
                top_dt = max(dt - self._decode_level_sum, 0.0)
                self.samples["top_level_increment"].append(top_dt)
                self.samples["local_tail_decode"].append(self._decode_level_sum)
            self._step += 1
            return logits, cache_out

        def timed_decode_level(*args, **kwargs):  # type: ignore[no-untyped-def]
            t0 = time.perf_counter()
            out = orig_decode_level(*args, **kwargs)
            mx.eval(out)
            self._decode_level_sum += time.perf_counter() - t0
            return out

        # Reset per-call state for this generate() invocation.
        self._step = 0
        self._decode_level_sum = 0.0
        self._last_encode_dt = 0.0

        model._encode_bottom_up = timed_encode  # type: ignore[assignment,method-assign]
        model._decode_from_enc_outputs = timed_decode  # type: ignore[assignment,method-assign]
        model._decode_level = timed_decode_level  # type: ignore[assignment,method-assign]
        try:
            yield
        finally:
            model._encode_bottom_up = orig_encode  # type: ignore[method-assign]
            model._decode_from_enc_outputs = orig_decode  # type: ignore[method-assign]
            model._decode_level = orig_decode_level  # type: ignore[method-assign]


def _run_generate(
    model: PhotonModel,
    prompt: mx.array,
    max_new_tokens: int,
    use_kv_cache: bool,
    recorder: _PhaseRecorder | None = None,
) -> float:
    """Run generate once, return wall-clock seconds."""
    t0 = time.perf_counter()
    if recorder is not None:
        with recorder.patch(model):
            generated, _ = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                use_kv_cache=use_kv_cache,
            )
    else:
        generated, _ = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache,
        )
    mx.eval(generated)
    return time.perf_counter() - t0


def _phase_stats(samples: list[float]) -> dict[str, float | int]:
    """Return mean / p50 / p95 / total / count stats for a phase."""
    if not samples:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "total_ms": 0.0,
        }
    sorted_ms = sorted(s * 1000 for s in samples)
    n = len(sorted_ms)
    mean_ms = sum(sorted_ms) / n
    p50_ms = statistics.median(sorted_ms)
    # Nearest-rank p95 (n * 0.95, 1-indexed, clamped to last element).
    p95_idx = max(0, min(n - 1, int(round(0.95 * n)) - 1))
    p95_ms = sorted_ms[p95_idx]
    return {
        "count": n,
        "mean_ms": round(mean_ms, 4),
        "p50_ms": round(p50_ms, 4),
        "p95_ms": round(p95_ms, 4),
        "total_ms": round(sum(sorted_ms), 4),
    }


def benchmark(
    config_path: Path,
    prompt_len: int,
    max_new_tokens: int,
    n_runs: int,
) -> dict:
    cfg = load_photon_config(config_path)

    # Build model with random init (benchmark only, not for quality).
    mx.random.seed(42)
    model = PhotonModel(cfg)
    mx.eval(model.parameters())

    vocab_size = cfg.tokenizer.vocab_size
    prompt = _make_prompt(B=1, T=prompt_len, vocab_size=vocab_size)

    # Warmup once per path.
    _run_generate(model, prompt, max_new_tokens=4, use_kv_cache=True)
    _run_generate(model, prompt, max_new_tokens=4, use_kv_cache=False)

    # Cached path — per-phase instrumented.
    reset_peak_memory()
    cache_times: list[float] = []
    recorder = _PhaseRecorder()
    for _ in range(n_runs):
        cache_times.append(
            _run_generate(
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                use_kv_cache=True,
                recorder=recorder,
            )
        )
    cache_mem = measure_memory()

    # No-cache reference path (no per-phase instrumentation — single pass).
    reset_peak_memory()
    nocache_times: list[float] = []
    for _ in range(n_runs):
        nocache_times.append(
            _run_generate(
                model, prompt, max_new_tokens=max_new_tokens, use_kv_cache=False
            )
        )
    nocache_mem = measure_memory()

    cache_avg = sum(cache_times) / len(cache_times)
    nocache_avg = sum(nocache_times) / len(nocache_times)
    speedup = nocache_avg / cache_avg if cache_avg > 0 else float("nan")

    phase_breakdown = {
        phase: _phase_stats(recorder.samples[phase]) for phase in _PHASES
    }

    return {
        "config": str(config_path),
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "n_runs": n_runs,
        "cached": {
            "avg_sec": round(cache_avg, 4),
            "per_token_ms": round(1000 * cache_avg / max_new_tokens, 3),
            "tokens_per_sec": round(max_new_tokens / cache_avg, 2),
            "peak_mb": round(cache_mem.peak_mb, 2),
            "runs_sec": [round(t, 4) for t in cache_times],
            "phases": phase_breakdown,
        },
        "nocache": {
            "avg_sec": round(nocache_avg, 4),
            "per_token_ms": round(1000 * nocache_avg / max_new_tokens, 3),
            "tokens_per_sec": round(max_new_tokens / nocache_avg, 2),
            "peak_mb": round(nocache_mem.peak_mb, 2),
            "runs_sec": [round(t, 4) for t in nocache_times],
        },
        "speedup_x": round(speedup, 2),
    }


def _report_path(timestamp: str) -> Path:
    # Fixed directory under the repo — no user-supplied path accepted (§7).
    out_dir = _REPO_ROOT / "bench" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"kv_cache_speedup_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure KV-cache speedup")
    parser.add_argument(
        "--config", default="configs/photon_tiny.yaml", help="Photon config YAML."
    )
    parser.add_argument(
        "--prompt-len", type=int, default=2048, help="Prompt length in tokens."
    )
    parser.add_argument(
        "--gen", type=int, default=64, help="Number of tokens to generate."
    )
    parser.add_argument("--runs", type=int, default=3, help="Measurement runs.")
    args = parser.parse_args()

    # Fail-closed CLI validation (CB-003 / DR4-002):
    # Reject non-positive counts (0 division on avg) and totals that exceed
    # the model's max_position_embeddings (OOM / RoPE overrun).
    if args.prompt_len < 1:
        raise ValueError(f"--prompt-len must be >= 1, got {args.prompt_len}")
    if args.gen < 1:
        raise ValueError(f"--gen must be >= 1, got {args.gen}")
    if args.runs < 1:
        raise ValueError(f"--runs must be >= 1, got {args.runs}")

    config_path = (_REPO_ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    cfg = load_photon_config(config_path)
    max_pos = cfg.model.max_position_embeddings
    total_horizon = args.prompt_len + args.gen
    if total_horizon > max_pos:
        raise ValueError(
            f"--prompt-len + --gen = {total_horizon} exceeds "
            f"max_position_embeddings={max_pos}"
        )

    result = benchmark(
        config_path=config_path,
        prompt_len=args.prompt_len,
        max_new_tokens=args.gen,
        n_runs=args.runs,
    )
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_path = _report_path(ts)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
