"""bench/issue61_prune_batch.py — before/after benchmark for prune_evidence.

Measures the per-call latency of ``PhotonInference.prune_evidence`` for the
pre-Issue-#61 sequential path versus the batched path implemented for the
Issue, on a synthetic mixed-length workload of 64 candidate chunks.

The legacy sequential implementation is inlined as ``_legacy_prune_evidence``
so the bench is self-contained — it does not require git checkout of the
previous file. Both code paths use the same ``PhotonInference`` instance,
so PHOTON model weights and session state are identical.

Acceptance target (Issue #61, 性能・実機検証):
    >= 1.5x speedup of the batched path over the sequential path on
    N=64 chunks at max_len up to ``cfg.model.max_position_embeddings``.

Usage:
    python bench/issue61_prune_batch.py
    python bench/issue61_prune_batch.py --warmup 5 --measure 20 --max-len 2048
    python bench/issue61_prune_batch.py --report-path reports/issue-61-prune-batch.md
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from math import prod
from pathlib import Path
from typing import Any

import mlx.core as mx

# Make the package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from photon_mlx.inference import PhotonInference  # noqa: E402
from photon_mlx.model import PhotonModel  # noqa: E402
from torch_ref.config import (  # noqa: E402
    HierarchyConfig,
    ModelConfig,
    PhotonConfig,
    TokenizerConfig,
)


def _bench_cfg(max_position_embeddings: int) -> PhotonConfig:
    """Tiny but realistic config for benchmark workloads."""
    return PhotonConfig(
        model=ModelConfig(
            base_embed_dim=16,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=max_position_embeddings,
        ),
        hierarchy=HierarchyConfig(
            levels=2,
            chunk_sizes=[4, 4],
            converter_prefix_lengths=[2, 2],
            encoder_layers_per_level=[1, 1],
            decoder_layers_per_level=[1, 1],
        ),
        tokenizer=TokenizerConfig(vocab_size=256),
    )


def _legacy_prune_evidence(
    inference: PhotonInference,
    chunk_texts: list[str],
    chunk_ids: list[str],
    session_id: str,
    max_chunks: int,
) -> list[int]:
    """Inlined copy of the pre-Issue-#61 sequential ``prune_evidence``.

    Kept verbatim from the develop branch to provide a fair baseline. Any
    discrepancy versus the live implementation is intentional — this is the
    point of comparison.
    """
    session = inference._sessions.get(session_id)
    all_indices = list(range(len(chunk_texts)))

    if (
        session is None
        or session.current_state is None
        or not session.current_state.level_states
    ):
        return all_indices

    if len(chunk_texts) <= max_chunks:
        return all_indices

    coarse_state = session.current_state.level_states[-1]
    coarse_vec = mx.mean(
        coarse_state.astype(mx.float32),
        axis=tuple(range(coarse_state.ndim - 1)),
    )

    padding_multiple = prod(inference.cfg.hierarchy.chunk_sizes)
    max_len = inference.cfg.model.max_position_embeddings

    scores: list[tuple[int, float]] = []
    for idx, text in enumerate(chunk_texts):
        if not text.strip():
            scores.append((idx, -1.0))
            continue

        token_ids = [
            b % inference.cfg.tokenizer.vocab_size for b in text.encode("utf-8")
        ]
        if not token_ids:
            scores.append((idx, -1.0))
            continue

        remainder = len(token_ids) % padding_multiple
        if remainder != 0:
            token_ids = token_ids + [0] * (padding_multiple - remainder)

        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            remainder = len(token_ids) % padding_multiple
            if remainder != 0:
                token_ids = token_ids[: len(token_ids) - remainder]

        if not token_ids:
            scores.append((idx, -1.0))
            continue

        input_ids = mx.array(token_ids, dtype=mx.int32).reshape(1, -1)
        _, h_state = inference.hierarchical_prefill(input_ids)

        chunk_top = h_state.level_states[-1]
        chunk_vec = mx.mean(
            chunk_top.astype(mx.float32),
            axis=tuple(range(chunk_top.ndim - 1)),
        )

        dot = mx.sum(coarse_vec * chunk_vec)
        norm_a = mx.sqrt(mx.sum(coarse_vec * coarse_vec))
        norm_b = mx.sqrt(mx.sum(chunk_vec * chunk_vec))
        sim = dot / (norm_a * norm_b + 1e-8)
        mx.eval(sim)
        scores.append((idx, float(sim.item())))

    scores.sort(key=lambda x: x[1], reverse=True)
    return sorted([s[0] for s in scores[:max_chunks]])


def _make_workload(
    n_chunks: int,
    max_len: int,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Synthesise ``n_chunks`` mixed-length text chunks.

    Lengths are drawn deterministically so before/after numbers are comparable
    across runs. The longest chunk targets ``max_len`` bytes which exercises the
    truncation path.
    """
    import random

    rng = random.Random(seed)
    texts: list[str] = []
    for i in range(n_chunks):
        # Mixed lengths: short / medium / long / near-max
        bucket = i % 4
        if bucket == 0:
            length = rng.randint(20, 60)
        elif bucket == 1:
            length = rng.randint(120, 240)
        elif bucket == 2:
            length = rng.randint(400, 800)
        else:
            length = max(max_len - rng.randint(0, 200), 800)
        body = "".join(chr(33 + (rng.randint(0, 90))) for _ in range(length))
        texts.append(f"chunk_{i:03d}: {body}")
    ids = [f"c{i:03d}" for i in range(n_chunks)]
    return texts, ids


def _time_calls(
    fn,
    *,
    warmup: int,
    measure: int,
) -> list[float]:
    """Warm up then collect ``measure`` per-call wall times in seconds."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(measure):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def _summary(samples: list[float]) -> dict[str, float]:
    sorted_s = sorted(samples)
    return {
        "min_ms": min(sorted_s) * 1000,
        "p50_ms": statistics.median(sorted_s) * 1000,
        "p95_ms": sorted_s[int(0.95 * (len(sorted_s) - 1))] * 1000,
        "max_ms": max(sorted_s) * 1000,
        "mean_ms": statistics.fmean(sorted_s) * 1000,
        "n": len(sorted_s),
    }


def _format_summary(label: str, s: dict[str, float]) -> str:
    return (
        f"{label:<24} "
        f"min={s['min_ms']:.2f} ms  "
        f"p50={s['p50_ms']:.2f} ms  "
        f"p95={s['p95_ms']:.2f} ms  "
        f"max={s['max_ms']:.2f} ms  "
        f"mean={s['mean_ms']:.2f} ms  "
        f"n={s['n']}"
    )


def run_benchmark(
    n_chunks: int,
    max_len: int,
    max_chunks: int,
    warmup: int,
    measure: int,
    seed: int,
) -> dict[str, Any]:
    """Run the before/after benchmark and return a structured result."""
    mx.random.seed(seed)

    cfg = _bench_cfg(max_position_embeddings=max_len)
    model = PhotonModel(cfg)
    inference = PhotonInference(model, cfg)

    # Establish session state so prune_evidence takes the scoring path
    # (turn 2+ behaviour). 16 random tokens is enough to populate
    # ``level_states[-1]`` with the (1, T_top, D) coarse vector.
    setup_ids = mx.random.randint(0, 256, (1, 16))
    inference.session_forward(setup_ids, "bench-session", "repo", "abc")

    chunk_texts, chunk_ids = _make_workload(n_chunks, max_len, seed=seed)

    def call_legacy() -> None:
        _legacy_prune_evidence(
            inference,
            chunk_texts,
            chunk_ids,
            session_id="bench-session",
            max_chunks=max_chunks,
        )

    def call_batched() -> None:
        inference.prune_evidence(
            chunk_texts=chunk_texts,
            chunk_ids=chunk_ids,
            session_id="bench-session",
            max_chunks=max_chunks,
        )

    legacy_samples = _time_calls(call_legacy, warmup=warmup, measure=measure)
    batched_samples = _time_calls(call_batched, warmup=warmup, measure=measure)

    legacy_summary = _summary(legacy_samples)
    batched_summary = _summary(batched_samples)
    speedup_p50 = legacy_summary["p50_ms"] / max(batched_summary["p50_ms"], 1e-9)
    speedup_mean = legacy_summary["mean_ms"] / max(batched_summary["mean_ms"], 1e-9)

    legacy_result = _legacy_prune_evidence(
        inference, chunk_texts, chunk_ids, "bench-session", max_chunks
    )
    batched_result = inference.prune_evidence(
        chunk_texts=chunk_texts,
        chunk_ids=chunk_ids,
        session_id="bench-session",
        max_chunks=max_chunks,
    )

    return {
        "config": {
            "n_chunks": n_chunks,
            "max_len": max_len,
            "max_chunks": max_chunks,
            "warmup": warmup,
            "measure": measure,
            "seed": seed,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "mlx_version": getattr(mx, "__version__", "unknown"),
        },
        "legacy": legacy_summary,
        "batched": batched_summary,
        "speedup": {
            "p50": round(speedup_p50, 3),
            "mean": round(speedup_mean, 3),
        },
        "selection_match": legacy_result == batched_result,
        "selection_legacy": legacy_result,
        "selection_batched": batched_result,
    }


def _format_report(result: dict[str, Any]) -> str:
    cfg = result["config"]
    env = result["environment"]
    leg = result["legacy"]
    bat = result["batched"]
    sp = result["speedup"]
    target = 1.5
    verdict = (
        "PASS (>= 1.5x speedup)"
        if sp["p50"] >= target
        else f"FAIL (p50 speedup {sp['p50']}x < {target}x)"
    )
    return f"""# Issue #61 — `prune_evidence` バッチ化ベンチマーク結果

## 実行条件

| 項目 | 値 |
|------|-----|
| N chunks | {cfg["n_chunks"]} |
| max_len (max_position_embeddings) | {cfg["max_len"]} |
| max_chunks (top-K) | {cfg["max_chunks"]} |
| warmup runs | {cfg["warmup"]} |
| measure runs | {cfg["measure"]} |
| seed | {cfg["seed"]} |

## 実機環境

| 項目 | 値 |
|------|-----|
| Python | {env["python"]} |
| Platform | {env["platform"]} |
| Machine | {env["machine"]} |
| Processor | {env["processor"]} |
| MLX version | {env["mlx_version"]} |

## レイテンシ（per call）

| 経路 | min | p50 | p95 | max | mean | n |
|-----|-----|-----|-----|-----|-----|---|
| 逐次（legacy） | {leg["min_ms"]:.2f} ms | {leg["p50_ms"]:.2f} ms | {leg["p95_ms"]:.2f} ms | {leg["max_ms"]:.2f} ms | {leg["mean_ms"]:.2f} ms | {leg["n"]} |
| バッチ（new） | {bat["min_ms"]:.2f} ms | {bat["p50_ms"]:.2f} ms | {bat["p95_ms"]:.2f} ms | {bat["max_ms"]:.2f} ms | {bat["mean_ms"]:.2f} ms | {bat["n"]} |

## 高速化倍率

- p50 speedup: **{sp["p50"]}x**
- mean speedup: **{sp["mean"]}x**

## 受入判定

**{verdict}**

## 選択結果の同等性

- 逐次選択: `{result["selection_legacy"]}`
- バッチ選択: `{result["selection_batched"]}`
- top-K 一致: **{result["selection_match"]}**

## OOM チェック

実行が完了し、上記レイテンシが記録できていることが OOM していないことの実機証拠である。
M2 Pro / M3 Max など実機での再現は本ファイルを直接実行して結果欄を更新してください。

## 補足

- 本レポートは `bench/issue61_prune_batch.py` の `--report-path` オプションで自動生成・上書きされる。
- 数値は `_tiny_cfg` 相当の小型 PHOTON config を使用しており、production の絶対値とは異なる。倍率（speedup）は同一インスタンス・同一入力での比較なので、実装上の高速化効果を直接示す。
- E2E follow-up latency への影響は本スクリプト単体では計測しない（既存 profiler の `total_ms` / `generation_ms` で間接確認）。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-chunks", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measure", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-path",
        type=str,
        default="reports/issue-61-prune-batch.md",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Optional path to also dump the raw JSON result.",
    )
    args = parser.parse_args()

    result = run_benchmark(
        n_chunks=args.n_chunks,
        max_len=args.max_len,
        max_chunks=args.max_chunks,
        warmup=args.warmup,
        measure=args.measure,
        seed=args.seed,
    )

    print(_format_summary("legacy (sequential)", result["legacy"]))
    print(_format_summary("batched (Issue #61)", result["batched"]))
    print(
        "speedup: "
        f"p50={result['speedup']['p50']}x  "
        f"mean={result['speedup']['mean']}x  "
        f"selection_match={result['selection_match']}"
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_format_report(result))
    print(f"report written: {report_path}")

    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2))
        print(f"json written: {json_path}")


if __name__ == "__main__":
    main()
