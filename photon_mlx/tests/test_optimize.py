"""Tests for Mac optimization utilities."""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch_ref.config import (
    HierarchyConfig,
    ModelConfig,
    PhotonConfig,
    TokenizerConfig,
)
from photon_mlx.model import PhotonModel
from photon_mlx.optimize import (
    LatencyResult,
    MemoryReport,
    benchmark_forward,
    measure_memory,
    pad_input_ids,
    pad_to_multiple,
    reset_peak_memory,
    warmup_model,
)


def _tiny_cfg() -> PhotonConfig:
    return PhotonConfig(
        model=ModelConfig(
            base_embed_dim=16,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=128,
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


# ---------------------------------------------------------------
# Padding tests
# ---------------------------------------------------------------

class TestPadding:
    def test_pad_to_multiple_exact(self) -> None:
        assert pad_to_multiple(16, 16) == 16

    def test_pad_to_multiple_rounds_up(self) -> None:
        assert pad_to_multiple(17, 16) == 32
        assert pad_to_multiple(1, 16) == 16

    def test_pad_input_ids_no_op(self) -> None:
        ids = mx.zeros((1, 16), dtype=mx.int32)
        out = pad_input_ids(ids, 16)
        assert out.shape == (1, 16)

    def test_pad_input_ids_extends(self) -> None:
        ids = mx.ones((2, 10), dtype=mx.int32)
        out = pad_input_ids(ids, 16, pad_id=0)
        assert out.shape == (2, 16)
        mx.eval(out)
        assert out[0, 9].item() == 1
        assert out[0, 10].item() == 0

    def test_pad_input_ids_already_longer(self) -> None:
        ids = mx.ones((1, 32), dtype=mx.int32)
        out = pad_input_ids(ids, 16)
        assert out.shape == (1, 32)


# ---------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------

class TestMemory:
    def test_measure_memory_returns_report(self) -> None:
        report = measure_memory()
        assert isinstance(report, MemoryReport)
        assert report.active_mb >= 0
        assert report.peak_mb >= 0

    def test_as_dict(self) -> None:
        report = MemoryReport(active_bytes=1048576, peak_bytes=2097152)
        d = report.as_dict()
        assert d["active_mb"] == 1.0
        assert d["peak_mb"] == 2.0

    def test_reset_peak_no_error(self) -> None:
        reset_peak_memory()  # should not raise


# ---------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------

class TestWarmup:
    def test_warmup_runs(self) -> None:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)

        def fwd(ids):
            logits, _ = model(ids)
            return logits

        avg_ms = warmup_model(fwd, (1, 16), 256, n_warmup=2)
        assert avg_ms > 0


# ---------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------

class TestBenchmark:
    def test_benchmark_forward(self) -> None:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)

        def fwd(ids):
            logits, _ = model(ids)
            return logits

        ids = mx.random.randint(0, 256, (1, 16))
        result = benchmark_forward(fwd, ids, n_runs=3)
        assert isinstance(result, LatencyResult)
        assert result.total_ms > 0
        assert result.tokens_per_sec > 0

    def test_latency_result_as_dict(self) -> None:
        r = LatencyResult(total_ms=10.5, per_token_ms=0.65, tokens_per_sec=1500.0)
        d = r.as_dict()
        assert d["total_ms"] == 10.5
        assert d["tokens_per_sec"] == 1500.0
