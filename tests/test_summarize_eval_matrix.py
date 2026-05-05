from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.summarize_eval_matrix import (
    _looks_like_reasoning_leak,
    _summarize_multi_turn,
    _summarize_static,
    build_matrix_summary,
)


def test_looks_like_reasoning_leak_detects_known_markers() -> None:
    assert _looks_like_reasoning_leak("Thinking Process:\nfoo") is True
    assert _looks_like_reasoning_leak("<|channel|>thought\nbar") is True
    assert _looks_like_reasoning_leak("最終回答だけです。") is False


def test_summarize_static_counts_leaks_and_generator_usage() -> None:
    summary = _summarize_static(
        [
            {
                "answer": "Thinking Process: foo",
                "no_citation": False,
                "cited_chunk_ids": ["c1", "c2"],
                "latency_ms": 10.0,
                "generation_ms": 7.0,
                "generator_used": "qwen",
                "generator_fallback_reason": None,
            },
            {
                "answer": "final answer",
                "no_citation": True,
                "cited_chunk_ids": [],
                "latency_ms": 20.0,
                "generation_ms": 15.0,
                "generator_used": "photon",
                "generator_fallback_reason": "empty_output",
            },
        ]
    )

    assert summary["count"] == 2
    assert summary["no_citation_rate"] == 0.5
    assert summary["leak_rate"] == 0.5
    assert summary["latency_ms"]["p50"] == 15.0
    assert summary["generator_used_counts"] == {"qwen": 1, "photon": 1}
    assert summary["generator_fallback_reasons"] == {"empty_output": 1}


def test_summarize_multi_turn_reports_followup_and_turn_5_6() -> None:
    summary = _summarize_multi_turn(
        [
            {
                "turn_id": 1,
                "answer": "ok",
                "no_citation": False,
                "cited_chunk_ids": ["c1"],
                "latency_ms": 10.0,
                "generation_ms": 5.0,
                "generator_used": "qwen",
            },
            {
                "turn_id": 5,
                "answer": "ok",
                "no_citation": True,
                "cited_chunk_ids": [],
                "latency_ms": 20.0,
                "generation_ms": 15.0,
                "generator_used": "qwen",
            },
            {
                "turn_id": 6,
                "answer": "Analyze the Request: foo",
                "no_citation": False,
                "cited_chunk_ids": ["c2"],
                "latency_ms": 30.0,
                "generation_ms": 25.0,
                "generator_used": "qwen",
            },
        ]
    )

    assert summary["count"] == 3
    assert summary["followup_latency_p50_ms"] == 25.0
    assert summary["turn_5_6_no_citation_rate"] == 0.5
    assert summary["leak_rate"] == round(1 / 3, 4)


def test_build_matrix_summary_reads_variant_metadata(tmp_path: Path) -> None:
    base_cfg = {
        "model": {
            "provider": "mlx_lm",
            "model_id": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
        },
        "inference": {"photon_generation_enabled": False},
    }
    base_cfg_path = tmp_path / "baseline.yaml"
    base_cfg_path.write_text(yaml.safe_dump(base_cfg), encoding="utf-8")

    eval_cfg = {
        "variants": [
            {
                "id": "baseline_qwen35_nothink",
                "label": "Baseline + Qwen3.5 no-think",
                "config_path": str(base_cfg_path),
                "override": {
                    "model": {"model_id": "mlx-community/Qwen3.5-9B-MLX-4bit"}
                },
            }
        ]
    }
    eval_cfg_path = tmp_path / "eval.yaml"
    eval_cfg_path.write_text(yaml.safe_dump(eval_cfg), encoding="utf-8")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prediction_path = run_dir / "bench_x_baseline_qwen35_nothink.jsonl"
    prediction_path.write_text(
        json.dumps(
            {
                "eval_id": "SE-001",
                "question": "Q",
                "answer": "A",
                "cited_chunk_ids": ["c1"],
                "no_citation": False,
                "latency_ms": 12.0,
                "generation_ms": 9.0,
                "generator_used": "qwen",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_matrix_summary(run_dir, eval_cfg_path)

    variant = summary["variants"][0]
    assert variant["variant_id"] == "baseline_qwen35_nothink"
    assert variant["config"]["model_id"] == "mlx-community/Qwen3.5-9B-MLX-4bit"
    assert variant["static"]["count"] == 1
