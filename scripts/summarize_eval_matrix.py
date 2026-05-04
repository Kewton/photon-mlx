from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_reporag.config import Config, deep_merge


_LEAK_MARKERS = (
    "Thinking Process:",
    "Here's a thinking process",
    "Analyze the Request:",
    "<|channel|>thought",
)


def _looks_like_reasoning_leak(answer: str) -> bool:
    return any(marker.lower() in answer.lower() for marker in _LEAK_MARKERS)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 2),
        "p50": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def _rate(num: int, den: int) -> float:
    return round((num / den) if den else 0.0, 4)


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _split_predictions(
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    static_records = [record for record in predictions if "eval_id" in record]
    multi_turn_records = [
        record
        for record in predictions
        if "session_id" in record and "turn_id" in record and "eval_id" not in record
    ]
    return static_records, multi_turn_records


def _summarize_static(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["latency_ms"]) for record in records]
    generations = [float(record["generation_ms"]) for record in records]
    citation_counts = [len(record.get("cited_chunk_ids", [])) for record in records]
    no_citation_count = sum(1 for record in records if record.get("no_citation"))
    leak_count = sum(
        1
        for record in records
        if _looks_like_reasoning_leak(str(record.get("answer", "")))
    )
    generator_used = Counter(
        record.get("generator_used") or "unknown" for record in records
    )
    fallback_reasons = Counter(
        record.get("generator_fallback_reason")
        for record in records
        if record.get("generator_fallback_reason")
    )
    return {
        "count": len(records),
        "no_citation_rate": _rate(no_citation_count, len(records)),
        "leak_rate": _rate(leak_count, len(records)),
        "latency_ms": _stats(latencies),
        "generation_ms": _stats(generations),
        "avg_citation_count": round(statistics.mean(citation_counts), 4)
        if citation_counts
        else 0.0,
        "generator_used_counts": dict(generator_used),
        "generator_fallback_reasons": dict(fallback_reasons),
    }


def _summarize_multi_turn(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["latency_ms"]) for record in records]
    generations = [float(record["generation_ms"]) for record in records]
    citation_counts = [len(record.get("cited_chunk_ids", [])) for record in records]
    no_citation_count = sum(1 for record in records if record.get("no_citation"))
    leak_count = sum(
        1
        for record in records
        if _looks_like_reasoning_leak(str(record.get("answer", "")))
    )
    followup_records = [
        record for record in records if int(record.get("turn_id", 0)) >= 2
    ]
    turn_5_6_records = [
        record for record in records if int(record.get("turn_id", 0)) in {5, 6}
    ]
    generator_used = Counter(
        record.get("generator_used") or "unknown" for record in records
    )
    fallback_reasons = Counter(
        record.get("generator_fallback_reason")
        for record in records
        if record.get("generator_fallback_reason")
    )
    return {
        "count": len(records),
        "no_citation_rate": _rate(no_citation_count, len(records)),
        "leak_rate": _rate(leak_count, len(records)),
        "latency_ms": _stats(latencies),
        "generation_ms": _stats(generations),
        "avg_citation_count": round(statistics.mean(citation_counts), 4)
        if citation_counts
        else 0.0,
        "followup_latency_p50_ms": round(
            statistics.median(
                float(record["latency_ms"]) for record in followup_records
            ),
            2,
        )
        if followup_records
        else 0.0,
        "turn_5_6_no_citation_rate": _rate(
            sum(1 for record in turn_5_6_records if record.get("no_citation")),
            len(turn_5_6_records),
        ),
        "generator_used_counts": dict(generator_used),
        "generator_fallback_reasons": dict(fallback_reasons),
    }


def _load_variant_runtime_config(
    variant_cfg: dict[str, Any],
) -> dict[str, Any]:
    with open(variant_cfg["config_path"], encoding="utf-8") as f:
        base_data = yaml.safe_load(f)
    override = variant_cfg.get("override", {})
    merged = deep_merge(base_data, override) if override else base_data
    cfg = Config(merged)
    inference_cfg = cfg.get("inference")
    tokenizer_cfg = cfg.get("tokenizer")
    return {
        "provider": getattr(cfg.model, "provider", None),
        "model_id": getattr(cfg.model, "model_id", None),
        "photon_generation_enabled": getattr(
            inference_cfg, "photon_generation_enabled", False
        )
        if inference_cfg is not None
        else False,
        "tokenizer_id": getattr(tokenizer_cfg, "tokenizer_id", None)
        if tokenizer_cfg is not None
        else None,
        "checkpoint_path": getattr(cfg.model, "checkpoint_path", None),
    }


def build_matrix_summary(
    run_dir: Path,
    eval_config_path: Path,
) -> dict[str, Any]:
    with open(eval_config_path, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    variants = eval_cfg.get("variants", [])
    run_dir = run_dir.resolve()
    summary_variants: list[dict[str, Any]] = []

    for variant in variants:
        matches = sorted(run_dir.glob(f"*_{variant['id']}.jsonl"))
        if not matches:
            raise FileNotFoundError(f"missing predictions for variant {variant['id']}")
        prediction_path = matches[-1]
        predictions = _load_predictions(prediction_path)
        static_records, multi_turn_records = _split_predictions(predictions)
        runtime_cfg = _load_variant_runtime_config(variant)
        summary_variants.append(
            {
                "variant_id": variant["id"],
                "label": variant.get("label", variant["id"]),
                "prediction_path": str(prediction_path),
                "config": runtime_cfg,
                "static": _summarize_static(static_records),
                "multi_turn": _summarize_multi_turn(multi_turn_records),
            }
        )

    return {
        "run_dir": str(run_dir),
        "eval_config_path": str(eval_config_path.resolve()),
        "variants": summary_variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a full eval model matrix")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/eval_qwen_model_matrix.yaml")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    summary = build_matrix_summary(Path(args.run_dir), Path(args.config))
    output_text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(output_text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
