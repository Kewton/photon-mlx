"""
run_all.py  –  Run all benchmark variants defined in eval.yaml.

Usage:
    python bench/run_all.py --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _build_variant_pipeline(variant_cfg: dict) -> Any:
    """Build pipeline for a variant using build_pipeline factory."""
    from baseline_reporag.config import deep_merge
    from baseline_reporag.photon_pipeline import build_pipeline

    import yaml

    config_path = variant_cfg["config_path"]
    with open(config_path, encoding="utf-8") as f:
        base_data = yaml.safe_load(f)

    override = variant_cfg.get("override", {})
    if override:
        merged_data = deep_merge(base_data, override)
    else:
        merged_data = base_data

    from baseline_reporag.config import Config

    cfg = Config(merged_data)
    return build_pipeline(cfg)


# ---------------------------------------------------------------------------
# Eval set runners
# ---------------------------------------------------------------------------


def _run_static_eval(pipeline: Any, ds_cfg: dict) -> list[dict]:
    """Run static (single-question) eval set."""
    path = ds_cfg["path"]
    max_cases = ds_cfg.get("max_cases", 0)

    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line))
    if max_cases > 0:
        questions = questions[:max_cases]

    predictions = []
    for q in questions:
        result = pipeline.query(
            question=q["question"],
            session_id=f"eval-{q['id']}",
            repo_id="",
        )
        predictions.append(
            {
                "eval_id": q["id"],
                "category": q.get("category", ""),
                "question": q["question"],
                "answer": result.answer,
                "cited_chunk_ids": result.cited_chunk_ids,
                "no_citation": result.no_citation,
                "latency_ms": result.latency.total_ms,
                "retrieval_ms": result.latency.retrieval_ms,
                "generation_ms": result.latency.generation_ms,
                "memory_peak_mb": result.memory.peak_mb,
            }
        )
    return predictions


def _run_multi_turn_eval(pipeline: Any, ds_cfg: dict) -> list[dict]:
    """Run multi-turn session eval set."""
    path = ds_cfg["path"]
    max_sessions = ds_cfg.get("max_sessions", 0)
    max_turns = ds_cfg.get("max_turns_per_session", 99)

    sessions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            sessions.append(json.loads(line))
    if max_sessions > 0:
        sessions = sessions[:max_sessions]

    predictions = []
    for session in sessions:
        sid = session["session_id"]
        turns = session.get("turns", [])[:max_turns]
        for turn in turns:
            result = pipeline.query(
                question=turn["question"],
                session_id=sid,
                repo_id="",
            )
            predictions.append(
                {
                    "session_id": sid,
                    "turn_id": turn.get("turn_id", 0),
                    "question": turn["question"],
                    "answer": result.answer,
                    "cited_chunk_ids": result.cited_chunk_ids,
                    "no_citation": result.no_citation,
                    "latency_ms": result.latency.total_ms,
                    "retrieval_ms": result.latency.retrieval_ms,
                    "generation_ms": result.latency.generation_ms,
                    "memory_peak_mb": result.memory.peak_mb,
                }
            )
    return predictions


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------


def run_variant(variant_cfg: dict, eval_cfg: dict) -> list[dict]:
    """
    Run a single benchmark variant against all enabled eval sets.
    Returns a list of prediction records.
    """
    pipeline = _build_variant_pipeline(variant_cfg)
    datasets = eval_cfg.get("datasets", {})
    predictions: list[dict] = []

    static_cfg = datasets.get("static_eval", {})
    if static_cfg.get("enabled"):
        predictions.extend(_run_static_eval(pipeline, static_cfg))

    mt_cfg = datasets.get("multi_turn_eval", {})
    if mt_cfg.get("enabled"):
        predictions.extend(_run_multi_turn_eval(pipeline, mt_cfg))

    return predictions


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def save_run_predictions(
    run_id: str,
    variant_id: str,
    predictions: list[dict],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_id}_{variant_id}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all benchmark variants")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or (
        f"bench_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    output_dir = Path(cfg["run"]["output_dir"]) / run_id
    print(f"run_id:     {run_id}")
    print(f"output_dir: {output_dir}\n")

    for variant in cfg.get("variants", []):
        print(f"  variant: {variant['id']} ...")
        predictions = run_variant(variant, cfg)
        path = save_run_predictions(run_id, variant["id"], predictions, output_dir)
        print(f"    saved {len(predictions)} predictions -> {path}")

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
