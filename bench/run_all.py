"""
run_all.py  –  Run all benchmark variants defined in eval.yaml.

Usage:
    python bench/run_all.py --config configs/eval.yaml
    python bench/run_all.py --config configs/eval.yaml --variants id1,id2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# --variants CSV helpers (Issue #92 T-0b)
# ---------------------------------------------------------------------------


def parse_variants_csv(raw: str | None) -> list[str]:
    """Split a CSV string into a list of variant ids.

    Pure-Python ``str.split(',')`` only — no shell / subprocess (DR4-002
    security contract).

    Codex CB-003 fail-closed (Issue #92 T-0b): there are three inputs we
    must distinguish so malformed ``--variants`` cannot silently escalate
    to "run all variants":

    * ``raw is None``       → flag not passed → return ``[]`` (no filter).
    * ``raw`` is non-empty and every comma-separated token is a non-empty
      non-whitespace string → return the stripped token list.
    * Otherwise (``raw == ""``, ``","``, ``"a,,b"``, ``" "``, ``",a"``,
      ``"a,"``, ...) the CSV is malformed. Raise
      :class:`argparse.ArgumentTypeError` WITHOUT embedding the raw value
      (DR4-001 no-leak — the attacker-controlled string must not reach
      logs / tracebacks).

    The ``None`` vs. empty-CSV distinction is what keeps :func:`filter_variants`
    from fail-OPEN-ing on a typo: previously ``--variants ','`` collapsed
    to ``[]`` and then the ``if not selected`` branch in ``filter_variants``
    treated it as "no filter → all variants run".
    """
    if raw is None:
        return []
    # Empty / whitespace-only CSV is a hard error (distinct from ``None``).
    if not raw.strip():
        raise argparse.ArgumentTypeError(
            "--variants must be a non-empty comma-separated list of ids"
        )
    tokens = raw.split(",")
    stripped = [tok.strip() for tok in tokens]
    if any(not tok for tok in stripped):
        # Any empty middle / leading / trailing token fails closed.
        raise argparse.ArgumentTypeError(
            "--variants must not contain empty or whitespace-only tokens"
        )
    return stripped


def filter_variants(
    variants: list[dict],
    selected: list[str] | None,
) -> list[dict]:
    """Filter ``variants`` to those whose id is in ``selected`` (CSV-ordered).

    ``selected`` semantics:

    * ``None`` or ``[]`` → return ``variants`` unchanged (no filter applied).
    * Otherwise: every id in ``selected`` MUST match a variant's ``id`` via
      exact string equality. An unknown id raises
      :class:`argparse.ArgumentError` with a fail-closed message that
      intentionally excludes the raw invalid token (DR4-001 no-leak; only
      the allowed-id count is surfaced).

    Order in the returned list matches the CSV order so downstream reports
    are deterministic when re-running a subset.
    """
    if not selected:
        return list(variants)

    by_id: dict[str, dict] = {v["id"]: v for v in variants}
    result: list[dict] = []
    for sid in selected:
        if sid not in by_id:
            # No-leak: the unknown id is attacker-controlled; surface only
            # the count of allowed ids so operators can diagnose without
            # seeing attacker payload. Raise via argparse so CLI entry
            # point fails closed with a clean usage error.
            raise argparse.ArgumentError(
                None,
                f"--variants contains an unknown id; allowed ids count = {len(by_id)}",
            )
        result.append(by_id[sid])
    return result


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
                "generator_used": getattr(result, "generator_used", None),
                "generator_fallback_reason": getattr(
                    result, "generator_fallback_reason", None
                ),
                "latency_ms": result.latency.total_ms,
                "retrieval_ms": result.latency.retrieval_ms,
                "generation_ms": result.latency.generation_ms,
                "memory_peak_mb": result.memory.peak_mb,
            }
        )
    return predictions


def _iter_static_eval_predictions(pipeline: Any, ds_cfg: dict):
    """Yield static eval predictions one by one."""
    path = ds_cfg["path"]
    max_cases = ds_cfg.get("max_cases", 0)

    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line))
    if max_cases > 0:
        questions = questions[:max_cases]

    for q in questions:
        result = pipeline.query(
            question=q["question"],
            session_id=f"eval-{q['id']}",
            repo_id="",
        )
        yield {
            "eval_id": q["id"],
            "category": q.get("category", ""),
            "question": q["question"],
            "answer": result.answer,
            "cited_chunk_ids": result.cited_chunk_ids,
            "no_citation": result.no_citation,
            "generator_used": getattr(result, "generator_used", None),
            "generator_fallback_reason": getattr(
                result, "generator_fallback_reason", None
            ),
            "latency_ms": result.latency.total_ms,
            "retrieval_ms": result.latency.retrieval_ms,
            "generation_ms": result.latency.generation_ms,
            "memory_peak_mb": result.memory.peak_mb,
        }


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
                    "generator_used": getattr(result, "generator_used", None),
                    "generator_fallback_reason": getattr(
                        result, "generator_fallback_reason", None
                    ),
                    "latency_ms": result.latency.total_ms,
                    "retrieval_ms": result.latency.retrieval_ms,
                    "generation_ms": result.latency.generation_ms,
                    "memory_peak_mb": result.memory.peak_mb,
                }
            )
    return predictions


def _iter_multi_turn_eval_predictions(pipeline: Any, ds_cfg: dict):
    """Yield multi-turn eval predictions one by one."""
    path = ds_cfg["path"]
    max_sessions = ds_cfg.get("max_sessions", 0)
    max_turns = ds_cfg.get("max_turns_per_session", 99)

    sessions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            sessions.append(json.loads(line))
    if max_sessions > 0:
        sessions = sessions[:max_sessions]

    for session in sessions:
        sid = session["session_id"]
        turns = session.get("turns", [])[:max_turns]
        for turn in turns:
            result = pipeline.query(
                question=turn["question"],
                session_id=sid,
                repo_id="",
            )
            yield {
                "session_id": sid,
                "turn_id": turn.get("turn_id", 0),
                "question": turn["question"],
                "answer": result.answer,
                "cited_chunk_ids": result.cited_chunk_ids,
                "no_citation": result.no_citation,
                "generator_used": getattr(result, "generator_used", None),
                "generator_fallback_reason": getattr(
                    result, "generator_fallback_reason", None
                ),
                "latency_ms": result.latency.total_ms,
                "retrieval_ms": result.latency.retrieval_ms,
                "generation_ms": result.latency.generation_ms,
                "memory_peak_mb": result.memory.peak_mb,
            }


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------


def iter_variant_predictions(variant_cfg: dict, eval_cfg: dict):
    """Yield predictions for a single variant."""
    pipeline = _build_variant_pipeline(variant_cfg)
    datasets = eval_cfg.get("datasets", {})

    static_cfg = datasets.get("static_eval", {})
    if static_cfg.get("enabled"):
        yield from _iter_static_eval_predictions(pipeline, static_cfg)

    mt_cfg = datasets.get("multi_turn_eval", {})
    if mt_cfg.get("enabled"):
        yield from _iter_multi_turn_eval_predictions(pipeline, mt_cfg)


def run_variant(variant_cfg: dict, eval_cfg: dict) -> list[dict]:
    """
    Run a single benchmark variant against all enabled eval sets.
    Returns a list of prediction records.
    """
    return list(iter_variant_predictions(variant_cfg, eval_cfg))


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


def append_prediction(path: Path, prediction: dict) -> None:
    """Append a single JSONL prediction record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        f.flush()


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _dataset_case_count(
    path: Path, max_items: int, *, multi_turn: bool, max_turns: int = 0
) -> int:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_items > 0:
        records = records[:max_items]
    if not multi_turn:
        return len(records)
    return sum(len(record.get("turns", [])[:max_turns]) for record in records)


def _expected_predictions_for_eval(eval_cfg: dict) -> int:
    datasets = eval_cfg.get("datasets", {})
    total = 0

    static_cfg = datasets.get("static_eval", {})
    if static_cfg.get("enabled"):
        total += _dataset_case_count(
            Path(static_cfg["path"]),
            int(static_cfg.get("max_cases", 0)),
            multi_turn=False,
        )

    mt_cfg = datasets.get("multi_turn_eval", {})
    if mt_cfg.get("enabled"):
        total += _dataset_case_count(
            Path(mt_cfg["path"]),
            int(mt_cfg.get("max_sessions", 0)),
            multi_turn=True,
            max_turns=int(mt_cfg.get("max_turns_per_session", 99)),
        )

    return total


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all benchmark variants")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip variants whose output JSONL already exists with the expected record count.",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help=(
            "Optional comma-separated list of variant ids to run. Unknown "
            "ids fail closed (Issue #92 T-0b)."
        ),
    )
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or (
        f"bench_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    output_dir = Path(cfg["run"]["output_dir"]) / run_id
    progress_path = output_dir / f"{run_id}_progress.json"
    print(f"run_id:     {run_id}")
    print(f"output_dir: {output_dir}\n")

    # Codex CB-003: ``parse_variants_csv`` raises ``ArgumentTypeError`` on
    # malformed CSV (empty tokens). Route it through ``parser.error`` so
    # the CLI exits with a clean usage message (same treatment as the
    # unknown-id case below). The raw value is already sanitized away
    # inside ``parse_variants_csv``.
    try:
        selected_ids = parse_variants_csv(args.variants)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    try:
        variants_to_run = filter_variants(cfg.get("variants", []), selected_ids)
    except argparse.ArgumentError as exc:
        parser.error(str(exc))

    expected_per_variant = _expected_predictions_for_eval(cfg)
    total_expected = expected_per_variant * len(variants_to_run)
    total_completed = 0
    completed_variants: list[str] = []

    for variant in variants_to_run:
        variant_id = variant["id"]
        path = output_dir / f"{run_id}_{variant_id}.jsonl"
        existing_count = _count_jsonl_records(path)

        if (
            args.resume
            and existing_count == expected_per_variant
            and existing_count > 0
        ):
            completed_variants.append(variant_id)
            total_completed += existing_count
            print(
                f"  variant: {variant_id} ... skipped (resume, {existing_count}/{expected_per_variant})"
            )
            _write_progress(
                progress_path,
                {
                    "run_id": run_id,
                    "status": "running",
                    "current_variant": None,
                    "completed_variants": completed_variants,
                    "total_variants": len(variants_to_run),
                    "variant_expected_predictions": expected_per_variant,
                    "total_expected_predictions": total_expected,
                    "total_completed_predictions": total_completed,
                },
            )
            continue

        if path.exists():
            if existing_count:
                print(
                    f"  variant: {variant_id} ... restarting from scratch "
                    f"(existing partial output {existing_count}/{expected_per_variant})"
                )
            path.unlink()
        else:
            print(f"  variant: {variant_id} ...")

        _write_progress(
            progress_path,
            {
                "run_id": run_id,
                "status": "running",
                "current_variant": variant_id,
                "completed_variants": completed_variants,
                "total_variants": len(variants_to_run),
                "variant_expected_predictions": expected_per_variant,
                "current_variant_completed_predictions": 0,
                "total_expected_predictions": total_expected,
                "total_completed_predictions": total_completed,
            },
        )

        variant_count = 0
        for prediction in iter_variant_predictions(variant, cfg):
            append_prediction(path, prediction)
            variant_count += 1
            total_completed += 1
            if variant_count % 10 == 0 or variant_count == expected_per_variant:
                print(
                    f"    progress {variant_count}/{expected_per_variant}", flush=True
                )
            _write_progress(
                progress_path,
                {
                    "run_id": run_id,
                    "status": "running",
                    "current_variant": variant_id,
                    "completed_variants": completed_variants,
                    "total_variants": len(variants_to_run),
                    "variant_expected_predictions": expected_per_variant,
                    "current_variant_completed_predictions": variant_count,
                    "current_output_path": str(path),
                    "total_expected_predictions": total_expected,
                    "total_completed_predictions": total_completed,
                },
            )

        completed_variants.append(variant_id)
        print(f"    saved {variant_count} predictions -> {path}")
        _write_progress(
            progress_path,
            {
                "run_id": run_id,
                "status": "running",
                "current_variant": None,
                "completed_variants": completed_variants,
                "total_variants": len(variants_to_run),
                "variant_expected_predictions": expected_per_variant,
                "total_expected_predictions": total_expected,
                "total_completed_predictions": total_completed,
            },
        )

    _write_progress(
        progress_path,
        {
            "run_id": run_id,
            "status": "completed",
            "current_variant": None,
            "completed_variants": completed_variants,
            "total_variants": len(variants_to_run),
            "variant_expected_predictions": expected_per_variant,
            "total_expected_predictions": total_expected,
            "total_completed_predictions": total_completed,
        },
    )

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
