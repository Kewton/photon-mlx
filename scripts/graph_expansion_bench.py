"""Graph-expansion / neighborhood-expansion grid search driver (Issue #91).

Mirrors ``scripts/retrieval_grid_search.py``: single loaded pipeline,
per-config override via ``Config.merge_override``, atomic per-config JSON,
and SIGTERM resume.

See ``workspace/design/issue-91-graph-neighborhood-design-policy.md`` for
the full design.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._graph_bench_core import (  # noqa: E402
    GraphBenchParams,
    atomic_write_json,
    generate_graph_bench_grid,
    generate_graph_bench_phase2,
)
from scripts._grid_search_core import aggregate_metrics  # noqa: E402

if TYPE_CHECKING:
    from baseline_reporag.config import Config
    from baseline_reporag.pipeline import RepoRAGPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graph/Neighborhood expansion grid search (Issue #91)."
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--eval-set", default="data/eval_sets/static_v1.jsonl")
    parser.add_argument(
        "--output",
        default="reports/graph_expansion_bench.json",
        help="Atomic per-config JSON state path.",
    )
    parser.add_argument("--phase", choices=["1", "2"], default="1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first 2 phase-1 configs (for CI / local sanity).",
    )
    parser.add_argument("--max-questions", type=int, default=40)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument(
        "--seed-json",
        default=None,
        help="Phase 2 only: path to a phase-1 JSON whose top configs will seed phase 2.",
    )
    parser.add_argument("--top-n-for-phase2", type=int, default=3)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Pipeline build + eval helpers
# ---------------------------------------------------------------------------


def build_pipeline_for_sweep(
    base_cfg: "Config",
    scratch_log_root: Path,
) -> "RepoRAGPipeline":
    from baseline_reporag.pipeline_factory import build_pipeline

    scratch_log_root.mkdir(parents=True, exist_ok=True)
    sweep_cfg = base_cfg.merge_override({"paths": {"log_root": str(scratch_log_root)}})
    return build_pipeline(sweep_cfg)  # type: ignore[return-value]


def load_questions(path: Path, max_questions: int) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line))
    if max_questions > 0:
        questions = questions[:max_questions]
    return questions


def unanswerable_ids_from(questions: list[dict[str, Any]]) -> set[str]:
    return {q["id"] for q in questions if q.get("answerable") is False}


def run_eval_inproc(
    pipeline: "RepoRAGPipeline",
    questions: list[dict[str, Any]],
    *,
    config_idx: int,
    repo_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for q in questions:
        result = pipeline.query(
            question=q["question"],
            session_id=f"graphbench-{config_idx}-{q['id']}",
            repo_id=repo_id,
        )
        latency_ms = float(getattr(result.latency, "total_ms", 0.0))
        memory_peak = float(getattr(result.memory, "peak_mb", 0.0))
        records.append(
            {
                "eval_id": q["id"],
                "no_citation": bool(result.no_citation),
                "wrong_citation_indices": list(result.wrong_citation_indices or []),
                "latency_ms": latency_ms,
                "memory_peak_mb": memory_peak,
            }
        )
    return records


# ---------------------------------------------------------------------------
# State + sweep loop
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _params_key(params: GraphBenchParams) -> str:
    return json.dumps(params.to_override_dict(), sort_keys=True)


def _build_config_entry(
    config_idx: int,
    params: GraphBenchParams,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    started_at: str,
    completed_at: str,
    duration_seconds: float,
) -> dict[str, Any]:
    memory_peaks = [
        r.get("memory_peak_mb", 0.0)
        for r in records
        if r.get("memory_peak_mb") is not None
    ]
    peak = max(memory_peaks) if memory_peaks else None
    entry: dict[str, Any] = {
        "config_idx": config_idx,
        "params": params.to_override_dict(),
        "raw_no_citation_rate": float(metrics.get("no_citation_rate", 0.0)),
        "true_nc_rate": float(metrics.get("true_nc_rate", 0.0) or 0.0),
        "wrong_citation_count": int(metrics.get("wrong_citation_count", 0)),
        "latency_p50_ms": float(metrics.get("latency_p50", 0.0)),
        "latency_p95_ms": float(metrics.get("latency_p95", 0.0)),
        "n_questions": int(metrics.get("n_questions", len(records))),
        "n_no_citation": int(metrics.get("n_no_citation", 0)),
        "duration_seconds": duration_seconds,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    if peak is not None:
        entry["memory_peak_mb"] = peak
    return entry


class _InterruptedError(BaseException):
    pass


def _install_signal_handlers(flag: dict[str, bool]) -> None:
    def _handler(signum: int, _frame: FrameType | None) -> None:
        logger.warning("Received signal %s, flushing and exiting.", signum)
        flag["stop"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_grid(
    base_cfg: "Config",
    grid: list[GraphBenchParams],
    *,
    pipeline: "RepoRAGPipeline",
    questions: list[dict[str, Any]],
    unanswerable_ids: set[str],
    output_path: Path,
    repo_id: str,
    resume: bool,
    phase_name: str,
    stop_flag: dict[str, bool],
) -> dict[str, Any]:
    existing = _load_state(output_path) if resume else None
    if existing and resume:
        state = existing
    else:
        state = {
            "phase": phase_name,
            "base_config_path": getattr(base_cfg, "_config_path", ""),
            "max_questions_per_config": len(questions),
            "started_at": _now_iso(),
            "completed_at": "",
            "configs": [],
            "status": "running",
        }

    done_keys = {
        json.dumps(c.get("params", {}), sort_keys=True)
        for c in state.get("configs", [])
    }

    sweep_start = time.perf_counter()
    total = len(grid)
    for idx, params in enumerate(grid):
        if stop_flag.get("stop"):
            state["status"] = "interrupted"
            state["interrupted_at"] = _now_iso()
            atomic_write_json(output_path, state)
            raise _InterruptedError("interrupted by signal")

        key = _params_key(params)
        if resume and key in done_keys:
            logger.info(
                "[%s] skipping already-completed config %d/%d",
                phase_name,
                idx + 1,
                total,
            )
            continue

        started_at = _now_iso()
        t0 = time.perf_counter()
        logger.info(
            "[%s] [Config %d/%d] starting params=%s",
            phase_name,
            idx + 1,
            total,
            params.to_override_dict(),
        )

        cfg2 = base_cfg.merge_override({"retrieval": params.to_override_dict()})
        pipeline.cfg = cfg2  # type: ignore[attr-defined]

        try:
            records = run_eval_inproc(
                pipeline, questions, config_idx=idx, repo_id=repo_id
            )
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            state["interrupted_at"] = _now_iso()
            atomic_write_json(output_path, state)
            raise

        metrics = aggregate_metrics(records, unanswerable_ids)
        duration = time.perf_counter() - t0
        completed_at = _now_iso()
        entry = _build_config_entry(
            idx,
            params,
            records,
            metrics,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )
        state["configs"].append(entry)
        state["completed_at"] = completed_at
        done_keys.add(key)
        atomic_write_json(output_path, state)

        elapsed = time.perf_counter() - sweep_start
        logger.info(
            "[%s] [Config %d/%d] NC=%.1f%% p50=%.0fms elapsed=%.1fs",
            phase_name,
            idx + 1,
            total,
            100.0 * entry["raw_no_citation_rate"],
            entry["latency_p50_ms"],
            elapsed,
        )

    state["status"] = f"{phase_name}_complete"
    state["completed_at"] = _now_iso()
    atomic_write_json(output_path, state)
    return state


# ---------------------------------------------------------------------------
# Phase 2 seed selection
# ---------------------------------------------------------------------------


def _seeds_from_phase1_json(path: Path, top_n: int) -> list[GraphBenchParams]:
    data = json.loads(path.read_text(encoding="utf-8"))
    configs = data.get("configs", [])
    ordered = sorted(
        configs,
        key=lambda c: (
            c.get("raw_no_citation_rate", 1.0),
            c.get("latency_p50_ms", 0.0),
        ),
    )[:top_n]
    seeds: list[GraphBenchParams] = []
    for c in ordered:
        p = c.get("params", {})
        ge = p.get("graph_expansion", {})
        ne = p.get("neighborhood_expansion", {})
        ew = ge.get("edge_weights", {})
        seeds.append(
            GraphBenchParams(
                max_hops=int(ge.get("max_hops", 1)),
                max_nodes=int(ge.get("max_nodes", 24)),
                neighborhood_before=int(ne.get("before", 1)),
                neighborhood_after=int(ne.get("after", 1)),
                edge_weights_call=float(ew.get("call", 1.0)),
                edge_weights_inherit=float(ew.get("inherit", 0.8)),
                edge_weights_import=float(ew.get("import", 0.5)),
                adaptive_neighborhood=bool(ne.get("adaptive", False)),
            )
        )
    return seeds


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    eval_set_path = Path(args.eval_set)
    if not eval_set_path.is_absolute():
        eval_set_path = REPO_ROOT / eval_set_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_absolute():
        logs_dir = REPO_ROOT / logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    from baseline_reporag.config import load_config

    base_cfg = load_config(config_path)
    repo_id = base_cfg.repo.repo_id

    if args.phase == "1":
        grid = generate_graph_bench_grid()
        phase_name = "phase1"
    else:
        seed_json = args.seed_json
        if seed_json is None:
            # Default: look for the phase-1 state sitting next to --output.
            default_seed = output_path.with_name("graph_expansion_bench_phase1.json")
            if default_seed.exists():
                seed_json = str(default_seed)
            else:
                logger.error(
                    "phase 2 requires --seed-json pointing at a phase-1 JSON "
                    "(or a sibling graph_expansion_bench_phase1.json)."
                )
                return 2
        seeds = _seeds_from_phase1_json(Path(seed_json), args.top_n_for_phase2)
        grid = generate_graph_bench_phase2(seeds)
        phase_name = "phase2"

    if args.smoke:
        grid = grid[:2]
        logger.info("SMOKE: limiting grid to %d configs", len(grid))

    if not grid:
        logger.error("Empty grid — nothing to run.")
        return 2

    questions = load_questions(eval_set_path, args.max_questions)
    unanswerable = unanswerable_ids_from(questions)
    logger.info(
        "Loaded %d questions (unanswerable: %d)", len(questions), len(unanswerable)
    )

    stop_flag: dict[str, bool] = {"stop": False}
    _install_signal_handlers(stop_flag)

    pipeline = build_pipeline_for_sweep(base_cfg, logs_dir / "graph_bench_scratch")
    try:
        run_grid(
            base_cfg,
            grid,
            pipeline=pipeline,
            questions=questions,
            unanswerable_ids=unanswerable,
            output_path=output_path,
            repo_id=repo_id,
            resume=args.resume,
            phase_name=phase_name,
            stop_flag=stop_flag,
        )
    except _InterruptedError:
        logger.warning("Sweep interrupted; state flushed to %s", output_path)
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
