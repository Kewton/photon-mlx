"""Retrieval parameter grid-search driver (Issue #88).

Sweeps ``configs/baseline.yaml`` ``retrieval.*`` parameters in a single
process, reusing one loaded pipeline across all configs. Writes per-config
progress to ``reports/retrieval_grid_search.json`` atomically and renders
``reports/retrieval_grid_search.md`` at the end.

See ``workspace/design/issue-88-retrieval-grid-search-design-policy.md``
for the full design rationale. The pure-function helpers live in
``scripts/_grid_search_core`` and are unit-tested without MLX.
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

from scripts._grid_search_core import (  # noqa: E402
    ConfigParams,
    ConfigResult,
    aggregate_metrics,
    atomic_write_json,
    generate_phase1_grid,
    generate_phase2_grid,
    validate_override,
    write_markdown_report,
)

if TYPE_CHECKING:
    from baseline_reporag.config import Config
    from baseline_reporag.pipeline import RepoRAGPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI parsing + path validation
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval parameter grid search (Issue #88)."
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--eval-set", default="data/eval_sets/static_eval.jsonl")
    parser.add_argument("--phase", choices=["1", "2", "both"], default="both")
    parser.add_argument("--max-questions", type=int, default=40)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--top-n-for-phase2", type=int, default=5)
    parser.add_argument("--phase2-topk-delta", type=int, default=5)
    parser.add_argument("--phase2-weight-delta", type=float, default=0.05)
    return parser.parse_args(argv)


def _repo_relative(p: str | Path) -> Path:
    """Resolve ``p`` and assert it stays under ``REPO_ROOT``.

    Rejects absolute paths, ``..`` traversal, and symlink escapes. The
    grid-search tool is trusted-operator-only; this guard exists to keep
    an accidental ``--config /etc/shadow`` from reaching ``pickle.load``.
    """
    raw = Path(p)
    if raw.is_absolute():
        raise SystemExit(f"error: path must be repo-relative, got absolute: {p!r}")
    resolved = (REPO_ROOT / raw).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(f"error: path escapes repo root: {p!r}") from exc
    return resolved


def _validate_paths(args: argparse.Namespace) -> None:
    _repo_relative(args.config)
    _repo_relative(args.eval_set)
    _repo_relative(args.output_dir)
    _repo_relative(args.logs_dir)


# ---------------------------------------------------------------------------
# Pipeline build + eval helpers
# ---------------------------------------------------------------------------


def build_pipeline_for_sweep(
    base_cfg: Config,
    scratch_log_root: Path,
) -> "RepoRAGPipeline":
    """Load a single pipeline instance, pointing scratch logs to *scratch_log_root*."""
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
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run ``pipeline.query`` once per question, collect citation records.

    Each question uses ``session_id = f"grid-{config_idx}-{q['id']}"`` to
    keep session history from leaking across configs.

    Issue #143 (DR3-001): ``seed`` is forwarded into every ``query``
    call so the grid sweep observes the same Qwen sampling stream as
    the production eval scripts.  Default ``None`` keeps the legacy
    call shape used by ``test_retrieval_grid_search_smoke.py``.
    """
    records: list[dict[str, Any]] = []
    for q in questions:
        result = pipeline.query(
            question=q["question"],
            session_id=f"grid-{config_idx}-{q['id']}",
            repo_id=repo_id,
            seed=seed,
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
# Sweep loop + resume + atomic write
# ---------------------------------------------------------------------------


def _params_key(params: ConfigParams) -> str:
    return json.dumps(params.to_override_dict(), sort_keys=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _load_state(state_path: Path) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read existing state at %s", state_path)
        return None


def _build_config_result(
    config_idx: int,
    params: ConfigParams,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    started_at: str,
    completed_at: str,
    duration_seconds: float,
) -> ConfigResult:
    memory_peaks = [
        r.get("memory_peak_mb", 0.0)
        for r in records
        if r.get("memory_peak_mb") is not None
    ]
    peak = max(memory_peaks) if memory_peaks else None
    return ConfigResult(
        config_idx=config_idx,
        params=params,
        raw_no_citation_rate=float(metrics.get("no_citation_rate", 0.0)),
        true_nc_rate=float(metrics.get("true_nc_rate", 0.0) or 0.0),
        wrong_citation_count=int(metrics.get("wrong_citation_count", 0)),
        latency_p50_ms=float(metrics.get("latency_p50", 0.0)),
        latency_p95_ms=float(metrics.get("latency_p95", 0.0)),
        n_questions=int(metrics.get("n_questions", len(records))),
        n_no_citation=int(metrics.get("n_no_citation", 0)),
        duration_seconds=duration_seconds,
        started_at=started_at,
        completed_at=completed_at,
        memory_peak_mb=peak,
    )


class _InterruptedError(BaseException):
    """Raised when a signal handler requests graceful shutdown."""


def _install_signal_handlers(flag: dict[str, bool]) -> None:
    def _handler(signum: int, _frame: FrameType | None) -> None:
        logger.warning("Received signal %s, flushing and exiting.", signum)
        flag["stop"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # SIGTERM not available on every platform / thread context.
            pass


def run_phase(
    base_cfg: Config,
    grid: list[ConfigParams],
    *,
    questions: list[dict[str, Any]],
    unanswerable_ids: set[str],
    pipeline: "RepoRAGPipeline",
    state_path: Path,
    resume: bool,
    phase_name: str,
    repo_id: str,
    stop_flag: dict[str, bool],
    seed: int | None = None,
) -> list[ConfigResult]:
    """Execute one phase of the sweep, atomically persisting per-config state."""
    state: dict[str, Any]
    existing = _load_state(state_path) if resume else None
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

    results: list[ConfigResult] = []
    total = len(grid)
    sweep_start = time.perf_counter()

    for idx, params in enumerate(grid):
        if stop_flag.get("stop"):
            state["status"] = "interrupted"
            state["interrupted_at"] = _now_iso()
            atomic_write_json(state_path, state)
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
                pipeline,
                questions,
                config_idx=idx,
                repo_id=repo_id,
                seed=seed,
            )
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            state["interrupted_at"] = _now_iso()
            atomic_write_json(state_path, state)
            raise

        metrics = aggregate_metrics(records, unanswerable_ids)
        duration = time.perf_counter() - t0
        completed_at = _now_iso()
        result = _build_config_result(
            idx, params, records, metrics, started_at, completed_at, duration
        )
        results.append(result)

        state["configs"].append(result.to_json())
        state["completed_at"] = completed_at
        done_keys.add(key)
        atomic_write_json(state_path, state)

        elapsed = time.perf_counter() - sweep_start
        logger.info(
            "[%s] [Config %d/%d] NC=%.1f%% (%d/%d) p50=%.0fms elapsed=%.1fs",
            phase_name,
            idx + 1,
            total,
            100.0 * result.raw_no_citation_rate,
            result.n_no_citation,
            result.n_questions,
            result.latency_p50_ms,
            elapsed,
        )

    state["status"] = f"{phase_name}_complete"
    state["completed_at"] = _now_iso()
    atomic_write_json(state_path, state)
    return results


# ---------------------------------------------------------------------------
# Best config selection + phase 2 gating
# ---------------------------------------------------------------------------


def pick_best(results: list[ConfigResult]) -> ConfigResult | None:
    if not results:
        return None
    return min(
        results,
        key=lambda r: (r.raw_no_citation_rate, r.latency_p50_ms),
    )


def top_n(results: list[ConfigResult], n: int) -> list[ConfigResult]:
    return sorted(results, key=lambda r: (r.raw_no_citation_rate, r.latency_p50_ms))[:n]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    _validate_paths(args)

    config_path = _repo_relative(args.config)
    eval_set_path = _repo_relative(args.eval_set)
    output_dir = _repo_relative(args.output_dir)
    logs_dir = _repo_relative(args.logs_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    from baseline_reporag.config import load_config

    # Issue #143 / Step 3 (DR3-001): grid sweep observes the same Qwen
    # sampling stream as production eval scripts via ``cfg.run.seed``.
    from baseline_reporag.eval.run_config import resolve_eval_seed

    base_cfg = load_config(config_path)
    repo_id = base_cfg.repo.repo_id
    seed = resolve_eval_seed(base_cfg)

    phase1_grid = generate_phase1_grid()
    for params in phase1_grid:
        validate_override(base_cfg, params)

    logger.info("Phase 1 grid: %d configs", len(phase1_grid))

    if args.dry_run:
        _print_dry_run(phase1_grid)
        return 0

    questions = load_questions(eval_set_path, args.max_questions)
    unanswerable = unanswerable_ids_from(questions)
    logger.info(
        "Loaded %d questions (unanswerable: %d)", len(questions), len(unanswerable)
    )

    scratch_root = logs_dir / "grid_search_scratch"
    pipeline = build_pipeline_for_sweep(base_cfg, scratch_root)

    state_path = output_dir / "retrieval_grid_search.json"
    report_path = output_dir / "retrieval_grid_search.md"

    stop_flag: dict[str, bool] = {"stop": False}
    _install_signal_handlers(stop_flag)

    all_results: list[ConfigResult] = []

    try:
        if args.phase in ("1", "both"):
            phase1_results = run_phase(
                base_cfg,
                phase1_grid,
                questions=questions,
                unanswerable_ids=unanswerable,
                pipeline=pipeline,
                state_path=state_path,
                resume=args.resume,
                phase_name="phase1",
                repo_id=repo_id,
                stop_flag=stop_flag,
                seed=seed,
            )
            all_results.extend(phase1_results)

        if args.phase in ("2", "both"):
            seeds_source = all_results or _reload_results_from_state(state_path)
            seeds = [r.params for r in top_n(seeds_source, args.top_n_for_phase2)]
            if not seeds:
                logger.warning("No Phase 1 seeds available; skipping Phase 2.")
            else:
                best_so_far = pick_best(seeds_source)
                if (
                    args.phase == "both"
                    and best_so_far is not None
                    and best_so_far.raw_no_citation_rate > 0.18
                ):
                    logger.warning(
                        "Phase 1 best NC=%.1f%% > 18%%; skipping Phase 2.",
                        100.0 * best_so_far.raw_no_citation_rate,
                    )
                else:
                    phase2_grid = generate_phase2_grid(
                        seeds,
                        topk_delta=args.phase2_topk_delta,
                        weight_delta=args.phase2_weight_delta,
                    )
                    logger.info(
                        "Phase 2 grid: %d configs around top-%d seeds",
                        len(phase2_grid),
                        len(seeds),
                    )
                    phase2_results = run_phase(
                        base_cfg,
                        phase2_grid,
                        questions=questions,
                        unanswerable_ids=unanswerable,
                        pipeline=pipeline,
                        state_path=state_path,
                        resume=args.resume,
                        phase_name="phase2",
                        repo_id=repo_id,
                        stop_flag=stop_flag,
                        seed=seed,
                    )
                    all_results.extend(phase2_results)
    except _InterruptedError:
        logger.warning("Sweep interrupted; partial state preserved in %s", state_path)
        return 130

    final_state = _load_state(state_path) or {}
    all_config_results = _reload_results_from_state(state_path)
    best = pick_best(all_config_results)
    if best is not None:
        write_markdown_report(final_state, best, report_path)
        logger.info(
            "Best config: idx=%d raw_nc=%.2f%% p50=%.0fms",
            best.config_idx,
            100.0 * best.raw_no_citation_rate,
            best.latency_p50_ms,
        )
    else:
        logger.warning("No results available; Markdown report not written.")
    return 0


def _reload_results_from_state(state_path: Path) -> list[ConfigResult]:
    state = _load_state(state_path)
    if not state:
        return []
    out: list[ConfigResult] = []
    for entry in state.get("configs", []):
        params = ConfigParams.from_dict(entry.get("params", {}))
        out.append(
            ConfigResult(
                config_idx=int(entry.get("config_idx", 0)),
                params=params,
                raw_no_citation_rate=float(entry.get("raw_no_citation_rate", 0.0)),
                true_nc_rate=float(entry.get("true_nc_rate", 0.0) or 0.0),
                wrong_citation_count=int(entry.get("wrong_citation_count", 0)),
                latency_p50_ms=float(entry.get("latency_p50_ms", 0.0)),
                latency_p95_ms=float(entry.get("latency_p95_ms", 0.0)),
                n_questions=int(entry.get("n_questions", 0)),
                n_no_citation=int(entry.get("n_no_citation", 0)),
                duration_seconds=float(entry.get("duration_seconds", 0.0)),
                started_at=str(entry.get("started_at", "")),
                completed_at=str(entry.get("completed_at", "")),
                memory_peak_mb=entry.get("memory_peak_mb"),
            )
        )
    return out


def _print_dry_run(grid: list[ConfigParams]) -> None:
    print(f"Phase 1 grid contains {len(grid)} configs:")
    for idx, params in enumerate(grid):
        print(f"  [{idx:2d}] {json.dumps(params.to_override_dict(), sort_keys=True)}")


if __name__ == "__main__":
    sys.exit(main())
