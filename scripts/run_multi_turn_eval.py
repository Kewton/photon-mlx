"""
run_multi_turn_eval.py  –  Run multi-turn session eval against baseline.

Usage:
    python scripts/run_multi_turn_eval.py --max-sessions 5
    python scripts/run_multi_turn_eval.py  # all 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_reporag.citation import is_refusal_answer
from baseline_reporag.config import load_config
from baseline_reporag.contracts import QueryResult

# Issue #143 / Step 3: eval scripts resolve ``cfg.run.seed`` /
# ``cfg.run.deterministic`` and forward the resolved seed into
# ``pipeline.query`` so MLX-LM sampling is deterministic across runs.
from baseline_reporag.eval.run_config import resolve_eval_seed

# CB-004 (codex-fix): lightweight factory import — baseline-only envs no
# longer have to install MLX to run multi-turn evaluation.
from baseline_reporag.pipeline_factory import build_pipeline


def build_prediction_record(
    *,
    session_id: str,
    turn_id: Any,
    question: str,
    result: QueryResult,
) -> dict:
    """Build one JSONL row with refusal-aware accounting (Issue #156).

    ``no_citation`` records raw "did the answer carry any [C:N]" — kept for
    backward compatibility with prior eval logs. ``is_refusal`` flags
    legitimate abstentions ("根拠が不足しています ..."), and ``true_failure``
    is the only field that should be aggregated as a real miss.
    """
    answer = result.answer or ""
    no_citation = bool(not result.cited_chunk_ids)
    is_refusal = is_refusal_answer(answer)
    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "question": question,
        "answer": result.answer,
        "cited_chunk_ids": result.cited_chunk_ids,
        "no_citation": no_citation,
        "is_refusal": is_refusal,
        "true_failure": bool(no_citation and not is_refusal),
        "latency_ms": result.latency.total_ms,
        "retrieval_ms": result.latency.retrieval_ms,
        "generation_ms": result.latency.generation_ms,
        "memory_peak_mb": result.memory.peak_mb,
    }


def _row_is_refusal(row: dict) -> bool:
    """Return ``is_refusal`` if recorded; otherwise recover from ``answer``.

    Older prediction JSONLs (pre-Issue #156) lack ``is_refusal``, so a
    refusal-aware regrade has to fall back to the same phrase check that
    the eval grader uses.
    """
    if "is_refusal" in row:
        return bool(row.get("is_refusal"))
    return is_refusal_answer(str(row.get("answer", "") or ""))


def _row_true_failure(row: dict) -> bool:
    if "true_failure" in row:
        return bool(row.get("true_failure"))
    return bool(row.get("no_citation")) and not _row_is_refusal(row)


def summarize_run_predictions(rows: Iterable[dict]) -> dict:
    """Aggregate prediction rows into a refusal-aware run summary.

    Returns counts and rates for total / no_citation (raw) / refusal /
    true_failure, plus ``p50_latency_ms``. Legacy rows without the new
    fields are recovered via the refusal-phrase check so this same
    function powers ``regrade_eval_with_refusal.py``.
    """
    total = 0
    no_cite = 0
    refusal = 0
    true_fail = 0
    latencies: list[float] = []
    for row in rows:
        total += 1
        if row.get("no_citation"):
            no_cite += 1
        if _row_is_refusal(row):
            refusal += 1
        if _row_true_failure(row):
            true_fail += 1
        lat = row.get("latency_ms", 0.0)
        if isinstance(lat, (int, float)):
            latencies.append(float(lat))
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    return {
        "total_turns": total,
        "no_citation_turns": no_cite,
        "refusal_turns": refusal,
        "true_failure_turns": true_fail,
        "raw_nc_rate": (no_cite / total) if total else 0.0,
        "refusal_rate": (refusal / total) if total else 0.0,
        "true_failure_rate": (true_fail / total) if total else 0.0,
        "p50_latency_ms": p50,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-turn eval")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--eval-set", default="data/eval_sets/multi_turn_eval.jsonl")
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--output", default="")
    # Issue #82 Wave 4: ``--repo-id`` lets the Streamlit eval runner
    # override the repo_id (the default path only has cfg.repo.repo_id).
    parser.add_argument("--repo-id", default="")
    # Issue #82 Wave 4: when the Streamlit app invokes this script it
    # passes a ``--marker-file`` path; we ``touch()`` it on successful
    # completion so the async sync loop can detect success reliably.
    parser.add_argument("--marker-file", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = args.repo_id or cfg.repo.repo_id
    # #154 follow-up: ``build_pipeline(cfg)`` loads the index at
    # ``data/indexes/{cfg.repo.repo_id}``, so ``--repo-id`` must be
    # propagated into ``cfg`` *before* the pipeline is built. Without
    # this, the eval silently loads the YAML default index (e.g.
    # ``fastapi_fastapi``) while ``pipeline.query(repo_id=...)`` filters
    # for the requested repo, leaving retrieval empty. Surfaced when the
    # #154 cross-repo filter started dropping mismatched chunks.
    cfg.repo.repo_id = repo_id

    # Issue #143 / Step 3: resolve the eval seed once per run.  The
    # resolver fails fast on malformed ``cfg.run`` blocks (YAML bool /
    # str / out-of-range int).
    seed = resolve_eval_seed(cfg)

    run_id = f"mt_eval_{repo_id}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Stage 3 DR3-001: route via build_pipeline so PHOTON / baseline
    # providers both take the same path.
    pipeline = build_pipeline(cfg)

    # Load sessions
    sessions = []
    for line in Path(args.eval_set).read_text(encoding="utf-8").splitlines():
        if line.strip():
            sessions.append(json.loads(line))
    if args.max_sessions > 0:
        sessions = sessions[: args.max_sessions]

    print(f"run_id: {run_id}")
    print(f"sessions: {len(sessions)}")
    print()

    # Issue #82 Wave 4: when ``--output`` is a .json path (used by the
    # Streamlit eval runner) predictions land in a sibling .jsonl file so
    # the .json path can carry the UI summary instead.  Classic CLI usage
    # (no --output or .jsonl) keeps the pre-Wave-4 behaviour.
    if args.output:
        output_arg = Path(args.output)
        if output_arg.suffix == ".json":
            predictions_path = output_arg.with_suffix(".predictions.jsonl")
        else:
            predictions_path = output_arg
    else:
        predictions_path = Path(f"logs/{run_id}_predictions.jsonl")
    output_path = predictions_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session_stats: list[dict] = []

    with open(output_path, "w", encoding="utf-8") as f:
        for si, session_data in enumerate(sessions, 1):
            sid = session_data["session_id"]
            turns = session_data["turns"]
            print(f"[Session {si}/{len(sessions)}] {sid} ({len(turns)} turns)")

            turn_latencies: list[float] = []
            turn_citations: list[int] = []
            turn_refusals: list[bool] = []
            turn_true_failures: list[bool] = []

            for turn in turns:
                result = pipeline.query(
                    question=turn["question"],
                    session_id=f"eval-{sid}",
                    repo_id=repo_id,
                    seed=seed,
                )
                pred = build_prediction_record(
                    session_id=sid,
                    turn_id=turn["turn_id"],
                    question=turn["question"],
                    result=result,
                )
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

                turn_latencies.append(result.latency.total_ms)
                turn_citations.append(len(result.cited_chunk_ids))
                turn_refusals.append(bool(pred["is_refusal"]))
                turn_true_failures.append(bool(pred["true_failure"]))

                print(
                    f"  T{turn['turn_id']}: {result.latency.total_ms:.0f}ms "
                    f"cites={len(result.cited_chunk_ids)} "
                    f"refusal={'Y' if pred['is_refusal'] else 'N'}"
                )

            # Session summary
            avg_lat = sum(turn_latencies) / len(turn_latencies)
            followup_lat = sum(turn_latencies[1:]) / max(len(turn_latencies) - 1, 1)
            session_stats.append(
                {
                    "session_id": sid,
                    "avg_latency_ms": avg_lat,
                    "followup_avg_latency_ms": followup_lat,
                    "first_turn_latency_ms": turn_latencies[0],
                    "total_citations": sum(turn_citations),
                    "no_citation_turns": sum(1 for c in turn_citations if c == 0),
                    "refusal_turns": sum(1 for r in turn_refusals if r),
                    "true_failure_turns": sum(1 for t in turn_true_failures if t),
                }
            )
            print(f"  → avg {avg_lat:.0f}ms, follow-up avg {followup_lat:.0f}ms\n")

    # Summary
    print(f"\n{'=' * 50}")
    print("Session summary:")
    for s in session_stats:
        print(
            f"  {s['session_id']}: avg={s['avg_latency_ms']:.0f}ms "
            f"followup={s['followup_avg_latency_ms']:.0f}ms "
            f"cites={s['total_citations']} "
            f"no_cite={s['no_citation_turns']} "
            f"refusal={s['refusal_turns']} "
            f"true_fail={s['true_failure_turns']}"
        )

    # Save summary
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(session_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nPredictions -> {output_path}")
    print(f"Summary -> {summary_path}")

    # Issue #82 Wave 4: when invoked with ``--output`` pointing at a JSON
    # path, also emit the async-eval summary (done_q / total_q / p50 /
    # nc_rate) to that path so the UI sync loop can read it on success.
    # Issue #156 extends this with refusal-aware fields so callers can
    # surface true_failure_rate without re-grading.
    if args.output and args.output.endswith(".json"):
        rows: list[dict] = []
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        run_summary = summarize_run_predictions(rows)
        ui_summary = {
            "done_q": run_summary["total_turns"],
            "total_q": run_summary["total_turns"],
            "p50_latency_ms": run_summary["p50_latency_ms"],
            "nc_rate": run_summary["raw_nc_rate"],
            "refusal_turns": run_summary["refusal_turns"],
            "true_failure_turns": run_summary["true_failure_turns"],
            "true_failure_rate": run_summary["true_failure_rate"],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(ui_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"UI summary -> {args.output}")

    # Issue #82 Wave 4: touch marker_file on successful completion.
    if args.marker_file:
        marker = Path(args.marker_file)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        print(f"Marker -> {marker}")


if __name__ == "__main__":
    main()
