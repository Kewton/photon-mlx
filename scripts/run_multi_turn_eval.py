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

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_reporag.config import load_config

# CB-004 (codex-fix): lightweight factory import — baseline-only envs no
# longer have to install MLX to run multi-turn evaluation.
from baseline_reporag.pipeline_factory import build_pipeline


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

            for turn in turns:
                result = pipeline.query(
                    question=turn["question"],
                    session_id=f"eval-{sid}",
                    repo_id=repo_id,
                )
                pred = {
                    "session_id": sid,
                    "turn_id": turn["turn_id"],
                    "question": turn["question"],
                    "answer": result.answer,
                    "cited_chunk_ids": result.cited_chunk_ids,
                    "no_citation": result.no_citation,
                    "latency_ms": result.latency.total_ms,
                    "retrieval_ms": result.latency.retrieval_ms,
                    "generation_ms": result.latency.generation_ms,
                    "memory_peak_mb": result.memory.peak_mb,
                }
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

                turn_latencies.append(result.latency.total_ms)
                turn_citations.append(len(result.cited_chunk_ids))

                print(
                    f"  T{turn['turn_id']}: {result.latency.total_ms:.0f}ms "
                    f"cites={len(result.cited_chunk_ids)}"
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
            f"no_cite_turns={s['no_citation_turns']}"
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
    if args.output and args.output.endswith(".json"):
        total_latencies: list[float] = []
        total_no_cite = 0
        total_q = 0
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_q += 1
            lat = rec.get("latency_ms", 0.0)
            if isinstance(lat, (int, float)):
                total_latencies.append(float(lat))
            if rec.get("no_citation"):
                total_no_cite += 1
        total_latencies.sort()
        p50 = total_latencies[len(total_latencies) // 2] if total_latencies else 0.0
        nc_rate = (total_no_cite / total_q) if total_q > 0 else 0.0
        ui_summary = {
            "done_q": total_q,
            "total_q": total_q,
            "p50_latency_ms": p50,
            "nc_rate": nc_rate,
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
