"""
run_baseline_eval.py  –  Run baseline RepoRAG against the static eval set.

Usage:
    python scripts/run_baseline_eval.py --max-questions 10
    python scripts/run_baseline_eval.py  # all 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_reporag.config import load_config

# Issue #143 / Step 3: eval scripts resolve ``cfg.run.seed`` /
# ``cfg.run.deterministic`` and forward the resolved seed into
# ``pipeline.query`` so MLX-LM sampling is deterministic across runs.
from baseline_reporag.eval.run_config import resolve_eval_seed

# CB-004 (codex-fix): lightweight factory import — baseline-only envs no
# longer have to install MLX to run this evaluation script.
from baseline_reporag.pipeline_factory import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline eval")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--eval-set", default="data/eval_sets/static_eval.jsonl")
    parser.add_argument("--max-questions", type=int, default=0)
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
    # Issue #143 / Step 3: ``--repo-id`` silent bug fix.  The factory
    # below loads the index at ``data/indexes/{cfg.repo.repo_id}``, so
    # we MUST mutate ``cfg.repo.repo_id`` *before* ``build_pipeline``
    # runs — otherwise the index of the YAML default repo loads while
    # ``pipeline.query(repo_id=...)`` filters for the requested repo,
    # silently producing empty retrieval.  ``run_multi_turn_eval.py``
    # already does this; ``run_baseline_eval.py`` was missing the
    # mirroring fix until now.
    cfg.repo.repo_id = repo_id

    # Resolve the eval seed once per run.  The resolver fails fast on
    # malformed ``cfg.run`` blocks (YAML bool / str / out-of-range int).
    seed = resolve_eval_seed(cfg)

    # Route via ``build_pipeline`` so PHOTON / baseline providers both
    # receive a fully-wired pipeline and so evaluation runs observe
    # ``photon_generation_enabled`` (Stage 3 DR3-001).
    pipeline = build_pipeline(cfg)

    # The factory builds the run logger internally; reuse its id when it
    # exists so log line-up is preserved (baseline path only exposes it
    # indirectly — use cfg/repo metadata in the output filename).
    import time

    run_id = f"baseline_eval_{repo_id}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Load eval questions
    questions = []
    for line in Path(args.eval_set).read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line))
    if args.max_questions > 0:
        questions = questions[: args.max_questions]

    print(f"run_id: {run_id}")
    print(f"questions: {len(questions)}")
    print()

    # Issue #82 Wave 4: when ``--output`` is a .json path (used by the
    # Streamlit eval runner) predictions land in a sibling .jsonl file so
    # the .json path can carry the UI summary instead.  Classic CLI usage
    # (no --output or .jsonl) keeps the pre-Wave-4 behaviour.
    if args.output:
        output_arg = Path(args.output)
        if output_arg.suffix == ".json":
            output_path = output_arg.with_suffix(".predictions.jsonl")
        else:
            output_path = output_arg
    else:
        output_path = Path(f"logs/{run_id}_predictions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            print(
                f"[{i}/{len(questions)}] {q['id']} ({q['category']}) ...",
                end="",
                flush=True,
            )
            result = pipeline.query(
                question=q["question"],
                session_id=f"eval-{q['id']}",
                repo_id=repo_id,
                seed=seed,
            )
            pred = {
                "eval_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": result.answer,
                "cited_chunk_ids": result.cited_chunk_ids,
                "no_citation": result.no_citation,
                "latency_ms": result.latency.total_ms,
                "retrieval_ms": result.latency.retrieval_ms,
                "generation_ms": result.latency.generation_ms,
                "memory_peak_mb": result.memory.peak_mb,
            }
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            print(
                f" {result.latency.total_ms:.0f}ms  cites={len(result.cited_chunk_ids)}"
            )

    print(f"\nPredictions saved -> {output_path}")
    print(f"Run log -> logs/{run_id}.jsonl")

    # Issue #82 Wave 4: when invoked with ``--output`` pointing at a JSON
    # path (vs. the default JSONL predictions file), also emit a summary
    # JSON with aggregated progress fields.  The async sync loop reads
    # this JSON to populate ``EvalJob`` progress on success.
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
        summary = {
            "done_q": total_q,
            "total_q": total_q,
            "p50_latency_ms": p50,
            "nc_rate": nc_rate,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Summary -> {args.output}")

    # Issue #82 Wave 4: touch marker_file on successful completion.
    if args.marker_file:
        marker = Path(args.marker_file)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        print(f"Marker -> {marker}")


if __name__ == "__main__":
    main()
