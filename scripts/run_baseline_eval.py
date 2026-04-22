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

# CB-004 (codex-fix): lightweight factory import — baseline-only envs no
# longer have to install MLX to run this evaluation script.
from baseline_reporag.pipeline_factory import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline eval")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--eval-set", default="data/eval_sets/static_eval.jsonl")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = cfg.repo.repo_id

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

    output_path = (
        Path(args.output) if args.output else Path(f"logs/{run_id}_predictions.jsonl")
    )
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


if __name__ == "__main__":
    main()
