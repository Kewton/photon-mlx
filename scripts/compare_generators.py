"""
compare_generators.py  –  Qwen vs PHOTON side-by-side generator comparison.

Runs the same cfg with ``inference.photon_generation_enabled`` first False
then True, writing a per-question JSONL row (latency, answer, citations,
etc.) so the two generators can be compared fairly (Issue #62 Phase 1
acceptance criterion 4).

Usage:
    python scripts/compare_generators.py \
        --config configs/photon_tiny.yaml \
        --questions data/eval_sets/smoke.jsonl \
        --repo-id fastapi_fastapi
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


DEFAULT_OUTPUT_DIR = "reports"


def load_questions(path: str | Path) -> list[dict]:
    """Load questions from a JSONL file.

    Each line is expected to be a JSON object with at least a ``question``
    field.  Optional fields (``id``, ``session_id``, ``category``) are
    passed through to the output rows.
    """
    questions: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        questions.append(json.loads(line))
    return questions


def build_output_path(output_dir: str | Path | None) -> Path:
    """Build the output JSONL path using a timestamp suffix."""
    base = Path(output_dir) if output_dir else Path(DEFAULT_OUTPUT_DIR)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return base / f"compare_generators_{ts}.jsonl"


def override_photon_generation(cfg: Any, enabled: bool) -> None:
    """Set ``cfg.inference.photon_generation_enabled`` in-place.

    The config is a :class:`baseline_reporag.config.Config` dot-access
    wrapper; :meth:`Config.get` creates sections lazily so tests can pass
    cfgs that omit the ``inference`` section. ``Config`` is a plain
    mutable class (no ``__setattr__`` override, no ``__slots__``), so
    normal attribute assignment is sufficient (Issue #62 Phase 1 refactor
    R-2: replaced prior ``object.__setattr__`` bypass which was
    defensive-but-fragile).
    """
    inference = cfg.get("inference")
    if inference is None:
        # Import locally to avoid pulling the whole package at module
        # import time (the test suite imports us for helpers before the
        # full config stack is available).
        from baseline_reporag.config import Config

        inference = Config({})
        cfg.inference = inference
    inference.photon_generation_enabled = bool(enabled)


def run_variant(
    cfg: Any,
    questions: list[dict],
    repo_id: str,
    *,
    photon_generation_enabled: bool,
    seed: int | None = None,
) -> list[dict]:
    """Run all questions through a single generator variant.

    Returns a list of JSONL-friendly dicts (one per question).

    Issue #143 / Step 7: ``seed`` (keyword-only, default ``None``) is
    forwarded into ``pipeline.query`` so the side-by-side comparison
    observes the same Qwen sampling stream as ``run_baseline_eval``
    when callers resolve ``seed`` from ``cfg.run`` upstream. ``None``
    keeps the legacy call shape used by the existing MagicMock
    ``test_compare_generators.py::TestRunVariantAttachesGeneratorUsed``.
    """
    # CB-004 (codex-fix): lightweight factory import — no MLX required to
    # execute the Qwen-only variant (the PHOTON variant lazy-loads MLX on
    # demand inside ``build_pipeline`` when ``provider == "photon"``).
    from baseline_reporag.pipeline_factory import build_pipeline

    override_photon_generation(cfg, photon_generation_enabled)
    pipeline = build_pipeline(cfg)
    variant = "photon" if photon_generation_enabled else "qwen"

    rows: list[dict] = []
    for i, q in enumerate(questions, 1):
        question = q["question"]
        session_id = q.get("session_id") or f"compare-{q.get('id', f'row{i}')}"
        t0 = time.perf_counter()
        result = pipeline.query(
            question=question,
            session_id=session_id,
            repo_id=repo_id,
            seed=seed,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(
            {
                "variant_requested": variant,
                "question_id": q.get("id"),
                "category": q.get("category"),
                "question": question,
                "answer": result.answer,
                # CB-003 (codex-fix): ``generator_used`` is now a first-class
                # field on ``QueryResult`` so the side-by-side comparison can
                # distinguish a real PHOTON answer from a Qwen fallback.
                "generator_used": result.generator_used,
                "generator_fallback_reason": result.generator_fallback_reason,
                "cited_chunk_ids": result.cited_chunk_ids,
                "no_citation": result.no_citation,
                "latency_ms": result.latency.total_ms,
                "retrieval_ms": result.latency.retrieval_ms,
                "generation_ms": result.latency.generation_ms,
                "memory_peak_mb": result.memory.peak_mb,
                "elapsed_wall_ms": elapsed_ms,
            }
        )
    return rows


def write_rows(rows: list[dict], output_path: Path) -> None:
    """Emit ``rows`` as JSONL to ``output_path`` (directory auto-created)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen vs PHOTON side-by-side generator comparison (Issue #62 Phase 1)."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--repo-id", default="")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for the compare_generators_<timestamp>.jsonl output "
            "(default: reports/)."
        ),
    )
    args = parser.parse_args()

    from baseline_reporag.config import load_config

    # Issue #143 / Step 7: resolve ``cfg.run.seed`` once and forward it
    # into both Qwen + PHOTON variants so the two outputs are directly
    # comparable.  ``resolve_eval_seed`` validates the run block and
    # returns ``None`` when ``deterministic=False``.
    from baseline_reporag.eval.run_config import resolve_eval_seed

    cfg = load_config(args.config)
    repo_id = args.repo_id or cfg.repo.repo_id
    seed = resolve_eval_seed(cfg)
    questions = load_questions(args.questions)

    if not questions:
        print(f"No questions loaded from {args.questions}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(questions)} question(s) from {args.questions}")

    all_rows: list[dict] = []

    for enabled in (False, True):
        label = "PHOTON" if enabled else "Qwen"
        print(
            f"\n=== Running {label} variant (photon_generation_enabled={enabled}) ==="
        )
        rows = run_variant(
            cfg,
            questions,
            repo_id=repo_id,
            photon_generation_enabled=enabled,
            seed=seed,
        )
        all_rows.extend(rows)
        for row in rows:
            print(
                f"  [{row.get('question_id', '?')}] "
                f"generator_used={row.get('generator_used')} "
                f"latency_ms={row.get('latency_ms', 0):.0f}"
            )

    output_path = build_output_path(args.output_dir)
    write_rows(all_rows, output_path)
    print(f"\nWrote {len(all_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
