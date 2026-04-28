"""CLI wrapper for institutional eval-set generation (Issue #110).

Delegates to ``baseline_reporag.eval.institutional.*`` helpers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline_reporag.eval.institutional.corpus import build_doc_index  # noqa: E402
from baseline_reporag.eval.institutional.generator import (  # noqa: E402
    GenerationFailure,
    generate_question,
)
from baseline_reporag.eval.institutional.llm_client import select_llm_client  # noqa: E402
from baseline_reporag.eval.institutional.multi_turn import (  # noqa: E402
    generate_multi_turn_set,
)
from baseline_reporag.eval.institutional.prompt import SUPPORTED_CATEGORIES  # noqa: E402
from baseline_reporag.eval.institutional.sampler import pick_category_docs  # noqa: E402
from baseline_reporag.eval.institutional.writer import (  # noqa: E402
    read_existing_ids,
    write_generation_summary,
    write_jsonl,
)


def _resolve_corpus_dir(args: argparse.Namespace) -> Path:
    value = args.corpus_dir or os.environ.get("INSTITUTIONAL_CORPUS_DIR", "")
    if not value:
        print(
            "error: --corpus-dir not provided and INSTITUTIONAL_CORPUS_DIR unset",
            file=sys.stderr,
        )
        sys.exit(2)
    path = Path(value)
    if not path.exists() or not path.is_dir():
        print(f"error: corpus directory does not exist: {path}", file=sys.stderr)
        sys.exit(2)
    return path


def _build_static(
    *,
    index: list,
    client,
    count: int,
    seed: int,
    existing_ids: set[str],
) -> tuple[list[dict], int, int]:
    rng = random.Random(seed)
    per_cat = max(1, count // len(SUPPORTED_CATEGORIES))
    rows: list[dict] = []
    succeeded = 0
    failed = 0
    for category in SUPPORTED_CATEGORIES:
        docs = pick_category_docs(index, category, per_cat, rng)
        for seq_offset, doc in enumerate(docs, start=1):
            row_id = f"INST-{category.upper().replace('_', '-')}-{seq_offset:03d}"
            if row_id in existing_ids:
                continue
            try:
                row = generate_question(
                    doc=doc, category=category, seq=seq_offset, client=client
                )
                rows.append(row)
                succeeded += 1
            except GenerationFailure:
                failed += 1
    return rows, succeeded, failed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate institutional eval set")
    parser.add_argument("--corpus-dir", default="")
    parser.add_argument(
        "--provider", default="auto", choices=["auto", "openai", "qwen"]
    )
    parser.add_argument(
        "--mode", default="static", choices=["static", "multi_turn", "both"]
    )
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--output", default="data/eval_sets")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--failure-log", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus_dir = _resolve_corpus_dir(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    started_perf = time.perf_counter()

    index = build_doc_index(corpus_dir)

    if args.dry_run:
        print(f"dry-run: corpus={corpus_dir}, docs={len(index)}")
        return 0

    client = select_llm_client(args.provider)

    total_succeeded = 0
    total_failed = 0
    resumed = bool(args.resume)

    if args.mode in ("static", "both"):
        static_path = output_dir / "institutional_static_eval.jsonl"
        existing = read_existing_ids(static_path) if args.resume else set()
        rows, succeeded, failed = _build_static(
            index=index,
            client=client,
            count=args.count,
            seed=args.seed,
            existing_ids=existing,
        )
        write_jsonl(static_path, rows, validate=True)
        total_succeeded += succeeded
        total_failed += failed

    if args.mode in ("multi_turn", "both"):
        mt_path = output_dir / "institutional_multi_turn_eval.jsonl"
        sessions = generate_multi_turn_set(index=index, client=client)
        mt_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(mt_path, sessions, validate=False)
        total_succeeded += len(sessions)
        total_failed += max(0, args.sessions - len(sessions))

    elapsed = time.perf_counter() - started_perf
    ended_at = dt.datetime.now(dt.timezone.utc).isoformat()

    summary_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "provider": client.name,
        "generator_model": client.model,
        "seed": args.seed,
        "mode": args.mode,
        "attempted": total_succeeded + total_failed,
        "succeeded": total_succeeded,
        "failed": total_failed,
        "resumed": resumed,
        "elapsed_seconds": elapsed,
        "started_at": started_at,
        "ended_at": ended_at,
    }
    summary_path = (
        Path("reports") / f"institutional_generation_summary_{summary_ts}.json"
    )
    write_generation_summary(summary_path, summary)

    fail_rate = total_failed / max(1, total_succeeded + total_failed)
    return 1 if fail_rate > 0.05 else 0


if __name__ == "__main__":
    raise SystemExit(main())
