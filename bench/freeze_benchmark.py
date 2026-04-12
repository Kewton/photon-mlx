"""
freeze_benchmark.py  –  Freeze the benchmark at the current eval set state.

Records the freeze metadata (date, repo commit, eval set checksums)
to reports/benchmark_freeze.json.

Usage:
    python bench/freeze_benchmark.py \
        --repo-commit <SHA> \
        --note "Week 2 freeze"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the benchmark")
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--eval-dir", default="data/eval_sets")
    parser.add_argument("--output", default="reports/benchmark_freeze.json")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    checksums: dict[str, str] = {}
    for p in sorted(eval_dir.glob("*.jsonl")):
        checksums[p.name] = file_sha256(p)

    freeze_record = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_commit": args.repo_commit,
        "note": args.note,
        "eval_set_checksums": checksums,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(freeze_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Benchmark frozen -> {out_path}")
    print(json.dumps(freeze_record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
