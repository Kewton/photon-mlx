"""
export_report.py  –  Export a benchmark report from run logs.

Usage:
    python scripts/export_report.py --run-id latest
    python scripts/export_report.py --run-id baseline_fastapi_fastapi_20260412_eba8942
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_run(log_dir: Path, run_id: str) -> list[dict]:
    if run_id == "latest":
        jsonl_files = sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not jsonl_files:
            print("No run logs found.")
            sys.exit(1)
        path = jsonl_files[-1]
        print(f"Using latest: {path.name}")
    else:
        path = log_dir / f"{run_id}.jsonl"
    if not path.exists():
        print(f"Run log not found: {path}")
        sys.exit(1)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def compute_report(records: list[dict]) -> dict:
    latencies = [r["latency"]["total_ms"] for r in records if "latency" in r]
    retrieval = [r["latency"]["retrieval_ms"] for r in records if "latency" in r]
    generation = [r["latency"]["generation_ms"] for r in records if "latency" in r]
    memory_peaks = [r["memory"]["peak_mb"] for r in records if "memory" in r]
    no_cite = sum(1 for r in records if r.get("no_citation"))
    wrong_cite = sum(1 for r in records if r.get("wrong_citation_indices"))

    def stats(arr: list[float]) -> dict:
        if not arr:
            return {}
        a = np.array(arr)
        return {
            "count": len(a),
            "mean": round(float(a.mean()), 2),
            "p50": round(float(np.percentile(a, 50)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "min": round(float(a.min()), 2),
            "max": round(float(a.max()), 2),
        }

    return {
        "total_turns": len(records),
        "latency_total_ms": stats(latencies),
        "latency_retrieval_ms": stats(retrieval),
        "latency_generation_ms": stats(generation),
        "memory_peak_mb": stats(memory_peaks),
        "no_citation_count": no_cite,
        "wrong_citation_count": wrong_cite,
        "no_citation_rate": round(no_cite / len(records), 4) if records else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a benchmark report")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    records = load_run(Path(args.log_dir), args.run_id)
    report = compute_report(records)

    output_text = json.dumps(report, indent=2, ensure_ascii=False)
    print(output_text)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
