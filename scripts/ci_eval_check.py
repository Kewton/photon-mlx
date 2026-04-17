"""CI eval threshold checker.

Reads eval run logs and checks against regression thresholds.
Exit code 0 = pass, 1 = threshold violated.

Usage:
    python scripts/ci_eval_check.py \
        --static-log logs/baseline_eval_*.jsonl \
        --mt-log logs/mt_eval_*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys


# ---------------------------------------------------------------------------
# Thresholds (generous to account for LLM non-determinism)
# ---------------------------------------------------------------------------
STATIC_NC_MAX = 0.30  # no-citation rate
MT_NC_MAX = 0.35  # no-citation rate
WRONG_CITE_MAX = 0  # wrong citation count
LATENCY_P50_MAX = 25_000  # ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_latest(pattern: str) -> str | None:
    """Resolve a glob pattern to the latest matching file (by name)."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def _load_records(path: str) -> list[dict]:
    """Load JSONL records from a file."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _get_latency(record: dict) -> float | None:
    """Extract total latency in ms from a record.

    Supports both nested (run log) and flat (prediction log) formats.
    """
    if "latency" in record and isinstance(record["latency"], dict):
        return record["latency"].get("total_ms")
    return record.get("latency_ms")


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------


def check_static(log_path: str) -> dict:
    """Check static eval thresholds."""
    records = _load_records(log_path)
    total = len(records)
    if total == 0:
        return {"total": 0, "error": "no records found"}

    no_cite = sum(1 for r in records if r.get("no_citation"))
    wrong_cite = sum(1 for r in records if r.get("wrong_citation_indices"))
    latencies = [lat for r in records if (lat := _get_latency(r)) is not None]

    return {
        "total": total,
        "no_citation_rate": no_cite / total,
        "wrong_citation_count": wrong_cite,
        "latency_p50": statistics.median(latencies) if latencies else 0,
    }


def check_mt(log_path: str) -> dict:
    """Check multi-turn eval thresholds."""
    records = _load_records(log_path)
    total = len(records)
    if total == 0:
        return {"total": 0, "error": "no records found"}

    no_cite = sum(1 for r in records if r.get("no_citation"))

    return {
        "total": total,
        "no_citation_rate": no_cite / total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CI eval threshold checker for regression monitoring"
    )
    parser.add_argument(
        "--static-log",
        required=True,
        help="Glob pattern for static eval log (e.g. logs/baseline_eval_*.jsonl)",
    )
    parser.add_argument(
        "--mt-log",
        required=True,
        help="Glob pattern for multi-turn eval log (e.g. logs/mt_eval_*.jsonl)",
    )
    args = parser.parse_args()

    violations: list[str] = []

    # --- Static eval ---
    static_path = _resolve_latest(args.static_log)
    if static_path is None:
        print(f"FAIL: No static eval log found matching: {args.static_log}")
        sys.exit(1)

    print(f"Static eval log: {static_path}")
    static = check_static(static_path)
    print(f"  Total records:        {static['total']}")
    print(f"  No-citation rate:     {static['no_citation_rate']:.2%}")
    print(f"  Wrong citation count: {static['wrong_citation_count']}")
    print(f"  Latency P50:          {static['latency_p50']:.0f} ms")

    if static["no_citation_rate"] > STATIC_NC_MAX:
        violations.append(
            f"Static no-citation rate {static['no_citation_rate']:.2%} "
            f"> {STATIC_NC_MAX:.0%}"
        )
    if static["wrong_citation_count"] > WRONG_CITE_MAX:
        violations.append(
            f"Static wrong citation count {static['wrong_citation_count']} "
            f"> {WRONG_CITE_MAX}"
        )
    if static["latency_p50"] > LATENCY_P50_MAX:
        violations.append(
            f"Static latency P50 {static['latency_p50']:.0f} ms > {LATENCY_P50_MAX} ms"
        )

    # --- Multi-turn eval ---
    mt_path = _resolve_latest(args.mt_log)
    if mt_path is None:
        print(f"\nFAIL: No multi-turn eval log found matching: {args.mt_log}")
        sys.exit(1)

    print(f"\nMulti-turn eval log: {mt_path}")
    mt = check_mt(mt_path)
    print(f"  Total records:    {mt['total']}")
    print(f"  No-citation rate: {mt['no_citation_rate']:.2%}")

    if mt["no_citation_rate"] > MT_NC_MAX:
        violations.append(
            f"MT no-citation rate {mt['no_citation_rate']:.2%} > {MT_NC_MAX:.0%}"
        )

    # --- Verdict ---
    print()
    if violations:
        print("=" * 50)
        print("FAIL: Threshold violations detected")
        print("=" * 50)
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("=" * 50)
        print("PASS: All thresholds within acceptable range")
        print("=" * 50)


if __name__ == "__main__":
    main()
