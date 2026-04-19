"""
measure_fallback_recall.py  --  Measure Safe RecGen fallback recall from MT eval logs.

Reads JSONL logs produced by run_multi_turn_eval.py (or the PHOTON pipeline)
and computes:

  - fallback_rate       = fallback_flag=True / total turns
  - fallback_recall     = (fallback_flag AND no_citation) / no_citation
  - fallback_precision  = (fallback_flag AND no_citation) / fallback_flag

``no_citation=True`` is used as a proxy for "should have re-retrieved".

Usage:
    python scripts/measure_fallback_recall.py --log logs/mt_eval_*.jsonl
    python scripts/measure_fallback_recall.py --log logs/photon_mt_20260417.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_log_path(pattern: str) -> Path:
    """Resolve a file path or glob pattern to the latest matching file."""
    path = Path(pattern)
    if path.is_file():
        return path
    # Treat as glob and pick the latest (lexicographic sort)
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"ERROR: No log file found matching: {pattern}", file=sys.stderr)
        sys.exit(1)
    return Path(matches[-1])


def _load_records(path: Path) -> list[dict]:
    """Load JSONL records from a file."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(records: list[dict]) -> dict:
    """Compute fallback rate, recall, and precision from log records.

    Returns a dict with computed metrics and supporting counts.
    """
    total = len(records)
    if total == 0:
        return {"total": 0, "error": "no records found"}

    fallback_true = sum(1 for r in records if r.get("fallback_flag"))
    no_citation_true = sum(1 for r in records if r.get("no_citation"))
    both_true = sum(
        1 for r in records if r.get("fallback_flag") and r.get("no_citation")
    )

    # fallback_rate = turns where fallback fired / total
    fallback_rate = fallback_true / total

    # fallback_recall = (fallback AND no_citation) / no_citation
    # "Of turns that *should* have re-retrieved, how many did we catch?"
    fallback_recall = both_true / no_citation_true if no_citation_true > 0 else None

    # fallback_precision = (fallback AND no_citation) / fallback
    # "Of turns where fallback fired, how many actually needed it?"
    fallback_precision = both_true / fallback_true if fallback_true > 0 else None

    return {
        "total": total,
        "fallback_true": fallback_true,
        "no_citation_true": no_citation_true,
        "both_true": both_true,
        "fallback_rate": fallback_rate,
        "fallback_recall": fallback_recall,
        "fallback_precision": fallback_precision,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_pct(value: float | None) -> str:
    """Format a float as a percentage string, or N/A if None."""
    if value is None:
        return "N/A (denominator = 0)"
    return f"{value:.2%}"


def print_report(metrics: dict, log_path: Path) -> None:
    """Print a human-readable report to stdout."""
    print("=" * 60)
    print("  Safe RecGen Fallback Recall Report")
    print("=" * 60)
    print(f"  Log file:  {log_path}")
    print(f"  Total turns:           {metrics['total']}")
    print()

    print("  Counts:")
    print(f"    fallback_flag=True:  {metrics['fallback_true']}")
    print(f"    no_citation=True:    {metrics['no_citation_true']}")
    print(f"    both=True:           {metrics['both_true']}")
    print()

    print("  Metrics:")
    print(f"    fallback_rate      = {_fmt_pct(metrics['fallback_rate'])}")
    print(f"    fallback_recall    = {_fmt_pct(metrics['fallback_recall'])}")
    print(f"    fallback_precision = {_fmt_pct(metrics['fallback_precision'])}")
    print()

    print("  Definitions:")
    print("    fallback_rate      = fallback_flag / total")
    print("    fallback_recall    = (fallback AND no_citation) / no_citation")
    print("    fallback_precision = (fallback AND no_citation) / fallback_flag")
    print()
    print("  Note: no_citation=True is used as a proxy for")
    print('  "should have re-retrieved".')
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Safe RecGen fallback recall from MT eval logs. "
            "Accepts a file path or glob pattern via --log."
        ),
    )
    parser.add_argument(
        "--log",
        required=True,
        help=(
            "Path to JSONL log file, or a glob pattern "
            "(e.g. logs/mt_eval_*.jsonl). "
            "When a pattern is given the latest matching file is used."
        ),
    )
    args = parser.parse_args()

    log_path = _resolve_log_path(args.log)
    records = _load_records(log_path)

    metrics = compute_metrics(records)

    if "error" in metrics:
        print(f"ERROR: {metrics['error']}", file=sys.stderr)
        sys.exit(1)

    print_report(metrics, log_path)


if __name__ == "__main__":
    main()
