"""Re-aggregate an eval predictions JSONL with refusal-aware accounting.

Issue #156: ``run_multi_turn_eval.py`` only recently learned to emit
``is_refusal`` / ``true_failure``. Past institutional MT eval logs
(``logs/*.predictions.jsonl`` from #113, #135, #148, ...) therefore
record only raw ``no_citation``, which over-counts misses by treating
legitimate "根拠が不足しています" refusals as hallucination failures.

This script re-reads such a log line-by-line and writes a refusal-aware
summary JSON. Lines without ``is_refusal`` are recovered from ``answer``
via the same phrase check the production grader uses, so the output is a
faithful regrade of historical runs without rerunning the model.

Usage:
    python scripts/regrade_eval_with_refusal.py \
        --predictions logs/mt_eval_xxx_predictions.jsonl \
        --output reports/mt_eval_xxx_refusal_aware.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_multi_turn_eval import summarize_run_predictions  # noqa: E402


def regrade_predictions(predictions_path: Path) -> dict:
    """Read a predictions JSONL and return a refusal-aware run summary."""
    rows: list[dict] = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    summary = summarize_run_predictions(rows)
    summary["source"] = predictions_path.name
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regrade an eval predictions JSONL with refusal-aware accounting."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a predictions JSONL produced by run_multi_turn_eval.py.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Where to write the summary JSON. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"error: predictions file not found: {pred_path}", file=sys.stderr)
        return 2

    summary = regrade_predictions(pred_path)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
        print(f"Summary -> {out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
