"""aggregate_institutional_baseline.py — Aggregate predictions JSONL into report markdown.

Issue #127: predictions JSONL (output of ``scripts/run_baseline_eval.py``) を
``reports/institutional_baseline_static.md`` §3-§6 形式の markdown table に集計する。

Usage:
    python scripts/aggregate_institutional_baseline.py \\
        --predictions logs/institutional/baseline_eval_*.predictions.jsonl \\
        --output - \\
        --section overall,category,latency,failures

    # in-place で既存 report.md の sentinel ブロックを置換
    python scripts/aggregate_institutional_baseline.py \\
        --predictions <jsonl> \\
        --output reports/institutional_baseline_static.md \\
        --in-place

Design: workspace/design/issue-127-aggregate-script-design-policy.md
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, NamedTuple

REQUIRED_FIELDS: tuple[str, ...] = (
    "eval_id",
    "category",
    "question",
    "answer",
    "cited_chunk_ids",
    "no_citation",
    "latency_ms",
    "retrieval_ms",
    "generation_ms",
    "memory_peak_mb",
)

PredictionRecord = dict[str, Any]
OverallStat = dict[str, float]
CategoryStat = dict[str, dict[str, float]]
LatencyStat = dict[str, dict[str, float]]
FailurePick = tuple[PredictionRecord, bool]


def is_no_citation(record: PredictionRecord) -> bool:
    return bool(record["no_citation"]) or not record["cited_chunk_ids"]


def expand_prediction_paths(args_predictions: list[str]) -> list[Path]:
    paths: list[Path] = []
    for arg in args_predictions:
        if any(ch in arg for ch in ("*", "?", "[")):
            matched = sorted(glob_module.glob(arg, recursive=True))
            if not matched:
                raise FileNotFoundError(f"No files matched pattern: {arg}")
            paths.extend(Path(p) for p in matched)
        else:
            p = Path(arg)
            if not p.is_file():
                raise FileNotFoundError(f"Not a file: {arg}")
            paths.append(p)
    return paths


def load_predictions(paths: list[Path]) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for path in paths:
        for line_num, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in REQUIRED_FIELDS:
                if field not in record:
                    raise KeyError(
                        f"Missing required field '{field}' in {path}:{line_num}"
                    )
            records.append(record)
    return records


def compute_overall(records: list[PredictionRecord]) -> OverallStat:
    total = len(records)
    nc = sum(1 for r in records if is_no_citation(r))
    nc_rate = nc / total * 100 if total else 0.0
    return {"total": float(total), "nc": float(nc), "nc_rate": nc_rate}


def compute_category(records: list[PredictionRecord]) -> CategoryStat:
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        per_cat[r["category"]][1] += 1
        if is_no_citation(r):
            per_cat[r["category"]][0] += 1
    return {
        cat: {
            "total": float(tot),
            "nc": float(nc),
            "nc_rate": (nc / tot * 100) if tot else 0.0,
        }
        for cat, (nc, tot) in sorted(per_cat.items())
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[k]


def compute_latency(records: list[PredictionRecord]) -> LatencyStat:
    fields = (
        ("total", "latency_ms"),
        ("retrieval", "retrieval_ms"),
        ("generation", "generation_ms"),
        ("memory_peak", "memory_peak_mb"),
    )
    out: LatencyStat = {}
    for label, key in fields:
        vals = [float(r[key]) for r in records]
        out[label] = {
            "p50": _percentile(vals, 50),
            "p95": _percentile(vals, 95),
            "max": max(vals) if vals else 0.0,
            "mean": (sum(vals) / len(vals)) if vals else 0.0,
        }
    return out


def pick_failure_examples(
    records: list[PredictionRecord],
) -> dict[str, FailurePick]:
    by_cat: dict[str, list[PredictionRecord]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    picks: dict[str, FailurePick] = {}
    for cat, items in sorted(by_cat.items()):
        nc_rec = next((r for r in items if is_no_citation(r)), None)
        if nc_rec is not None:
            picks[cat] = (nc_rec, False)
        else:
            picks[cat] = (items[0], True)
    return picks


def _fmt_rate(rate: float, digits: int = 2) -> str:
    return f"{rate:.{digits}f} %"


def _fmt_ms(ms: float) -> str:
    return f"{ms:,.0f}"


def _fmt_mb(mb: float) -> str:
    return f"{mb:.1f}"


def _truncate(text: str, max_chars: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def render_overall_block(stat: OverallStat) -> str:
    return (
        "| 指標 | 値 |\n"
        "|------|-----|\n"
        f"| 全質問数 | {int(stat['total'])} |\n"
        f"| NC (no-citation) 件数 | {int(stat['nc'])} |\n"
        f"| **NC rate** | **{_fmt_rate(stat['nc_rate'], digits=2)}** |"
    )


def render_category_block(stats: CategoryStat) -> str:
    lines = [
        "| Category | 件数 | NC 件数 | NC rate |",
        "|----------|------|---------|---------|",
    ]
    total_q = 0
    total_nc = 0
    for cat, s in stats.items():
        tot = int(s["total"])
        nc = int(s["nc"])
        total_q += tot
        total_nc += nc
        lines.append(f"| {cat} | {tot} | {nc} | {_fmt_rate(s['nc_rate'], digits=1)} |")
    overall_rate = (total_nc / total_q * 100) if total_q else 0.0
    lines.append(
        f"| **合計** | **{total_q}** | **{total_nc}** | **{_fmt_rate(overall_rate, digits=2)}** |"
    )
    return "\n".join(lines)


def render_latency_block(stat: LatencyStat) -> str:
    lines = ["| 指標 | 値 |", "|------|-----|"]
    t = stat["total"]
    lines += [
        f"| 全体 p50 | {_fmt_ms(t['p50'])} ms |",
        f"| 全体 p95 | {_fmt_ms(t['p95'])} ms |",
        f"| 全体 max | {_fmt_ms(t['max'])} ms |",
        f"| 全体 mean | {_fmt_ms(t['mean'])} ms |",
    ]
    r = stat["retrieval"]
    lines += [
        f"| Retrieval p50 | {_fmt_ms(r['p50'])} ms |",
        f"| Retrieval p95 | {_fmt_ms(r['p95'])} ms |",
    ]
    g = stat["generation"]
    lines += [
        f"| Generation p50 | {_fmt_ms(g['p50'])} ms |",
        f"| Generation p95 | {_fmt_ms(g['p95'])} ms |",
    ]
    m = stat["memory_peak"]
    lines += [
        f"| Memory peak (p50) | {_fmt_mb(m['p50'])} MB |",
        f"| Memory peak (p95) | {_fmt_mb(m['p95'])} MB |",
        f"| Memory peak (max) | {_fmt_mb(m['max'])} MB |",
    ]
    return "\n".join(lines)


def render_failures_block(
    picks: dict[str, FailurePick], category_stats: CategoryStat
) -> str:
    sections: list[str] = []
    for idx, (cat, (rec, is_successful)) in enumerate(picks.items(), start=1):
        cs = category_stats.get(cat, {"nc": 0.0, "total": 0.0})
        nc = int(cs["nc"])
        tot = int(cs["total"])
        marker = " (successful sample)" if is_successful else ""
        block = [f"### {idx}. {cat}{marker}（NC {nc}/{tot}）"]
        block.append(f"- **eval_id**: `{rec['eval_id']}`")
        block.append(f"- **question**: {_truncate(rec['question'])}")
        if is_successful:
            block.append(f"- **cites**: {len(rec['cited_chunk_ids'])} chunks")
        else:
            block.append(f"- **answer (抜粋)**: {_truncate(rec['answer'])}")
        block.append(
            f"- **latency_ms**: {_fmt_ms(rec['latency_ms'])}"
            f"（retrieval {_fmt_ms(rec['retrieval_ms'])}"
            f" / generation {_fmt_ms(rec['generation_ms'])}）"
        )
        sections.append("\n".join(block))
    return "\n\n".join(sections)


class SectionSpec(NamedTuple):
    compute: Callable[[list[PredictionRecord]], Any]
    render: Callable[[Any], str]


SECTION_REGISTRY: dict[str, SectionSpec] = {
    "overall": SectionSpec(compute_overall, render_overall_block),
    "category": SectionSpec(compute_category, render_category_block),
    "latency": SectionSpec(compute_latency, render_latency_block),
    "failures": SectionSpec(pick_failure_examples, render_failures_block),
}


SENTINEL_RE: dict[str, re.Pattern[str]] = {
    sec: re.compile(
        rf"<!--\s*aggregate:{sec}:start\s*-->.*?<!--\s*aggregate:{sec}:end\s*-->",
        re.DOTALL,
    )
    for sec in SECTION_REGISTRY
}


def _build_blocks(
    records: list[PredictionRecord], sections: list[str]
) -> dict[str, str]:
    blocks: dict[str, str] = {}
    cat_stats_for_failures: CategoryStat | None = (
        compute_category(records) if "failures" in sections else None
    )
    for sec in sections:
        if sec not in SECTION_REGISTRY:
            raise KeyError(f"Unknown section: {sec}")
        spec = SECTION_REGISTRY[sec]
        if sec == "failures":
            assert cat_stats_for_failures is not None
            blocks[sec] = render_failures_block(
                spec.compute(records), cat_stats_for_failures
            )
        else:
            blocks[sec] = spec.render(spec.compute(records))
    return blocks


def apply_in_place(report_path: Path, sec_to_block: dict[str, str]) -> None:
    text = report_path.read_text(encoding="utf-8")
    for sec, new_block in sec_to_block.items():
        pattern = SENTINEL_RE[sec]
        replacement = (
            f"<!-- aggregate:{sec}:start -->\n{new_block}\n<!-- aggregate:{sec}:end -->"
        )
        if pattern.search(text):
            text = pattern.sub(replacement, text)
        else:
            print(
                f"WARNING: sentinel for section '{sec}' not found in {report_path}. "
                f"Skipped. (Insert <!-- aggregate:{sec}:start --><!-- aggregate:{sec}:end -->"
                f" manually first.)",
                file=sys.stderr,
            )
    report_path.write_text(text, encoding="utf-8")


def _emit_output(blocks: dict[str, str], args: argparse.Namespace) -> None:
    if args.in_place:
        if args.output == "-":
            raise ValueError("--in-place requires --output to be a file path, not '-'")
        out_path = Path(args.output)
        if not out_path.is_file():
            raise FileNotFoundError(f"--in-place target not found: {out_path}")
        apply_in_place(out_path, blocks)
        return

    combined = "\n\n".join(blocks[sec] for sec in blocks)
    if args.output == "-":
        sys.stdout.write(combined)
        if not combined.endswith("\n"):
            sys.stdout.write("\n")
    else:
        Path(args.output).write_text(
            combined + ("" if combined.endswith("\n") else "\n"), encoding="utf-8"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate predictions JSONL into report markdown sections."
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Predictions JSONL path(s) or glob pattern(s). Supports recursive '**'.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output path or '-' for stdout (default).",
    )
    parser.add_argument(
        "--section",
        default="overall,category,latency,failures",
        help="Comma-separated sections to emit (default: all 4).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace sentinel-bracketed sections in --output file in place.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    sections = [s.strip() for s in args.section.split(",") if s.strip()]
    paths = expand_prediction_paths(args.predictions)
    records = load_predictions(paths)
    blocks = _build_blocks(records, sections)
    _emit_output(blocks, args)


if __name__ == "__main__":
    main()
