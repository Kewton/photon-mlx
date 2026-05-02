"""Score baseline vs PHOTON runs for ``workspace/テストシナリオ2.md``.

The scorer is intentionally deterministic and rule-based.  It does not try to
replace human review; it makes regression comparison repeatable by checking
turn-level expectations such as required cited documents, forbidden drift, and
unsupported inclusion answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Expectation:
    required_answer: tuple[tuple[str, ...], ...] = ()
    required_cited_paths: tuple[tuple[str, ...], ...] = ()
    forbidden_answer: tuple[str, ...] = ()
    forbidden_cited_paths: tuple[str, ...] = ()
    must_be_cautious: bool = False
    max_citations: int | None = None


@dataclass(frozen=True)
class TurnScore:
    key: str
    label: str
    scenario_id: str
    turn: int
    total: int
    answer_score: int
    evidence_recall_score: int
    evidence_precision_score: int
    citation_score: int
    safety_score: int
    latency_ms: float
    citation_count: int
    wrong_citation_count: int
    notes: tuple[str, ...]


MAX_SCORE = 10


EXPECTATIONS: dict[tuple[str, int], Expectation] = {
    ("A1", 1): Expectation(
        required_answer=(("1号", "１号"), ("認定", "基準")),
        required_cited_paths=(("セーフティネット保証1号",),),
        forbidden_cited_paths=("セーフティネット保証4号", "セーフティネット保証6号"),
        max_citations=6,
    ),
    ("A1", 2): Expectation(
        required_answer=(("1号", "１号"), ("2号", "２号"), ("違い", "異な")),
        required_cited_paths=(("セーフティネット保証1号",), ("セーフティネット保証2号",)),
        forbidden_cited_paths=("セーフティネット保証4号", "セーフティネット保証6号"),
        max_citations=10,
    ),
    ("A1", 3): Expectation(
        required_answer=(("1号", "１号"), ("2号", "２号"), ("必要書類", "書類")),
        required_cited_paths=(("セーフティネット保証1号",), ("セーフティネット保証2号",)),
        forbidden_cited_paths=("セーフティネット保証4号", "セーフティネット保証6号"),
        max_citations=10,
    ),
    ("A2", 1): Expectation(
        required_answer=(("4号", "４号"), ("認定", "条件")),
        required_cited_paths=(("セーフティネット保証4号",),),
        forbidden_cited_paths=("セーフティネット保証1号", "セーフティネット保証2号"),
        max_citations=7,
    ),
    ("A2", 2): Expectation(
        required_answer=(
            ("様式第4－②", "様式第4-2", "様式4-2", "4－②", "4-2"),
            ("様式第4－③", "様式第4-3", "様式4-3", "4－③", "4-3"),
        ),
        required_cited_paths=(("セーフティネット保証4号",),),
        forbidden_cited_paths=("セーフティネット保証1号", "セーフティネット保証2号"),
        max_citations=6,
    ),
    ("A2", 3): Expectation(
        required_answer=(("災害発生前",), ("売上",), ("ある", "ない")),
        required_cited_paths=(("セーフティネット保証4号",),),
        forbidden_cited_paths=("セーフティネット保証1号", "セーフティネット保証2号"),
        max_citations=7,
    ),
    ("A3", 1): Expectation(
        required_answer=(("起業家", "創業支援"), ("融資",)),
        required_cited_paths=(("起業家・創業支援融資",),),
        max_citations=10,
    ),
    ("A3", 2): Expectation(
        required_answer=(("計画書",), ("事業", "起業")),
        required_cited_paths=(("起業計画書",), ("起業家・創業支援融資",)),
        max_citations=12,
    ),
    ("A3", 3): Expectation(
        required_answer=(("資金計画", "資金調達", "運転資金"), ("計画書",)),
        required_cited_paths=(("起業計画書",),),
        max_citations=8,
    ),
    ("A4", 1): Expectation(
        required_answer=(("生産性向上",), ("事業拡大",)),
        required_cited_paths=(("生産性向上・事業拡大",),),
        max_citations=9,
    ),
    ("A4", 2): Expectation(
        required_answer=(("事業拡大",), ("店舗改善",), ("違い", "異な")),
        required_cited_paths=(("事業拡大",), ("店舗改善",)),
        max_citations=8,
    ),
    ("A4", 3): Expectation(
        required_answer=(("実施場所", "場所"), ("事業拡大",), ("店舗改善",)),
        required_cited_paths=(("事業拡大",), ("店舗改善",)),
        max_citations=8,
    ),
    ("A5", 1): Expectation(
        required_answer=(("1号", "１号"), ("必要書類", "書類")),
        required_cited_paths=(("セーフティネット保証1号",),),
        forbidden_cited_paths=("デジタル化支援", "セーフティネット保証4号"),
        max_citations=5,
    ),
    ("A5", 2): Expectation(
        required_answer=(("デジタル化支援",), ("対象",)),
        required_cited_paths=(("デジタル化支援",),),
        forbidden_cited_paths=("セーフティネット保証1号", "セーフティネット保証2号"),
        max_citations=6,
    ),
    ("A5", 3): Expectation(
        required_answer=(("デジタル化支援",), ("必要書類", "書類")),
        required_cited_paths=(("デジタル化支援",),),
        forbidden_answer=("セーフティネット保証1号", "セーフティネット保証 1 号"),
        forbidden_cited_paths=("セーフティネット保証1号", "セーフティネット保証2号"),
        max_citations=6,
    ),
    ("B1", 1): Expectation(
        required_answer=(("収益構造改善",), ("融資",)),
        required_cited_paths=(("収益構造改善",),),
        max_citations=8,
    ),
    ("B1", 2): Expectation(
        required_answer=(("売上高減少",), ("利益率減少",), ("違い", "異な")),
        required_cited_paths=(("売上高減少",), ("利益率減少",)),
        max_citations=8,
    ),
    ("B1", 3): Expectation(
        required_answer=(("利益率",), ("売上総利益率", "営業利益率")),
        required_cited_paths=(("利益率減少",),),
        forbidden_cited_paths=("売上高減少",),
        max_citations=6,
    ),
    ("B2", 1): Expectation(
        required_answer=(("法人用",), ("個人用",)),
        required_cited_paths=(("法人用",), ("個人用",)),
        max_citations=8,
    ),
    ("B2", 2): Expectation(
        required_answer=(("法人",), ("記入", "項目")),
        required_cited_paths=(("法人用",),),
        forbidden_cited_paths=("デジタル化支援", "起業計画書", "セーフティネット保証"),
        max_citations=5,
    ),
    ("B2", 3): Expectation(
        required_answer=(("法人",), ("個人",), ("違い", "異な")),
        required_cited_paths=(("法人用",), ("個人用",)),
        forbidden_cited_paths=("デジタル化支援", "セーフティネット保証"),
        max_citations=8,
    ),
    ("B3", 1): Expectation(
        required_answer=(("事業承継",), ("対象",)),
        required_cited_paths=(("事業承継支援融資",),),
        max_citations=9,
    ),
    ("B3", 2): Expectation(
        required_answer=(("計画書",), ("事業承継",)),
        required_cited_paths=(("事業承継計画書",),),
        max_citations=10,
    ),
    ("B3", 3): Expectation(
        required_answer=(("M&A", "株式譲渡"), ("様式", "計画書")),
        required_cited_paths=(("M&A", "株式譲渡", "承継"),),
        max_citations=9,
    ),
    ("B4", 1): Expectation(
        required_answer=(("環境", "省エネルギー"), ("融資",)),
        required_cited_paths=(("環境・省エネルギー",),),
        max_citations=12,
    ),
    ("B4", 2): Expectation(
        required_answer=(("環境", "省エネルギー"), ("公害防止",), ("違い", "異な")),
        required_cited_paths=(("環境・省エネルギー",), ("公害防止",)),
        max_citations=10,
    ),
    ("B4", 3): Expectation(
        required_answer=(("実施場所", "場所"),),
        required_cited_paths=(("環境・省エネルギー", "公害防止"),),
        max_citations=8,
    ),
    ("B5", 1): Expectation(
        required_answer=(("6号", "６号"), ("7号", "７号"), ("8号", "８号")),
        required_cited_paths=(("セーフティネット保証6号",), ("セーフティネット保証7号",), ("セーフティネット保証8号",)),
        forbidden_cited_paths=("セーフティネット保証1号", "セーフティネット保証2号"),
        max_citations=12,
    ),
    ("B5", 2): Expectation(
        required_answer=(("7号", "７号"),),
        required_cited_paths=(("セーフティネット保証7号",),),
        forbidden_cited_paths=("セーフティネット保証1号", "セーフティネット保証2号"),
        max_citations=10,
    ),
    ("B5", 3): Expectation(
        required_answer=(("6号", "６号"), ("7号", "７号"), ("8号", "８号"), ("必要書類", "書類")),
        required_cited_paths=(("セーフティネット保証6号",), ("セーフティネット保証7号",), ("セーフティネット保証8号",)),
        forbidden_cited_paths=("セーフティネット保証1号", "セーフティネット保証2号"),
        max_citations=10,
    ),
    ("C1", 1): Expectation(
        required_answer=(("デジタル化支援",), ("対象経費", "経費")),
        required_cited_paths=(("デジタル化支援",),),
        max_citations=6,
    ),
    ("C1", 2): Expectation(
        required_answer=(("補助金",), ("融資",)),
        required_cited_paths=(("デジタル化支援",),),
        forbidden_cited_paths=("中小企業融資", "セーフティネット保証"),
        max_citations=6,
    ),
    ("C1", 3): Expectation(
        required_answer=(("申請",), ("注意", "期限", "事業開始前")),
        required_cited_paths=(("デジタル化支援",),),
        forbidden_cited_paths=("中小企業融資", "セーフティネット保証"),
        max_citations=8,
    ),
    ("C2", 1): Expectation(
        required_answer=(("起業家", "創業支援"), ("必要書類", "書類")),
        required_cited_paths=(("起業家・創業支援融資",),),
        max_citations=12,
    ),
    ("C2", 2): Expectation(
        required_answer=(("起業家", "創業支援"), ("生産性向上", "事業拡大")),
        required_cited_paths=(("起業家・創業支援融資",), ("生産性向上・事業拡大",)),
        max_citations=12,
    ),
    ("C2", 3): Expectation(
        required_answer=(("事業計画書", "計画書"), ("起業家", "創業支援"), ("生産性向上", "事業拡大")),
        required_cited_paths=(("起業家・創業支援融資", "起業計画書"), ("生産性向上・事業拡大",)),
        max_citations=8,
    ),
    ("C3", 1): Expectation(
        required_answer=(("事業承継",), ("計画書",)),
        required_cited_paths=(("事業承継",),),
        max_citations=14,
    ),
    ("C3", 2): Expectation(
        required_answer=(("承継後",), ("違い", "異な")),
        required_cited_paths=(("事業承継",), ("承継後", "事業計画書")),
        max_citations=10,
    ),
    ("C3", 3): Expectation(
        required_answer=(("売上高",), ("どちら", "計画書")),
        required_cited_paths=(("事業承継", "承継後", "事業計画書"),),
        max_citations=8,
    ),
    ("C4", 1): Expectation(
        required_answer=(("商店街活性化",), ("対象",)),
        required_cited_paths=(("商店街活性化",),),
        forbidden_cited_paths=("収益構造改善", "セーフティネット保証"),
        max_citations=6,
    ),
    ("C4", 2): Expectation(
        required_answer=(("オンライン販売",), ("確認できません", "確認できない", "明記されていません", "明記がない")),
        required_cited_paths=(("商店街活性化",),),
        forbidden_cited_paths=("収益構造改善", "セーフティネット保証"),
        must_be_cautious=True,
        max_citations=5,
    ),
    ("C4", 3): Expectation(
        required_answer=(("オンライン販売",), ("確認できません", "確認できない", "明記されていません", "根拠は不足")),
        required_cited_paths=(("商店街活性化",),),
        forbidden_cited_paths=("収益構造改善", "セーフティネット保証"),
        must_be_cautious=True,
        max_citations=5,
    ),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _turn_key(record: dict[str, Any]) -> str:
    return f"{record.get('label')}:{record.get('scenario_id')}:T{record.get('turn')}"


def _normalise(text: str) -> str:
    return "".join(str(text).casefold().split())


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalised = _normalise(text)
    return any(_normalise(term) in normalised for term in terms)


def _group_hits(text: str, groups: tuple[tuple[str, ...], ...]) -> int:
    return sum(1 for group in groups if _contains_any(text, group))


def _path_text(record: dict[str, Any]) -> str:
    paths: list[str] = []
    for item in record.get("cited_info") or []:
        paths.append(str(item.get("rel_path") or ""))
        paths.append(str(item.get("section") or ""))
    return "\n".join(paths)


def _count_forbidden_paths(record: dict[str, Any], forbidden: tuple[str, ...]) -> int:
    count = 0
    for item in record.get("cited_info") or []:
        text = f"{item.get('rel_path') or ''}\n{item.get('section') or ''}"
        if _contains_any(text, forbidden):
            count += 1
    return count


def _score_answer(record: dict[str, Any], expectation: Expectation, notes: list[str]) -> int:
    if not expectation.required_answer:
        return 2
    answer = str(record.get("answer") or "")
    hit_count = _group_hits(answer, expectation.required_answer)
    if hit_count < len(expectation.required_answer):
        notes.append(
            f"answer_missing={len(expectation.required_answer) - hit_count}"
        )
    forbidden_hit = _contains_any(answer, expectation.forbidden_answer)
    if forbidden_hit:
        notes.append("forbidden_answer")
    if hit_count == len(expectation.required_answer) and not forbidden_hit:
        return 2
    if hit_count > 0 and not forbidden_hit:
        return 1
    return 0


def _score_evidence_recall(
    record: dict[str, Any], expectation: Expectation, notes: list[str]
) -> int:
    if not expectation.required_cited_paths:
        return 2
    paths = _path_text(record)
    hit_count = _group_hits(paths, expectation.required_cited_paths)
    if hit_count < len(expectation.required_cited_paths):
        notes.append(
            f"evidence_missing={len(expectation.required_cited_paths) - hit_count}"
        )
    if hit_count == len(expectation.required_cited_paths):
        return 2
    if hit_count > 0:
        return 1
    return 0


def _score_evidence_precision(
    record: dict[str, Any], expectation: Expectation, notes: list[str]
) -> int:
    issues = 0
    forbidden_count = _count_forbidden_paths(record, expectation.forbidden_cited_paths)
    if forbidden_count:
        issues += min(forbidden_count, 2)
        notes.append(f"forbidden_cited_paths={forbidden_count}")
    citation_count = len(record.get("cited_chunk_ids") or [])
    if expectation.max_citations is not None and citation_count > expectation.max_citations:
        issues += 1
        notes.append(f"too_many_citations={citation_count}>{expectation.max_citations}")
    if issues == 0:
        return 2
    if issues == 1:
        return 1
    return 0


def _score_citation(record: dict[str, Any], notes: list[str]) -> int:
    wrong = record.get("wrong_citation_indices") or []
    cited = record.get("cited_chunk_ids") or []
    no_citation = bool(record.get("no_citation"))
    if wrong:
        notes.append(f"wrong_citation={wrong}")
        return 0
    if no_citation or not cited:
        notes.append("no_citation")
        return 0
    return 2


def _score_safety(record: dict[str, Any], expectation: Expectation, notes: list[str]) -> int:
    if not expectation.must_be_cautious:
        return 2
    answer = str(record.get("answer") or "")
    cautious = _contains_any(
        answer,
        (
            "確認できません",
            "確認できない",
            "明記されていません",
            "明記がない",
            "根拠が不足",
            "断定できません",
        ),
    )
    affirmative = _contains_any(
        answer,
        (
            "対象になります",
            "対象です",
            "対象に含まれます",
            "含まれます",
        ),
    )
    guard = bool((record.get("claim_support_guard") or {}).get("applied"))
    if (cautious or guard) and not affirmative:
        return 2
    if cautious or guard:
        notes.append("cautious_but_mixed")
        return 1
    notes.append("missing_cautious_answer")
    return 0


def score_record(record: dict[str, Any]) -> TurnScore:
    scenario_id = str(record.get("scenario_id"))
    turn = int(record.get("turn"))
    expectation = EXPECTATIONS.get((scenario_id, turn), Expectation())
    notes: list[str] = []
    answer_score = _score_answer(record, expectation, notes)
    evidence_recall_score = _score_evidence_recall(record, expectation, notes)
    evidence_precision_score = _score_evidence_precision(record, expectation, notes)
    citation_score = _score_citation(record, notes)
    safety_score = _score_safety(record, expectation, notes)
    total = (
        answer_score
        + evidence_recall_score
        + evidence_precision_score
        + citation_score
        + safety_score
    )
    return TurnScore(
        key=_turn_key(record),
        label=str(record.get("label")),
        scenario_id=scenario_id,
        turn=turn,
        total=total,
        answer_score=answer_score,
        evidence_recall_score=evidence_recall_score,
        evidence_precision_score=evidence_precision_score,
        citation_score=citation_score,
        safety_score=safety_score,
        latency_ms=float(record.get("latency_ms") or 0.0),
        citation_count=len(record.get("cited_chunk_ids") or []),
        wrong_citation_count=len(record.get("wrong_citation_indices") or []),
        notes=tuple(notes),
    )


def _score_records(records: list[dict[str, Any]]) -> dict[str, TurnScore]:
    return {_turn_key(record): score_record(record) for record in records}


def _winner(baseline: TurnScore, photon: TurnScore) -> str:
    if photon.total > baseline.total:
        return "photon"
    if baseline.total > photon.total:
        return "baseline"
    if photon.latency_ms < baseline.latency_ms * 0.85:
        return "photon_latency"
    if baseline.latency_ms < photon.latency_ms * 0.85:
        return "baseline_latency"
    return "tie"


def _summary(scores: list[TurnScore]) -> dict[str, Any]:
    return {
        "turns": len(scores),
        "avg_score": round(statistics.mean(s.total for s in scores), 3) if scores else 0,
        "avg_latency_ms": round(statistics.mean(s.latency_ms for s in scores), 1)
        if scores
        else 0,
        "avg_citations": round(statistics.mean(s.citation_count for s in scores), 3)
        if scores
        else 0,
        "perfect_turns": sum(1 for s in scores if s.total == MAX_SCORE),
        "wrong_citation_turns": sum(1 for s in scores if s.wrong_citation_count),
    }


def compare_runs(
    *,
    baseline_records: list[dict[str, Any]],
    photon_records: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_scores = _score_records(baseline_records)
    photon_scores = _score_records(photon_records)
    keys = sorted(set(baseline_scores) & set(photon_scores))
    rows: list[dict[str, Any]] = []
    for key in keys:
        baseline = baseline_scores[key]
        photon = photon_scores[key]
        rows.append(
            {
                "key": key,
                "scenario_id": photon.scenario_id,
                "turn": photon.turn,
                "winner": _winner(baseline, photon),
                "baseline_total": baseline.total,
                "photon_total": photon.total,
                "delta": photon.total - baseline.total,
                "baseline_latency_ms": baseline.latency_ms,
                "photon_latency_ms": photon.latency_ms,
                "baseline_citations": baseline.citation_count,
                "photon_citations": photon.citation_count,
                "baseline_notes": ";".join(baseline.notes),
                "photon_notes": ";".join(photon.notes),
                "baseline_breakdown": {
                    "answer": baseline.answer_score,
                    "evidence_recall": baseline.evidence_recall_score,
                    "evidence_precision": baseline.evidence_precision_score,
                    "citation": baseline.citation_score,
                    "safety": baseline.safety_score,
                },
                "photon_breakdown": {
                    "answer": photon.answer_score,
                    "evidence_recall": photon.evidence_recall_score,
                    "evidence_precision": photon.evidence_precision_score,
                    "citation": photon.citation_score,
                    "safety": photon.safety_score,
                },
            }
        )

    winner_counts: dict[str, int] = {}
    for row in rows:
        winner = str(row["winner"])
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

    return {
        "max_score_per_turn": MAX_SCORE,
        "matched_turns": len(rows),
        "baseline": _summary([baseline_scores[k] for k in keys]),
        "photon": _summary([photon_scores[k] for k in keys]),
        "winner_counts": winner_counts,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "key",
        "scenario_id",
        "turn",
        "winner",
        "baseline_total",
        "photon_total",
        "delta",
        "baseline_latency_ms",
        "photon_latency_ms",
        "baseline_citations",
        "photon_citations",
        "baseline_notes",
        "photon_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = report["baseline"]
    photon = report["photon"]
    lines = [
        "# Scenario 2 Baseline vs PHOTON Score Report",
        "",
        f"- Matched turns: {report['matched_turns']}",
        f"- Max score per turn: {report['max_score_per_turn']}",
        f"- Baseline avg score: {baseline['avg_score']}",
        f"- PHOTON avg score: {photon['avg_score']}",
        f"- Baseline avg latency: {baseline['avg_latency_ms']} ms",
        f"- PHOTON avg latency: {photon['avg_latency_ms']} ms",
        f"- Winner counts: `{json.dumps(report['winner_counts'], ensure_ascii=False)}`",
        "",
        "| Turn | Winner | Baseline | PHOTON | Delta | Notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        notes = " / ".join(
            part
            for part in (row["baseline_notes"], row["photon_notes"])
            if part
        )
        lines.append(
            "| {key} | {winner} | {baseline_total} | {photon_total} | {delta} | {notes} |".format(
                key=row["key"],
                winner=row["winner"],
                baseline_total=row["baseline_total"],
                photon_total=row["photon_total"],
                delta=row["delta"],
                notes=notes.replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-log", required=True, type=Path)
    parser.add_argument("--photon-log", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = compare_runs(
        baseline_records=_load_jsonl(args.baseline_log),
        photon_records=_load_jsonl(args.photon_log),
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.output_csv:
        write_csv(args.output_csv, report["rows"])
    if args.output_md:
        write_markdown(args.output_md, report)
    print(
        json.dumps(
            {
                "matched_turns": report["matched_turns"],
                "baseline_avg_score": report["baseline"]["avg_score"],
                "photon_avg_score": report["photon"]["avg_score"],
                "winner_counts": report["winner_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
