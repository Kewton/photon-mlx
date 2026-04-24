"""Post-hoc citation evaluator for institutional eval set (Issue #110 B-5)."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Protocol


class Grade(str, Enum):
    CORRECT_CHUNK = "correct_chunk"
    ARTICLE_LEVEL_CORRECT = "article_level_correct"
    WRONG_CITATION = "wrong_citation"
    NO_CITATION = "no_citation"


class ChunkLookup(Protocol):
    """Minimal read-only view over a chunk store (ISP-narrow)."""

    def get_chunk_text(self, chunk_id: str) -> str: ...
    def get_doc_id(self, chunk_id: str) -> str: ...


@dataclass(frozen=True)
class DictChunkLookup:
    """Dict-backed ``ChunkLookup`` for tests / standalone CLI fixtures."""

    data: dict[str, tuple[str, str]]

    def get_chunk_text(self, chunk_id: str) -> str:
        return self.data.get(chunk_id, ("", ""))[0]

    def get_doc_id(self, chunk_id: str) -> str:
        return self.data.get(chunk_id, ("", ""))[1]


_ARTICLE_REF_RE = re.compile(r"第\d+条(?:第\d+項)?(?:第\d+号)?")


@lru_cache(maxsize=512)
def _compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _patterns_match(body: str, patterns: Iterable[str]) -> bool:
    """Return True if any pattern fullmatches a 第...条 reference in body."""
    found = _ARTICLE_REF_RE.findall(body)
    if not found:
        return False
    compiled = [_compile_pattern(p) for p in patterns]
    for ref in found:
        for regex in compiled:
            if regex.fullmatch(ref):
                return True
    return False


def grade_prediction(
    pred: dict,
    eval_q: dict,
    lookup: ChunkLookup,
) -> Grade:
    """Grade one prediction row against the eval-set row + chunk lookup."""
    cited = list(pred.get("cited_chunk_ids", []) or [])
    wrong_indices = list(pred.get("wrong_citation_indices", []) or [])
    expected = set(eval_q.get("reference_chunk_ids", []) or [])
    patterns = list(eval_q.get("expected_citation_patterns", []) or [])
    src_doc = str(eval_q.get("source_document_id", "") or "")

    if wrong_indices:
        return Grade.WRONG_CITATION

    cited_set = set(cited)
    if cited_set & expected:
        return Grade.CORRECT_CHUNK

    if src_doc and patterns:
        for cid in cited:
            if lookup.get_doc_id(cid) != src_doc:
                continue
            body = lookup.get_chunk_text(cid)
            if _patterns_match(body, patterns):
                return Grade.ARTICLE_LEVEL_CORRECT

    if pred.get("no_citation") or not cited:
        return Grade.NO_CITATION
    return Grade.WRONG_CITATION


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _eval_id(row: dict) -> str:
    return str(row.get("eval_id") or row.get("id") or "")


def grade_eval_set(
    *,
    eval_set_path: Path,
    lookup: ChunkLookup,
    run_log_path: Path | None = None,
    predictions_path: Path | None = None,
) -> dict:
    """Grade a full eval set against run-log (preferred) or predictions."""
    if run_log_path is None and predictions_path is None:
        raise ValueError("one of run_log_path / predictions_path must be provided")

    questions = {_eval_id(row): row for row in _read_jsonl(eval_set_path)}
    input_mode = "run_log" if run_log_path is not None else "predictions"
    pred_path = run_log_path if run_log_path is not None else predictions_path
    assert pred_path is not None
    pred_rows = _read_jsonl(pred_path)

    per_category: dict[str, Counter[str]] = {}
    totals: Counter[str] = Counter()
    wrong_index_count = 0
    details: list[dict] = []

    for pred in pred_rows:
        eid = _eval_id(pred)
        eval_q = questions.get(eid)
        if eval_q is None:
            continue
        grade = grade_prediction(pred, eval_q, lookup)
        category = str(eval_q.get("category", ""))
        totals[grade.value] += 1
        per_category.setdefault(category, Counter())[grade.value] += 1
        if pred.get("wrong_citation_indices"):
            wrong_index_count += len(pred["wrong_citation_indices"])
        details.append({"eval_id": eid, "category": category, "grade": grade.value})

    total = sum(totals.values())
    report: dict = {
        "eval_set": eval_set_path.name,
        "input_log": pred_path.name if pred_path is not None else "",
        "input_mode": input_mode,
        "total": total,
        "correct_chunk": totals.get(Grade.CORRECT_CHUNK.value, 0),
        "article_level_correct": totals.get(Grade.ARTICLE_LEVEL_CORRECT.value, 0),
        "wrong_citation": totals.get(Grade.WRONG_CITATION.value, 0),
        "no_citation": totals.get(Grade.NO_CITATION.value, 0),
        "wrong_index_count": wrong_index_count,
        "wrong_citation_rate": (
            totals.get(Grade.WRONG_CITATION.value, 0) / total if total else 0.0
        ),
        "no_citation_rate": (
            totals.get(Grade.NO_CITATION.value, 0) / total if total else 0.0
        ),
        "per_category": {
            cat: dict(counter) for cat, counter in sorted(per_category.items())
        },
        "details": details,
    }
    return report
