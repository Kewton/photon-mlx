"""JSONL atomic writer + schema validator for institutional eval set (Issue #110)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

REQUIRED_STATIC_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "category",
        "difficulty",
        "question",
        "reference_answer",
        "reference_chunk_ids",
        "grading_notes",
        "rubric",
        "answerable",
    }
)

OPTIONAL_STATIC_KEYS: frozenset[str] = frozenset(
    {
        "expected_citation_patterns",
        "source_document_id",
        "human_verified",
        "verified_by",
        "verified_at",
        "generator_model",
    }
)

_CITATION_PATTERN_RE = re.compile(r"^第\d+条(?:の\d+)?(?:第\d+項)?(?:第\d+号)?$")

_NON_EMPTY_STRING_KEYS: frozenset[str] = frozenset({"question", "reference_answer"})


def _validate_patterns(patterns: object) -> None:
    if not isinstance(patterns, list):
        raise AssertionError("expected_citation_patterns must be list[str]")
    for pat in patterns:
        if not isinstance(pat, str) or not _CITATION_PATTERN_RE.fullmatch(pat):
            raise AssertionError(f"Invalid citation pattern: {pat!r}")


def validate_record(record: dict) -> None:
    """Assert that ``record`` matches the institutional static-eval schema."""
    missing = REQUIRED_STATIC_KEYS - record.keys()
    if missing:
        raise AssertionError(f"Missing required keys: {sorted(missing)}")
    allowed = REQUIRED_STATIC_KEYS | OPTIONAL_STATIC_KEYS
    unknown = set(record.keys()) - allowed
    if unknown:
        raise AssertionError(f"Unknown keys in record: {sorted(unknown)}")
    for key in _NON_EMPTY_STRING_KEYS:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise AssertionError(f"Field {key!r} must be a non-empty string")
    if "expected_citation_patterns" in record:
        _validate_patterns(record["expected_citation_patterns"])


def append_jsonl(path: Path, record: dict, *, validate: bool = True) -> None:
    """Append a record to JSONL with atomic tmp+rename semantics (DR3-004)."""
    if validate:
        validate_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")

    line = json.dumps(record, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(existing + line, encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, records: Iterable[dict], *, validate: bool = True) -> None:
    """Write an iterable of records atomically (replaces any existing file)."""
    records = list(records)
    if validate:
        for record in records:
            validate_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_existing_ids(path: Path, key: str = "id") -> set[str]:
    """Read an existing JSONL file and return the set of values for ``key``."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        val = row.get(key)
        if isinstance(val, str):
            ids.add(val)
    return ids


def write_generation_summary(path: Path, summary: dict) -> None:
    """Persist the terminal generation-summary artifact (DR3-006)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
