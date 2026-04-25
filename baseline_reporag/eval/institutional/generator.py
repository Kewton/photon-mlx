"""One-question generation core for institutional eval set (Issue #110)."""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

from .corpus import DocIndex, build_context
from .llm_client import LLMClient
from .prompt import build_prompt

_REQUIRED_LLM_KEYS: frozenset[str] = frozenset(
    {"question", "reference_answer", "expected_citation_patterns", "grading_notes"}
)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CITATION_PATTERN_RE = re.compile(r"^第\d+条(?:の\d+)?(?:第\d+項)?(?:第\d+号)?$")

MAX_RETRIES: int = 3


class GenerationFailure(RuntimeError):
    """Raised after ``MAX_RETRIES`` parse / key-check failures for one Q."""


def _parse_json_strict(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON is not an object")
    missing = _REQUIRED_LLM_KEYS - value.keys()
    if missing:
        raise ValueError(f"LLM response missing keys: {sorted(missing)}")
    for key in ("question", "reference_answer"):
        field = value.get(key)
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"LLM response field {key!r} must be a non-empty string")
    return value


def _emit_failure_line(record: dict, *, stream=sys.stderr) -> None:
    print(json.dumps(record, ensure_ascii=False), file=stream)


def _read_document(doc: DocIndex, max_chars: int = 8000) -> str:
    try:
        return doc.path.read_text(encoding="utf-8")[:max_chars]
    except OSError:
        return ""


def _build_row(
    *,
    doc: DocIndex,
    category: str,
    seq: int,
    parsed: dict,
    client: LLMClient,
    difficulty: str,
) -> dict:
    row_id = f"INST-{category.upper().replace('_', '-')}-{seq:03d}"
    row: dict[str, Any] = {
        "id": row_id,
        "category": category,
        "difficulty": difficulty,
        "question": str(parsed.get("question", "")).strip(),
        "reference_answer": str(parsed.get("reference_answer", "")).strip(),
        "reference_chunk_ids": list(parsed.get("reference_chunk_ids", []) or []),
        "grading_notes": str(parsed.get("grading_notes", "")).strip(),
        "rubric": {
            "correctness": {"max": 2, "notes": ""},
            "grounding": {"max": 2, "notes": ""},
            "usefulness": {"max": 1, "notes": ""},
        },
        "answerable": bool(parsed.get("answerable", True)),
        "source_document_id": doc.doc_id,
        "generator_model": client.model,
        "human_verified": False,
    }
    patterns = parsed.get("expected_citation_patterns")
    if patterns:
        cleaned = [
            p
            for p in patterns
            if isinstance(p, str) and _CITATION_PATTERN_RE.fullmatch(p)
        ]
        if cleaned:
            row["expected_citation_patterns"] = cleaned
    return row


def generate_question(
    *,
    doc: DocIndex,
    category: str,
    seq: int,
    client: LLMClient,
    difficulty: str = "medium",
    max_retries: int = MAX_RETRIES,
    sleep_fn=time.sleep,
) -> dict:
    """Generate 1 eval row for ``doc`` / ``category`` using ``client``.

    Retries up to ``max_retries`` times on JSON parse / key-check failure.
    On exhaustion raises ``GenerationFailure`` and emits a stderr JSON line.
    """
    document_text = _read_document(doc)
    prompt = build_prompt(
        category, doc.metadata, build_context(doc.metadata, document_text)
    )

    last_error: str = ""
    for attempt in range(max_retries):
        try:
            raw = client.generate(prompt)
            parsed = _parse_json_strict(raw)
            return _build_row(
                doc=doc,
                category=category,
                seq=seq,
                parsed=parsed,
                client=client,
                difficulty=difficulty,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            _emit_failure_line(
                {
                    "doc_id": doc.doc_id,
                    "category": category,
                    "attempt": attempt + 1,
                    "error": last_error,
                }
            )
            if attempt < max_retries - 1:
                sleep_fn(2**attempt)

    raise GenerationFailure(
        f"Failed to generate Q for {doc.doc_id}/{category} after {max_retries} tries: {last_error}"
    )
