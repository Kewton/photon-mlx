"""Six-turn session builder for institutional multi-turn eval (Issue #110)."""

from __future__ import annotations

import json
import re
import time
import warnings
from typing import Any

from .corpus import DocIndex, build_context
from .generator import (
    GenerationFailure,
    MAX_RETRIES,
    _emit_failure_line,
    _read_document,
)
from .llm_client import LLMClient
from .prompt import CATEGORY_CONFIG

TURN_TEMPLATES: tuple[dict[str, str], ...] = (
    {"topic": "定義", "category": "definition", "hint": "metadata.title / 前文"},
    {"topic": "適用範囲", "category": "scope", "hint": "第1〜2条"},
    {"topic": "中核条項", "category": "article_lookup", "hint": "中核の条文"},
    {"topic": "罰則", "category": "penalty", "hint": "罰則章"},
    {"topic": "例外・但書", "category": "exception", "hint": "但書・経過措置条文"},
    {"topic": "概観", "category": "overview", "hint": "全体要約"},
)

SESSION_PATTERNS: dict[str, dict[str, int]] = {
    "drill_down": {"count": 15, "start": 1},
    "cross_reference": {"count": 10, "start": 16},
    "real_scenario": {"count": 5, "start": 26},
}

_SESSION_SYSTEM = (
    "あなたは日本の法令・制度文書のエキスパートです。"
    "次の文書に対し、6 ターンの連続した質問シナリオを生成してください。"
    "各ターンの focus は: 1=定義 / 2=適用範囲 / 3=中核条項 / 4=罰則 / 5=例外・但書 / 6=全体概観 です。"
    "出力は JSON オブジェクト 1 件で、キーは turns。"
    "turns は長さ 6 の配列で、各要素は "
    "{question, reference_answer, expected_citation_patterns, grading_notes, tags} を持つ。"
    "各ターンの reference は出来る限り異なる条文 / セクションを指すように組み立てる。"
    "出力は JSON オブジェクト 1 件のみ。前後に markdown フェンスや説明を含めない。"
)

_SESSION_REQUIRED = frozenset({"turns"})


def _session_id(seq: int) -> str:
    return f"INST-MT-{seq:03d}"


def assert_distinct_citations(session: dict) -> None:
    """Assert 6-turn reference_chunk_ids are all distinct within the session."""
    seen: set[str] = set()
    for turn in session.get("turns", []):
        for cid in turn.get("reference_chunk_ids", []) or []:
            if cid in seen:
                raise AssertionError(
                    f"Duplicate reference_chunk_id {cid!r} in session {session.get('session_id')}"
                )
            seen.add(cid)


def _parse_session(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("session JSON is not an object")
    missing = _SESSION_REQUIRED - value.keys()
    if missing:
        raise ValueError(f"session missing keys: {sorted(missing)}")
    turns = value.get("turns")
    if not isinstance(turns, list) or len(turns) != 6:
        raise ValueError(
            f"session must contain exactly 6 turns, got {len(turns or [])}"
        )
    return value


def _pick_fallback_doc(
    docs: list[DocIndex], *, require_penalty: bool, require_exception: bool
) -> DocIndex | None:
    for d in docs:
        if require_penalty and not d.has_penalty:
            continue
        if require_exception and not d.has_exception:
            continue
        return d
    return None


def _build_session_prompt(doc: DocIndex) -> str:
    doc_text = _read_document(doc)
    context = build_context(doc.metadata, doc_text)
    labels = ", ".join(f"{i + 1}:{t['topic']}" for i, t in enumerate(TURN_TEMPLATES))
    header = (
        f"タイトル: {doc.metadata.get('title', doc.doc_id)}\nトピック順序: {labels}\n"
    )
    return f"{_SESSION_SYSTEM}\n\n{header}{context}"


def _compose_session(
    *,
    doc: DocIndex,
    scenario: str,
    seq: int,
    parsed: dict,
    client: LLMClient,
) -> dict:
    turns_out: list[dict[str, Any]] = []
    for idx, turn in enumerate(parsed["turns"], start=1):
        cat = TURN_TEMPLATES[idx - 1]["category"]
        turn_row: dict[str, Any] = {
            "turn_id": idx,
            "question": str(turn.get("question", "")).strip(),
            "reference_answer": str(turn.get("reference_answer", "")).strip(),
            "reference_chunk_ids": list(turn.get("reference_chunk_ids", []) or []),
            "grading_notes": str(turn.get("grading_notes", "")).strip(),
            "tags": list(turn.get("tags", []) or [cat]),
        }
        patterns = turn.get("expected_citation_patterns")
        if patterns:
            turn_row["expected_citation_patterns"] = list(patterns)
        turns_out.append(turn_row)
    return {
        "session_id": _session_id(seq),
        "category": scenario,
        "scenario": (
            CATEGORY_CONFIG["overview"]["focus"]
            if scenario == "drill_down"
            else f"{scenario} multi-turn session"
        ),
        "source_document_id": doc.doc_id,
        "generator_model": client.model,
        "human_verified": False,
        "turns": turns_out,
        "session_tags": [scenario, "institutional"],
    }


def generate_session(
    *,
    doc: DocIndex,
    scenario: str,
    seq: int,
    client: LLMClient,
    max_retries: int = MAX_RETRIES,
    sleep_fn=time.sleep,
    base_seed: int = 42,
) -> dict:
    """Generate one 6-turn session for ``doc`` using ``client``.

    Issue #135 Day 3: ``QwenMLXAdapter.generate(prompt, seed=N)`` is
    deterministic for a given prompt + seed pair — naive retry on the
    same seed replays the identical (broken) JSON. We perturb ``seed``
    by ``attempt`` so each retry actually samples a different output;
    OpenAI / other clients that already vary outputs are unaffected
    because they treat ``seed`` as a hint, not a hard determinant.
    """
    prompt = _build_session_prompt(doc)
    last_error = ""
    for attempt in range(max_retries):
        try:
            raw = client.generate(prompt, seed=base_seed + attempt)
            parsed = _parse_session(raw)
            session = _compose_session(
                doc=doc, scenario=scenario, seq=seq, parsed=parsed, client=client
            )
            assert_distinct_citations(session)
            return session
        except (json.JSONDecodeError, ValueError, AssertionError) as exc:
            last_error = str(exc)
            _emit_failure_line(
                {
                    "doc_id": doc.doc_id,
                    "scenario": scenario,
                    "attempt": attempt + 1,
                    "error": last_error,
                }
            )
            if attempt < max_retries - 1:
                sleep_fn(2**attempt)
    raise GenerationFailure(
        f"Failed to generate session for {doc.doc_id}/{scenario}: {last_error}"
    )


def generate_multi_turn_set(
    *,
    index: list[DocIndex],
    client: LLMClient,
    patterns: dict[str, dict[str, int]] | None = None,
) -> list[dict]:
    """Generate the full 30-session multi-turn eval set."""
    patterns = patterns or SESSION_PATTERNS
    sessions: list[dict] = []
    available_docs = [d for d in index if d.has_articles]

    cursor = 0
    for scenario, plan in patterns.items():
        target = plan["count"]
        start = plan["start"]
        built = 0
        while built < target and cursor < len(available_docs):
            doc = available_docs[cursor]
            cursor += 1
            if scenario == "drill_down":
                if not (doc.has_penalty or doc.has_exception):
                    fallback = _pick_fallback_doc(
                        available_docs[cursor:],
                        require_penalty=True,
                        require_exception=False,
                    )
                    if fallback is not None:
                        doc = fallback
            try:
                session = generate_session(
                    doc=doc, scenario=scenario, seq=start + built, client=client
                )
            except GenerationFailure:
                continue
            sessions.append(session)
            built += 1
        if built < target:
            warnings.warn(
                (
                    f"scenario {scenario!r} shrunk: requested {target} session(s) but "
                    f"only {built} built (available_docs={len(available_docs)}, "
                    f"cursor={cursor}). DR2-006: session-count reduction is a last "
                    "resort — check corpus coverage."
                ),
                stacklevel=2,
            )
    return sessions
