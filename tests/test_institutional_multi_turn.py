"""Tests for baseline_reporag.eval.institutional.multi_turn."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline_reporag.eval.institutional.corpus import DocIndex
from baseline_reporag.eval.institutional.multi_turn import (
    SESSION_PATTERNS,
    TURN_TEMPLATES,
    assert_distinct_citations,
    generate_multi_turn_set,
    generate_session,
)


class _FakeSessionClient:
    name = "fake"
    model = "fake-model"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        seed: int | None = 42,
        response_format: str = "json_object",
    ) -> str:
        return self._responses.pop(0)


def _six_turn_payload(distinct: bool = True) -> str:
    turns: list[dict] = []
    for i in range(6):
        cid = f"chunk-{i if distinct else 0}"
        turns.append(
            {
                "question": f"Q{i + 1}?",
                "reference_answer": f"A{i + 1}",
                "reference_chunk_ids": [cid],
                "expected_citation_patterns": [f"第{i + 1}条"],
                "grading_notes": "note",
                "tags": [],
            }
        )
    return json.dumps({"turns": turns}, ensure_ascii=False)


def _make_doc(tmp_path: Path, *, doc_id: str = "0001_example") -> DocIndex:
    doc_dir = tmp_path / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_path = doc_dir / "document.md"
    doc_path.write_text(
        "第1条 定義。\n第2条 適用。\n罰則: 第30条。\n但し例外あり。", encoding="utf-8"
    )
    return DocIndex(
        doc_id=doc_id,
        path=doc_path,
        has_articles=True,
        has_penalty=True,
        has_exception=True,
        metadata={"title": "Example Law"},
    )


def test_turn_templates_are_six() -> None:
    assert len(TURN_TEMPLATES) == 6


def test_session_patterns_total_thirty() -> None:
    total = sum(p["count"] for p in SESSION_PATTERNS.values())
    assert total == 30


def test_generate_session_has_six_turns(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    client = _FakeSessionClient([_six_turn_payload()])
    session = generate_session(doc=doc, scenario="drill_down", seq=1, client=client)
    assert len(session["turns"]) == 6
    assert session["session_id"] == "INST-MT-001"
    assert session["source_document_id"] == "0001_example"
    assert session["category"] == "drill_down"


def test_assert_distinct_citations_passes_on_distinct(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    client = _FakeSessionClient([_six_turn_payload(distinct=True)])
    session = generate_session(doc=doc, scenario="drill_down", seq=1, client=client)
    assert_distinct_citations(session)  # should not raise


def test_assert_distinct_citations_fails_on_duplicate() -> None:
    session = {
        "session_id": "INST-MT-099",
        "turns": [
            {"reference_chunk_ids": ["a"]},
            {"reference_chunk_ids": ["a"]},
        ],
    }
    with pytest.raises(AssertionError):
        assert_distinct_citations(session)


def test_session_id_naming_is_zero_padded(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    client = _FakeSessionClient([_six_turn_payload()])
    session = generate_session(
        doc=doc, scenario="cross_reference", seq=16, client=client
    )
    assert session["session_id"] == "INST-MT-016"


def test_generate_session_retries_on_duplicate_citations(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path)
    client = _FakeSessionClient(
        [_six_turn_payload(distinct=False), _six_turn_payload(distinct=True)]
    )
    session = generate_session(
        doc=doc, scenario="drill_down", seq=1, client=client, sleep_fn=lambda _s: None
    )
    assert session["session_id"] == "INST-MT-001"


def test_generate_multi_turn_set_respects_patterns(tmp_path: Path) -> None:
    docs: list[DocIndex] = []
    for i in range(40):
        d = _make_doc(tmp_path, doc_id=f"doc_{i:03d}")
        docs.append(d)

    responses = [_six_turn_payload() for _ in range(30)]
    client = _FakeSessionClient(responses)
    sessions = generate_multi_turn_set(index=docs, client=client)
    assert len(sessions) == 30
    categories = [s["category"] for s in sessions]
    assert categories.count("drill_down") == 15
    assert categories.count("cross_reference") == 10
    assert categories.count("real_scenario") == 5


def test_generate_multi_turn_set_warns_on_doc_shortage(tmp_path: Path) -> None:
    docs = [_make_doc(tmp_path, doc_id=f"doc_{i:03d}") for i in range(2)]
    responses = [_six_turn_payload() for _ in range(2)]
    client = _FakeSessionClient(responses)
    with pytest.warns(UserWarning, match="shrunk"):
        sessions = generate_multi_turn_set(index=docs, client=client)
    assert len(sessions) == 2


# ---------------------------------------------------------------------------
# Issue #135 Day 3: retry must vary the seed so deterministic local LLMs
# (mlx_lm Qwen) actually try a different sample on each attempt instead of
# replaying the identical malformed JSON.
# ---------------------------------------------------------------------------


class _SeedRecordingClient:
    """Records every ``seed`` it sees and lets us shape responses per attempt."""

    name = "fake"
    model = "fake-model"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.seeds_seen: list[int | None] = []

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        seed: int | None = 42,
        response_format: str = "json_object",
    ) -> str:
        self.seeds_seen.append(seed)
        if not self._responses:
            raise AssertionError("more generate calls than canned responses")
        return self._responses.pop(0)


def test_generate_session_varies_seed_per_retry(tmp_path: Path) -> None:
    """Issue #135: with mlx_lm Qwen, ``client.generate(prompt, seed=42)``
    is deterministic — every retry replays the same broken JSON. Verify
    that ``generate_session`` perturbs the seed on each retry attempt so
    the LLM actually has a chance to produce different output.
    """
    doc = _make_doc(tmp_path)
    # First two responses fail JSON parse; third succeeds.
    bad = "not valid json {{}"
    client = _SeedRecordingClient([bad, bad, _six_turn_payload(distinct=True)])
    session = generate_session(
        doc=doc,
        scenario="drill_down",
        seq=1,
        client=client,
        sleep_fn=lambda _s: None,
    )
    assert session["session_id"] == "INST-MT-001"
    # Three calls must have used three distinct seeds — otherwise the
    # retry is just replaying the same deterministic LLM output.
    assert len(client.seeds_seen) == 3
    assert len(set(client.seeds_seen)) == 3, (
        f"retries must use distinct seeds, got {client.seeds_seen}"
    )
