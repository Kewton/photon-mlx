"""Tests for baseline_reporag.generation.prompt — _SYSTEM, _FORMAT_HINT, build_messages."""

from __future__ import annotations

import re

from baseline_reporag.generation.prompt import _SYSTEM, _FORMAT_HINT, build_messages
from baseline_reporag.generation.evidence_pack import _EVIDENCE_HEADER


# ------------------------------------------------------------------
# _SYSTEM / _FORMAT_HINT constants
# ------------------------------------------------------------------


def test_system_message_contains_citation_rules() -> None:
    assert "[C:N]" in _SYSTEM
    assert "Cite" in _SYSTEM or "cite" in _SYSTEM.lower()


def test_format_hint_contains_few_shot_examples() -> None:
    """_FORMAT_HINT must contain at least one Q:/A: example block."""
    assert "Q:" in _FORMAT_HINT
    assert "A:" in _FORMAT_HINT


def test_format_hint_examples_use_concrete_indices() -> None:
    """Few-shot examples must use concrete [C:<digit>] citations."""
    concrete = re.findall(r"\[C:\d+\]", _FORMAT_HINT)
    assert len(concrete) >= 2, f"Expected >=2 concrete citations, got {concrete}"


def test_citation_notation_consistency() -> None:
    """_EVIDENCE_HEADER, _SYSTEM, _FORMAT_HINT all mention [C:N]."""
    for source_name, source in [
        ("_EVIDENCE_HEADER", _EVIDENCE_HEADER),
        ("_SYSTEM", _SYSTEM),
        ("_FORMAT_HINT", _FORMAT_HINT),
    ]:
        assert "[C:N]" in source or re.search(r"\[C:\d+\]", source), (
            f"{source_name} missing citation notation"
        )


# ------------------------------------------------------------------
# build_messages
# ------------------------------------------------------------------


def test_build_messages_structure() -> None:
    msgs = build_messages("What?", "evidence")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_build_messages_includes_evidence() -> None:
    msgs = build_messages("What?", "evidence text here")
    user_content = msgs[1]["content"]
    assert "evidence text here" in user_content


def test_build_messages_includes_format_hint() -> None:
    msgs = build_messages("What?", "ev")
    user_content = msgs[1]["content"]
    assert "Answer format" in user_content


def test_build_messages_with_history() -> None:
    msgs = build_messages("What?", "ev", history_text="prev turn")
    user_content = msgs[1]["content"]
    assert "prev turn" in user_content


def test_build_messages_without_history() -> None:
    msgs = build_messages("What?", "ev")
    user_content = msgs[1]["content"]
    assert "Conversation History" not in user_content
