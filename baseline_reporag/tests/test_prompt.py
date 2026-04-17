from __future__ import annotations

import re

from baseline_reporag.generation.prompt import (
    _FORMAT_HINT,
    build_messages,
)


class TestBuildMessages:
    """build_messages() の出力構造を検証する。"""

    def test_build_messages_returns_system_and_user(self) -> None:
        """build_messages() が system + user の 2 メッセージを返すこと。"""
        msgs = build_messages(
            question="What is the main router?",
            evidence_text="[C:1] app/main.py\ndef main(): pass",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_build_messages_includes_format_hint(self) -> None:
        """user message に _FORMAT_HINT が含まれること。"""
        msgs = build_messages(
            question="Where is auth?",
            evidence_text="[C:1] auth/main.py\nclass Auth: pass",
        )
        user_content = msgs[1]["content"]
        assert _FORMAT_HINT in user_content

    def test_build_messages_with_history(self) -> None:
        """history_text が含まれる場合の出力構造確認。"""
        history = "Q1: What is X?\nA1: X is Y."
        msgs = build_messages(
            question="Follow up",
            evidence_text="[C:1] foo.py\npass",
            history_text=history,
        )
        user_content = msgs[1]["content"]
        assert "Conversation History" in user_content
        assert history in user_content


class TestFormatHintFewShot:
    """_FORMAT_HINT 内の few-shot example を検証する。"""

    def test_format_hint_contains_few_shot_examples(self) -> None:
        """_FORMAT_HINT に _FEW_SHOT_EXAMPLES が含まれること。"""
        from baseline_reporag.generation.prompt import _FEW_SHOT_EXAMPLES

        assert _FEW_SHOT_EXAMPLES in _FORMAT_HINT

    def test_few_shot_examples_contain_citation_pattern(self) -> None:
        """_FEW_SHOT_EXAMPLES 内に [C:\\d+] パターンが含まれること。"""
        from baseline_reporag.generation.prompt import _FEW_SHOT_EXAMPLES

        assert re.search(r"\[C:\d+\]", _FEW_SHOT_EXAMPLES)

    def test_few_shot_examples_count(self) -> None:
        """example 数が 7 であること（"Q:" で始まる行が 7 つ）。"""
        from baseline_reporag.generation.prompt import _FEW_SHOT_EXAMPLES

        q_count = len(re.findall(r"^Q:", _FEW_SHOT_EXAMPLES, re.MULTILINE))
        assert q_count == 7

    def test_format_hint_token_budget(self) -> None:
        """_FORMAT_HINT が 1500 トークン（約 6000 文字）以内であること。"""
        assert len(_FORMAT_HINT) <= 6000
