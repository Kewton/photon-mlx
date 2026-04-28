"""Tests for baseline_reporag.eval.institutional.prompt."""

from __future__ import annotations

import pytest

from baseline_reporag.eval.institutional.prompt import (
    CATEGORY_CONFIG,
    SUPPORTED_CATEGORIES,
    build_prompt,
    build_system_prompt,
    build_user_prompt,
)


def test_all_six_categories_configured() -> None:
    expected = {
        "definition",
        "article_lookup",
        "overview",
        "scope",
        "penalty",
        "exception",
    }
    assert set(SUPPORTED_CATEGORIES) == expected
    assert len(CATEGORY_CONFIG) == 6


def test_build_system_prompt_contains_category_label() -> None:
    prompt = build_system_prompt("definition")
    assert "定義" in prompt
    assert "JSON" in prompt


def test_build_system_prompt_rejects_unknown_category() -> None:
    with pytest.raises(ValueError):
        build_system_prompt("unknown")


def test_build_user_prompt_injects_title() -> None:
    prompt = build_user_prompt({"title": "My Law"}, "some context body")
    assert "タイトル: My Law" in prompt
    assert "some context body" in prompt


def test_build_user_prompt_without_title() -> None:
    prompt = build_user_prompt({}, "body")
    assert prompt == "body"


def test_build_prompt_combines_system_and_user() -> None:
    combined = build_prompt("penalty", {"title": "P"}, "ctx")
    assert "罰則" in combined
    assert "タイトル: P" in combined
    assert "ctx" in combined
