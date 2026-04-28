from __future__ import annotations

from baseline_reporag.generation.qwen_thinking import (
    is_qwen35_model,
    normalize_qwen_thinking,
)


def test_is_qwen35_model() -> None:
    assert is_qwen35_model("mlx-community/Qwen3.5-9B-MLX-4bit") is True
    assert is_qwen35_model("mlx-community/Qwen3.6-35B-A3B-4bit") is False
    assert is_qwen35_model("mlx-community/gemma-4-e4b-it-8bit") is False


def test_normalize_qwen_thinking_defaults_qwen35_to_no_think() -> None:
    messages = [{"role": "user", "content": "Hello"}]

    normalized, enable_thinking = normalize_qwen_thinking(
        messages,
        "mlx-community/Qwen3.5-9B-MLX-4bit",
        enable_thinking=False,
    )

    assert normalized == messages
    assert enable_thinking is False


def test_normalize_qwen_thinking_strips_nothink_directive() -> None:
    messages = [{"role": "user", "content": "/nothink Hello"}]

    normalized, enable_thinking = normalize_qwen_thinking(
        messages,
        "mlx-community/Qwen3.5-9B-MLX-4bit",
        enable_thinking=False,
    )

    assert normalized == [{"role": "user", "content": "Hello"}]
    assert enable_thinking is False
    assert messages == [{"role": "user", "content": "/nothink Hello"}]


def test_normalize_qwen_thinking_ignores_non_qwen35_models() -> None:
    messages = [{"role": "user", "content": "/nothink Hello"}]

    normalized, enable_thinking = normalize_qwen_thinking(
        messages,
        "mlx-community/gemma-4-e4b-it-8bit",
        enable_thinking=False,
    )

    assert normalized == messages
    assert enable_thinking is None
