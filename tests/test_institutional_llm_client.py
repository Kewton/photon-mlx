"""Tests for baseline_reporag.eval.institutional.llm_client."""

from __future__ import annotations

import pytest

from baseline_reporag.eval.institutional import llm_client as llm_module
from baseline_reporag.eval.institutional.llm_client import QwenMLXAdapter
from baseline_reporag.eval.institutional.llm_client import select_llm_client


class _StubClient:
    name = "stub"
    model = "stub-model"

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        seed: int | None = 42,
        response_format: str = "json_object",
    ) -> str:
        return "{}"


def test_select_llm_client_auto_prefers_openai_when_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(llm_module._PROVIDERS, "openai", lambda: _StubClient())
    client = select_llm_client("auto")
    assert client.name == "stub"


def test_select_llm_client_auto_falls_back_to_qwen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _QwenStub(_StubClient):
        name = "qwen"

    monkeypatch.setitem(llm_module._PROVIDERS, "qwen", lambda: _QwenStub())
    client = select_llm_client("auto")
    assert client.name == "qwen"


def test_select_llm_client_explicit_qwen_ignores_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _QwenStub(_StubClient):
        name = "qwen"

    monkeypatch.setitem(llm_module._PROVIDERS, "qwen", lambda: _QwenStub())
    client = select_llm_client("qwen")
    assert client.name == "qwen"


def test_select_llm_client_unknown_raises() -> None:
    with pytest.raises(ValueError):
        select_llm_client("gemini")


def test_select_llm_client_wraps_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise ImportError("openai not installed")

    monkeypatch.setitem(llm_module._PROVIDERS, "openai", _raise)
    with pytest.raises(RuntimeError):
        select_llm_client("openai")


def test_qwen_mlx_adapter_passes_no_think_to_qwen35(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTokenizer:
        def __init__(self) -> None:
            self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return "PROMPT"

    fake_tokenizer = _FakeTokenizer()
    captured: dict[str, object] = {}

    adapter = QwenMLXAdapter.__new__(QwenMLXAdapter)
    adapter.model = "mlx-community/Qwen3.5-9B-MLX-4bit"
    adapter._tokenizer = fake_tokenizer
    adapter._model = object()
    adapter._make_sampler = lambda temp: ("sampler", temp)

    def _fake_generate_fn(model, tokenizer, *, prompt, sampler, max_tokens):
        captured["prompt"] = prompt
        captured["sampler"] = sampler
        captured["max_tokens"] = max_tokens
        return "ANSWER"

    adapter._generate_fn = _fake_generate_fn

    out = adapter.generate("Hello", response_format="text")

    assert out == "ANSWER"
    assert captured == {
        "prompt": "PROMPT",
        "sampler": ("sampler", 0.2),
        "max_tokens": 1024,
    }
    assert fake_tokenizer.calls == [
        (
            [{"role": "user", "content": "Hello"}],
            {
                "add_generation_prompt": True,
                "tokenize": False,
                "enable_thinking": False,
            },
        )
    ]
