"""LLM client Protocol + OpenAI / Qwen MLX adapters (Issue #110 DR1-002)."""

from __future__ import annotations

import os
from typing import Callable, Literal, Protocol


class LLMClient(Protocol):
    """Minimal LLM surface the generator depends on (ISP-narrow)."""

    name: str
    model: str

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        seed: int | None = 42,
        response_format: Literal["json_object", "text"] = "json_object",
    ) -> str: ...


class OpenAIAdapter:
    """Adapter for OpenAI Chat Completions (gpt-4o-mini family)."""

    name = "openai"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini-2024-07-18",
        api_key: str | None = None,
    ) -> None:
        import openai  # lazy import (DR3-003)

        self.model = model
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        seed: int | None = 42,
        response_format: Literal["json_object", "text"] = "json_object",
    ) -> str:
        fmt = (
            {"type": "json_object"}
            if response_format == "json_object"
            else {"type": "text"}
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            seed=seed,
            response_format=fmt,
        )
        content = resp.choices[0].message.content
        return content or ""


class QwenMLXAdapter:
    """Adapter for Qwen2.5-Coder served via mlx_lm (offline fallback)."""

    name = "qwen"

    def __init__(
        self,
        *,
        model: str = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    ) -> None:
        from mlx_lm import generate as mlx_generate  # lazy import
        from mlx_lm import load as mlx_load

        self.model = model
        self._generate_fn = mlx_generate
        self._model, self._tokenizer = mlx_load(model)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        seed: int | None = 42,
        response_format: Literal["json_object", "text"] = "json_object",
    ) -> str:
        suffix = (
            "\n\n出力は JSON オブジェクト 1 件のみ。前後に markdown フェンスや説明を含めない。"
            if response_format == "json_object"
            else ""
        )
        return self._generate_fn(
            self._model,
            self._tokenizer,
            prompt=prompt + suffix,
            temp=temperature,
            max_tokens=1024,
        )


_PROVIDERS: dict[str, Callable[[], LLMClient]] = {
    "openai": lambda: OpenAIAdapter(),
    "qwen": lambda: QwenMLXAdapter(),
}


def select_llm_client(provider: str = "auto") -> LLMClient:
    """Return an LLMClient. ``auto`` prefers openai when OPENAI_API_KEY is set."""
    if provider == "auto":
        provider = "openai" if os.environ.get("OPENAI_API_KEY") else "qwen"
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r}")
    try:
        return _PROVIDERS[provider]()
    except ImportError as exc:
        raise RuntimeError(f"No LLM provider available: {exc}") from exc
