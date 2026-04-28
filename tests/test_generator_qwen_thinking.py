from __future__ import annotations

from baseline_reporag.generation.generator import Generator


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "PROMPT"


def test_generator_passes_enable_thinking_false_to_qwen35(monkeypatch) -> None:
    fake_tokenizer = _FakeTokenizer()
    captured: dict[str, object] = {}

    def _fake_load(self) -> None:
        self._model = object()
        self._tokenizer = fake_tokenizer

    def _fake_generate(model, tokenizer, *, prompt, max_tokens, sampler):
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        return "ANSWER"

    monkeypatch.setattr(Generator, "_load", _fake_load)
    monkeypatch.setattr(
        "baseline_reporag.generation.generator.mlx_lm.generate", _fake_generate
    )

    gen = Generator("mlx-community/Qwen3.5-9B-MLX-4bit")
    out = gen.generate([{"role": "user", "content": "Hello"}], max_new_tokens=12)

    assert out == "ANSWER"
    assert captured["prompt"] == "PROMPT"
    assert captured["max_tokens"] == 12
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


def test_generator_does_not_pass_enable_thinking_for_non_qwen35(monkeypatch) -> None:
    fake_tokenizer = _FakeTokenizer()

    def _fake_load(self) -> None:
        self._model = object()
        self._tokenizer = fake_tokenizer

    monkeypatch.setattr(Generator, "_load", _fake_load)
    monkeypatch.setattr(
        "baseline_reporag.generation.generator.mlx_lm.generate",
        lambda *args, **kwargs: "ANSWER",
    )

    gen = Generator("mlx-community/gemma-4-e4b-it-8bit")
    gen.generate([{"role": "user", "content": "Hello"}], max_new_tokens=12)

    assert fake_tokenizer.calls == [
        (
            [{"role": "user", "content": "Hello"}],
            {"add_generation_prompt": True, "tokenize": False},
        )
    ]
