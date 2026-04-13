from __future__ import annotations

from typing import Callable

try:
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


class Generator:
    """Thin wrapper around mlx_lm for text generation."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 768,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._model = None
        self._tokenizer = None
        self._sampler: Callable | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not _HAS_MLX:
            raise ImportError("mlx_lm is required: pip install mlx-lm")
        self._model, self._tokenizer = mlx_lm.load(self._model_id)
        self._sampler = make_sampler(
            temp=self._temperature,
            top_p=self._top_p,
        )

    def generate(self, messages: list[dict]) -> str:
        self._load()
        prompt: str = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self._max_new_tokens,
            sampler=self._sampler,
        )
