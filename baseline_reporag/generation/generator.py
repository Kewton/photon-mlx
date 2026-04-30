from __future__ import annotations

import functools
import subprocess
import sys
from typing import Any, Callable

from .qwen_thinking import normalize_qwen_thinking


class _UnavailableMlxLm:
    def load(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ImportError("mlx_lm is required and must have an accessible Metal device")

    def generate(self, *_args: Any, **_kwargs: Any) -> str:
        raise ImportError("mlx_lm is required and must have an accessible Metal device")


mlx_lm: Any = _UnavailableMlxLm()
make_sampler: Callable[..., Callable] | None = None
_HAS_MLX: bool | None = None


@functools.lru_cache(maxsize=1)
def _mlx_metal_available() -> bool:
    probe = "import mlx.core as mx; mx.array([1]); print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _ensure_mlx_lm() -> None:
    global _HAS_MLX, make_sampler, mlx_lm

    if _HAS_MLX is True:
        return
    if not _mlx_metal_available():
        _HAS_MLX = False
        raise ImportError("mlx_lm is required and must have an accessible Metal device")
    try:
        import mlx_lm as loaded_mlx_lm
        from mlx_lm.sample_utils import make_sampler as loaded_make_sampler
    except ImportError:
        _HAS_MLX = False
        raise

    mlx_lm = loaded_mlx_lm
    make_sampler = loaded_make_sampler
    _HAS_MLX = True


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
        _ensure_mlx_lm()
        self._model, self._tokenizer = mlx_lm.load(self._model_id)
        assert make_sampler is not None
        self._sampler = make_sampler(
            temp=self._temperature,
            top_p=self._top_p,
        )

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        *,
        seed: int | None = None,
    ) -> str:
        """Generate a completion for *messages* with optional MLX seeding.

        Issue #143: when ``seed`` is provided, ``mx.random.seed(seed)`` is
        invoked immediately before ``mlx_lm.generate`` so eval scripts can
        pin Qwen-14B sampling to a deterministic stream. ``seed`` is
        keyword-only so the legacy 4 callsites (``cli`` / ``server`` /
        ``photon_pipeline`` Qwen-only / Qwen-fallback) and the 17+ existing
        ``MagicMock`` tests in ``test_pipeline_integration.py`` keep working
        unchanged.

        Critical (DR3-002): ``if seed is not None`` — NOT ``if seed:``.
        ``seed=0`` is a valid deterministic seed that ``if seed:`` would
        silently drop, leaving the run nondeterministic.
        """
        self._load()
        if seed is not None:
            # Local import keeps baseline-only environments (no MLX) able
            # to import this module for type-checking; the seeding branch
            # only runs once mlx-lm is loaded above.
            import mlx.core as mx

            mx.random.seed(seed)
        normalized_messages, enable_thinking = normalize_qwen_thinking(
            messages,
            self._model_id,
            enable_thinking=False,
        )
        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        prompt: str = self._tokenizer.apply_chat_template(
            normalized_messages,
            **template_kwargs,
        )
        return mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens or self._max_new_tokens,
            sampler=self._sampler,
        )
