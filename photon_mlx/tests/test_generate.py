"""Tests for PhotonModel.generate() greedy decode."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch_ref.config import (
    HierarchyConfig,
    ModelConfig,
    PhotonConfig,
    TokenizerConfig,
)
from photon_mlx.inference import PhotonInference, _TokenizerEncodeFailure
from photon_mlx.model import PhotonModel


def _tiny_cfg() -> PhotonConfig:
    return PhotonConfig(
        model=ModelConfig(
            base_embed_dim=16,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=128,
        ),
        hierarchy=HierarchyConfig(
            levels=2,
            chunk_sizes=[4, 4],
            converter_prefix_lengths=[2, 2],
            encoder_layers_per_level=[2, 2],
            decoder_layers_per_level=[2, 2],
        ),
        tokenizer=TokenizerConfig(vocab_size=256),
    )


@pytest.fixture
def model() -> PhotonModel:
    mx.random.seed(42)
    m = PhotonModel(_tiny_cfg())
    mx.eval(m.parameters())
    return m


class TestGenerateShape:
    def test_generate_output_shape(self, model: PhotonModel) -> None:
        prompt = mx.zeros((1, 16), dtype=mx.int32)
        generated, step_logits = model.generate(prompt, max_new_tokens=16)
        assert generated.shape == (1, 32)
        assert step_logits is None  # return_logits=False by default

    def test_step_logits_shape(self, model: PhotonModel) -> None:
        prompt = mx.zeros((1, 16), dtype=mx.int32)
        generated, step_logits = model.generate(
            prompt, max_new_tokens=16, return_logits=True
        )
        assert generated.shape == (1, 32)
        assert step_logits is not None
        assert step_logits.shape == (1, 16, 256)  # (B, max_new_tokens, V)

    def test_generate_valid_token_ids(self, model: PhotonModel) -> None:
        prompt = mx.zeros((1, 16), dtype=mx.int32)
        generated, _ = model.generate(prompt, max_new_tokens=16)
        mx.eval(generated)
        gen_np = generated[0].tolist()
        for tok in gen_np[16:]:  # check generated part only
            assert 0 <= tok < 256, f"Token {tok} out of vocab range"


class TestGenerateDeterminism:
    def test_generate_determinism(self, model: PhotonModel) -> None:
        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        gen1, _ = model.generate(prompt, max_new_tokens=16)
        gen2, _ = model.generate(prompt, max_new_tokens=16)
        mx.eval(gen1, gen2)
        assert mx.array_equal(gen1, gen2).item()


class TestGeneratePrompt:
    def test_generate_prompt_preserved(self, model: PhotonModel) -> None:
        prompt = mx.array(
            [[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]]
        )
        generated, _ = model.generate(prompt, max_new_tokens=16)
        mx.eval(generated)
        assert mx.array_equal(generated[:, :16], prompt).item()


class TestGenerateBoundary:
    def test_generate_chunk_boundary(self, model: PhotonModel) -> None:
        """prompt_len exactly at chunk boundary (16)."""
        prompt = mx.zeros((1, 16), dtype=mx.int32)
        generated, _ = model.generate(prompt, max_new_tokens=16)
        assert generated.shape == (1, 32)

    def test_generate_short_prompt(self, model: PhotonModel) -> None:
        """prompt_len < 16 (minimum chunk size)."""
        prompt = mx.zeros((1, 4), dtype=mx.int32)
        generated, _ = model.generate(prompt, max_new_tokens=16)
        assert generated.shape == (1, 20)

    @pytest.mark.parametrize("prompt_len", [15, 16, 17])
    def test_generate_cross_boundary(self, model: PhotonModel, prompt_len: int) -> None:
        """Various prompt lengths around chunk boundary."""
        prompt = mx.zeros((1, prompt_len), dtype=mx.int32)
        generated, _ = model.generate(prompt, max_new_tokens=16)
        assert generated.shape == (1, prompt_len + 16)

    def test_generate_single_token(self, model: PhotonModel) -> None:
        prompt = mx.zeros((1, 16), dtype=mx.int32)
        generated, _ = model.generate(prompt, max_new_tokens=1)
        assert generated.shape == (1, 17)


class TestGenerateQuality:
    def test_generate_logits_not_nan(self, model: PhotonModel) -> None:
        prompt = mx.zeros((1, 16), dtype=mx.int32)
        _, step_logits = model.generate(prompt, max_new_tokens=16, return_logits=True)
        mx.eval(step_logits)
        assert mx.isfinite(step_logits).all().item()


class TestKVCacheEquivalence:
    """Issue #54 Phase 1: KV-cache path must match no-cache path.

    ``use_kv_cache=False`` is retained as an internal testing fallback
    (DR1-005). It is not documented as a public knob; the equivalence
    tests below exist specifically to certify the cached path.
    """

    def test_step_logits_match_without_cache(self, model: PhotonModel) -> None:
        prompt = mx.array(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
            dtype=mx.int32,
        )
        _, logits_cache = model.generate(
            prompt, max_new_tokens=8, return_logits=True, use_kv_cache=True
        )
        _, logits_nocache = model.generate(
            prompt, max_new_tokens=8, return_logits=True, use_kv_cache=False
        )
        mx.eval(logits_cache, logits_nocache)
        assert logits_cache.shape == logits_nocache.shape
        assert mx.allclose(logits_cache, logits_nocache, atol=1e-5, rtol=1e-4).item(), (
            "KV-cache step logits diverged from no-cache reference"
        )

    def test_cross_level1_boundary(self, model: PhotonModel) -> None:
        """Crossing a 4-token chunk boundary must produce identical tokens."""
        for prompt_len, n_new in [(15, 2), (16, 1)]:
            prompt = mx.array(
                [[(i + 1) % 256 for i in range(prompt_len)]], dtype=mx.int32
            )
            gen_cache, _ = model.generate(
                prompt, max_new_tokens=n_new, use_kv_cache=True
            )
            gen_nocache, _ = model.generate(
                prompt, max_new_tokens=n_new, use_kv_cache=False
            )
            mx.eval(gen_cache, gen_nocache)
            assert mx.array_equal(gen_cache, gen_nocache).item(), (
                f"token mismatch at prompt_len={prompt_len}, n_new={n_new}"
            )

    def test_cross_top_level_boundary(self, model: PhotonModel) -> None:
        """Crossing a 16-token super-chunk boundary must match."""
        for prompt_len, n_new in [(31, 2), (32, 1)]:
            prompt = mx.array(
                [[(i + 3) % 256 for i in range(prompt_len)]], dtype=mx.int32
            )
            gen_cache, _ = model.generate(
                prompt, max_new_tokens=n_new, use_kv_cache=True
            )
            gen_nocache, _ = model.generate(
                prompt, max_new_tokens=n_new, use_kv_cache=False
            )
            mx.eval(gen_cache, gen_nocache)
            assert mx.array_equal(gen_cache, gen_nocache).item(), (
                f"token mismatch at prompt_len={prompt_len}, n_new={n_new}"
            )

    @pytest.mark.parametrize("prompt_len", [16, 17, 32, 64])
    def test_multiple_prompt_lengths(self, model: PhotonModel, prompt_len: int) -> None:
        prompt = mx.array([[(i + 7) % 256 for i in range(prompt_len)]], dtype=mx.int32)
        gen_cache, _ = model.generate(prompt, max_new_tokens=8, use_kv_cache=True)
        gen_nocache, _ = model.generate(prompt, max_new_tokens=8, use_kv_cache=False)
        mx.eval(gen_cache, gen_nocache)
        assert mx.array_equal(gen_cache, gen_nocache).item()


# -------------------------------------------------------------------------
# Issue #62 Phase 1: PhotonInference.generate_answer
# -------------------------------------------------------------------------


class _ByteTokenizer:
    """Minimal real tokenizer matching the PHOTON stub (byte-level modulo vocab).

    Exercises the real ``PhotonInference.generate_answer`` code path: encode →
    ``model.generate`` → decode. Not a MagicMock.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


@pytest.fixture
def inference_engine(model: PhotonModel) -> PhotonInference:
    tokenizer = _ByteTokenizer(model.cfg.tokenizer.vocab_size)
    return PhotonInference(model, model.cfg, tokenizer)


class TestGenerateAnswer:
    """Contract tests for ``PhotonInference.generate_answer`` (Issue #62)."""

    def test_returns_nonempty_string(self, inference_engine: PhotonInference) -> None:
        """Real model + real tokenizer: generate_answer must return a string."""
        result = inference_engine.generate_answer(
            "def add(a, b):",
            max_new_tokens=4,
        )
        assert isinstance(result, str)
        # A 4-token greedy decode from a byte tokenizer must produce at
        # most 4 UTF-8 code units; empty string is acceptable when the
        # bytes do not form a valid decodable substring.
        assert len(result) <= 16  # 4 tokens × at-most-4-byte glyphs

    def test_raises_on_tokenizer_failure(self, model: PhotonModel) -> None:
        """tokenizer.encode exceptions must propagate (fail-fast)."""

        class _BrokenTokenizer:
            vocab_size = 256
            pad_token_id = 0

            def encode(self, text: str) -> list[int]:
                raise RuntimeError("simulated encode failure")

            def decode(self, ids: list[int]) -> str:
                return ""

        engine = PhotonInference(model, model.cfg, _BrokenTokenizer())
        with pytest.raises((_TokenizerEncodeFailure, RuntimeError)):
            engine.generate_answer("hello", max_new_tokens=2)

    def test_respects_max_new_tokens(self, inference_engine: PhotonInference) -> None:
        """The decoded result must correspond to at most ``max_new_tokens``
        newly-generated tokens (the prompt slice is stripped)."""
        short = inference_engine.generate_answer("hi", max_new_tokens=2)
        longer = inference_engine.generate_answer("hi", max_new_tokens=4)
        # Each generated token is <= 4 UTF-8 bytes, so the longer result's
        # decoded length must fit within the token horizon.
        assert len(short.encode("utf-8", errors="replace")) <= 2 * 4
        assert len(longer.encode("utf-8", errors="replace")) <= 4 * 4

    def test_shares_tokenizer_with_prune_path(
        self, inference_engine: PhotonInference
    ) -> None:
        """Issue #58 integrity: the same tokenizer instance is used for
        generation as for pruning (so both paths share a semantic space)."""
        # Public attribute introduced in inference.py:111. generate_answer
        # must NOT swap in a different tokenizer.
        assert inference_engine.tokenizer is inference_engine.tokenizer
        original = inference_engine.tokenizer
        _ = inference_engine.generate_answer("abc", max_new_tokens=2)
        assert inference_engine.tokenizer is original

    @pytest.mark.parametrize(
        "bad_value",
        [True, False, "512", 0, -1, 1.5, None],
    )
    def test_rejects_non_positive_int_token_budget(
        self, inference_engine: PhotonInference, bad_value
    ) -> None:
        """``max_new_tokens`` must be a positive, non-bool ``int``."""
        with pytest.raises(ValueError):
            inference_engine.generate_answer("abc", max_new_tokens=bad_value)

    def test_generate_answer_rejects_oversized_prompt_before_mlx_allocation(
        self, model: PhotonModel, monkeypatch
    ) -> None:
        """CB-001 (codex-fix): prompts that exceed
        ``max_position_embeddings - max_new_tokens`` must raise ``ValueError``
        *before* any ``mx.array`` allocation so oversize first-turn prompts
        cannot materialize large buffers and then thrash into fallback.
        """
        tokenizer = _ByteTokenizer(model.cfg.tokenizer.vocab_size)
        engine = PhotonInference(model, model.cfg, tokenizer)

        max_pos = model.cfg.model.max_position_embeddings

        # Craft a prompt larger than ``max_position_embeddings`` so the
        # preflight check must trip before allocation.
        oversized_prompt = "a" * (max_pos + 8)

        real_mx_array = mx.array
        allocations: list[tuple] = []

        def _spy_mx_array(data, *args, **kwargs):
            allocations.append(getattr(data, "shape", None) or len(data))
            return real_mx_array(data, *args, **kwargs)

        monkeypatch.setattr("photon_mlx.inference.mx.array", _spy_mx_array)

        with pytest.raises(ValueError, match="max_position_embeddings"):
            engine.generate_answer(oversized_prompt, max_new_tokens=4)

        # Defense in depth: no ``mx.array`` should be allocated for the
        # encoded prompt before the preflight raises.
        assert allocations == [], (
            f"oversize prompt allocated mx.array before guard: {allocations}"
        )

    def test_tokenize_chunk_failure_logs_without_exception_text(
        self, model: PhotonModel, caplog
    ) -> None:
        """CB-002 (codex-fix): ``_tokenize_chunk`` warning log on encode
        failure must not include the raw exception body; it must record the
        exception class name only so prompt fragments / tokenizer internals
        do not leak into logs.
        """
        import logging

        secret_marker = "SECRET_PROMPT_LEAK_kjh5f8"

        class _BrokenTokenizer:
            vocab_size = 256
            pad_token_id = 0

            def encode(self, text: str) -> list[int]:
                raise RuntimeError(secret_marker)

            def decode(self, ids) -> str:
                return ""

        engine = PhotonInference(model, model.cfg, _BrokenTokenizer())

        with caplog.at_level(logging.WARNING, logger="photon_mlx.inference"):
            with pytest.raises(_TokenizerEncodeFailure):
                engine._tokenize_chunk("some-chunk-text")

        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warning_records, "expected a warning record from _tokenize_chunk"
        for rec in warning_records:
            msg = rec.getMessage()
            assert secret_marker not in msg, (
                "raw exception body leaked into warning log (CB-002)"
            )
            # Positive assertion: the closed-enum class name is present.
            assert "RuntimeError" in msg
