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
