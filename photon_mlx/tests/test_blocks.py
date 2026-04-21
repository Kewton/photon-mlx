"""Tests for ``photon_mlx.blocks.precompute_rope`` RoPE scaling extensions.

Covers Issue #55 Phase 2:

- ``scaling='none'`` equivalence with the legacy (no-kwarg) call.
- ``scaling='ntk'`` with ``scale_factor=1.0`` equals ``scaling='none'``.
- ``scale_factor=32.0`` produces different cos/sin than ``scale_factor=1.0``.
- YAML→ModelConfig→precompute_rope integration (Phase 2 → Phase 4 bridge).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from photon_mlx.blocks import precompute_rope  # noqa: E402
from torch_ref.config import ModelConfig  # noqa: E402


# ---------------------------------------------------------------------------
# scaling='none' equivalence
# ---------------------------------------------------------------------------


def test_rope_scaling_none_equivalence() -> None:
    """``scaling='none', scale_factor=1.0`` must match the legacy signature.

    The legacy call (no scaling kwargs) must produce byte-equal tables to
    the explicit ``scaling='none', scale_factor=1.0`` call so existing
    checkpoints keep loading unchanged.
    """
    cos_legacy, sin_legacy = precompute_rope(dim=64, max_len=2048, theta=1e6)
    cos_new, sin_new = precompute_rope(
        dim=64,
        max_len=2048,
        theta=1e6,
        scaling="none",
        scale_factor=1.0,
    )
    assert mx.array_equal(cos_legacy, cos_new).item()
    assert mx.array_equal(sin_legacy, sin_new).item()


# ---------------------------------------------------------------------------
# scaling='ntk' + scale_factor=1.0 equivalence with 'none'
# ---------------------------------------------------------------------------


def test_rope_ntk_scale_factor_one_equals_none() -> None:
    """NTK with ``scale_factor=1.0`` collapses to vanilla RoPE (array-equal).

    ``theta * (1.0 ** (dim / (dim - 2))) == theta``, so the result must be
    bit-identical to ``scaling='none'``.
    """
    cos_none, sin_none = precompute_rope(
        dim=64,
        max_len=2048,
        theta=1e6,
        scaling="none",
        scale_factor=1.0,
    )
    cos_ntk, sin_ntk = precompute_rope(
        dim=64,
        max_len=2048,
        theta=1e6,
        scaling="ntk",
        scale_factor=1.0,
    )
    assert mx.array_equal(cos_none, cos_ntk).item()
    assert mx.array_equal(sin_none, sin_ntk).item()


# ---------------------------------------------------------------------------
# scaling='ntk' + scale_factor=32.0 differs from 1.0
# ---------------------------------------------------------------------------


def test_rope_ntk_scale_factor_32_differs() -> None:
    """``scale_factor=32`` must produce a different (cos, sin) than ``1.0``.

    Otherwise the NTK path is effectively a no-op.
    """
    cos_1, sin_1 = precompute_rope(
        dim=64,
        max_len=2048,
        theta=1e6,
        scaling="ntk",
        scale_factor=1.0,
    )
    cos_32, sin_32 = precompute_rope(
        dim=64,
        max_len=2048,
        theta=1e6,
        scaling="ntk",
        scale_factor=32.0,
    )
    assert not mx.array_equal(cos_1, cos_32).item()
    assert not mx.array_equal(sin_1, sin_32).item()


# ---------------------------------------------------------------------------
# YAML/ModelConfig → precompute_rope propagation integration
# ---------------------------------------------------------------------------


def test_rope_scaling_propagation_integration() -> None:
    """ModelConfig.rope_scaling_from → precompute_rope must not raise.

    Verifies the round trip from ModelConfig field to precompute_rope
    kwargs works cleanly (the concrete types and keyword names line up).
    """
    cfg_model = ModelConfig(
        head_dim=64,
        max_position_embeddings=2048,
        rope_theta=1e6,
        rope_scaling="ntk",
        rope_scale_factor=4.0,
    )
    scaling, factor = ModelConfig.rope_scaling_from(cfg_model)
    cos, sin = precompute_rope(
        cfg_model.head_dim,
        cfg_model.max_position_embeddings,
        cfg_model.rope_theta,
        scaling=scaling,
        scale_factor=factor,
    )
    assert cos.shape == (2048, 32)
    assert sin.shape == (2048, 32)


# ---------------------------------------------------------------------------
# Phase 4: local RoPE must NOT apply scaling
# ---------------------------------------------------------------------------


def test_local_rope_scaling_is_none() -> None:
    """PhotonModel._local_cos/_local_sin must be vanilla RoPE even when the
    top-level RoPE uses ``rope_scaling='ntk'``.

    The local decoder only covers ``max_local`` positions (well within the
    training range), so NTK-style extrapolation is unnecessary — and
    applying it would silently drift the local attention.
    """
    from photon_mlx.model import PhotonModel
    from torch_ref.config import (
        HierarchyConfig,
        PhotonConfig,
        TokenizerConfig,
    )

    cfg = PhotonConfig(
        model=ModelConfig(
            base_embed_dim=16,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=128,
            rope_theta=1e6,
            rope_scaling="ntk",
            rope_scale_factor=32.0,
        ),
        hierarchy=HierarchyConfig(
            levels=2,
            chunk_sizes=[4, 4],
            converter_prefix_lengths=[2, 2],
            encoder_layers_per_level=[1, 1],
            decoder_layers_per_level=[1, 1],
        ),
        tokenizer=TokenizerConfig(vocab_size=256),
    )

    mx.random.seed(0)
    model = PhotonModel(cfg)

    # Recompute the expected local RoPE with NO scaling applied.
    m = cfg.model
    CS = cfg.hierarchy.chunk_sizes
    PL = cfg.hierarchy.converter_prefix_lengths
    max_local = max(c * (p + 1) for c, p in zip(CS, PL))
    vanilla_cos, vanilla_sin = precompute_rope(m.head_dim, max_local, m.rope_theta)

    assert mx.array_equal(model._local_cos, vanilla_cos).item()
    assert mx.array_equal(model._local_sin, vanilla_sin).item()

    # Sanity: the TOP-level RoPE *must* differ from vanilla because
    # scaling='ntk', factor=32 is in effect there.
    top_vanilla_cos, _ = precompute_rope(
        m.head_dim, m.max_position_embeddings, m.rope_theta
    )
    assert not mx.array_equal(model._rope_cos, top_vanilla_cos).item()
