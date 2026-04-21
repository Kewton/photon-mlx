"""Tests for ``torch_ref.model._precompute_rope`` RoPE-scaling signature sync.

Issue #55: ``_precompute_rope`` accepts the same ``scaling`` / ``scale_factor``
kwargs as the MLX counterpart to keep the two APIs substitutable (LSP),
but raises ``NotImplementedError`` for anything other than ``scaling='none'``
because the torch_ref reference is not meant to handle long contexts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch_ref.model import _precompute_rope  # noqa: E402


def test_precompute_rope_none_unchanged() -> None:
    """Default call (no kwargs) must produce a valid complex64 RoPE table."""
    rope = _precompute_rope(64, 2048, 1e6)
    assert rope.shape == (2048, 32)
    assert rope.dtype == torch.complex64
    # The modulus of every polar-form element must be ~1.0.
    mod = rope.abs()
    assert torch.allclose(mod, torch.ones_like(mod), atol=1e-6)

    # Explicit scaling='none', scale_factor=1.0 must return the same table.
    rope_explicit = _precompute_rope(64, 2048, 1e6, scaling="none", scale_factor=1.0)
    assert torch.equal(rope, rope_explicit)


def test_precompute_rope_scaling_raises() -> None:
    """Passing scaling != 'none' must raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="does not support RoPE scaling"):
        _precompute_rope(64, 2048, 1e6, scaling="ntk")

    # scale_factor != 1.0 alone (with scaling='none') is tolerated; torch_ref
    # matches ModelConfig's warning-only contract for that silent misuse.
    rope = _precompute_rope(64, 2048, 1e6, scaling="none", scale_factor=4.0)
    assert rope.shape == (2048, 32)
