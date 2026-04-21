"""Shared pytest fixtures for photon_mlx tests.

Provides a factory fixture that builds a stub tokenizer aligned with a
:class:`PhotonConfig` instance.  The tokenizer mimics the byte-level stub
used by :class:`baseline_reporag.photon_pipeline._StubTokenizer` so
``PhotonInference(model, cfg, tokenizer)`` receives a single unified
tokenizer instance in tests (Issue #58).
"""

from __future__ import annotations

import pytest


class _StubTokenizer:
    """Minimal stub tokenizer: encodes UTF-8 bytes modulo ``vocab_size``."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")


@pytest.fixture
def stub_tokenizer_for_cfg():
    """Return a factory that builds a stub tokenizer for a given PhotonConfig.

    Usage::

        def test_something(stub_tokenizer_for_cfg):
            cfg = _tiny_cfg()
            tokenizer = stub_tokenizer_for_cfg(cfg)
            engine = PhotonInference(model, cfg, tokenizer)
    """

    def _make(cfg) -> _StubTokenizer:
        return _StubTokenizer(cfg.tokenizer.vocab_size)

    return _make
