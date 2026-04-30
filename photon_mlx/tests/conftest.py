"""Shared pytest fixtures for photon_mlx tests.

Provides a factory fixture that builds a test-only stub tokenizer aligned
with a :class:`PhotonConfig` instance.  The tokenizer encodes UTF-8 bytes
modulo ``vocab_size`` and is used purely as a deterministic test fixture
for ``PhotonInference(model, cfg, tokenizer)`` (Issue #58).

This stub is local to the test module and is **not** the production
tokenizer path. Production PHOTON pipelines use ``transformers.AutoTokenizer``
loaded via ``baseline_reporag.photon_pipeline._load_hf_tokenizer``;
test fixtures that omit a ``tokenizer:`` block previously fell back to a
production stub which was removed in Issue #139.
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path

import pytest


@functools.lru_cache(maxsize=1)
def _mlx_metal_available() -> bool:
    """Return whether MLX can create a Metal-backed array on this runner.

    Some headless macOS / sandboxed sessions have the ``mlx`` package
    installed but no accessible Metal device. Importing ``mlx.core`` in the
    main pytest process can abort Python before tests can skip, so probe in a
    subprocess and use the exit status as the collection guard.
    """

    probe = "import mlx.core as mx; mx.array([1]); print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    if collection_path.suffix == ".py" and not _mlx_metal_available():
        return True
    return False


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
