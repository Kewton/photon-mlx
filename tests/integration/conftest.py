"""Shared fixtures for ``tests/integration/`` (Issue #145).

Scope notes (DR1-003 / DR4-005):

- All autouse fixtures here are ``scope="function"`` (pytest default), so any
  ``monkeypatch`` they perform is automatically reverted at test teardown.
- This conftest is read only for tests under ``tests/integration/`` because
  conftest discovery follows directory locality.  Sibling autouse fixtures
  in ``baseline_reporag/tests/conftest.py`` (e.g. ``_patch_heavy_deps`` in
  ``test_photon_pipeline_checkpoint_load.py``) do not apply here.
- ``_real_photon_model_guard`` runs after ``_photon_env_isolation`` (parameter
  dependency, DR2-003) and asserts that ``photon_mlx.model.PhotonModel`` is
  not a ``MagicMock`` so we catch any leakage from sibling test files.
"""

from __future__ import annotations

import functools
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# Environment variables the production checkpoint-load path reads.  Cleared at
# test entry so an environment leak (e.g. a developer-set
# ``PHOTON_ALLOW_RANDOM_INIT=1``) cannot mask a real failure.
_ISOLATED_ENV_VARS = ("PHOTON_CHECKPOINT_ROOT", "PHOTON_ALLOW_RANDOM_INIT")


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


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    if collection_path.suffix == ".py" and not _mlx_metal_available():
        return True
    return False


@pytest.fixture(autouse=True)
def _photon_env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove PHOTON_* env vars at test start (restored at teardown)."""
    for name in _ISOLATED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _real_photon_model_guard(_photon_env_isolation: None) -> None:
    """Fail fast if a sibling autouse fixture left PhotonModel patched.

    Parameter dependency on ``_photon_env_isolation`` pins the order
    (DR2-003): env isolation runs first, then this guard.
    """
    photon_model_mod = importlib.import_module("photon_mlx.model")
    assert not isinstance(photon_model_mod.PhotonModel, MagicMock), (
        "photon_mlx.model.PhotonModel is patched with MagicMock — autouse "
        "fixture leakage detected from a sibling conftest. Integration tests "
        "must construct the real PhotonModel."
    )


@pytest.fixture(autouse=True)
def _mlx_available_or_skip() -> None:
    """Skip the integration test on environments where MLX cannot import.

    ``mlx.core`` only loads on Apple Silicon; the weekly_eval workflow runs on
    a self-hosted M-series runner where this always succeeds.  Local linux/x86
    development boxes hit ``ImportError`` and we skip there rather than fail.
    """
    if not _mlx_metal_available():
        pytest.skip("MLX Metal device is not available on this runner")


# ---------------------------------------------------------------------------
# Fake tokenizer (DR1-008 / DR4-005)
# ---------------------------------------------------------------------------
#
# ``_build_photon_deps`` calls ``_load_hf_tokenizer`` unconditionally.  We
# never want the integration test to reach the network so the fixture below
# returns a ``MagicMock`` whose ``vocab_size`` matches the cfg's
# ``tokenizer.vocab_size``.  The methods listed below cover the call graph
# of ``deps['photon_inference'].generate_answer(max_new_tokens=1)``:
# ``encode`` (encoded prompt) and ``decode`` (final string).
# ``_check_weight_initialization`` only touches ``model.token_embed`` (DR2-009),
# so the fake tokenizer does not need to model embedding internals.


@pytest.fixture
def fake_tokenizer() -> Any:
    """Return a MagicMock tokenizer with the methods used by smoke generate."""
    fake = MagicMock()
    fake.vocab_size = 256
    fake.pad_token_id = 0
    # Smoke generate: prompt → list[int] → mx.array. Three distinct token ids
    # avoid PAD-token edge cases inside PhotonInference.generate_answer.
    fake.encode.return_value = [1, 2, 3]
    fake.decode.return_value = ""
    return fake


@pytest.fixture
def patched_hf_tokenizer(monkeypatch: pytest.MonkeyPatch, fake_tokenizer: Any) -> Any:
    """Replace the production HF tokenizer loader with the fake fixture.

    Patches at the module boundary used by ``_build_photon_deps`` (the
    only production call site).  Function-scope teardown restores the
    real loader (DR4-005).
    """
    monkeypatch.setattr(
        "baseline_reporag.photon_pipeline._load_hf_tokenizer",
        lambda tokenizer_id, expected_vocab_size: fake_tokenizer,
    )
    return fake_tokenizer
