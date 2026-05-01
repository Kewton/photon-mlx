"""Collection guards for baseline_reporag tests."""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path

import pytest


_PHOTON_PIPELINE_TESTS = {
    "test_photon_pipeline.py",
    "test_photon_pipeline_checkpoint_load.py",
}


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
    if collection_path.name in _PHOTON_PIPELINE_TESTS and not _mlx_metal_available():
        return True
    return False
