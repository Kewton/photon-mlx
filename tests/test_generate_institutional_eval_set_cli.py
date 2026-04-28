"""CLI smoke tests for scripts/generate_institutional_eval_set.py (Issue #110 AC-A-9)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "generate_institutional_eval_set.py"


def _run(
    args: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    merged_env = {
        k: v for k, v in os.environ.items() if k != "INSTITUTIONAL_CORPUS_DIR"
    }
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=merged_env,
    )


def test_help_exits_zero_and_lists_expected_flags() -> None:
    proc = _run(["--help"])
    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    for flag in (
        "--corpus-dir",
        "--provider",
        "--mode",
        "--resume",
        "--seed",
        "--temperature",
        "--count",
        "--sessions",
        "--output",
        "--failure-log",
    ):
        assert flag in combined, f"{flag} not documented in --help output"


def test_missing_corpus_exits_two_with_hint() -> None:
    proc = _run([])
    assert proc.returncode == 2
    assert "corpus" in (proc.stdout + proc.stderr).lower()


def test_nonexistent_corpus_dir_exits_two() -> None:
    proc = _run(["--corpus-dir", "/nonexistent/path/that/does/not/exist"])
    assert proc.returncode == 2
    assert "corpus" in (proc.stdout + proc.stderr).lower()
