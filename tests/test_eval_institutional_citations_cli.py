"""CLI smoke tests for scripts/eval_institutional_citations.py (Issue #110 AC-A-9)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "eval_institutional_citations.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_exits_zero_and_lists_expected_flags() -> None:
    proc = _run(["--help"])
    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    for flag in (
        "--eval-set",
        "--run-log",
        "--predictions",
        "--chunk-store",
        "--in-memory",
        "--output",
    ):
        assert flag in combined, f"{flag} not documented in --help output"


def test_no_args_exits_two() -> None:
    proc = _run([])
    assert proc.returncode == 2


def test_missing_run_log_and_predictions_exits_two(tmp_path: Path) -> None:
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text("", encoding="utf-8")
    proc = _run(["--eval-set", str(eval_set), "--in-memory"])
    assert proc.returncode == 2
    assert "run-log" in (proc.stdout + proc.stderr) or "predictions" in (
        proc.stdout + proc.stderr
    )
