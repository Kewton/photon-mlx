"""Unit tests for pure helpers in app/photon_app.py.

These tests avoid importing Streamlit by importing the module via its file
path and relying only on the helper functions that do not touch ``st``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
PHOTON_APP_PATH = PROJECT_ROOT / "app" / "photon_app.py"


def _load_photon_app_module():
    module_name = "photon_app_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, PHOTON_APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so dataclass introspection works.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


photon_app = _load_photon_app_module()


# ---------------------------------------------------------------
# CB-002: checkpoint discovery excludes `.tmp` directories
# ---------------------------------------------------------------


class TestDiscoverCheckpoints:
    def test_discover_excludes_best_tmp(self, tmp_path: Path) -> None:
        """best.tmp/ (the atomic-save scratch dir) must NOT appear in listings."""
        # Simulated on-disk layout captured mid atomic replacement.
        (tmp_path / "repo" / "job" / "best").mkdir(parents=True)
        (tmp_path / "repo" / "job" / "best" / "weights.npz").write_bytes(b"\x00")
        (tmp_path / "repo" / "job" / "best.tmp").mkdir(parents=True)
        (tmp_path / "repo" / "job" / "best.tmp" / "weights.npz").write_bytes(b"\x00")

        entries = photon_app._discover_checkpoints(tmp_path)
        paths = [e[1] for e in entries]
        assert any(p.endswith("/best") for p in paths)
        assert not any(p.endswith("/best.tmp") for p in paths)
        assert not any("best.tmp" in e[2] for e in entries)

    def test_discover_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        assert photon_app._discover_checkpoints(missing) == []

    def test_discover_sorts_best_then_final_then_step(self, tmp_path: Path) -> None:
        for name in ("step_000100", "final", "best", "step_000050"):
            (tmp_path / "repo" / "job" / name).mkdir(parents=True)
            (tmp_path / "repo" / "job" / name / "weights.npz").write_bytes(b"\x00")

        entries = photon_app._discover_checkpoints(tmp_path)
        priorities = [e[0] for e in entries]
        # Sorted ascending by priority: best(0), final(1), step(2), step(2)
        assert priorities == sorted(priorities)
        assert entries[0][1].endswith("/best")
        assert entries[1][1].endswith("/final")


# ---------------------------------------------------------------
# CB-001: page_index no longer uses shell=True, and _safe_id rejects injection
# ---------------------------------------------------------------


class TestSafeId:
    def test_safe_id_accepts_allowlist(self) -> None:
        assert photon_app._safe_id("my_repo_01", label="repo_id") == "my_repo_01"

    @pytest.mark.parametrize(
        "bad",
        [
            "foo; rm -rf /",
            "foo|bar",
            "foo && bar",
            "../etc/passwd",
            "foo/bar",
            "foo\\bar",
            "foo bar",  # space
            "foo$bar",
            "",
        ],
    )
    def test_safe_id_rejects_metacharacters(self, bad: str) -> None:
        with pytest.raises(ValueError):
            photon_app._safe_id(bad, label="repo_id")


class TestPageIndexNoShellTrue:
    """Guardrail: no `shell=True` should remain in photon_app.py.

    This is a smoke-style regression test rather than a full UI test, since
    the goal is to prevent future regressions that reintroduce shell=True in
    ANY subprocess spawn inside the app module.
    """

    def test_source_has_no_shell_true(self) -> None:
        src = PHOTON_APP_PATH.read_text(encoding="utf-8")
        # Strip single-line Python comments so a comment-level mention of
        # shell=True wouldn't fail the guardrail. Adequate for this file's
        # coding style (no multi-line strings contain the literal).
        code_lines = [
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        ]
        # Strip the module docstring too (a triple-quoted block at the top)
        # to keep the check focused on actual executable code.
        joined = "\n".join(code_lines)
        # Quick and simple: remove the first triple-double-quoted block if
        # it starts at the very beginning of the file.
        if joined.startswith('"""'):
            end = joined.find('"""', 3)
            if end >= 0:
                joined = joined[end + 3 :]
        assert "shell=True" not in joined, (
            "shell=True must not appear in app/photon_app.py — use argv list + shell=False"
        )


class TestSubprocessImportAvailable:
    """Smoke: photon_app imports ``subprocess`` so argv-list spawns work."""

    def test_subprocess_module_referenced(self) -> None:
        assert subprocess.Popen  # sanity
        src = PHOTON_APP_PATH.read_text(encoding="utf-8")
        assert "import subprocess" in src
