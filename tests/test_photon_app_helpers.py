"""Unit tests for pure helpers in app/photon_app.py.

These tests avoid importing Streamlit by importing the module via its file
path and relying only on the helper functions that do not touch ``st``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------
# Issue #82 Wave 1 (W1-T1): _run_query routes through build_pipeline(cfg)
# ---------------------------------------------------------------


def _make_proj(tmp_path: Path, name: str = "demo") -> "photon_app.Project":
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("model:\n  provider: baseline\n")
    return photon_app.Project(
        name=name,
        repo_id="demo_repo",
        index_dir=str(tmp_path / "idx"),
        config_path=str(cfg_path),
        photon_config_path="",
        checkpoint_dir="",
        use_photon=False,
        created_at="2026-04-20T00:00:00",
    )


class _FakeSessionState(dict):
    """Minimal stand-in for ``streamlit.session_state``.

    Supports both ``obj["k"]`` and ``obj.k`` access so code that mixes the
    two (as photon_app does) keeps working under test.
    """

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value) -> None:
        self[key] = value


class TestRunQueryUsesBuildPipeline:
    """W1-T1: _run_query must go through baseline_reporag.pipeline_factory."""

    def test_run_query_uses_build_pipeline(self, tmp_path: Path) -> None:
        proj = _make_proj(tmp_path)

        # Fake QueryResult returned by the mocked pipeline.
        result = SimpleNamespace(
            answer="hello",
            session_id="s1",
            turn_id=1,
            cited_chunk_ids=["c1"],
            wrong_citation_indices=[],
            no_citation=False,
            latency=SimpleNamespace(total_ms=123.0),
            memory=SimpleNamespace(),
            drift_metrics=None,
        )
        fake_pipeline = MagicMock()
        fake_pipeline.query.return_value = result

        fake_cfg = SimpleNamespace(model=SimpleNamespace(provider="baseline"))

        fake_session_state = _FakeSessionState()

        with (
            patch.object(photon_app, "load_config", create=True, return_value=fake_cfg),
            patch.object(
                photon_app,
                "build_pipeline",
                create=True,
                return_value=fake_pipeline,
            ) as mock_build,
            patch.object(photon_app.st, "session_state", fake_session_state),
        ):
            answer, metadata = photon_app._run_query(proj, "q?", "sess")

        mock_build.assert_called_once_with(fake_cfg)
        fake_pipeline.query.assert_called_once()
        assert answer == "hello"
        assert metadata["latency_ms"] == 123.0
        assert metadata["cited_count"] == 1

    def test_run_query_handles_mlx_import_error(self, tmp_path: Path) -> None:
        proj = _make_proj(tmp_path, name="demo_photon")
        fake_cfg = SimpleNamespace(model=SimpleNamespace(provider="photon"))

        fake_session_state = _FakeSessionState()

        def _raise_mlx(*_args, **_kwargs):
            raise ModuleNotFoundError("No module named 'mlx'")

        with (
            patch.object(photon_app, "load_config", create=True, return_value=fake_cfg),
            patch.object(
                photon_app, "build_pipeline", create=True, side_effect=_raise_mlx
            ),
            patch.object(photon_app.st, "session_state", fake_session_state),
        ):
            answer, metadata = photon_app._run_query(proj, "q?", "sess")

        assert "photon_unavailable_demo_photon" in fake_session_state
        assert answer.startswith("エラー")
        # Metadata contract preserved even on error.
        assert "latency_ms" in metadata
        assert "cited_count" in metadata
        assert "pack_size" in metadata
        assert "no_citation" in metadata
        assert "drift_metrics" in metadata
        assert "turn_id" in metadata
