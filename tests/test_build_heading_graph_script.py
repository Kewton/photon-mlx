"""Tests for scripts/build_heading_graph.py CLI (Issue #180, DR2-014)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _ROOT / "scripts" / "build_heading_graph.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_heading_graph_cli_under_test", _SCRIPT_PATH
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_cfg(path: Path, *, enabled: bool) -> None:
    path.write_text(
        "repo:\n"
        "  repo_id: demo\n"
        "  repo_commit: head\n"
        "paths:\n"
        f"  data_root: {path.parent / 'data'}\n"
        "  log_root: logs\n"
        "indexing:\n"
        "  heading_graph:\n"
        f"    enabled: {'true' if enabled else 'false'}\n"
    )


class TestBuildHeadingGraphSkip:
    def test_enabled_false_skips_build_and_save(self, tmp_path, capsys):
        """is_heading_graph_enabled=False → build and save are never called."""
        cfg_path = tmp_path / "cfg.yaml"
        _write_cfg(cfg_path, enabled=False)

        mod = _load_script_module()

        with (
            patch.object(mod, "HeadingGraph") as mock_graph_cls,
            patch.object(mod, "ChunkStore") as mock_store_cls,
            patch.object(mod, "_validate_repo_id"),
        ):
            with patch(
                "sys.argv",
                [
                    "build_heading_graph.py",
                    "--repo-id",
                    "demo",
                    "--config",
                    str(cfg_path),
                ],
            ):
                mod.main()

        mock_store_cls.assert_not_called()
        mock_graph_cls.assert_not_called()
        captured = capsys.readouterr()
        assert "Skipped" in captured.out

    def test_enabled_true_builds_and_saves(self, tmp_path, capsys):
        """is_heading_graph_enabled=True → HeadingGraph.build + save called."""
        cfg_path = tmp_path / "cfg.yaml"
        _write_cfg(cfg_path, enabled=True)

        mod = _load_script_module()
        mock_graph = MagicMock()
        mock_store = MagicMock()

        with (
            patch.object(mod, "HeadingGraph") as mock_graph_cls,
            patch.object(mod, "ChunkStore") as mock_store_cls,
            patch.object(mod, "_validate_repo_id"),
        ):
            mock_graph_cls.return_value = mock_graph
            mock_store_cls.return_value = mock_store
            with patch(
                "sys.argv",
                [
                    "build_heading_graph.py",
                    "--repo-id",
                    "demo",
                    "--config",
                    str(cfg_path),
                ],
            ):
                mod.main()

        mock_graph.build.assert_called_once()
        mock_graph.save.assert_called_once()
        mock_store.close.assert_called_once()

    def test_validate_repo_id_called(self, tmp_path):
        """_validate_repo_id must be called before any file access (DR2-014)."""
        cfg_path = tmp_path / "cfg.yaml"
        _write_cfg(cfg_path, enabled=False)

        mod = _load_script_module()
        with (
            patch.object(mod, "_validate_repo_id") as mock_validate,
            patch.object(mod, "ChunkStore"),
        ):
            with patch(
                "sys.argv",
                [
                    "build_heading_graph.py",
                    "--repo-id",
                    "demo",
                    "--config",
                    str(cfg_path),
                ],
            ):
                mod.main()
        mock_validate.assert_called_once_with("demo")

    def test_store_closed_on_build_failure(self, tmp_path):
        """ChunkStore.close() called even if HeadingGraph.build raises (CB-006)."""
        cfg_path = tmp_path / "cfg.yaml"
        _write_cfg(cfg_path, enabled=True)

        mod = _load_script_module()
        mock_graph = MagicMock()
        mock_graph.build.side_effect = RuntimeError("build failed")
        mock_store = MagicMock()

        with (
            patch.object(mod, "HeadingGraph") as mock_graph_cls,
            patch.object(mod, "ChunkStore") as mock_store_cls,
            patch.object(mod, "_validate_repo_id"),
        ):
            mock_graph_cls.return_value = mock_graph
            mock_store_cls.return_value = mock_store
            with patch(
                "sys.argv",
                [
                    "build_heading_graph.py",
                    "--repo-id",
                    "demo",
                    "--config",
                    str(cfg_path),
                ],
            ):
                with pytest.raises(RuntimeError):
                    mod.main()

        mock_store.close.assert_called_once()
