"""Tests for the ``scripts/build_symbol_graph.py`` CLI (Issue #109).

Exercises the ``indexing.symbol_graph.enabled`` conditional skip: when
disabled, the script must not call ``SymbolGraph.build`` or
``SymbolGraph.save`` and should emit a skip log line on stdout.

CB-006: also verifies ``ChunkStore.close()`` is called even when
``SymbolGraph.build`` / ``.save`` raises.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _ROOT / "scripts" / "build_symbol_graph.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_symbol_graph_cli_under_test", _SCRIPT_PATH
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
        "  symbol_graph:\n"
        f"    enabled: {'true' if enabled else 'false'}\n"
    )


class TestBuildSymbolGraphConditionalSkip:
    def test_enabled_false_skips_build_and_save(self, tmp_path, capsys):
        cfg_path = tmp_path / "cfg.yaml"
        _write_cfg(cfg_path, enabled=False)

        mod = _load_script_module()

        with (
            patch.object(mod, "SymbolGraph") as mock_graph_cls,
            patch.object(mod, "ChunkStore") as mock_store_cls,
            patch.object(
                sys,
                "argv",
                [str(_SCRIPT_PATH), "--repo-id", "demo", "--config", str(cfg_path)],
            ),
        ):
            mock_graph_cls.return_value = MagicMock()
            mock_store_cls.return_value = MagicMock()
            mod.main()

        # Neither build nor save was invoked.
        mock_graph_cls.return_value.build.assert_not_called()
        mock_graph_cls.return_value.save.assert_not_called()

        captured = capsys.readouterr()
        assert "Skipped" in captured.out
        assert "indexing.symbol_graph.enabled" in captured.out

    def test_enabled_true_builds_and_saves(self, tmp_path, capsys):
        cfg_path = tmp_path / "cfg.yaml"
        _write_cfg(cfg_path, enabled=True)

        mod = _load_script_module()

        with (
            patch.object(mod, "SymbolGraph") as mock_graph_cls,
            patch.object(mod, "ChunkStore") as mock_store_cls,
            patch.object(
                sys,
                "argv",
                [str(_SCRIPT_PATH), "--repo-id", "demo", "--config", str(cfg_path)],
            ),
        ):
            mock_graph_cls.return_value = MagicMock()
            mock_store_cls.return_value = MagicMock()
            mod.main()

        mock_graph_cls.return_value.build.assert_called_once()
        mock_graph_cls.return_value.save.assert_called_once()


class TestBuildSymbolGraphStoreCloseOnFailure:
    """CB-006: ``ChunkStore`` must close even on ``build()``/``save()`` failure."""

    def test_store_closed_when_build_raises(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        _write_cfg(cfg_path, enabled=True)

        mod = _load_script_module()

        store = MagicMock()
        graph = MagicMock()
        graph.build.side_effect = RuntimeError("boom")

        with (
            patch.object(mod, "SymbolGraph") as mock_graph_cls,
            patch.object(mod, "ChunkStore") as mock_store_cls,
            patch.object(
                sys,
                "argv",
                [str(_SCRIPT_PATH), "--repo-id", "demo", "--config", str(cfg_path)],
            ),
        ):
            mock_store_cls.return_value = store
            mock_graph_cls.return_value = graph
            with pytest.raises(RuntimeError):
                mod.main()

        store.close.assert_called_once()
        graph.save.assert_not_called()

    def test_store_closed_when_save_raises(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        _write_cfg(cfg_path, enabled=True)

        mod = _load_script_module()

        store = MagicMock()
        graph = MagicMock()
        graph.save.side_effect = OSError("disk full")

        with (
            patch.object(mod, "SymbolGraph") as mock_graph_cls,
            patch.object(mod, "ChunkStore") as mock_store_cls,
            patch.object(
                sys,
                "argv",
                [str(_SCRIPT_PATH), "--repo-id", "demo", "--config", str(cfg_path)],
            ),
        ):
            mock_store_cls.return_value = store
            mock_graph_cls.return_value = graph
            with pytest.raises(OSError):
                mod.main()

        store.close.assert_called_once()
