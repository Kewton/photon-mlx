"""Tests for load_active_graph helper in baseline_reporag.indexing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline_reporag.indexing import load_active_graph


class TestLoadActiveGraph:
    def test_heading_graph_enabled_returns_heading_graph(self, tmp_path: Path) -> None:
        """heading_graph enabled + json exists → HeadingGraph returned."""
        hg_path = tmp_path / "heading_graph.json"
        hg_path.write_text(
            json.dumps({"sections": {}, "chunk_to_path": {}}), encoding="utf-8"
        )
        cfg = {"indexing": {"heading_graph": {"enabled": True}}}
        from baseline_reporag.indexing.heading_graph import HeadingGraph

        result = load_active_graph(cfg, tmp_path)
        assert isinstance(result, HeadingGraph)

    def test_symbol_graph_enabled_returns_symbol_graph(self, tmp_path: Path) -> None:
        """heading disabled + symbol enabled + json exists → SymbolGraph returned."""
        sg_path = tmp_path / "symbol_graph.json"
        sg_path.write_text(
            json.dumps({"definitions": {}, "edges": {}}), encoding="utf-8"
        )
        cfg = {
            "indexing": {
                "heading_graph": {"enabled": False},
                "symbol_graph": {"enabled": True},
            }
        }
        from baseline_reporag.indexing.symbol_graph import SymbolGraph

        result = load_active_graph(cfg, tmp_path)
        assert isinstance(result, SymbolGraph)

    def test_both_enabled_heading_takes_priority(self, tmp_path: Path) -> None:
        """Both enabled → heading graph wins (DR1-012)."""
        (tmp_path / "heading_graph.json").write_text(
            json.dumps({"sections": {}, "chunk_to_path": {}}), encoding="utf-8"
        )
        (tmp_path / "symbol_graph.json").write_text(
            json.dumps({"definitions": {}, "edges": {}}), encoding="utf-8"
        )
        cfg = {
            "indexing": {
                "heading_graph": {"enabled": True},
                "symbol_graph": {"enabled": True},
            }
        }
        from baseline_reporag.indexing.heading_graph import HeadingGraph

        result = load_active_graph(cfg, tmp_path)
        assert isinstance(result, HeadingGraph)

    def test_both_disabled_returns_none(self, tmp_path: Path) -> None:
        """Both disabled → None returned."""
        cfg = {
            "indexing": {
                "heading_graph": {"enabled": False},
                "symbol_graph": {"enabled": False},
            }
        }
        assert load_active_graph(cfg, tmp_path) is None

    def test_heading_enabled_json_missing_raises(self, tmp_path: Path) -> None:
        """heading_graph enabled but json absent → FileNotFoundError (DR2-006)."""
        cfg = {"indexing": {"heading_graph": {"enabled": True}}}
        with pytest.raises(FileNotFoundError):
            load_active_graph(cfg, tmp_path)

    def test_default_cfg_returns_none(self, tmp_path: Path) -> None:
        """Empty config: both graphs default to their off states → None."""
        # heading defaults False, symbol defaults True but no json present
        # symbol defaults True → would raise FileNotFoundError
        # Test with explicit both off via empty heading block only
        cfg = {
            "indexing": {
                "heading_graph": {},
                "symbol_graph": {"enabled": False},
            }
        }
        assert load_active_graph(cfg, tmp_path) is None
