"""Tests for Issue #109 ``indexing.symbol_graph.enabled=false`` path in
``pipeline_factory._build_baseline_deps_no_mlx``.

Ensures the factory honours the dead-flag:
- enabled=false  → ``SymbolGraph.load`` is NOT called; ``graph`` is ``None``.
- enabled=true   → existing behaviour (``SymbolGraph.load`` invoked).
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

# Stub MLX if missing (pipeline_factory lazily imports generator.py → mlx_lm).
if importlib.util.find_spec("mlx") is None:
    for _mod in ("mlx", "mlx.core", "mlx_lm", "mlx_lm.sample_utils"):
        if _mod not in sys.modules:
            _stub = ModuleType(_mod)
            _stub.make_sampler = lambda **kw: None  # type: ignore[attr-defined]
            sys.modules[_mod] = _stub

from baseline_reporag.config import Config  # noqa: E402
from baseline_reporag.pipeline_factory import _build_baseline_deps_no_mlx  # noqa: E402


def _cfg_heading(
    *, heading_enabled: bool | None, symbol_enabled: bool | None, tmp_path
) -> Config:
    data = {
        "repo": {"repo_id": "demo", "repo_commit": "head"},
        "paths": {
            "data_root": str(tmp_path / "data"),
            "log_root": str(tmp_path / "logs"),
        },
        "model": {"model_id": "test-model"},
        "generation": {
            "max_new_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "retrieval": {"reranker": {"enabled": False}},
        "indexing": {},
    }
    if heading_enabled is not None:
        data["indexing"]["heading_graph"] = {"enabled": heading_enabled}
    if symbol_enabled is not None:
        data["indexing"]["symbol_graph"] = {"enabled": symbol_enabled}
    return Config(data)


def _cfg(*, enabled: bool | None, tmp_path) -> Config:
    data = {
        "repo": {"repo_id": "demo", "repo_commit": "head"},
        "paths": {
            "data_root": str(tmp_path / "data"),
            "log_root": str(tmp_path / "logs"),
        },
        "model": {
            "model_id": "test-model",
        },
        "generation": {
            "max_new_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "retrieval": {
            "reranker": {"enabled": False},
        },
    }
    if enabled is not None:
        data["indexing"] = {"symbol_graph": {"enabled": enabled}}
    return Config(data)


class TestPipelineFactoryConditionalGraph:
    def test_enabled_false_sets_graph_to_none(self, tmp_path):
        cfg = _cfg(enabled=False, tmp_path=tmp_path)

        with (
            patch("baseline_reporag.ingestion.store.ChunkStore"),
            patch("baseline_reporag.indexing.lexical.LexicalIndex"),
            patch("baseline_reporag.indexing.embedding.EmbeddingIndex"),
            patch(
                "baseline_reporag.indexing.symbol_graph.SymbolGraph"
            ) as mock_graph_cls,
            patch("baseline_reporag.memory.session.SessionManager"),
            patch("baseline_reporag.generation.generator.Generator"),
            patch("baseline_reporag.logger.RunLogger"),
        ):
            deps = _build_baseline_deps_no_mlx(cfg)

        assert deps["graph"] is None
        mock_graph_cls.load.assert_not_called()

    def test_enabled_true_loads_symbol_graph(self, tmp_path):
        cfg = _cfg(enabled=True, tmp_path=tmp_path)
        mock_graph = MagicMock()

        with (
            patch("baseline_reporag.ingestion.store.ChunkStore"),
            patch("baseline_reporag.indexing.lexical.LexicalIndex"),
            patch("baseline_reporag.indexing.embedding.EmbeddingIndex"),
            patch(
                "baseline_reporag.indexing.symbol_graph.SymbolGraph"
            ) as mock_graph_cls,
            patch("baseline_reporag.memory.session.SessionManager"),
            patch("baseline_reporag.generation.generator.Generator"),
            patch("baseline_reporag.logger.RunLogger"),
        ):
            mock_graph_cls.load.return_value = mock_graph
            deps = _build_baseline_deps_no_mlx(cfg)

        assert deps["graph"] is mock_graph
        mock_graph_cls.load.assert_called_once()

    def test_missing_indexing_block_defaults_to_loading(self, tmp_path):
        cfg = _cfg(enabled=None, tmp_path=tmp_path)
        mock_graph = MagicMock()

        with (
            patch("baseline_reporag.ingestion.store.ChunkStore"),
            patch("baseline_reporag.indexing.lexical.LexicalIndex"),
            patch("baseline_reporag.indexing.embedding.EmbeddingIndex"),
            patch(
                "baseline_reporag.indexing.symbol_graph.SymbolGraph"
            ) as mock_graph_cls,
            patch("baseline_reporag.memory.session.SessionManager"),
            patch("baseline_reporag.generation.generator.Generator"),
            patch("baseline_reporag.logger.RunLogger"),
        ):
            mock_graph_cls.load.return_value = mock_graph
            deps = _build_baseline_deps_no_mlx(cfg)

        assert deps["graph"] is mock_graph
        mock_graph_cls.load.assert_called_once()


class TestPipelineFactoryHeadingGraph:
    """heading_graph conditional tests (DR1-012, DR2-006)."""

    _COMMON_PATCHES = [
        "baseline_reporag.ingestion.store.ChunkStore",
        "baseline_reporag.indexing.lexical.LexicalIndex",
        "baseline_reporag.indexing.embedding.EmbeddingIndex",
        "baseline_reporag.memory.session.SessionManager",
        "baseline_reporag.generation.generator.Generator",
        "baseline_reporag.logger.RunLogger",
    ]

    def _common_ctx(self):
        from contextlib import ExitStack

        stack = ExitStack()
        for target in self._COMMON_PATCHES:
            stack.enter_context(patch(target))
        return stack

    def test_heading_graph_block_absent_defaults_to_none(self, tmp_path):
        """heading_graph block absent + symbol disabled → graph None (default False)."""
        cfg = _cfg_heading(
            heading_enabled=None, symbol_enabled=False, tmp_path=tmp_path
        )
        with self._common_ctx():
            deps = _build_baseline_deps_no_mlx(cfg)
        assert deps["graph"] is None

    def test_heading_graph_enabled_false_sets_graph_none(self, tmp_path):
        """heading_graph.enabled=false + symbol disabled → graph None."""
        cfg = _cfg_heading(
            heading_enabled=False, symbol_enabled=False, tmp_path=tmp_path
        )
        with self._common_ctx():
            deps = _build_baseline_deps_no_mlx(cfg)
        assert deps["graph"] is None

    def test_both_enabled_heading_takes_priority(self, tmp_path):
        """Both enabled → heading_graph wins (DR1-012)."""
        cfg = _cfg_heading(heading_enabled=True, symbol_enabled=True, tmp_path=tmp_path)
        mock_heading = MagicMock()
        with (
            self._common_ctx(),
            patch(
                "baseline_reporag.indexing.heading_graph.HeadingGraph"
            ) as mock_hg_cls,
            patch("baseline_reporag.indexing.symbol_graph.SymbolGraph") as mock_sg_cls,
        ):
            mock_hg_cls.load.return_value = mock_heading
            deps = _build_baseline_deps_no_mlx(cfg)

        assert deps["graph"] is mock_heading
        mock_hg_cls.load.assert_called_once()
        mock_sg_cls.load.assert_not_called()

    def test_heading_enabled_json_missing_raises_file_not_found(self, tmp_path):
        """heading_graph.enabled=true + json absent → FileNotFoundError (DR2-006)."""
        import pytest

        cfg = _cfg_heading(
            heading_enabled=True, symbol_enabled=False, tmp_path=tmp_path
        )
        with self._common_ctx(), pytest.raises(FileNotFoundError):
            _build_baseline_deps_no_mlx(cfg)
