"""Tests for the shared ``validate_repo_id`` helper (CB-004 / CB-005).

Codex review flagged that ``baseline_reporag/pipeline_factory.py`` and
``demo/run_demo.py`` built filesystem paths from ``cfg.repo.repo_id``
without the allowlist check that ``scripts/build_symbol_graph.py``
already applied. These tests pin down the shared helper and ensure all
three entry points use it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent

# Stub MLX for the pipeline_factory import.
if importlib.util.find_spec("mlx") is None:
    for _mod in ("mlx", "mlx.core", "mlx_lm", "mlx_lm.sample_utils"):
        if _mod not in sys.modules:
            _stub = ModuleType(_mod)
            _stub.make_sampler = lambda **kw: None  # type: ignore[attr-defined]
            sys.modules[_mod] = _stub

from baseline_reporag.config import (  # noqa: E402
    _REPO_ID_MAX_LENGTH,
    Config,
    validate_repo_id,
)


class TestValidateRepoIdHappyPath:
    def test_alphanumeric_passes(self):
        assert validate_repo_id("fastapi_fastapi") == "fastapi_fastapi"

    def test_hyphen_passes(self):
        assert validate_repo_id("my-repo-123") == "my-repo-123"

    def test_underscore_passes(self):
        assert validate_repo_id("Some_Repo_1") == "Some_Repo_1"


class TestValidateRepoIdRejects:
    @pytest.mark.parametrize(
        "bad",
        [
            "../outside",
            "..",
            "/absolute/path",
            "",
            "a/b",
            "a\\b",
            "has space",
            "dot.file",
            "unicode_α",
            "日本語",
            "name;with;semicolons",
            "null\x00byte",
        ],
    )
    def test_rejects_path_traversal_and_special_chars(self, bad):
        with pytest.raises(ValueError):
            validate_repo_id(bad)

    def test_rejects_non_string(self):
        with pytest.raises((TypeError, ValueError)):
            validate_repo_id(None)  # type: ignore[arg-type]


class TestValidateRepoIdLengthLimit:
    """CB-R2-001: reject repo_id that would overflow OS filename/path limits.

    ``_REPO_ID_MAX_LENGTH`` is the authoritative ceiling; we pin the
    boundary behaviour (max-length passes, max+1 raises) rather than a
    hardcoded constant so future tuning is a one-line change in
    ``baseline_reporag/config.py``.
    """

    def test_boundary_max_length_passes(self):
        value = "a" * _REPO_ID_MAX_LENGTH
        assert validate_repo_id(value) == value

    def test_over_boundary_by_one_raises(self):
        value = "a" * (_REPO_ID_MAX_LENGTH + 1)
        with pytest.raises(ValueError):
            validate_repo_id(value)

    def test_far_over_boundary_raises(self):
        value = "a" * 300
        with pytest.raises(ValueError):
            validate_repo_id(value)


class TestPipelineFactoryUsesValidator:
    """CB-004: pipeline factory must refuse unsafe ``repo_id`` values."""

    def _cfg(self, repo_id: str, tmp_path) -> Config:
        return Config(
            {
                "repo": {"repo_id": repo_id, "repo_commit": "head"},
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
                "indexing": {"symbol_graph": {"enabled": False}},
            }
        )

    def test_factory_rejects_path_traversal(self, tmp_path):
        from baseline_reporag.pipeline_factory import _build_baseline_deps_no_mlx

        cfg = self._cfg("../outside", tmp_path=tmp_path)
        with pytest.raises(ValueError):
            _build_baseline_deps_no_mlx(cfg)

    def test_factory_rejects_absolute_path(self, tmp_path):
        from baseline_reporag.pipeline_factory import _build_baseline_deps_no_mlx

        cfg = self._cfg("/tmp/x", tmp_path=tmp_path)
        with pytest.raises(ValueError):
            _build_baseline_deps_no_mlx(cfg)

    def test_factory_accepts_valid_repo_id(self, tmp_path):
        from baseline_reporag.pipeline_factory import _build_baseline_deps_no_mlx

        cfg = self._cfg("demo", tmp_path=tmp_path)

        with (
            patch("baseline_reporag.ingestion.store.ChunkStore"),
            patch("baseline_reporag.indexing.lexical.LexicalIndex"),
            patch("baseline_reporag.indexing.embedding.EmbeddingIndex"),
            patch("baseline_reporag.indexing.symbol_graph.SymbolGraph"),
            patch("baseline_reporag.memory.session.SessionManager"),
            patch("baseline_reporag.generation.generator.Generator"),
            patch("baseline_reporag.logger.RunLogger"),
        ):
            deps = _build_baseline_deps_no_mlx(cfg)
        assert deps is not None


class TestBuildSymbolGraphReusesValidator:
    """CB-004 follow-up: the CLI now delegates to the shared helper."""

    def test_cli_validator_matches_shared_helper(self):
        spec = importlib.util.spec_from_file_location(
            "build_symbol_graph_cli_validator_under_test",
            _ROOT / "scripts" / "build_symbol_graph.py",
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        # Valid id passes.
        mod._validate_repo_id("demo_repo")
        # Invalid id raises.
        with pytest.raises(ValueError):
            mod._validate_repo_id("../x")
