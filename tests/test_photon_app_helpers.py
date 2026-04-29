"""Unit tests for pure helpers in app/photon_app.py.

These tests avoid importing Streamlit by importing the module via its file
path and relying only on the helper functions that do not touch ``st``.
"""

from __future__ import annotations

import importlib.util
import sqlite3
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
    """Guardrail: no `shell=True` should remain in photon_app.py or eval_panel.py.

    This is a smoke-style regression test rather than a full UI test, since
    the goal is to prevent future regressions that reintroduce shell=True in
    ANY subprocess spawn inside the app module or its eval-runner helpers.
    Wave 4 (W4-T1, T-E6) extended the scan to ``app/components/eval_panel.py``
    so the new subprocess orchestration helpers are also covered.
    """

    @staticmethod
    def _strip_comments_and_module_docstring(src: str) -> str:
        code_lines = [
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        ]
        joined = "\n".join(code_lines)
        if joined.startswith('"""'):
            end = joined.find('"""', 3)
            if end >= 0:
                joined = joined[end + 3 :]
        return joined

    def test_source_has_no_shell_true(self) -> None:
        src = PHOTON_APP_PATH.read_text(encoding="utf-8")
        joined = self._strip_comments_and_module_docstring(src)
        assert "shell=True" not in joined, (
            "shell=True must not appear in app/photon_app.py — use argv list + shell=False"
        )

    def test_eval_panel_has_no_shell_true(self) -> None:
        eval_panel_path = PROJECT_ROOT / "app" / "components" / "eval_panel.py"
        assert eval_panel_path.exists(), f"missing {eval_panel_path}"
        src = eval_panel_path.read_text(encoding="utf-8")
        # AST-based check: reject any keyword literal ``shell=True`` passed
        # to any call expression.  This is stricter than a plain-text scan
        # because it ignores docstrings / comments that legitimately
        # mention the token (e.g. warnings in the function docstring) and
        # catches whitespace variations like ``shell = True``.
        import ast

        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if (
                    kw.arg == "shell"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    raise AssertionError(
                        "shell=True literal found in app/components/eval_panel.py "
                        f"at line {node.lineno}"
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

        fake_cfg = SimpleNamespace(
            model=SimpleNamespace(provider="baseline"),
            repo=SimpleNamespace(repo_id="demo_repo", repo_commit="orig"),
            paths=SimpleNamespace(data_root=str(tmp_path)),
        )

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
        fake_cfg = SimpleNamespace(
            model=SimpleNamespace(provider="photon"),
            repo=SimpleNamespace(repo_id="demo_repo", repo_commit="orig"),
            paths=SimpleNamespace(data_root=str(tmp_path)),
        )

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


# ---------------------------------------------------------------
# Issue #115 CB-001: _run_query honours wizard-generated photon_config_path
# ---------------------------------------------------------------


class TestResolveActiveConfigPath:
    """`photon_config_path` MUST take priority over `config_path` (mirrors
    the resolution rule already enforced by ``_launch_eval_job``)."""

    def test_returns_photon_config_when_set(self, tmp_path: Path) -> None:
        proj = _make_proj(tmp_path)
        proj.photon_config_path = str(tmp_path / "wizard.yaml")
        assert photon_app._resolve_active_config_path(proj) == proj.photon_config_path

    def test_falls_back_to_config_path_when_photon_blank(self, tmp_path: Path) -> None:
        proj = _make_proj(tmp_path)
        proj.photon_config_path = ""
        assert photon_app._resolve_active_config_path(proj) == proj.config_path


class TestPipelineCacheKey:
    """Cache key MUST embed the resolved config path so a swap invalidates
    any previously-built pipeline for the same project."""

    def test_key_includes_project_name_and_path(self) -> None:
        key = photon_app._pipeline_cache_key("demo", "/tmp/cfg.yaml")
        assert "demo" in key
        assert "/tmp/cfg.yaml" in key
        assert key.startswith("pipeline_")

    def test_distinct_paths_yield_distinct_keys(self) -> None:
        a = photon_app._pipeline_cache_key("demo", "/tmp/baseline.yaml")
        b = photon_app._pipeline_cache_key("demo", "/tmp/photon.yaml")
        assert a != b


class TestRunQueryUsesPhotonConfigPath:
    """CB-001 regression: when the wizard has emitted a PHOTON YAML, the
    chat path (``_run_query``) MUST load THAT YAML — not the bare
    ``proj.config_path`` — so domain templates / best-practice merges
    actually take effect."""

    def test_run_query_loads_photon_config_path_when_set(self, tmp_path: Path) -> None:
        proj = _make_proj(tmp_path)
        wizard_yaml = tmp_path / "photon.wizard.yaml"
        wizard_yaml.write_text("model:\n  provider: photon\n")
        proj.photon_config_path = str(wizard_yaml)

        result = SimpleNamespace(
            answer="hello",
            session_id="s1",
            turn_id=1,
            cited_chunk_ids=[],
            wrong_citation_indices=[],
            no_citation=True,
            latency=SimpleNamespace(total_ms=42.0),
            memory=SimpleNamespace(),
            drift_metrics=None,
        )
        fake_pipeline = MagicMock()
        fake_pipeline.query.return_value = result
        fake_cfg = SimpleNamespace(
            model=SimpleNamespace(provider="photon"),
            repo=SimpleNamespace(repo_id="demo_repo", repo_commit="orig"),
            paths=SimpleNamespace(data_root=str(tmp_path)),
        )
        fake_session_state = _FakeSessionState()

        with (
            patch.object(
                photon_app, "load_config", create=True, return_value=fake_cfg
            ) as mock_load,
            patch.object(
                photon_app,
                "build_pipeline",
                create=True,
                return_value=fake_pipeline,
            ),
            patch.object(photon_app.st, "session_state", fake_session_state),
        ):
            photon_app._run_query(proj, "q?", "sess")

        mock_load.assert_called_once_with(str(wizard_yaml))
        # And the pipeline must be cached under a key that embeds the wizard
        # path — not the bare config_path — so a future swap re-builds it.
        expected_key = photon_app._pipeline_cache_key(proj.name, str(wizard_yaml))
        assert expected_key in fake_session_state

    def test_run_query_falls_back_to_config_path_when_photon_blank(
        self, tmp_path: Path
    ) -> None:
        proj = _make_proj(tmp_path)
        proj.photon_config_path = ""

        result = SimpleNamespace(
            answer="ok",
            session_id="s1",
            turn_id=1,
            cited_chunk_ids=[],
            wrong_citation_indices=[],
            no_citation=True,
            latency=SimpleNamespace(total_ms=1.0),
            memory=SimpleNamespace(),
            drift_metrics=None,
        )
        fake_pipeline = MagicMock()
        fake_pipeline.query.return_value = result
        fake_cfg = SimpleNamespace(
            model=SimpleNamespace(provider="baseline"),
            repo=SimpleNamespace(repo_id="demo_repo", repo_commit="orig"),
            paths=SimpleNamespace(data_root=str(tmp_path)),
        )
        fake_session_state = _FakeSessionState()

        with (
            patch.object(
                photon_app, "load_config", create=True, return_value=fake_cfg
            ) as mock_load,
            patch.object(
                photon_app,
                "build_pipeline",
                create=True,
                return_value=fake_pipeline,
            ),
            patch.object(photon_app.st, "session_state", fake_session_state),
        ):
            photon_app._run_query(proj, "q?", "sess")

        mock_load.assert_called_once_with(proj.config_path)

    def test_run_query_cache_invalidates_when_config_path_changes(
        self, tmp_path: Path
    ) -> None:
        """Swapping ``photon_config_path`` MUST trigger a fresh ``build_pipeline``
        — even though ``proj.name`` stays the same — because the cache key
        embeds the resolved path."""
        proj = _make_proj(tmp_path)
        first_yaml = tmp_path / "first.yaml"
        first_yaml.write_text("model:\n  provider: baseline\n")
        proj.photon_config_path = str(first_yaml)

        result = SimpleNamespace(
            answer="ok",
            session_id="s1",
            turn_id=1,
            cited_chunk_ids=[],
            wrong_citation_indices=[],
            no_citation=True,
            latency=SimpleNamespace(total_ms=1.0),
            memory=SimpleNamespace(),
            drift_metrics=None,
        )
        fake_pipeline = MagicMock()
        fake_pipeline.query.return_value = result
        fake_cfg = SimpleNamespace(
            model=SimpleNamespace(provider="baseline"),
            repo=SimpleNamespace(repo_id="demo_repo", repo_commit="orig"),
            paths=SimpleNamespace(data_root=str(tmp_path)),
        )
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
            photon_app._run_query(proj, "q?", "sess")
            first_key = photon_app._pipeline_cache_key(proj.name, str(first_yaml))
            assert first_key in fake_session_state
            # Same path → cache hit, build_pipeline not called again.
            photon_app._run_query(proj, "q?", "sess")
            assert mock_build.call_count == 1

            # Swap to a different wizard-generated YAML → new cache key →
            # build_pipeline called again.
            second_yaml = tmp_path / "second.yaml"
            second_yaml.write_text("model:\n  provider: photon\n")
            proj.photon_config_path = str(second_yaml)
            photon_app._run_query(proj, "q?", "sess")
            assert mock_build.call_count == 2

            # CB-002: the stale cache entry under the previous path MUST
            # have been evicted so the previous pipeline (which would
            # otherwise pin MLX weights in memory) becomes collectable.
            second_key = photon_app._pipeline_cache_key(proj.name, str(second_yaml))
            assert second_key in fake_session_state
            assert first_key not in fake_session_state, (
                "stale pipeline cache for the previous config path was not evicted"
            )


class TestRunQueryOverridesRepoFromProject:
    """``_run_query`` MUST override ``cfg.repo.repo_id`` / ``cfg.repo.repo_commit``
    from the UI-selected ``proj.repo_id`` so ``build_pipeline`` loads the right
    corpus's indexes (``data/indexes/{cfg.repo.repo_id}``). This guards the
    silent-retrieval-failure trap observed when a project's ``repo_id`` differs
    from the config's hardcoded value (e.g. ``inst_test`` paired with
    ``configs/institutional_docs_photon.yaml`` whose ``repo_id`` is
    ``institutional_documents``).
    """

    def test_run_query_mutates_cfg_repo_to_match_proj_repo_id(
        self, tmp_path: Path
    ) -> None:
        # Create a chunks.db so override_repo_for_pipeline can resolve the
        # actual repo_commit from disk.
        idx_dir = tmp_path / "indexes" / "alpha_repo"
        idx_dir.mkdir(parents=True)
        db_path = idx_dir / "chunks.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                """CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    repo_commit TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    language TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    section_header TEXT,
                    content TEXT
                )"""
            )
            conn.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    "alpha_repo::doc.md::1-10",
                    "alpha_repo",
                    "real-commit-abc",
                    "doc.md",
                    "markdown",
                    1,
                    10,
                    "",
                    "x",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        proj = _make_proj(tmp_path)
        proj.repo_id = "alpha_repo"  # UI selection
        proj.config_path = str(tmp_path / "cfg.yaml")
        Path(proj.config_path).write_text("model:\n  provider: baseline\n")

        # config has the WRONG repo (the trap reproduced)
        fake_cfg = SimpleNamespace(
            model=SimpleNamespace(provider="baseline"),
            repo=SimpleNamespace(
                repo_id="institutional_documents",
                repo_commit="9e500539",
            ),
            paths=SimpleNamespace(data_root=str(tmp_path)),
        )

        # capture cfg state at build_pipeline call time
        captured = {}

        def fake_build(cfg):
            captured["repo_id"] = cfg.repo.repo_id
            captured["repo_commit"] = cfg.repo.repo_commit
            return MagicMock(
                query=MagicMock(
                    return_value=SimpleNamespace(
                        answer="ok",
                        session_id="s",
                        turn_id=1,
                        cited_chunk_ids=[],
                        wrong_citation_indices=[],
                        no_citation=False,
                        latency=SimpleNamespace(total_ms=0.0),
                        memory=SimpleNamespace(),
                        drift_metrics=None,
                    )
                )
            )

        fake_session_state = _FakeSessionState()
        with (
            patch.object(photon_app, "load_config", create=True, return_value=fake_cfg),
            patch.object(
                photon_app, "build_pipeline", create=True, side_effect=fake_build
            ),
            patch.object(photon_app.st, "session_state", fake_session_state),
        ):
            photon_app._run_query(proj, "q?", "sess")

        # build_pipeline saw the OVERRIDDEN repo metadata, not the config's hardcoded value
        assert captured["repo_id"] == "alpha_repo", (
            "cfg.repo.repo_id must be overridden to the project's repo_id "
            "before build_pipeline is invoked"
        )
        assert captured["repo_commit"] == "real-commit-abc", (
            "cfg.repo.repo_commit must be resolved from chunks.db "
            "(otherwise graph_expansion's iter_repo SQL filter sees no rows)"
        )
