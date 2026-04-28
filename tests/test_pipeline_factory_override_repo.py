"""Tests for ``pipeline_factory.override_repo_for_pipeline`` helper.

Bug background: ``build_pipeline(cfg)`` resolves the index directory from
``cfg.repo.repo_id`` (``data/indexes/{cfg.repo.repo_id}``). When a caller
supplies their own ``repo_id`` (CLI ``--repo-id``, Streamlit project
selection, ...) without mutating ``cfg`` first, the indexes loaded are
the wrong corpus → silent retrieval failure with confusing errors.

The helper centralises the documented "mutate-cfg-before-build_pipeline"
pattern (already used by ``scripts/run_baseline_eval.py`` and
``run_multi_turn_eval.py``) and additionally resolves the actual
``repo_commit`` from ``chunks.db``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from baseline_reporag.config import Config
from baseline_reporag.pipeline_factory import (
    _lookup_repo_commit_from_db,
    override_repo_for_pipeline,
)


def _make_chunks_db(db_path: Path, repo_id: str, repo_commit: str) -> None:
    """Minimal chunks.db with one row matching the schema in store.py."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
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
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"{repo_id}::doc.md::1-10",
                repo_id,
                repo_commit,
                "doc.md",
                "markdown",
                1,
                10,
                "",
                "test",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _make_minimal_config(data_root: Path, repo_id: str, repo_commit: str) -> Config:
    return Config(
        {
            "paths": {"data_root": str(data_root), "log_root": str(data_root / "logs")},
            "repo": {"repo_id": repo_id, "repo_commit": repo_commit},
            "model": {"provider": "baseline", "model_id": "fake"},
        }
    )


# ---------------------------------------------------------------
# _lookup_repo_commit_from_db
# ---------------------------------------------------------------


class TestLookupRepoCommitFromDb:
    def test_returns_actual_commit(self, tmp_path: Path) -> None:
        _make_chunks_db(
            tmp_path / "indexes" / "myrepo" / "chunks.db",
            repo_id="myrepo",
            repo_commit="manual-12345",
        )
        commit = _lookup_repo_commit_from_db("myrepo", str(tmp_path))
        assert commit == "manual-12345"

    def test_returns_none_when_db_missing(self, tmp_path: Path) -> None:
        commit = _lookup_repo_commit_from_db("ghost", str(tmp_path))
        assert commit is None

    def test_returns_none_when_repo_id_not_in_db(self, tmp_path: Path) -> None:
        _make_chunks_db(
            tmp_path / "indexes" / "other" / "chunks.db",
            repo_id="other",
            repo_commit="abc",
        )
        # Look up a repo_id that has its own dir but no matching row inside
        # → fall through to None (cannot happen in practice but covers the
        # edge case of an empty / hand-edited DB).
        empty_db = tmp_path / "indexes" / "empty" / "chunks.db"
        empty_db.parent.mkdir(parents=True)
        conn = sqlite3.connect(str(empty_db))
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
            conn.commit()
        finally:
            conn.close()
        assert _lookup_repo_commit_from_db("empty", str(tmp_path)) is None


# ---------------------------------------------------------------
# override_repo_for_pipeline
# ---------------------------------------------------------------


class TestOverrideRepoForPipeline:
    def test_overrides_repo_id_and_commit_from_db(self, tmp_path: Path) -> None:
        _make_chunks_db(
            tmp_path / "indexes" / "inst_test" / "chunks.db",
            repo_id="inst_test",
            repo_commit="manual-99999",
        )
        cfg = _make_minimal_config(
            tmp_path,
            repo_id="institutional_documents",
            repo_commit="9e500539",  # sentinel for old corpus
        )

        override_repo_for_pipeline(cfg, "inst_test")

        assert cfg.repo.repo_id == "inst_test"
        assert cfg.repo.repo_commit == "manual-99999"

    def test_keeps_repo_commit_when_db_missing(self, tmp_path: Path) -> None:
        """index dir/chunks.db が無い repo_id の場合、repo_id だけ更新し
        repo_commit は元の値を保つ (silent な空文字置換を避ける)。"""
        cfg = _make_minimal_config(tmp_path, repo_id="orig", repo_commit="orig-commit")

        override_repo_for_pipeline(cfg, "fresh_unindexed")

        assert cfg.repo.repo_id == "fresh_unindexed"
        assert cfg.repo.repo_commit == "orig-commit"

    def test_none_repo_id_is_noop(self, tmp_path: Path) -> None:
        cfg = _make_minimal_config(tmp_path, "orig", "orig-commit")
        override_repo_for_pipeline(cfg, None)
        assert cfg.repo.repo_id == "orig"
        assert cfg.repo.repo_commit == "orig-commit"

    def test_empty_repo_id_is_noop(self, tmp_path: Path) -> None:
        cfg = _make_minimal_config(tmp_path, "orig", "orig-commit")
        override_repo_for_pipeline(cfg, "")
        assert cfg.repo.repo_id == "orig"
        assert cfg.repo.repo_commit == "orig-commit"

    def test_explicit_data_root_kw_arg(self, tmp_path: Path) -> None:
        """``data_root`` kwarg overrides ``cfg.paths.data_root`` for the lookup
        — letting callers point at an alternate index location."""
        alt_root = tmp_path / "alt"
        _make_chunks_db(
            alt_root / "indexes" / "myrepo" / "chunks.db",
            repo_id="myrepo",
            repo_commit="alt-commit",
        )
        cfg = _make_minimal_config(tmp_path, "orig", "orig-commit")

        override_repo_for_pipeline(cfg, "myrepo", data_root=str(alt_root))

        assert cfg.repo.repo_id == "myrepo"
        assert cfg.repo.repo_commit == "alt-commit"


# ---------------------------------------------------------------
# Regression guard: existing callers must not regress
# ---------------------------------------------------------------


class TestExistingCallerCompat:
    """Existing scripts (run_baseline_eval, run_multi_turn_eval) already
    mutate ``cfg.repo.repo_id`` directly. The new helper must not introduce
    any new globals or break the cfg shape they depend on.
    """

    def test_helper_does_not_change_attributes_other_than_repo(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_minimal_config(tmp_path, "orig", "orig-commit")
        before_paths = (cfg.paths.data_root, cfg.paths.log_root)
        before_model = (cfg.model.provider, cfg.model.model_id)

        override_repo_for_pipeline(cfg, "newrepo")  # DB missing → no-op on commit

        assert (cfg.paths.data_root, cfg.paths.log_root) == before_paths
        assert (cfg.model.provider, cfg.model.model_id) == before_model
