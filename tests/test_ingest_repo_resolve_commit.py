"""Tests for ``scripts/ingest_repo.py::resolve_commit``.

Issue: 制度文書 PDF/markdown など非 git ディレクトリも ingest できるよう
``resolve_commit`` を拡張した。本テストは:

- git repo: 既存の SHA 解決動作を保つこと
- 非 git directory: ``manual-<mtime>`` 形式の 40 文字 id を返すこと
- determinism: 同じ snapshot に対して同じ id を返すこと
- file edit: ファイルを更新すると id が変わること
- empty directory: 親 dir の mtime にフォールバックすること
- non-existent path: FileNotFoundError を上げること
- explicit 40-char SHA: 解決をスキップしてそのまま返すこと

を検証する。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    """Load scripts/ingest_repo.py via importlib (scripts/ is not a package)."""
    script_path = REPO_ROOT / "scripts" / "ingest_repo.py"
    spec = importlib.util.spec_from_file_location("ingest_repo", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ingest_repo"] = module
    spec.loader.exec_module(module)
    return module


class TestResolveCommitGitRepo:
    """既存挙動: git repo では git rev-parse HEAD の出力を返す。"""

    def test_returns_real_sha_for_git_repo(self, tmp_path: Path) -> None:
        # tmp_path に新規 git repo を作成
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
        subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

        module = _load_module()
        sha = module.resolve_commit(str(tmp_path), "HEAD")
        # SHA-1 = 40 hex chars
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)


class TestResolveCommitNonGit:
    """新規挙動: 非 git directory では manual-<mtime> を合成して返す。"""

    def test_synthesizes_manual_id_for_non_git(self, tmp_path: Path, capsys) -> None:
        # tmp_path は git init していない普通のディレクトリ
        (tmp_path / "doc.md").write_text("hello\n", encoding="utf-8")
        module = _load_module()

        commit_id = module.resolve_commit(str(tmp_path), "HEAD")

        assert commit_id.startswith("manual-")
        assert len(commit_id) == 40
        # warning は stderr に出る
        err = capsys.readouterr().err
        assert "is not a git repository" in err
        assert commit_id in err

    def test_deterministic_when_files_unchanged(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_text("v1\n", encoding="utf-8")
        module = _load_module()
        first = module.resolve_commit(str(tmp_path), "HEAD")
        second = module.resolve_commit(str(tmp_path), "HEAD")
        assert first == second

    def test_changes_when_a_file_is_modified(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.md"
        target.write_text("v1\n", encoding="utf-8")
        module = _load_module()
        first = module.resolve_commit(str(tmp_path), "HEAD")

        # mtime resolution に依存しないよう sleep + 明示 mtime セット
        time.sleep(0.05)
        new_mtime = target.stat().st_mtime + 10.0
        os.utime(target, (new_mtime, new_mtime))

        second = module.resolve_commit(str(tmp_path), "HEAD")
        assert first != second

    def test_empty_directory_uses_top_level_mtime(self, tmp_path: Path) -> None:
        module = _load_module()
        # 中身がない dir でも例外を出さず id を返す
        commit_id = module.resolve_commit(str(tmp_path), "HEAD")
        assert commit_id.startswith("manual-")
        assert len(commit_id) == 40


class TestResolveCommitErrors:
    def test_non_existent_path_raises(self, tmp_path: Path) -> None:
        module = _load_module()
        with pytest.raises(FileNotFoundError):
            module.resolve_commit(str(tmp_path / "does_not_exist"), "HEAD")


class TestResolveCommitExplicitSha:
    def test_40_char_sha_is_returned_unchanged(self, tmp_path: Path) -> None:
        """明示的に 40 文字を渡すと git にも fs にも触れずそのまま返す。"""
        module = _load_module()
        sha = "a" * 40
        result = module.resolve_commit("/path/that/does/not/exist", sha)
        assert result == sha
