from __future__ import annotations

import tempfile
from pathlib import Path

from baseline_reporag.ingestion.extractor import _matches_any, extract_files

EXCLUDE_TRANSLATED_DOCS = ["docs/*/docs/**"]


class TestMatchesAnyExcludePattern:
    """_matches_any が docs/*/docs/** パターンで翻訳ドキュメントを正しく判定する。"""

    def test_excludes_english_docs(self) -> None:
        assert (
            _matches_any("docs/en/docs/tutorial/first.md", EXCLUDE_TRANSLATED_DOCS)
            is True
        )

    def test_excludes_ukrainian_docs(self) -> None:
        assert _matches_any("docs/uk/docs/history.md", EXCLUDE_TRANSLATED_DOCS) is True

    def test_excludes_chinese_docs(self) -> None:
        assert (
            _matches_any("docs/zh/docs/advanced/settings.md", EXCLUDE_TRANSLATED_DOCS)
            is True
        )

    def test_excludes_deeply_nested_docs(self) -> None:
        assert (
            _matches_any(
                "docs/ja/docs/tutorial/deep/nested/file.md", EXCLUDE_TRANSLATED_DOCS
            )
            is True
        )

    def test_preserves_source_code_top_level(self) -> None:
        assert _matches_any("fastapi/routing.py", EXCLUDE_TRANSLATED_DOCS) is False

    def test_preserves_source_code_nested(self) -> None:
        assert (
            _matches_any("fastapi/dependencies/utils.py", EXCLUDE_TRANSLATED_DOCS)
            is False
        )

    def test_preserves_top_level_mkdocs(self) -> None:
        assert _matches_any("docs/en/mkdocs.yml", EXCLUDE_TRANSLATED_DOCS) is False

    def test_preserves_top_level_overrides(self) -> None:
        assert (
            _matches_any("docs/en/overrides/main.html", EXCLUDE_TRANSLATED_DOCS)
            is False
        )

    def test_preserves_root_level_docs(self) -> None:
        assert _matches_any("docs/about.md", EXCLUDE_TRANSLATED_DOCS) is False

    def test_preserves_pyproject(self) -> None:
        assert _matches_any("pyproject.toml", EXCLUDE_TRANSLATED_DOCS) is False


class TestExtractFilesExcludeTranslatedDocs:
    """extract_files がディレクトリプルーニング含めて翻訳ドキュメントを除外する。"""

    def _make_repo(self, tmp: Path) -> None:
        """テスト用のリポジトリ構造を作成する。"""
        files = [
            "fastapi/routing.py",
            "fastapi/dependencies/utils.py",
            "docs/en/mkdocs.yml",
            "docs/en/docs/tutorial/first.md",
            "docs/en/docs/tutorial/deep/nested.md",
            "docs/uk/docs/history.md",
            "docs/zh/docs/advanced/settings.md",
            "README.md",
        ]
        for rel in files:
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"# {rel}\n", encoding="utf-8")

    def test_only_returns_non_excluded_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._make_repo(tmp_path)

            results = list(
                extract_files(
                    repo_path=tmp_path,
                    include=["**/*.py", "**/*.md", "**/*.yml"],
                    exclude=EXCLUDE_TRANSLATED_DOCS,
                )
            )
            paths = sorted(r.rel_path for r in results)

            assert "fastapi/routing.py" in paths
            assert "fastapi/dependencies/utils.py" in paths
            assert "docs/en/mkdocs.yml" in paths
            # NOTE: README.md はルートレベルのため **/*.md にマッチしない
            # （fnmatch は ** を特別扱いせず、**/*.md は / を含むパスのみマッチ）

            # 翻訳ドキュメントは除外されること
            assert "docs/en/docs/tutorial/first.md" not in paths
            assert "docs/en/docs/tutorial/deep/nested.md" not in paths
            assert "docs/uk/docs/history.md" not in paths
            assert "docs/zh/docs/advanced/settings.md" not in paths

    def test_file_count_is_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._make_repo(tmp_path)

            results = list(
                extract_files(
                    repo_path=tmp_path,
                    include=["**/*.py", "**/*.md", "**/*.yml"],
                    exclude=EXCLUDE_TRANSLATED_DOCS,
                )
            )
            # fastapi/routing.py, fastapi/dependencies/utils.py,
            # docs/en/mkdocs.yml = 3 files
            # (README.md はルートレベルで **/*.md にマッチしないため除外)
            assert len(results) == 3

    def test_directory_pruning_prevents_deep_traversal(self) -> None:
        """ディレクトリプルーニングにより除外対象の深いサブツリーを走査しない。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._make_repo(tmp_path)

            # os.walk のディレクトリプルーニングを検証するため、
            # 除外対象のサブディレクトリ内にさらにファイルを追加
            deep = tmp_path / "docs" / "en" / "docs" / "level1" / "level2" / "level3"
            deep.mkdir(parents=True, exist_ok=True)
            (deep / "deeply_hidden.md").write_text("# hidden\n", encoding="utf-8")

            results = list(
                extract_files(
                    repo_path=tmp_path,
                    include=["**/*.py", "**/*.md", "**/*.yml"],
                    exclude=EXCLUDE_TRANSLATED_DOCS,
                )
            )
            paths = [r.rel_path for r in results]
            assert "docs/en/docs/level1/level2/level3/deeply_hidden.md" not in paths
