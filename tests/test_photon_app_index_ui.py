"""Tests for the ベクトルDB作成 UI helpers in ``app/photon_app.py``.

Covers two follow-up changes after PR #169 (non-git ingest):

1. ``_discover_user_configs`` enumerates ``configs/*.yaml`` while excluding
   backup files, eval-matrix configs, and training-only configs.
2. ``_INDEX_PIPELINE_DRIVER`` Phase 0 uses the shared
   ``scripts.ingest_repo.resolve_commit`` (so non-git directories no longer
   crash the Streamlit pipeline driver).

We import ``app/photon_app.py`` via importlib to avoid pulling Streamlit's
runtime path-config side effects.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PHOTON_APP_PATH = PROJECT_ROOT / "app" / "photon_app.py"


def _load_photon_app_module():
    module_name = "photon_app_under_test_index_ui"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, PHOTON_APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


photon_app = _load_photon_app_module()


# ---------------------------------------------------------------
# _discover_user_configs
# ---------------------------------------------------------------


class TestDiscoverUserConfigs:
    def test_returns_baseline_and_institutional(self) -> None:
        configs = photon_app._discover_user_configs()
        assert "configs/baseline.yaml" in configs
        assert "configs/institutional_docs.yaml" in configs
        assert "configs/photon_small.yaml" in configs

    def test_excludes_wave6_backup(self) -> None:
        configs = photon_app._discover_user_configs()
        assert not any(c.endswith(".wave6_backup") for c in configs)

    def test_excludes_eval_matrix_configs(self) -> None:
        configs = photon_app._discover_user_configs()
        assert "configs/eval_qwen_model_matrix.yaml" not in configs
        assert "configs/eval_qwen_model_matrix_400.yaml" not in configs

    def test_excludes_retrain_config(self) -> None:
        configs = photon_app._discover_user_configs()
        assert "configs/institutional_docs_photon_retrain.yaml" not in configs

    def test_alphabetical_order(self) -> None:
        configs = photon_app._discover_user_configs()
        assert configs == sorted(configs)

    def test_synthetic_directory_with_excluded_patterns(self, tmp_path: Path) -> None:
        # 合成ディレクトリで全除外パターンが効くことを確認
        (tmp_path / "good.yaml").write_text("")
        (tmp_path / "good2.yaml").write_text("")
        (tmp_path / "old.yaml.wave6_backup").write_text("")
        (tmp_path / "eval_matrix.yaml").write_text("")
        (tmp_path / "something_retrain.yaml").write_text("")

        # 既存除外 stem は "configs/" 以下のファイル名に依存するので、
        # synthetic dir では default の動作だけ確認 (eval_/.wave6_backup の除外)
        configs = photon_app._discover_user_configs(configs_dir=tmp_path)
        # tmp_path 配下なので relative_to(PROJECT_ROOT) は失敗 → fallback で絶対パス
        names = [Path(c).name for c in configs]
        assert "good.yaml" in names
        assert "good2.yaml" in names
        assert "old.yaml.wave6_backup" not in names
        assert "eval_matrix.yaml" not in names

    def test_missing_configs_dir_returns_empty(self, tmp_path: Path) -> None:
        configs = photon_app._discover_user_configs(configs_dir=tmp_path / "nope")
        assert configs == []


# ---------------------------------------------------------------
# _INDEX_PIPELINE_DRIVER Phase 0: shared resolver
# ---------------------------------------------------------------


class TestDriverPhase0SharedResolver:
    """Phase 0 が ``scripts.ingest_repo.resolve_commit`` を経由することを確認。"""

    def test_driver_imports_resolve_commit(self) -> None:
        driver = photon_app._INDEX_PIPELINE_DRIVER
        assert "from scripts.ingest_repo import resolve_commit" in driver
        assert "resolve_commit(repo_dir, 'HEAD')" in driver
        # 旧コード (git rev-parse 直叩き) が残っていないこと
        assert "rev-parse" not in driver, (
            "Phase 0 が git rev-parse を直叩きしていると、非 git directory で "
            "強制 fail してしまう。共有 resolver 経由に統一すること。"
        )

    def test_driver_resolves_non_git_via_synthesized_id(self, tmp_path: Path) -> None:
        """Driver の Phase 0 を実機で動かし、non-git でも 40 文字 id を返すこと。"""
        (tmp_path / "doc.md").write_text("hello\n", encoding="utf-8")

        # Phase 0 だけを取り出して実行
        phase0_script = (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from scripts.ingest_repo import resolve_commit\n"
            f"commit = resolve_commit({str(tmp_path)!r}, 'HEAD')\n"
            "print(commit)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", phase0_script],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=True,
            timeout=30,
        )
        commit = result.stdout.strip()
        assert commit.startswith("manual-")
        assert len(commit) == 40
