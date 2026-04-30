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


class TestDriverPhase35HeadingGraph:
    """_INDEX_PIPELINE_DRIVER contains Phase 3.5 heading graph step (DR2-004)."""

    def test_driver_contains_phase35(self):
        """Driver string must include 'Phase 3.5' heading graph invocation."""
        mod = _load_photon_app_module()
        driver = mod._INDEX_PIPELINE_DRIVER
        assert "Phase 3.5" in driver

    def test_driver_phase35_uses_shell_false(self):
        """Phase 3.5 subprocess call must not use shell=True (DR4-003)."""
        mod = _load_photon_app_module()
        driver = mod._INDEX_PIPELINE_DRIVER
        assert "shell=True" not in driver

    def test_driver_phase35_passes_commit(self):
        """Phase 3.5 must pass --commit to build_heading_graph (CB-002)."""
        mod = _load_photon_app_module()
        driver = mod._INDEX_PIPELINE_DRIVER
        phase35_start = driver.find("Phase 3.5")
        done_pos = driver.find("'DONE'")
        phase35_block = driver[phase35_start:done_pos]
        assert "'--commit'" in phase35_block or '"--commit"' in phase35_block

    def test_driver_phase35_before_done(self):
        """Phase 3.5 must appear before DONE sentinel in driver."""
        mod = _load_photon_app_module()
        driver = mod._INDEX_PIPELINE_DRIVER
        phase35_pos = driver.find("Phase 3.5")
        done_pos = driver.find("'DONE'")
        assert phase35_pos != -1
        assert done_pos != -1
        assert phase35_pos < done_pos

    def test_sync_index_job_phase35_detected_before_phase3(self, tmp_path):
        """'Phase 3.5' in log content sets phase='heading_graph', not 'symbol_graph'."""
        mod = _load_photon_app_module()
        IndexJob = mod.IndexJob

        job = IndexJob(
            job_id="j1",
            repo_dir="/repo",
            repo_id="demo",
            config_path="/cfg.yaml",
            pid=None,
            status="running",
        )
        log_path = tmp_path / "run.log"
        log_path.write_text("Phase 1: Ingest\nPhase 3.5: Heading Graph\n")
        job.log_file = str(log_path)

        from unittest.mock import patch

        with patch.object(mod, "_is_process_running", return_value=True):
            mod._sync_index_job(job)

        assert job.phase == "heading_graph"

    def test_sync_index_job_phase35_failed_does_not_complete(self, tmp_path):
        """'Phase 3.5: Heading Graph FAILED' without 'DONE' → job fails."""
        mod = _load_photon_app_module()
        IndexJob = mod.IndexJob

        job = IndexJob(
            job_id="j2",
            repo_dir="/repo",
            repo_id="demo",
            config_path="/cfg.yaml",
            pid=9999999,
            status="running",
        )
        log_path = tmp_path / "run.log"
        log_path.write_text(
            "Phase 1: Ingest\nPhase 3: Symbol Graph\nPhase 3.5: Heading Graph FAILED\n"
        )
        job.log_file = str(log_path)

        from unittest.mock import patch

        with patch.object(mod, "_is_process_running", return_value=False):
            mod._sync_index_job(job)

        assert job.status == "failed"
