"""Tests for ``_is_process_running`` zombie detection.

Bug background: ``os.kill(pid, 0)`` returns success for zombie (defunct)
processes, so the legacy implementation kept ``_sync_index_job`` stuck on
``running`` after the index pipeline subprocess finished. This test suite
verifies the fix uses ``ps -o stat=`` to distinguish live processes from
zombies (state code starts with ``Z``).

We synthesize a real zombie process for an end-to-end check on the host's
``ps`` command, and we mock subprocess for white-box tests of the various
``ps`` outcomes (Z, S, R, ps failure, etc).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
PHOTON_APP_PATH = PROJECT_ROOT / "app" / "photon_app.py"


def _load_photon_app_module():
    module_name = "photon_app_under_test_zombie"
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
# Pure-Python edge cases (no real processes touched)
# ---------------------------------------------------------------


class TestIsProcessRunningEdges:
    def test_none_pid_returns_false(self) -> None:
        assert photon_app._is_process_running(None) is False

    def test_non_existent_pid_returns_false(self) -> None:
        # 999999 is well above the typical max pid range; if it happens to
        # exist locally, this single test may flake (very unlikely).
        assert photon_app._is_process_running(999_999) is False

    def test_current_process_returns_true(self) -> None:
        assert photon_app._is_process_running(os.getpid()) is True


# ---------------------------------------------------------------
# White-box: mock subprocess.run to test all ps outcomes
# ---------------------------------------------------------------


class TestIsProcessRunningPsBranches:
    def _patch_kill_and_run(
        self,
        monkeypatch,
        *,
        kill_ok: bool,
        ps_returncode: int,
        ps_stdout: str,
        ps_raises: Exception | None = None,
    ):
        def fake_kill(pid, sig):
            if not kill_ok:
                raise ProcessLookupError(f"no such pid {pid}")

        def fake_run(*args, **kwargs):
            if ps_raises is not None:
                raise ps_raises
            return MagicMock(returncode=ps_returncode, stdout=ps_stdout)

        monkeypatch.setattr(photon_app.os, "kill", fake_kill)
        monkeypatch.setattr(photon_app.subprocess, "run", fake_run)

    def test_zombie_state_returns_false(self, monkeypatch) -> None:
        """ps が "Z+" を返したら False (zombie 検出)。"""
        self._patch_kill_and_run(
            monkeypatch, kill_ok=True, ps_returncode=0, ps_stdout="Z+\n"
        )
        assert photon_app._is_process_running(12345) is False

    def test_zombie_uppercase_z_returns_false(self, monkeypatch) -> None:
        """`Z` (Linux 形式) でも False。"""
        self._patch_kill_and_run(
            monkeypatch, kill_ok=True, ps_returncode=0, ps_stdout="Z\n"
        )
        assert photon_app._is_process_running(12345) is False

    def test_sleeping_state_returns_true(self, monkeypatch) -> None:
        """`S` (sleeping/idle) は live。"""
        self._patch_kill_and_run(
            monkeypatch, kill_ok=True, ps_returncode=0, ps_stdout="S+\n"
        )
        assert photon_app._is_process_running(12345) is True

    def test_running_state_returns_true(self, monkeypatch) -> None:
        """`R` (running) は live。"""
        self._patch_kill_and_run(
            monkeypatch, kill_ok=True, ps_returncode=0, ps_stdout="R\n"
        )
        assert photon_app._is_process_running(12345) is True

    def test_ps_returncode_nonzero_returns_false(self, monkeypatch) -> None:
        """ps が pid を見つけられない (kill と ps の間で消えた) → False。"""
        self._patch_kill_and_run(
            monkeypatch, kill_ok=True, ps_returncode=1, ps_stdout=""
        )
        assert photon_app._is_process_running(12345) is False

    def test_ps_invocation_failure_falls_back_to_kill_ok(self, monkeypatch) -> None:
        """ps コマンドの起動自体が失敗 → 既存挙動 (kill OK = True) を維持。"""
        self._patch_kill_and_run(
            monkeypatch,
            kill_ok=True,
            ps_returncode=0,
            ps_stdout="",
            ps_raises=FileNotFoundError("ps not found"),
        )
        assert photon_app._is_process_running(12345) is True

    def test_ps_timeout_falls_back_to_kill_ok(self, monkeypatch) -> None:
        self._patch_kill_and_run(
            monkeypatch,
            kill_ok=True,
            ps_returncode=0,
            ps_stdout="",
            ps_raises=subprocess.TimeoutExpired(cmd="ps", timeout=5),
        )
        assert photon_app._is_process_running(12345) is True

    def test_kill_fails_short_circuits(self, monkeypatch) -> None:
        """kill 段階で失敗したら ps は呼ばれず False を返す。"""
        called = []

        def fake_kill(pid, sig):
            raise ProcessLookupError("no such pid")

        def fake_run(*args, **kwargs):
            called.append(args)
            return MagicMock(returncode=0, stdout="S\n")

        monkeypatch.setattr(photon_app.os, "kill", fake_kill)
        monkeypatch.setattr(photon_app.subprocess, "run", fake_run)

        assert photon_app._is_process_running(12345) is False
        assert called == [], "ps should not be invoked when kill already failed"


# ---------------------------------------------------------------
# End-to-end: synthesize a real zombie and probe it
# ---------------------------------------------------------------


class TestRealZombieDetection:
    """``fork`` で実際に zombie を作り、検出ロジックを通す。

    macOS / Linux で ``os.fork`` が利用可能。Windows でこのテストは skip。
    """

    @pytest.mark.skipif(
        not hasattr(os, "fork"),
        reason="zombie synthesis requires os.fork (POSIX-only)",
    )
    def test_real_zombie_returns_false(self) -> None:
        # 子プロセスを fork → 即 exit → 親は wait しない → 子は zombie 化
        pid = os.fork()
        if pid == 0:
            # 子: すぐ終了
            os._exit(0)

        try:
            # 子が実際に exit して zombie になるまで少し待つ
            for _ in range(50):
                time.sleep(0.05)
                stat_result = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(pid)],
                    capture_output=True,
                    text=True,
                )
                if (
                    stat_result.returncode == 0
                    and stat_result.stdout.strip().startswith("Z")
                ):
                    break
            else:
                pytest.skip(
                    "could not synthesize a zombie within 2.5s — host scheduler "
                    "may have already reaped"
                )

            # ここで pid は zombie 状態のはず
            assert photon_app._is_process_running(pid) is False
        finally:
            # 必ず wait して zombie を回収 (テスト後の系汚染防止)
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass


# ---------------------------------------------------------------
# Regression guard: _sync_index_job の状態遷移
# ---------------------------------------------------------------


class TestSyncIndexJobUsesNewDetection:
    def test_done_log_with_zombie_pid_transitions_to_completed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """log に DONE があり、_is_process_running が False なら completed に遷移。

        本 PR 修正後は zombie pid が False になるので、この経路が機能する。
        """
        # _is_process_running を強制的に False にする (本修正の効果を simulate)
        monkeypatch.setattr(photon_app, "_is_process_running", lambda pid: False)

        log_file = tmp_path / "fake.log"
        log_file.write_text(
            "Phase 1: Ingest\nDone: 10 files, 100 chunks\n"
            "Phase 2: BM25 + Embedding\nDone.\n"
            "Phase 3: Symbol Graph\nDONE\n",
            encoding="utf-8",
        )

        job = photon_app.IndexJob(
            job_id="test_job",
            repo_dir="/tmp/fake",
            repo_id="fake",
            config_path="configs/baseline.yaml",
            pid=12345,
            started_at="2026-04-29T00:00:00",
            status="running",
            log_file=str(log_file),
            phase="ingest",
            embedding_model="x",
        )

        changed = photon_app._sync_index_job(job)

        assert changed is True
        assert job.status == "completed"
        assert job.phase == "completed"
