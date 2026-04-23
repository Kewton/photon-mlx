"""Tests for app/photon_app.py EvalJob + AppState.eval_jobs (Issue #82 Wave 1).

These tests pin the new EvalJob dataclass, the AppState.eval_jobs mapping,
and the _load_state / _save_state round-trip behaviour so that legacy state
files (4-key shape) can be rehydrated without loss and always serialise back
as 5-key JSON including ``eval_jobs: {}``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import photon_app  # noqa: E402


class TestEvalJobDataclass(unittest.TestCase):
    def test_eval_job_defaults(self) -> None:
        job = photon_app.EvalJob()
        self.assertEqual(job.job_id, "")
        self.assertEqual(job.project_name, "")
        self.assertEqual(job.eval_type, "")
        self.assertEqual(job.status, "pending")
        self.assertEqual(job.pid, None)
        self.assertEqual(job.done_q, 0)
        self.assertEqual(job.total_q, 0)
        self.assertEqual(job.p50_latency_ms, 0.0)
        self.assertEqual(job.nc_rate, 0.0)
        self.assertEqual(job.error_message, "")

    def test_is_terminal_true_for_succeeded(self) -> None:
        self.assertTrue(photon_app.EvalJob(status="succeeded").is_terminal)

    def test_is_terminal_true_for_failed(self) -> None:
        self.assertTrue(photon_app.EvalJob(status="failed").is_terminal)

    def test_is_terminal_false_for_running(self) -> None:
        self.assertFalse(photon_app.EvalJob(status="running").is_terminal)

    def test_is_terminal_false_for_pending(self) -> None:
        self.assertFalse(photon_app.EvalJob(status="pending").is_terminal)


class TestFilterKnownFieldsForEvalJob(unittest.TestCase):
    """T-E3: _filter_known_fields drops unknown keys for EvalJob."""

    def test_drops_unknown_keeps_known(self) -> None:
        raw = {
            "job_id": "abc",
            "project_name": "demo",
            "status": "running",
            "unknown_key_1": 123,
            "another_unknown": "x",
        }
        filtered = photon_app._filter_known_fields(photon_app.EvalJob, raw)
        self.assertIn("job_id", filtered)
        self.assertIn("project_name", filtered)
        self.assertIn("status", filtered)
        self.assertNotIn("unknown_key_1", filtered)
        self.assertNotIn("another_unknown", filtered)
        self.assertEqual(filtered["job_id"], "abc")


class TestLoadStateMissingEvalJobsKey(unittest.TestCase):
    """T-E1: a legacy 4-key state file round-trips to a 5-key file."""

    def test_load_then_save_emits_five_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
            # 4-key legacy shape (no ``eval_jobs`` field).
            state_file.write_text(
                json.dumps(
                    {
                        "training_jobs": {},
                        "index_jobs": {},
                        "projects": {},
                        "chat_histories": {},
                    }
                )
            )

            with patch.object(photon_app, "STATE_FILE", state_file):
                state = photon_app._load_state()
                # New field defaults to empty dict.
                self.assertEqual(state.eval_jobs, {})
                photon_app._save_state(state)
                data = json.loads(state_file.read_text())

        self.assertEqual(
            set(data.keys()),
            {
                "training_jobs",
                "index_jobs",
                "projects",
                "chat_histories",
                "eval_jobs",
            },
        )
        self.assertEqual(data["eval_jobs"], {})


class TestSaveLoadRoundtripEvalJobs(unittest.TestCase):
    """T-E2: 2-entry EvalJob dict survives save → load unchanged."""

    def test_roundtrip_preserves_two_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"

            # Wave 2 (W2-T4): _load_state now enforces that eval_job paths
            # live under PROJECT_ROOT, so the round-trip fixture must use
            # paths inside the repo. Files do not need to actually exist.
            log_file = str(photon_app.PROJECT_ROOT / "logs" / "eval" / "j1.log")
            result_json = str(
                photon_app.PROJECT_ROOT / "reports" / "eval_runs" / "j1.json"
            )
            marker_file = str(
                photon_app.PROJECT_ROOT / "reports" / "eval_runs" / "j1.done"
            )

            original = photon_app.AppState()
            original.eval_jobs["j1"] = photon_app.EvalJob(
                job_id="j1",
                project_name="demo",
                eval_type="static",
                status="succeeded",
                started_at="2026-04-20T10:00:00",
                started_at_epoch=1_700_000_000.0,
                finished_at="2026-04-20T10:05:00",
                pid=4321,
                log_file=log_file,
                result_json=result_json,
                marker_file=marker_file,
                done_q=30,
                total_q=30,
                p50_latency_ms=1234.5,
                nc_rate=0.1,
            )
            original.eval_jobs["j2"] = photon_app.EvalJob(
                job_id="j2",
                project_name="demo",
                eval_type="multi_turn",
                status="failed",
                error_message="timeout",
            )

            with patch.object(photon_app, "STATE_FILE", state_file):
                photon_app._save_state(original)
                reloaded = photon_app._load_state()

        self.assertEqual(set(reloaded.eval_jobs), {"j1", "j2"})
        self.assertEqual(reloaded.eval_jobs["j1"].status, "succeeded")
        self.assertEqual(reloaded.eval_jobs["j1"].done_q, 30)
        self.assertAlmostEqual(reloaded.eval_jobs["j1"].p50_latency_ms, 1234.5)
        self.assertAlmostEqual(reloaded.eval_jobs["j1"].nc_rate, 0.1)
        self.assertEqual(reloaded.eval_jobs["j1"].marker_file, marker_file)
        self.assertEqual(reloaded.eval_jobs["j2"].status, "failed")
        self.assertEqual(reloaded.eval_jobs["j2"].error_message, "timeout")


class TestStateTamperingDetection(unittest.TestCase):
    """T-E7 (Issue #82 Wave 2): ``_load_state`` rejects tampered eval_jobs.

    The integrity checks added by W2-T4 / D4-004 cover three classes of
    tamper:
        1. ``log_file`` / ``result_json`` / ``marker_file`` pointing
           outside ``PROJECT_ROOT``
        2. ``status`` set to an unknown value
        3. ``pid`` stored as a non-int (e.g. a string)

    Each case should produce either ``status='failed'`` with an
    ``error_message`` mentioning ``tampering`` (for path escapes) or a
    silent normalization (for pid / status).
    """

    def _write_state(self, state_dir: Path, job_dict: dict) -> Path:
        state_file = state_dir / "state.json"
        state_file.write_text(
            json.dumps(
                {
                    "training_jobs": {},
                    "index_jobs": {},
                    "projects": {},
                    "chat_histories": {},
                    "eval_jobs": {"j1": job_dict},
                }
            )
        )
        return state_file

    def test_tampered_log_file_escapes_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = self._write_state(
                Path(td),
                {
                    "job_id": "j1",
                    "project_name": "demo",
                    "eval_type": "static",
                    "status": "running",
                    "log_file": "/etc/passwd",
                },
            )
            with patch.object(photon_app, "STATE_FILE", state_file):
                state = photon_app._load_state()

        job = state.eval_jobs["j1"]
        self.assertEqual(job.status, "failed")
        self.assertIn("tampering", job.error_message)

    def test_tampered_result_json_escapes_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = self._write_state(
                Path(td),
                {
                    "job_id": "j1",
                    "project_name": "demo",
                    "eval_type": "static",
                    "status": "running",
                    "result_json": "/tmp/../etc/passwd",
                },
            )
            with patch.object(photon_app, "STATE_FILE", state_file):
                state = photon_app._load_state()

        job = state.eval_jobs["j1"]
        self.assertEqual(job.status, "failed")
        self.assertIn("tampering", job.error_message)

    def test_unknown_status_normalized_to_failed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = self._write_state(
                Path(td),
                {
                    "job_id": "j1",
                    "project_name": "demo",
                    "eval_type": "static",
                    "status": "hacked",
                },
            )
            with patch.object(photon_app, "STATE_FILE", state_file):
                state = photon_app._load_state()

        self.assertEqual(state.eval_jobs["j1"].status, "failed")

    def test_non_int_pid_reset_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = self._write_state(
                Path(td),
                {
                    "job_id": "j1",
                    "project_name": "demo",
                    "eval_type": "static",
                    "status": "running",
                    "pid": "not-an-int",
                },
            )
            with patch.object(photon_app, "STATE_FILE", state_file):
                state = photon_app._load_state()

        self.assertIsNone(state.eval_jobs["j1"].pid)
        # Status must remain valid (still ``running`` because path
        # validation passed and the pid was merely normalized, not the
        # trigger for a failed state).
        self.assertEqual(state.eval_jobs["j1"].status, "running")

    def test_paths_inside_project_root_accepted(self) -> None:
        """Valid paths under PROJECT_ROOT must survive untouched."""

        with tempfile.TemporaryDirectory() as td:
            valid_log = photon_app.PROJECT_ROOT / "logs" / "eval" / "ok.log"
            valid_marker = photon_app.PROJECT_ROOT / "reports" / "eval_runs" / "ok.done"
            state_file = self._write_state(
                Path(td),
                {
                    "job_id": "j1",
                    "project_name": "demo",
                    "eval_type": "static",
                    "status": "running",
                    "pid": 4321,
                    "log_file": str(valid_log),
                    "marker_file": str(valid_marker),
                },
            )
            with patch.object(photon_app, "STATE_FILE", state_file):
                state = photon_app._load_state()

        job = state.eval_jobs["j1"]
        self.assertEqual(job.status, "running")
        self.assertEqual(job.pid, 4321)
        self.assertEqual(job.log_file, str(valid_log))
        self.assertEqual(job.marker_file, str(valid_marker))
        self.assertEqual(job.error_message, "")


class TestSyncEvalJob(unittest.TestCase):
    """W4-T2: _sync_eval_job transitions status from marker_file / timeout / dead pid."""

    def _make_job(self, **overrides) -> "photon_app.EvalJob":
        defaults = dict(
            job_id="j1",
            project_name="demo",
            eval_type="static",
            status="running",
            started_at="2026-04-20T10:00:00",
            started_at_epoch=1_000_000.0,
            pid=9999,
            log_file="",
            result_json="",
            marker_file="",
        )
        defaults.update(overrides)
        return photon_app.EvalJob(**defaults)

    def test_marker_file_transitions_to_succeeded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "j1.done"
            marker.touch()
            result_json = Path(td) / "j1.json"
            result_json.write_text(
                json.dumps(
                    {
                        "done_q": 120,
                        "total_q": 120,
                        "p50_latency_ms": 19400.5,
                        "nc_rate": 0.183,
                    }
                )
            )
            job = self._make_job(
                marker_file=str(marker),
                result_json=str(result_json),
            )
            changed = photon_app._sync_eval_job(job, now_epoch=1_000_100.0)

        self.assertTrue(changed)
        self.assertEqual(job.status, "succeeded")
        self.assertEqual(job.done_q, 120)
        self.assertEqual(job.total_q, 120)
        self.assertAlmostEqual(job.p50_latency_ms, 19400.5)
        self.assertAlmostEqual(job.nc_rate, 0.183)
        self.assertNotEqual(job.finished_at, "")

    def test_timeout_marks_failed(self) -> None:
        job = self._make_job(
            started_at_epoch=1_000_000.0,
        )
        # now is far past the timeout (3600s).
        with patch.object(photon_app, "_is_process_running", return_value=False):
            changed = photon_app._sync_eval_job(job, now_epoch=1_000_000.0 + 7200.0)

        self.assertTrue(changed)
        self.assertEqual(job.status, "failed")
        self.assertIn("timeout", job.error_message)

    def test_dead_process_no_marker_marks_failed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "j1.log"
            log.write_text("traceback: runtime error\n")
            # No marker file.
            job = self._make_job(
                log_file=str(log),
                started_at_epoch=1_000_000.0,
            )
            with patch.object(photon_app, "_is_process_running", return_value=False):
                changed = photon_app._sync_eval_job(job, now_epoch=1_000_010.0)

        self.assertTrue(changed)
        self.assertEqual(job.status, "failed")
        # error_message should contain tail of log.
        self.assertIn("traceback", job.error_message)

    def test_running_with_progress_updates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "j1.log"
            log.write_text(
                "starting\nPROGRESS done=30 total=120 p50_ms=19500 nc=0.15\n"
            )
            job = self._make_job(
                log_file=str(log),
                started_at_epoch=1_000_000.0,
            )
            with patch.object(photon_app, "_is_process_running", return_value=True):
                changed = photon_app._sync_eval_job(job, now_epoch=1_000_050.0)

        self.assertTrue(changed)
        self.assertEqual(job.status, "running")
        self.assertEqual(job.done_q, 30)
        self.assertEqual(job.total_q, 120)
        self.assertAlmostEqual(job.p50_latency_ms, 19500.0)
        self.assertAlmostEqual(job.nc_rate, 0.15)

    def test_running_no_progress_no_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "j1.log"
            log.write_text("starting\n")
            job = self._make_job(
                log_file=str(log),
                started_at_epoch=1_000_000.0,
            )
            with patch.object(photon_app, "_is_process_running", return_value=True):
                changed = photon_app._sync_eval_job(job, now_epoch=1_000_050.0)

        self.assertFalse(changed)
        self.assertEqual(job.status, "running")

    def test_terminal_job_is_not_reprocessed(self) -> None:
        job = self._make_job(status="succeeded", done_q=120, total_q=120)
        changed = photon_app._sync_eval_job(job, now_epoch=1_000_050.0)
        self.assertFalse(changed)
        self.assertEqual(job.status, "succeeded")

    def test_sync_all_jobs_iterates_eval_jobs(self) -> None:
        """_sync_all_jobs must also reconcile eval_jobs."""
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "j1.done"
            marker.touch()
            result_json = Path(td) / "j1.json"
            result_json.write_text(
                json.dumps(
                    {
                        "done_q": 10,
                        "total_q": 10,
                        "p50_latency_ms": 100.0,
                        "nc_rate": 0.0,
                    }
                )
            )
            state = photon_app.AppState()
            state.eval_jobs["j1"] = self._make_job(
                marker_file=str(marker),
                result_json=str(result_json),
            )
            changed = photon_app._sync_all_jobs(state)

        self.assertTrue(changed)
        self.assertEqual(state.eval_jobs["j1"].status, "succeeded")


if __name__ == "__main__":
    unittest.main()
