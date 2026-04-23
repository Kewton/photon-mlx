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
                log_file="/tmp/eval/j1.log",
                result_json="/tmp/eval/j1.json",
                marker_file="/tmp/eval/j1.done",
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
        self.assertEqual(reloaded.eval_jobs["j1"].marker_file, "/tmp/eval/j1.done")
        self.assertEqual(reloaded.eval_jobs["j2"].status, "failed")
        self.assertEqual(reloaded.eval_jobs["j2"].error_message, "timeout")


if __name__ == "__main__":
    unittest.main()
