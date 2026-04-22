"""Tests for app/photon_app.py state sync helpers (Issue #59).

The Streamlit app previously only reconciled job status during UI rendering,
so closing the browser or stopping Streamlit left `.cache/photon_app_state.json`
stuck at `status=running`. These tests cover the pure sync helpers that let a
background thread (and a one-shot startup pass) keep the file authoritative.
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


class TestSyncTrainingJob(unittest.TestCase):
    def _make_job(self, **overrides) -> photon_app.TrainingJob:
        defaults = dict(
            job_id="t1",
            repo_dir="/tmp/r",
            config_path="/tmp/c.yaml",
            pid=1,
            started_at="2026-04-20T10:00:00",
            status="running",
            log_file="",
            last_step=0,
            max_steps=1000,
            val_loss=0.0,
        )
        defaults.update(overrides)
        return photon_app.TrainingJob(**defaults)

    def test_running_process_keeps_status(self) -> None:
        job = self._make_job()
        with patch.object(photon_app, "_is_process_running", return_value=True):
            photon_app._sync_training_job(job, None)
        self.assertEqual(job.status, "running")

    def test_progress_advances_last_step_and_val_loss(self) -> None:
        job = self._make_job(last_step=10, val_loss=2.0)
        with patch.object(photon_app, "_is_process_running", return_value=True):
            changed = photon_app._sync_training_job(
                job, {"last_step": 500, "val_loss": 1.2345, "max_steps": 1000}
            )
        self.assertTrue(changed)
        self.assertEqual(job.last_step, 500)
        self.assertAlmostEqual(job.val_loss, 1.2345)

    def test_progress_does_not_regress(self) -> None:
        job = self._make_job(last_step=800, val_loss=1.1)
        with patch.object(photon_app, "_is_process_running", return_value=True):
            photon_app._sync_training_job(
                job, {"last_step": 100, "val_loss": 1.1, "max_steps": 1000}
            )
        self.assertEqual(job.last_step, 800)

    def test_dead_process_with_training_complete_marks_completed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "train.log"
            log.write_text("step 999\nTraining complete. Final loss: 0.0219\n")
            job = self._make_job(log_file=str(log))
            with patch.object(photon_app, "_is_process_running", return_value=False):
                changed = photon_app._sync_training_job(job, None)
        self.assertTrue(changed)
        self.assertEqual(job.status, "completed")

    def test_dead_process_with_steps_but_no_marker_marks_completed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "train.log"
            log.write_text("step 999 done\n")
            job = self._make_job(log_file=str(log), last_step=999)
            with patch.object(photon_app, "_is_process_running", return_value=False):
                photon_app._sync_training_job(job, None)
        self.assertEqual(job.status, "completed")

    def test_dead_process_with_no_progress_marks_failed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "train.log"
            log.write_text("Traceback ...\n")
            job = self._make_job(log_file=str(log), last_step=0)
            with patch.object(photon_app, "_is_process_running", return_value=False):
                photon_app._sync_training_job(job, None)
        self.assertEqual(job.status, "failed")


class TestSyncIndexJob(unittest.TestCase):
    def _make_job(self, **overrides) -> photon_app.IndexJob:
        defaults = dict(
            job_id="i1",
            repo_dir="/tmp/r",
            repo_id="r",
            config_path="configs/baseline.yaml",
            pid=1,
            started_at="2026-04-20T10:00:00",
            status="running",
            log_file="",
            phase="ingest",
            embedding_model="",
        )
        defaults.update(overrides)
        return photon_app.IndexJob(**defaults)

    def test_running_advances_phase_from_log(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "idx.log"
            log.write_text("Phase 1: Ingest\nPhase 2: BM25 + Embedding\n")
            job = self._make_job(log_file=str(log))
            with patch.object(photon_app, "_is_process_running", return_value=True):
                photon_app._sync_index_job(job)
        self.assertEqual(job.phase, "bm25_embed")
        self.assertEqual(job.status, "running")

    def test_dead_process_with_done_marks_completed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "idx.log"
            log.write_text("Phase 3: Symbol Graph\nDONE\n")
            job = self._make_job(log_file=str(log), phase="symbol_graph")
            with patch.object(photon_app, "_is_process_running", return_value=False):
                photon_app._sync_index_job(job)
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.phase, "completed")

    def test_dead_process_without_done_marks_failed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "idx.log"
            log.write_text("Phase 1: Ingest\nError: not found\n")
            job = self._make_job(log_file=str(log))
            with patch.object(photon_app, "_is_process_running", return_value=False):
                photon_app._sync_index_job(job)
        self.assertEqual(job.status, "failed")


class TestSyncStateFile(unittest.TestCase):
    """_sync_state_file reads the JSON, reconciles, and writes back if changed."""

    def test_running_to_completed_persisted_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
            log = Path(td) / "train.log"
            log.write_text("Training complete. Final loss: 0.0219\n")
            state_file.write_text(
                json.dumps(
                    {
                        "training_jobs": {
                            "t1": {
                                "job_id": "t1",
                                "repo_dir": "/tmp/r",
                                "config_path": "/tmp/c.yaml",
                                "pid": 1,
                                "started_at": "2026-04-20T10:00:00",
                                "status": "running",
                                "log_file": str(log),
                                "last_step": 0,
                                "max_steps": 1000,
                                "val_loss": 0.0,
                            }
                        },
                        "index_jobs": {},
                        "projects": {},
                        "chat_histories": {},
                    }
                )
            )

            with (
                patch.object(photon_app, "STATE_FILE", state_file),
                patch.object(photon_app, "_is_process_running", return_value=False),
            ):
                changed = photon_app._sync_state_file()

            data = json.loads(state_file.read_text())

        self.assertTrue(changed)
        self.assertEqual(data["training_jobs"]["t1"]["status"], "completed")

    def test_no_running_jobs_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
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
                changed = photon_app._sync_state_file()
        self.assertFalse(changed)


if __name__ == "__main__":
    unittest.main()
