from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from bench.run_all import run_variant, save_run_predictions


# ------------------------------------------------------------------
# Mock QueryResult for pipeline.query
# ------------------------------------------------------------------


@dataclass
class _MockLatency:
    retrieval_ms: float = 10.0
    graph_ms: float = 5.0
    evidence_ms: float = 5.0
    generation_ms: float = 30.0
    total_ms: float = 50.0


@dataclass
class _MockMemory:
    peak_mb: float = 100.0
    resident_mb: float = 50.0


@dataclass
class _MockQueryResult:
    answer: str = "test answer"
    session_id: str = "s1"
    turn_id: int = 1
    cited_chunk_ids: list = None
    wrong_citation_indices: list = None
    no_citation: bool = False
    latency: _MockLatency = None
    memory: _MockMemory = None
    drift_metrics: dict = None
    confidence: float = None
    fallback_decision: dict = None

    def __post_init__(self):
        if self.cited_chunk_ids is None:
            self.cited_chunk_ids = ["c1"]
        if self.wrong_citation_indices is None:
            self.wrong_citation_indices = []
        if self.latency is None:
            self.latency = _MockLatency()
        if self.memory is None:
            self.memory = _MockMemory()


# ------------------------------------------------------------------
# run_variant tests
# ------------------------------------------------------------------


class TestRunVariantStatic:
    def test_runs_static_eval(self, tmp_path):
        eval_file = tmp_path / "static.jsonl"
        eval_file.write_text(
            json.dumps(
                {
                    "id": "SE-001",
                    "category": "onboarding",
                    "question": "What is X?",
                    "reference_answer": "",
                    "reference_chunk_ids": [],
                    "grading_notes": "",
                }
            )
            + "\n"
        )
        variant_cfg = {
            "id": "baseline_rag",
            "kind": "baseline",
            "config_path": "./configs/baseline.yaml",
        }
        eval_cfg = {
            "datasets": {
                "static_eval": {
                    "enabled": True,
                    "path": str(eval_file),
                    "max_cases": 10,
                },
                "multi_turn_eval": {"enabled": False},
                "stress_eval": {"enabled": False},
            },
        }
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _MockQueryResult()

        with patch("bench.run_all._build_variant_pipeline", return_value=mock_pipeline):
            preds = run_variant(variant_cfg, eval_cfg)

        assert len(preds) == 1
        assert preds[0]["eval_id"] == "SE-001"
        assert preds[0]["answer"] == "test answer"
        assert "latency_ms" in preds[0]

    def test_respects_max_cases(self, tmp_path):
        lines = []
        for i in range(5):
            lines.append(
                json.dumps(
                    {
                        "id": f"SE-{i:03d}",
                        "category": "test",
                        "question": f"Q{i}",
                        "reference_answer": "",
                        "reference_chunk_ids": [],
                        "grading_notes": "",
                    }
                )
            )
        eval_file = tmp_path / "static.jsonl"
        eval_file.write_text("\n".join(lines) + "\n")

        variant_cfg = {"id": "test", "kind": "baseline", "config_path": "x.yaml"}
        eval_cfg = {
            "datasets": {
                "static_eval": {
                    "enabled": True,
                    "path": str(eval_file),
                    "max_cases": 2,
                },
                "multi_turn_eval": {"enabled": False},
                "stress_eval": {"enabled": False},
            },
        }
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _MockQueryResult()

        with patch("bench.run_all._build_variant_pipeline", return_value=mock_pipeline):
            preds = run_variant(variant_cfg, eval_cfg)

        assert len(preds) == 2


class TestRunVariantMultiTurn:
    def test_runs_multi_turn_eval(self, tmp_path):
        eval_file = tmp_path / "multi.jsonl"
        eval_file.write_text(
            json.dumps(
                {
                    "session_id": "MT-001",
                    "category": "test",
                    "scenario": "test scenario",
                    "turns": [
                        {"turn_id": 1, "question": "Q1"},
                        {"turn_id": 2, "question": "Q2"},
                    ],
                }
            )
            + "\n"
        )
        variant_cfg = {"id": "test", "kind": "baseline", "config_path": "x.yaml"}
        eval_cfg = {
            "datasets": {
                "static_eval": {"enabled": False},
                "multi_turn_eval": {
                    "enabled": True,
                    "path": str(eval_file),
                    "max_sessions": 10,
                    "max_turns_per_session": 6,
                },
                "stress_eval": {"enabled": False},
            },
        }
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _MockQueryResult()

        with patch("bench.run_all._build_variant_pipeline", return_value=mock_pipeline):
            preds = run_variant(variant_cfg, eval_cfg)

        assert len(preds) == 2
        assert preds[0]["session_id"] == "MT-001"
        assert preds[0]["turn_id"] == 1
        assert preds[1]["turn_id"] == 2


class TestRunVariantDisabledSets:
    def test_returns_empty_when_all_disabled(self):
        variant_cfg = {"id": "test", "kind": "baseline", "config_path": "x.yaml"}
        eval_cfg = {
            "datasets": {
                "static_eval": {"enabled": False},
                "multi_turn_eval": {"enabled": False},
                "stress_eval": {"enabled": False},
            },
        }
        mock_pipeline = MagicMock()
        with patch("bench.run_all._build_variant_pipeline", return_value=mock_pipeline):
            preds = run_variant(variant_cfg, eval_cfg)
        assert preds == []


class TestSaveRunPredictions:
    def test_creates_jsonl_file(self, tmp_path):
        preds = [{"eval_id": "SE-001", "answer": "test"}]
        path = save_run_predictions("run1", "baseline_rag", preds, tmp_path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["eval_id"] == "SE-001"
