from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from bench.run_all import (
    _count_jsonl_records,
    _expected_predictions_for_eval,
    filter_variants,
    parse_variants_csv,
    run_variant,
    save_run_predictions,
)


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
    citation_postprocessed: bool = False
    retrieval_debug: list = None

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


class TestProgressHelpers:
    def test_count_jsonl_records(self, tmp_path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
        assert _count_jsonl_records(path) == 2
        assert _count_jsonl_records(tmp_path / "missing.jsonl") == 0

    def test_expected_predictions_for_eval(self, tmp_path) -> None:
        static_path = tmp_path / "static.jsonl"
        static_path.write_text(
            "\n".join(
                json.dumps({"id": f"SE-{i}", "question": f"Q{i}"}) for i in range(3)
            )
            + "\n",
            encoding="utf-8",
        )
        mt_path = tmp_path / "mt.jsonl"
        mt_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "session_id": "MT-1",
                            "turns": [
                                {"turn_id": 1, "question": "Q1"},
                                {"turn_id": 2, "question": "Q2"},
                            ],
                        }
                    ),
                    json.dumps(
                        {
                            "session_id": "MT-2",
                            "turns": [
                                {"turn_id": 1, "question": "Q1"},
                                {"turn_id": 2, "question": "Q2"},
                                {"turn_id": 3, "question": "Q3"},
                            ],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        eval_cfg = {
            "datasets": {
                "static_eval": {
                    "enabled": True,
                    "path": str(static_path),
                    "max_cases": 2,
                },
                "multi_turn_eval": {
                    "enabled": True,
                    "path": str(mt_path),
                    "max_sessions": 2,
                    "max_turns_per_session": 2,
                },
            }
        }

        assert _expected_predictions_for_eval(eval_cfg) == 6


# ------------------------------------------------------------------
# Issue #92 T-0b: --variants CSV filter tests
# ------------------------------------------------------------------


class TestParseVariantsCsv:
    """``parse_variants_csv`` splits on ``,``, strips whitespace, fail-closed on empties.

    Codex CB-003 (Issue #92): empty / whitespace / empty-middle tokens
    are REJECTED (fail-closed), not silently dropped. ``None`` (flag not
    passed) is still distinct from an explicitly empty CSV.
    """

    def test_simple_csv_splits(self) -> None:
        assert parse_variants_csv("a,b,c") == ["a", "b", "c"]

    def test_strips_whitespace(self) -> None:
        assert parse_variants_csv(" a , b ,c ") == ["a", "b", "c"]

    def test_single_token_ok(self) -> None:
        assert parse_variants_csv("a") == ["a"]
        assert parse_variants_csv("a,b") == ["a", "b"]

    def test_none_returns_empty_list(self) -> None:
        # ``None`` means "flag not passed" — still the "no filter" signal.
        assert parse_variants_csv(None) == []

    # ---- Codex CB-003 fail-closed regression tests ----

    def test_empty_string_rejected(self) -> None:
        """``--variants ''`` must fail closed (was silently empty before)."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_variants_csv("")

    def test_whitespace_only_rejected(self) -> None:
        """``--variants ' '`` must fail closed."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_variants_csv(" ")
        with pytest.raises(argparse.ArgumentTypeError):
            parse_variants_csv("   \t  ")

    def test_bare_comma_rejected(self) -> None:
        """``--variants ','`` must fail closed (previously collapsed to ``[]``)."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_variants_csv(",")

    def test_empty_middle_token_rejected(self) -> None:
        """``--variants 'a,,b'`` must fail closed."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_variants_csv("a,,b")

    def test_trailing_comma_rejected(self) -> None:
        """``--variants 'a,'`` must fail closed."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_variants_csv("a,")

    def test_leading_comma_rejected(self) -> None:
        """``--variants ',a'`` must fail closed."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_variants_csv(",a")

    def test_error_messages_do_not_leak_raw_value(self) -> None:
        """DR4-001 no-leak: raw attacker-controlled value must not appear."""
        SENSITIVE_WS = "<ATTACK-WHITESPACE-PAYLOAD>"
        SENSITIVE_MIDDLE = "<ATTACK-MIDDLE-PAYLOAD>"
        # Whitespace-only: attacker could pass e.g. "   <sentinel>   " but
        # our reject message must not include it. Only a non-empty but
        # all-whitespace value hits this path; use a pure-whitespace string
        # to exercise the empty branch — and verify the attacker name used
        # in test identification never leaks.
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            parse_variants_csv("   ")
        assert SENSITIVE_WS not in str(excinfo.value)
        # Empty-middle branch: payload supplied as a token.
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            parse_variants_csv(f"a,,{SENSITIVE_MIDDLE}")
        assert SENSITIVE_MIDDLE not in str(excinfo.value)

    def test_pure_python_split_no_shell(self) -> None:
        """Security: parsing must not spawn subprocess / shell.

        We patch ``subprocess`` and ``os.system`` and confirm neither is
        consulted (DR4-002 no-shell guarantee).
        """
        with patch("subprocess.run") as mock_run, patch("os.system") as mock_sys:
            parse_variants_csv("a,b,c")
            assert not mock_run.called
            assert not mock_sys.called


class TestFilterVariants:
    """``filter_variants`` selects variants by id; fail-closed on unknowns."""

    def _mk_variants(self) -> list[dict]:
        return [
            {"id": "photon_rag_aggr_weighted"},
            {"id": "photon_rag_aggr_attention"},
            {"id": "photon_rag_aggr_last"},
            {"id": "baseline_rag"},
        ]

    def test_valid_csv_filters_variants_in_order(self) -> None:
        """Valid ids select a subset; result preserves the CSV order."""
        variants = self._mk_variants()
        selected = filter_variants(
            variants,
            ["photon_rag_aggr_attention", "photon_rag_aggr_weighted"],
        )
        assert [v["id"] for v in selected] == [
            "photon_rag_aggr_attention",
            "photon_rag_aggr_weighted",
        ]

    def test_none_selection_returns_all(self) -> None:
        variants = self._mk_variants()
        assert filter_variants(variants, None) == variants

    def test_empty_selection_returns_all(self) -> None:
        variants = self._mk_variants()
        assert filter_variants(variants, []) == variants

    def test_unknown_id_fails_closed(self) -> None:
        """Unknown id raises ``argparse.ArgumentError`` (fail-closed)."""
        variants = self._mk_variants()
        with pytest.raises(argparse.ArgumentError):
            filter_variants(variants, ["not_a_real_variant_id"])

    def test_unknown_id_error_message_does_not_leak_raw_value(self) -> None:
        """DR4-001 no-leak: raw invalid value must not appear in error text."""
        variants = self._mk_variants()
        SENSITIVE = "<ATTACK-PAYLOAD-VARIANT-SENTINEL>"
        with pytest.raises(argparse.ArgumentError) as excinfo:
            filter_variants(variants, [SENSITIVE])
        assert SENSITIVE not in str(excinfo.value)

    def test_partial_unknown_fails_closed(self) -> None:
        """A single unknown id in a mostly-valid CSV still fails closed."""
        variants = self._mk_variants()
        with pytest.raises(argparse.ArgumentError):
            filter_variants(
                variants,
                ["photon_rag_aggr_weighted", "unknown_x"],
            )

    def test_no_shell_calls_on_filter(self) -> None:
        """Filtering must not call subprocess / os.system (DR4-002)."""
        variants = self._mk_variants()
        with patch("subprocess.run") as mock_run, patch("os.system") as mock_sys:
            filter_variants(variants, ["baseline_rag"])
            assert not mock_run.called
            assert not mock_sys.called
