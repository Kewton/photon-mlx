"""Tests for scripts/eval_retrieval_recall.py (AC 5 secondary proxy recall)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _ROOT / "scripts" / "eval_retrieval_recall.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "eval_retrieval_recall_under_test", _SCRIPT_PATH
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestProxyGoldExtraction:
    def test_extracts_citation_patterns(self):
        mod = _load_script_module()
        record = {"expected_citation_patterns": ["第3条", "第5項"], "question": "test"}
        result = mod._extract_proxy_gold(record)
        assert "第3条" in result
        assert "第5項" in result

    def test_empty_patterns_returns_empty(self):
        mod = _load_script_module()
        record = {"expected_citation_patterns": [], "question": "test"}
        assert mod._extract_proxy_gold(record) == []

    def test_missing_patterns_key(self):
        mod = _load_script_module()
        record = {"question": "test"}
        assert mod._extract_proxy_gold(record) == []


class TestRecallComputation:
    def test_full_recall(self):
        mod = _load_script_module()
        evidence = ["第3条の内容", "第5項の規定"]
        gold = ["第3条", "第5項"]
        assert mod._compute_recall(evidence, gold) == 1.0

    def test_zero_recall(self):
        mod = _load_script_module()
        evidence = ["全く関係ない内容"]
        gold = ["第3条"]
        assert mod._compute_recall(evidence, gold) == 0.0

    def test_empty_gold_returns_zero(self):
        mod = _load_script_module()
        assert mod._compute_recall(["some text"], []) == 0.0


class TestRunEval:
    def test_run_eval_writes_output(self, tmp_path):
        """run_eval writes JSON output to the specified path."""
        mod = _load_script_module()
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "repo:\n  repo_id: demo\n  repo_commit: head\n"
            "paths:\n  data_root: data\n  log_root: logs\n"
            "indexing:\n  heading_graph:\n    enabled: false\n"
        )
        output_path = tmp_path / "out.json"
        mod.run_eval(str(cfg_path), ["on", "off"], str(output_path))
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "on" in data["variants"]
        assert "off" in data["variants"]
