"""Tests for scripts/compare_baseline_photon.py.

A-1 Phase 2 Step 1: 1 question で baseline と PHOTON を並べて比較するスクリプトの単体テスト。
build_pipeline と pipeline.query を mock することで、MLX 不在環境でも動作する。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_module():
    """compare_baseline_photon を importlib で読み込む (scripts/ は package ではない)。"""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "compare_baseline_photon.py"
    spec = importlib.util.spec_from_file_location(
        "compare_baseline_photon", script_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_baseline_photon"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_query_result():
    """軽量な QueryResult スタブ。"""
    from baseline_reporag.contracts import QueryResult
    from baseline_reporag.profiler import LatencyBreakdown, MemorySnapshot

    return QueryResult(
        answer="The entry point is main.py:42.",
        session_id="compare-baseline",
        turn_id=1,
        cited_chunk_ids=["c1", "c2"],
        wrong_citation_indices=[],
        no_citation=False,
        latency=LatencyBreakdown(
            retrieval_ms=100.0,
            generation_ms=900.0,
            citation_ms=0.0,
            total_ms=1000.0,
        ),
        memory=MemorySnapshot(peak_mb=50.0, current_mb=25.0),
        citation_postprocessed=False,
        generator_used="qwen",
        generator_fallback_reason=None,
    )


class TestRunVariant:
    def test_returns_variant_result_with_pipeline_data(
        self, monkeypatch, tmp_path: Path, fake_query_result
    ) -> None:
        """build_pipeline + pipeline.query の出力が VariantResult に正しく取り込まれる。"""
        module = _load_module()

        # 軽量 config を生成 (load_config が要求する最小フィールド)
        cfg_path = tmp_path / "fake.yaml"
        cfg_path.write_text(
            "version: 1\n"
            "paths:\n  data_root: ./data\n  log_root: ./logs\n"
            'repo:\n  repo_id: "fake_repo"\n  repo_commit: "deadbeef"\n'
            "model:\n"
            '  provider: "baseline"\n'
            '  model_id: "fake-model"\n',
            encoding="utf-8",
        )

        fake_pipeline = MagicMock()
        fake_pipeline.query.return_value = fake_query_result

        # script は ``from baseline_reporag.pipeline_factory import build_pipeline``
        # で module level にバインドしているため、script モジュール自体の
        # 属性を patch する。
        monkeypatch.setattr(module, "build_pipeline", lambda _cfg: fake_pipeline)

        result = module.run_variant(
            variant_id="baseline",
            config_path=str(cfg_path),
            question="Where is the entry point?",
            repo_id="fake_repo",
            session_id="test-1",
        )

        assert result.variant_id == "baseline"
        assert result.config_path == str(cfg_path)
        assert result.answer == "The entry point is main.py:42."
        assert result.cited_chunk_ids == ["c1", "c2"]
        assert result.no_citation is False
        assert result.latency_total_ms == 1000.0
        assert result.latency_retrieval_ms == 100.0
        assert result.latency_generation_ms == 900.0
        assert result.memory_peak_mb == 50.0

        # pipeline.query が正しい引数で呼ばれた
        fake_pipeline.query.assert_called_once()
        call_kwargs = fake_pipeline.query.call_args.kwargs
        assert call_kwargs["question"] == "Where is the entry point?"
        assert call_kwargs["session_id"] == "test-1"
        assert call_kwargs["repo_id"] == "fake_repo"

    def test_falls_back_to_cfg_repo_id_when_repo_id_empty(
        self, monkeypatch, tmp_path: Path, fake_query_result
    ) -> None:
        """空文字 repo_id の場合 cfg.repo.repo_id にフォールバックする。"""
        module = _load_module()

        cfg_path = tmp_path / "fake.yaml"
        cfg_path.write_text(
            "version: 1\n"
            "paths:\n  data_root: ./data\n  log_root: ./logs\n"
            'repo:\n  repo_id: "default_repo"\n  repo_commit: "deadbeef"\n'
            "model:\n"
            '  provider: "baseline"\n'
            '  model_id: "fake-model"\n',
            encoding="utf-8",
        )

        fake_pipeline = MagicMock()
        fake_pipeline.query.return_value = fake_query_result

        # script は ``from baseline_reporag.pipeline_factory import build_pipeline``
        # で module level にバインドしているため、script モジュール自体の
        # 属性を patch する。
        monkeypatch.setattr(module, "build_pipeline", lambda _cfg: fake_pipeline)

        module.run_variant(
            variant_id="baseline",
            config_path=str(cfg_path),
            question="Q?",
            repo_id="",  # empty → fallback
            session_id="test-1",
        )

        call_kwargs = fake_pipeline.query.call_args.kwargs
        assert call_kwargs["repo_id"] == "default_repo"


class TestPrintTextReport:
    def test_emits_question_variants_and_summary(
        self, capsys, fake_query_result
    ) -> None:
        module = _load_module()

        baseline_result = module.VariantResult(
            variant_id="baseline",
            config_path="configs/baseline.yaml",
            answer="ans-baseline",
            cited_chunk_ids=["c1"],
            no_citation=False,
            latency_total_ms=2000.0,
            latency_retrieval_ms=200.0,
            latency_generation_ms=1800.0,
            memory_peak_mb=60.0,
        )
        photon_result = module.VariantResult(
            variant_id="photon",
            config_path="configs/photon_small.yaml",
            answer="ans-photon",
            cited_chunk_ids=["c2"],
            no_citation=False,
            latency_total_ms=1000.0,
            latency_retrieval_ms=100.0,
            latency_generation_ms=900.0,
            memory_peak_mb=80.0,
        )

        module.print_text_report("Q?", [baseline_result, photon_result])
        captured = capsys.readouterr().out

        assert "Question: Q?" in captured
        assert "[baseline]" in captured
        assert "[photon]" in captured
        assert "ans-baseline" in captured
        assert "ans-photon" in captured
        # delta は photon - baseline なので -1000 ms (-50.0%)
        assert "-1000 ms" in captured
        assert "-50.0%" in captured

    def test_warns_on_no_citation(self, capsys) -> None:
        module = _load_module()

        result = module.VariantResult(
            variant_id="baseline",
            config_path="configs/baseline.yaml",
            answer="no-cite",
            cited_chunk_ids=[],
            no_citation=True,
            latency_total_ms=1000.0,
            latency_retrieval_ms=100.0,
            latency_generation_ms=900.0,
            memory_peak_mb=10.0,
        )
        module.print_text_report("Q?", [result])
        captured = capsys.readouterr().out
        assert "[WARNING] No citations" in captured


class TestToJsonPayload:
    def test_includes_question_and_variant_dicts(self) -> None:
        module = _load_module()

        result = module.VariantResult(
            variant_id="baseline",
            config_path="configs/baseline.yaml",
            answer="hello",
            cited_chunk_ids=["c1"],
            no_citation=False,
            latency_total_ms=1.0,
            latency_retrieval_ms=0.5,
            latency_generation_ms=0.5,
            memory_peak_mb=10.0,
        )
        payload = module.to_json_payload("Q?", [result])

        assert payload["question"] == "Q?"
        assert len(payload["variants"]) == 1
        assert payload["variants"][0]["variant_id"] == "baseline"
        assert payload["variants"][0]["answer"] == "hello"

        # JSON serialisable
        encoded = json.dumps(payload, ensure_ascii=False)
        decoded = json.loads(encoded)
        assert decoded["variants"][0]["cited_chunk_ids"] == ["c1"]


class TestIndent:
    def test_indents_each_line(self) -> None:
        module = _load_module()
        out = module._indent("a\nb\nc", prefix=">>")
        assert out == ">>a\n>>b\n>>c"
