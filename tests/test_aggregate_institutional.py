"""Tests for scripts/aggregate_institutional_baseline.py (Issue #127)."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import aggregate_institutional_baseline as agg  # noqa: E402


def _make_record(**overrides: object) -> dict:
    base = {
        "eval_id": "INST-OVERVIEW-001",
        "category": "overview",
        "question": "概要は？",
        "answer": "...",
        "cited_chunk_ids": ["c1", "c2"],
        "no_citation": False,
        "latency_ms": 1000.0,
        "retrieval_ms": 100.0,
        "generation_ms": 900.0,
        "memory_peak_mb": 20.0,
    }
    base.update(overrides)
    return base


# DR2-009: 6 category × 1Q 基本 + penalty に NC=false 追加 1Q = 計 7 records
DUMMY_PREDICTIONS = [
    _make_record(
        eval_id="INST-OVERVIEW-001",
        category="overview",
        question="この政令の目的は？",
        cited_chunk_ids=["c1", "c2", "c3"],
        no_citation=False,
        latency_ms=1000.0,
        retrieval_ms=100.0,
        generation_ms=900.0,
        memory_peak_mb=20.0,
    ),
    _make_record(
        eval_id="INST-EXCEPTION-001",
        category="exception",
        question="法第28条で定める例外は？",
        answer="根拠が不足しています。",
        cited_chunk_ids=[],
        no_citation=True,
        latency_ms=1500.0,
        retrieval_ms=150.0,
        generation_ms=1350.0,
        memory_peak_mb=25.0,
    ),
    _make_record(
        eval_id="INST-ARTICLE-LOOKUP-001",
        category="article_lookup",
        question="第4条に規定されている内容は？",
        answer="根拠が不足しています。",
        cited_chunk_ids=[],
        no_citation=True,
        latency_ms=2000.0,
        retrieval_ms=200.0,
        generation_ms=1800.0,
        memory_peak_mb=30.0,
    ),
    _make_record(
        eval_id="INST-DEFINITION-001",
        category="definition",
        question="保育所における自己評価の定義は？",
        answer="根拠不足。",
        cited_chunk_ids=[],
        no_citation=True,
        latency_ms=2500.0,
        retrieval_ms=250.0,
        generation_ms=2250.0,
        memory_peak_mb=35.0,
    ),
    _make_record(
        eval_id="INST-SCOPE-001",
        category="scope",
        question="この法律の適用を受けるのはどのような事業者？",
        cited_chunk_ids=["s1", "s2", "s3", "s4"],
        no_citation=False,
        latency_ms=3000.0,
        retrieval_ms=300.0,
        generation_ms=2700.0,
        memory_peak_mb=40.0,
    ),
    _make_record(
        eval_id="INST-PENALTY-001",
        category="penalty",
        question="罰則は？",
        answer="根拠不足。",
        cited_chunk_ids=[],
        no_citation=True,
        latency_ms=3500.0,
        retrieval_ms=350.0,
        generation_ms=3150.0,
        memory_peak_mb=45.0,
    ),
    _make_record(
        eval_id="INST-PENALTY-002",
        category="penalty",
        question="罰則の例外は？",
        cited_chunk_ids=["p1", "p2"],
        no_citation=False,
        latency_ms=4000.0,
        retrieval_ms=400.0,
        generation_ms=3600.0,
        memory_peak_mb=50.0,
    ),
]


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "sample.predictions.jsonl"
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in DUMMY_PREDICTIONS) + "\n",
        encoding="utf-8",
    )
    return p


# ----------------------------------------------------------------------
# T1: REQUIRED_FIELDS の 10 フィールド欠損 → KeyError (DR2-007 parametrize)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", agg.REQUIRED_FIELDS)
def test_load_predictions_required_fields(tmp_path: Path, missing_field: str) -> None:
    record = _make_record()
    record.pop(missing_field)
    p = tmp_path / "broken.predictions.jsonl"
    p.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(KeyError, match=missing_field):
        agg.load_predictions([p])


# ----------------------------------------------------------------------
# T2: compute_overall — NC rate (no_citation OR cited_chunk_ids 空)
# ----------------------------------------------------------------------


def test_compute_overall_nc_counts_both_flags() -> None:
    records = [
        _make_record(no_citation=True, cited_chunk_ids=["c1"]),  # NC by flag
        _make_record(no_citation=False, cited_chunk_ids=[]),  # NC by empty cites
        _make_record(no_citation=False, cited_chunk_ids=["c1"]),  # not NC
    ]
    stat = agg.compute_overall(records)
    assert stat["total"] == 3
    assert stat["nc"] == 2
    assert stat["nc_rate"] == pytest.approx(2 / 3 * 100)


def test_compute_overall_counts_refusal_with_citation_as_nc() -> None:
    """Issue #154 Bug 2: refusal answers with formal [C:N] must count as NC."""
    records = [
        _make_record(
            answer="根拠が不足しています。提供されたドキュメントには情報がありません。 [C:1]",
            no_citation=False,
            cited_chunk_ids=["c1"],
        ),
        _make_record(
            answer="The function is in cli.py [C:1]",
            no_citation=False,
            cited_chunk_ids=["c1"],
        ),
    ]
    stat = agg.compute_overall(records)
    assert stat["total"] == 2
    assert stat["nc"] == 1, "refusal with formal [C:1] should be counted as NC"


# ----------------------------------------------------------------------
# T3: compute_category — alphabetical sort
# ----------------------------------------------------------------------


def test_compute_category_alphabetical_sort() -> None:
    stats = agg.compute_category(DUMMY_PREDICTIONS)
    assert list(stats.keys()) == [
        "article_lookup",
        "definition",
        "exception",
        "overview",
        "penalty",
        "scope",
    ]
    assert stats["penalty"]["total"] == 2
    assert stats["penalty"]["nc"] == 1
    assert stats["overview"]["nc"] == 0


# ----------------------------------------------------------------------
# T4: compute_latency — percentile (sorted + k=round(p/100*(n-1)))
# ----------------------------------------------------------------------


def test_compute_latency_percentile_formula() -> None:
    records = [
        _make_record(
            latency_ms=float(v),
            retrieval_ms=10.0,
            generation_ms=float(v) - 10,
            memory_peak_mb=1.0,
        )
        for v in [100.0, 200.0, 300.0, 400.0, 500.0]
    ]
    lat = agg.compute_latency(records)
    # n=5, k for p=50 = round(0.5 * 4) = 2 -> sorted[2] = 300
    assert lat["total"]["p50"] == 300.0
    # k for p=95 = round(0.95 * 4) = 4 -> sorted[4] = 500
    assert lat["total"]["p95"] == 500.0
    assert lat["total"]["max"] == 500.0
    assert lat["total"]["mean"] == pytest.approx(300.0)


# ----------------------------------------------------------------------
# T5: pick_failure_examples — NC=true 優先、無ければ successful sample
# ----------------------------------------------------------------------


def test_pick_failure_examples_prefers_nc() -> None:
    picks = agg.pick_failure_examples(DUMMY_PREDICTIONS)
    assert picks["overview"][1] is True
    assert picks["scope"][1] is True
    assert picks["exception"][1] is False
    assert picks["penalty"][1] is False
    assert picks["penalty"][0]["eval_id"] == "INST-PENALTY-001"


# ----------------------------------------------------------------------
# T6: pick_failure_examples — tuple 形式 (record, is_successful_sample)
# ----------------------------------------------------------------------


def test_pick_failure_examples_returns_tuple() -> None:
    picks = agg.pick_failure_examples(DUMMY_PREDICTIONS)
    for cat, value in picks.items():
        assert isinstance(value, tuple)
        assert len(value) == 2
        rec, is_success = value
        assert isinstance(rec, dict)
        assert isinstance(is_success, bool)


# ----------------------------------------------------------------------
# T7-T10: Renderer snapshot (textwrap.dedent + strip)
# ----------------------------------------------------------------------


def _norm(s: str) -> str:
    return textwrap.dedent(s).strip()


def test_render_overall_block() -> None:
    stat = agg.compute_overall(DUMMY_PREDICTIONS)
    actual = agg.render_overall_block(stat).strip()
    expected = _norm(
        """
        | 指標 | 値 |
        |------|-----|
        | 全質問数 | 7 |
        | NC (no-citation) 件数 | 4 |
        | **NC rate** | **57.14 %** |
        """
    )
    assert actual == expected


def test_render_category_block_includes_total_row() -> None:
    stats = agg.compute_category(DUMMY_PREDICTIONS)
    actual = agg.render_category_block(stats).strip()
    assert "| article_lookup | 1 | 1 | 100.0 % |" in actual
    assert "| definition | 1 | 1 | 100.0 % |" in actual
    assert "| exception | 1 | 1 | 100.0 % |" in actual
    assert "| overview | 1 | 0 | 0.0 % |" in actual
    assert "| penalty | 2 | 1 | 50.0 % |" in actual
    assert "| scope | 1 | 0 | 0.0 % |" in actual
    # 合計行（DR2-004）
    assert "| **合計** | **7** | **4** | **57.14 %** |" in actual


def test_render_latency_block_with_units() -> None:
    stat = agg.compute_latency(DUMMY_PREDICTIONS)
    actual = agg.render_latency_block(stat)
    # 単位サフィックスは renderer 側で付与（DR2-003）
    assert " ms |" in actual
    assert " MB |" in actual
    assert "Memory peak (p50)" in actual
    assert "Memory peak (max)" in actual
    assert "全体 p50" in actual
    assert "Retrieval p50" in actual
    assert "Generation p95" in actual


def test_render_failures_block_with_counts_and_cites() -> None:
    picks = agg.pick_failure_examples(DUMMY_PREDICTIONS)
    cat_stats = agg.compute_category(DUMMY_PREDICTIONS)
    actual = agg.render_failures_block(picks, cat_stats)
    # NC X/Y 見出し（DR2-005）
    assert "（NC 1/1）" in actual  # exception
    assert "（NC 0/1）" in actual  # overview / scope
    assert "（NC 1/2）" in actual  # penalty
    # successful sample 注記
    assert "(successful sample)" in actual
    # cites 行（successful sample のみ）
    assert "cites" in actual
    assert "chunks" in actual


# ----------------------------------------------------------------------
# T11: CLI stdout output
# ----------------------------------------------------------------------


def test_cli_stdout_output(
    sample_jsonl: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agg.main(["--predictions", str(sample_jsonl), "--output", "-"])
    captured = capsys.readouterr()
    assert "全質問数" in captured.out
    assert "Category" in captured.out or "article_lookup" in captured.out
    assert "Memory peak" in captured.out
    assert "（NC " in captured.out


# ----------------------------------------------------------------------
# T12: CLI glob expansion (absolute / relative / shell-expanded)
# ----------------------------------------------------------------------


def test_cli_glob_expansion_absolute(tmp_path: Path) -> None:
    p = tmp_path / "abs.predictions.jsonl"
    p.write_text(json.dumps(DUMMY_PREDICTIONS[0]) + "\n", encoding="utf-8")
    paths = agg.expand_prediction_paths([str(tmp_path / "*.predictions.jsonl")])
    assert paths == [p]


def test_cli_glob_expansion_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "rel.predictions.jsonl"
    p.write_text(json.dumps(DUMMY_PREDICTIONS[0]) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    paths = agg.expand_prediction_paths(["*.predictions.jsonl"])
    assert len(paths) == 1
    assert paths[0].name == "rel.predictions.jsonl"


def test_cli_glob_expansion_shell_expanded_single_path(sample_jsonl: Path) -> None:
    paths = agg.expand_prediction_paths([str(sample_jsonl)])
    assert paths == [sample_jsonl]


def test_cli_glob_expansion_no_match_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No files matched"):
        agg.expand_prediction_paths([str(tmp_path / "nonexistent_*.jsonl")])


# ----------------------------------------------------------------------
# T13: in-place idempotent (正常 sentinel 4 ペア揃い)
# ----------------------------------------------------------------------


def _report_with_all_sentinels(tmp_path: Path) -> Path:
    report = tmp_path / "report.md"
    report.write_text(
        textwrap.dedent(
            """\
            # Report

            ## 3. Overall
            <!-- aggregate:overall:start -->
            (placeholder)
            <!-- aggregate:overall:end -->

            ### 判定結果（設計 §9）

            - [x] `< 50% NC` → 正常完了

            ## 4. Category
            <!-- aggregate:category:start -->
            (placeholder)
            <!-- aggregate:category:end -->

            **所見**: 手書き保持される

            ## 5. Latency
            <!-- aggregate:latency:start -->
            (placeholder)
            <!-- aggregate:latency:end -->

            ## 6. Failures
            <!-- aggregate:failures:start -->
            (placeholder)
            <!-- aggregate:failures:end -->
            """
        ),
        encoding="utf-8",
    )
    return report


def test_in_place_replacement_idempotent(tmp_path: Path, sample_jsonl: Path) -> None:
    report = _report_with_all_sentinels(tmp_path)
    args = ["--predictions", str(sample_jsonl), "--output", str(report), "--in-place"]
    agg.main(args)
    after_first = report.read_text(encoding="utf-8")
    agg.main(args)
    after_second = report.read_text(encoding="utf-8")
    assert after_first == after_second


# ----------------------------------------------------------------------
# T14: in-place preserves outside sentinel content
# ----------------------------------------------------------------------


def test_in_place_preserves_outside_sentinel(
    tmp_path: Path, sample_jsonl: Path
) -> None:
    report = _report_with_all_sentinels(tmp_path)
    agg.main(
        ["--predictions", str(sample_jsonl), "--output", str(report), "--in-place"]
    )
    text = report.read_text(encoding="utf-8")
    assert "### 判定結果（設計 §9）" in text
    assert "- [x] `< 50% NC` → 正常完了" in text
    assert "**所見**: 手書き保持される" in text


# ----------------------------------------------------------------------
# T15: missing sentinel warns and skips (no abort)
# ----------------------------------------------------------------------


def test_in_place_missing_sentinel_warns_and_skips(
    tmp_path: Path, sample_jsonl: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "no_sentinel.md"
    report.write_text(
        textwrap.dedent(
            """\
            # Report (no sentinels)

            手書き内容のみ。
            """
        ),
        encoding="utf-8",
    )
    agg.main(
        ["--predictions", str(sample_jsonl), "--output", str(report), "--in-place"]
    )
    captured = capsys.readouterr()
    for sec in ("overall", "category", "latency", "failures"):
        assert f"sentinel for section '{sec}' not found" in captured.err
    # 元テキストが破壊されていないこと
    assert "手書き内容のみ。" in report.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# T15-bis: 一部 sentinel 欠損で 2 回実行 idempotent (DR2-008)
# ----------------------------------------------------------------------


def test_in_place_partial_sentinel_idempotent(
    tmp_path: Path, sample_jsonl: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "partial.md"
    report.write_text(
        textwrap.dedent(
            """\
            # Report (only overall sentinel)

            <!-- aggregate:overall:start -->
            (placeholder)
            <!-- aggregate:overall:end -->

            手書き本文。
            """
        ),
        encoding="utf-8",
    )
    args = ["--predictions", str(sample_jsonl), "--output", str(report), "--in-place"]
    agg.main(args)
    after_first = report.read_text(encoding="utf-8")
    agg.main(args)
    after_second = report.read_text(encoding="utf-8")
    assert after_first == after_second
    captured = capsys.readouterr()
    # 欠損 3 section について警告が出ていること
    for sec in ("category", "latency", "failures"):
        assert f"sentinel for section '{sec}' not found" in captured.err
    # overall は更新されている
    assert "全質問数" in after_first
