from __future__ import annotations

import json
from pathlib import Path

from scripts.score_scenario2_comparison import (
    MAX_SCORE,
    compare_runs,
    main,
    score_record,
)


def _record(
    *,
    scenario_id: str = "A5",
    turn: int = 3,
    answer: str,
    paths: list[str],
    wrong: list[int] | None = None,
    label: str | None = None,
    latency_ms: float = 1000.0,
) -> dict:
    cited_ids = [f"c{i}" for i, _path in enumerate(paths, start=1)]
    return {
        "label": label or f"single_{scenario_id}",
        "scenario_id": scenario_id,
        "turn": turn,
        "question": "q",
        "answer": answer,
        "cited_chunk_ids": cited_ids,
        "wrong_citation_indices": wrong or [],
        "no_citation": not cited_ids,
        "latency_ms": latency_ms,
        "claim_support_guard": {"applied": False, "reason": None, "terms": []},
        "cited_info": [
            {"rel_path": path, "section": "", "source": "retrieval"} for path in paths
        ],
    }


def test_score_record_rewards_focused_topic_switch_answer() -> None:
    record = _record(
        answer="葛飾区デジタル化支援事業費補助金の必要書類です。",
        paths=["葛飾区デジタル化支援事業費補助金のご案内/document.md"],
    )

    score = score_record(record)

    assert score.total == MAX_SCORE
    assert score.notes == ()


def test_score_record_penalises_stale_context_mixing() -> None:
    record = _record(
        answer=(
            "葛飾区デジタル化支援事業費補助金の必要書類に加えて、"
            "セーフティネット保証1号の必要書類も説明します。"
        ),
        paths=[
            "葛飾区デジタル化支援事業費補助金のご案内/document.md",
            "セーフティネット保証1号認定のご案内/document.md",
        ],
    )

    score = score_record(record)

    assert score.total < MAX_SCORE
    assert "forbidden_answer" in score.notes
    assert any(note.startswith("forbidden_cited_paths=") for note in score.notes)


def test_compare_runs_counts_photon_win() -> None:
    baseline = _record(
        answer="必要書類です。セーフティネット保証1号も説明します。",
        paths=[
            "葛飾区デジタル化支援事業費補助金のご案内/document.md",
            "セーフティネット保証1号認定のご案内/document.md",
        ],
        latency_ms=2000,
    )
    photon = _record(
        answer="葛飾区デジタル化支援事業費補助金の必要書類です。",
        paths=["葛飾区デジタル化支援事業費補助金のご案内/document.md"],
        latency_ms=1000,
    )

    report = compare_runs(baseline_records=[baseline], photon_records=[photon])

    assert report["matched_turns"] == 1
    assert report["winner_counts"]["photon"] == 1
    assert report["rows"][0]["delta"] > 0


def test_c4_requires_cautious_answer() -> None:
    unsafe = _record(
        scenario_id="C4",
        turn=2,
        answer="オンライン販売も対象に含まれます。",
        paths=["商店街活性化資金融資認定申請書/document.md"],
    )
    safe = _record(
        scenario_id="C4",
        turn=2,
        answer="オンライン販売が対象に含まれるかは確認できません。",
        paths=["商店街活性化資金融資認定申請書/document.md"],
    )

    assert score_record(unsafe).safety_score == 0
    assert score_record(safe).safety_score == 2


def test_cli_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    photon_path = tmp_path / "photon.jsonl"
    baseline_path.write_text(
        json.dumps(
            _record(
                answer="必要書類です。セーフティネット保証1号も説明します。",
                paths=[
                    "葛飾区デジタル化支援事業費補助金のご案内/document.md",
                    "セーフティネット保証1号認定のご案内/document.md",
                ],
            ),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    photon_path.write_text(
        json.dumps(
            _record(
                answer="葛飾区デジタル化支援事業費補助金の必要書類です。",
                paths=["葛飾区デジタル化支援事業費補助金のご案内/document.md"],
            ),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "report.json"
    output_csv = tmp_path / "report.csv"
    output_md = tmp_path / "report.md"

    assert (
        main(
            [
                "--baseline-log",
                str(baseline_path),
                "--photon-log",
                str(photon_path),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["matched_turns"] == 1
    assert output_csv.read_text(encoding="utf-8").startswith("key,scenario_id")
    assert "Scenario 2" in output_md.read_text(encoding="utf-8")
