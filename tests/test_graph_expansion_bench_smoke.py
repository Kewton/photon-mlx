"""Smoke test for scripts/graph_expansion_bench.py (Issue #91).

Patches ``build_pipeline`` with a ``MagicMock`` so the sweep runs without
loading MLX. Verifies --smoke runs 2 configs and produces JSON with the
expected keys.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class _FakeLatency:
    total_ms: float = 7000.0


@dataclass
class _FakeMemory:
    peak_mb: float = 2000.0


@dataclass
class _FakeQueryResult:
    answer: str = "dummy"
    no_citation: bool = False
    wrong_citation_indices: list[int] | None = None
    latency: _FakeLatency = None  # type: ignore[assignment]
    memory: _FakeMemory = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.wrong_citation_indices is None:
            self.wrong_citation_indices = []
        if self.latency is None:
            self.latency = _FakeLatency()
        if self.memory is None:
            self.memory = _FakeMemory()


def _fake_eval_set(tmp_path: Path) -> Path:
    q_path = tmp_path / "eval.jsonl"
    lines = [
        json.dumps({"id": "q1", "question": "Q1"}),
        json.dumps({"id": "q2", "question": "Q2"}),
    ]
    q_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return q_path


def test_smoke_runs_two_configs_and_writes_json(tmp_path: Path) -> None:
    from scripts.graph_expansion_bench import main

    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = _FakeQueryResult()

    output_path = tmp_path / "bench.json"
    eval_path = _fake_eval_set(tmp_path)

    config_path = REPO_ROOT / "configs" / "baseline.yaml"
    rc = 1
    with patch(
        "scripts.graph_expansion_bench.build_pipeline_for_sweep",
        return_value=mock_pipeline,
    ):
        rc = main(
            [
                "--config",
                str(config_path.relative_to(REPO_ROOT)),
                "--eval-set",
                str(eval_path),
                "--output",
                str(output_path),
                "--smoke",
            ]
        )
    assert rc == 0, "CLI must return 0 on success"

    # Output JSON contains the per-config results list with required keys.
    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert "configs" in data
    assert len(data["configs"]) == 2
    for entry in data["configs"]:
        for key in (
            "config_idx",
            "params",
            "raw_no_citation_rate",
            "latency_p95_ms",
        ):
            assert key in entry, f"missing required key {key!r}"


def test_smoke_uses_small_config_subset(tmp_path: Path) -> None:
    """--smoke must run strictly 2 configs, independent of grid size."""
    from scripts import graph_expansion_bench as ge_bench

    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = _FakeQueryResult()

    output_path = tmp_path / "bench.json"
    eval_path = _fake_eval_set(tmp_path)
    config_path = REPO_ROOT / "configs" / "baseline.yaml"

    with patch(
        "scripts.graph_expansion_bench.build_pipeline_for_sweep",
        return_value=mock_pipeline,
    ):
        ge_bench.main(
            [
                "--config",
                str(config_path.relative_to(REPO_ROOT)),
                "--eval-set",
                str(eval_path),
                "--output",
                str(output_path),
                "--smoke",
            ]
        )

    # 2 configs * 2 questions = 4 calls
    assert mock_pipeline.query.call_count == 4


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__, "-v"])
