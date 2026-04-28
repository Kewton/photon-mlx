"""Smoke integration test for scripts/retrieval_grid_search (Issue #88).

Patches ``build_pipeline`` with a ``MagicMock`` so the sweep loop runs
without loading MLX / Qwen / sentence-transformers. Verifies:

- ``run_eval_inproc`` feeds ``session_id = grid-<idx>-<qid>`` correctly
- ``run_phase`` writes per-config results atomically
- ``--resume`` skips already-completed configs
- ``--dry-run`` prints the 24 Phase 1 configs and exits clean
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._grid_search_core import (  # noqa: E402
    ConfigParams,
    aggregate_metrics,
)
from scripts.retrieval_grid_search import (  # noqa: E402
    _params_key,
    main,
    run_eval_inproc,
    run_phase,
)


# ---------------------------------------------------------------------------
# Minimal fakes mirroring the bits of QueryResult the loop reads.
# ---------------------------------------------------------------------------


@dataclass
class _FakeLatency:
    total_ms: float = 12_000.0


@dataclass
class _FakeMemory:
    peak_mb: float = 4200.0


@dataclass
class _FakeQueryResult:
    answer: str
    no_citation: bool
    wrong_citation_indices: list[int]
    latency: _FakeLatency
    memory: _FakeMemory


def _mk_result(
    *, no_citation: bool, wrong: list[int] | None = None, latency: float = 12_000.0
) -> _FakeQueryResult:
    return _FakeQueryResult(
        answer="dummy",
        no_citation=no_citation,
        wrong_citation_indices=wrong or [],
        latency=_FakeLatency(total_ms=latency),
        memory=_FakeMemory(),
    )


def _load_base_cfg():  # type: ignore[no-untyped-def]
    from baseline_reporag.config import load_config

    return load_config(REPO_ROOT / "configs" / "baseline.yaml")


# ---------------------------------------------------------------------------
# run_eval_inproc
# ---------------------------------------------------------------------------


def test_smoke_run_eval_inproc_uses_session_id_scope() -> None:
    mock_pipeline = MagicMock()
    mock_pipeline.query.side_effect = [
        _mk_result(no_citation=False, latency=10_000.0),
        _mk_result(no_citation=True, latency=20_000.0),
        _mk_result(no_citation=False, wrong=[2], latency=30_000.0),
    ]
    questions = [
        {"id": "q1", "question": "Q1"},
        {"id": "q2", "question": "Q2"},
        {"id": "q3", "question": "Q3"},
    ]
    records = run_eval_inproc(
        mock_pipeline, questions, config_idx=7, repo_id="fastapi_fastapi"
    )
    assert len(records) == 3
    assert records[0]["eval_id"] == "q1"
    assert records[1]["no_citation"] is True
    assert records[2]["wrong_citation_indices"] == [2]

    calls = mock_pipeline.query.call_args_list
    assert calls[0].kwargs["session_id"] == "grid-7-q1"
    assert calls[1].kwargs["session_id"] == "grid-7-q2"
    assert calls[2].kwargs["session_id"] == "grid-7-q3"

    metrics = aggregate_metrics(records, unanswerable_ids=set())
    assert metrics["no_citation_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# run_phase writes JSON state atomically and supports resume.
# ---------------------------------------------------------------------------


def _params(
    lex: int = 20, emb: int = 20, fused: int = 16, rerank: int = 12
) -> ConfigParams:
    return ConfigParams(
        lexical_top_k=lex,
        embedding_top_k=emb,
        fused_top_k=fused,
        rerank_top_k=rerank,
        weights_lexical=0.45,
        weights_embedding=0.45,
    )


def test_run_phase_writes_state_per_config(tmp_path: Path) -> None:
    base_cfg = _load_base_cfg()
    mock_pipeline = MagicMock()
    mock_pipeline.cfg = base_cfg
    mock_pipeline.query.side_effect = [
        _mk_result(no_citation=False),
        _mk_result(no_citation=True),
    ]
    questions: list[dict[str, Any]] = [
        {"id": "q1", "question": "?"},
        {"id": "q2", "question": "?"},
    ]
    state_path = tmp_path / "state.json"

    grid = [_params(lex=20)]
    results = run_phase(
        base_cfg,
        grid,
        questions=questions,
        unanswerable_ids=set(),
        pipeline=mock_pipeline,
        state_path=state_path,
        resume=False,
        phase_name="phase1",
        repo_id="fastapi_fastapi",
        stop_flag={"stop": False},
    )

    assert len(results) == 1
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "phase1_complete"
    assert len(state["configs"]) == 1
    assert state["configs"][0]["params"]["lexical_top_k"] == 20


def test_run_phase_resume_skips_completed_configs(tmp_path: Path) -> None:
    base_cfg = _load_base_cfg()
    mock_pipeline = MagicMock()
    mock_pipeline.cfg = base_cfg
    state_path = tmp_path / "state.json"

    grid = [_params(lex=20), _params(lex=25)]

    # Run once with both configs...
    mock_pipeline.query.side_effect = [
        _mk_result(no_citation=False)
        for _ in range(2)  # q1 for config1
    ] + [
        _mk_result(no_citation=True)
        for _ in range(2)  # q1 for config2
    ]
    questions = [{"id": "q1", "question": "?"}]
    # But actually we only pass 1 question per config so side_effect length = 2.
    mock_pipeline.query.side_effect = [
        _mk_result(no_citation=False),
        _mk_result(no_citation=True),
    ]
    run_phase(
        base_cfg,
        grid,
        questions=questions,
        unanswerable_ids=set(),
        pipeline=mock_pipeline,
        state_path=state_path,
        resume=False,
        phase_name="phase1",
        repo_id="fastapi_fastapi",
        stop_flag={"stop": False},
    )

    # Now re-run with --resume; both configs are already in state so pipeline
    # must not be invoked again.
    call_count_before = mock_pipeline.query.call_count
    run_phase(
        base_cfg,
        grid,
        questions=questions,
        unanswerable_ids=set(),
        pipeline=mock_pipeline,
        state_path=state_path,
        resume=True,
        phase_name="phase1",
        repo_id="fastapi_fastapi",
        stop_flag={"stop": False},
    )
    assert mock_pipeline.query.call_count == call_count_before


def test_run_phase_resume_completes_missing_configs(tmp_path: Path) -> None:
    base_cfg = _load_base_cfg()
    mock_pipeline = MagicMock()
    mock_pipeline.cfg = base_cfg
    state_path = tmp_path / "state.json"

    # Pre-populate state with config A only.
    grid = [_params(lex=20), _params(lex=25)]
    key_done = _params_key(grid[0])
    state = {
        "phase": "phase1",
        "configs": [
            {
                "config_idx": 0,
                "params": json.loads(key_done),
                "raw_no_citation_rate": 0.1,
                "true_nc_rate": 0.1,
                "wrong_citation_count": 0,
                "latency_p50_ms": 12_000.0,
                "latency_p95_ms": 12_000.0,
                "n_questions": 1,
                "n_no_citation": 0,
                "duration_seconds": 1.0,
                "started_at": "x",
                "completed_at": "y",
            }
        ],
        "status": "running",
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    mock_pipeline.query.side_effect = [_mk_result(no_citation=True)]
    questions = [{"id": "q1", "question": "?"}]
    results = run_phase(
        base_cfg,
        grid,
        questions=questions,
        unanswerable_ids=set(),
        pipeline=mock_pipeline,
        state_path=state_path,
        resume=True,
        phase_name="phase1",
        repo_id="fastapi_fastapi",
        stop_flag={"stop": False},
    )
    # Only config B ran (config A skipped on resume).
    assert mock_pipeline.query.call_count == 1
    assert len(results) == 1
    assert results[0].params.lexical_top_k == 25


# ---------------------------------------------------------------------------
# main --dry-run prints the 24 Phase 1 configs and exits clean.
# ---------------------------------------------------------------------------


def test_main_dry_run_prints_grid(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("scripts.retrieval_grid_search.build_pipeline_for_sweep") as mock_build:
        exit_code = main(
            [
                "--config",
                "configs/baseline.yaml",
                "--dry-run",
            ]
        )
    # Dry-run must NOT build a pipeline.
    mock_build.assert_not_called()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Phase 1 grid contains 24 configs" in out


def test_main_rejects_absolute_config_path() -> None:
    with pytest.raises(SystemExit):
        main(["--config", "/etc/passwd", "--dry-run"])


def test_main_rejects_parent_traversal() -> None:
    with pytest.raises(SystemExit):
        main(["--config", "../../../etc/passwd", "--dry-run"])


# ---------------------------------------------------------------------------
# Sanity: pipeline.cfg is mutated in-place between configs.
# ---------------------------------------------------------------------------


def test_run_phase_mutates_pipeline_cfg_each_iteration(tmp_path: Path) -> None:
    base_cfg = _load_base_cfg()
    mock_pipeline = MagicMock()
    mock_pipeline.cfg = base_cfg

    seen_lex: list[int] = []

    def _capture_cfg(**_kwargs: Any) -> _FakeQueryResult:
        seen_lex.append(mock_pipeline.cfg.retrieval.lexical_top_k)
        return _mk_result(no_citation=False)

    mock_pipeline.query.side_effect = _capture_cfg
    questions = [{"id": "q1", "question": "?"}]
    grid = [_params(lex=20), _params(lex=25)]

    run_phase(
        base_cfg,
        grid,
        questions=questions,
        unanswerable_ids=set(),
        pipeline=mock_pipeline,
        state_path=tmp_path / "state.json",
        resume=False,
        phase_name="phase1",
        repo_id="fastapi_fastapi",
        stop_flag={"stop": False},
    )
    assert seen_lex == [20, 25]


# ---------------------------------------------------------------------------
# Sanity: stop_flag aborts before running the next config.
# ---------------------------------------------------------------------------


def test_run_phase_honors_stop_flag(tmp_path: Path) -> None:
    from scripts.retrieval_grid_search import _InterruptedError

    base_cfg = _load_base_cfg()
    mock_pipeline = MagicMock()
    mock_pipeline.cfg = base_cfg
    stop_flag = {"stop": True}
    grid = [_params(lex=20), _params(lex=25)]

    with pytest.raises(_InterruptedError):
        run_phase(
            base_cfg,
            grid,
            questions=[{"id": "q1", "question": "?"}],
            unanswerable_ids=set(),
            pipeline=mock_pipeline,
            state_path=tmp_path / "state.json",
            resume=False,
            phase_name="phase1",
            repo_id="fastapi_fastapi",
            stop_flag=stop_flag,
        )
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"


# ---------------------------------------------------------------------------
# Issue #143: ``run_eval_inproc`` forwards ``seed`` into ``pipeline.query``.
# ---------------------------------------------------------------------------


def test_run_eval_inproc_default_seed_none() -> None:
    """Default ``seed=None`` -> pipeline.query receives seed=None.

    ``pipeline.query`` (at the production layer) handles ``seed=None`` by
    skipping MLX seeding so interactive callers stay nondeterministic;
    this test only asserts the kwarg is forwarded transparently here.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = _mk_result(no_citation=False)
    questions = [{"id": "q1", "question": "Q1"}]
    run_eval_inproc(mock_pipeline, questions, config_idx=0, repo_id="r1")
    call = mock_pipeline.query.call_args
    assert call.kwargs.get("seed") is None


def test_run_eval_inproc_propagates_seed() -> None:
    """``seed=42`` -> pipeline.query receives seed=42 every iteration."""
    mock_pipeline = MagicMock()
    mock_pipeline.query.side_effect = [
        _mk_result(no_citation=False),
        _mk_result(no_citation=True),
    ]
    questions = [{"id": "q1", "question": "Q1"}, {"id": "q2", "question": "Q2"}]
    run_eval_inproc(mock_pipeline, questions, config_idx=0, repo_id="r1", seed=42)
    for call in mock_pipeline.query.call_args_list:
        assert call.kwargs.get("seed") == 42


def test_run_eval_inproc_seed_zero_propagates() -> None:
    """``seed=0`` MUST propagate (DR3-002 silent-bug guard)."""
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = _mk_result(no_citation=False)
    questions = [{"id": "q1", "question": "Q1"}]
    run_eval_inproc(mock_pipeline, questions, config_idx=0, repo_id="r1", seed=0)
    call = mock_pipeline.query.call_args
    assert call.kwargs.get("seed") == 0
