"""Smoke tests for ``scripts/run_stress_eval.py`` (Issue #143 / Step 3.5).

The heavy stress eval (LLM inference + concurrent sessions) is exercised
manually; CI only needs to confirm that the module imports cleanly and
that the seed plumbed via ``cfg.run`` reaches every ``pipeline.query``
call (DR3-001 / DR3-002).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stress_eval.py"


def _load_module():
    """Dynamically import ``scripts.run_stress_eval`` by path."""
    spec = importlib.util.spec_from_file_location("run_stress_eval", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_stress_eval"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Minimal fakes for QueryResult fields touched by run_real.
# ---------------------------------------------------------------------------


@dataclass
class _FakeLatency:
    retrieval_ms: float = 1.0
    generation_ms: float = 1.0
    total_ms: float = 2.0


@dataclass
class _FakeMemory:
    peak_mb: float = 100.0


@dataclass
class _FakeQueryResult:
    answer: str
    cited_chunk_ids: list[str]
    no_citation: bool
    latency: _FakeLatency
    memory: _FakeMemory


def _mk_result() -> _FakeQueryResult:
    return _FakeQueryResult(
        answer="hi",
        cited_chunk_ids=["c1"],
        no_citation=False,
        latency=_FakeLatency(),
        memory=_FakeMemory(),
    )


def _make_temp_yaml(tmp_path: Path) -> Path:
    """Build a minimal YAML config that load_config can parse + run.seed picks up."""
    cfg_path = tmp_path / "stress.yaml"
    cfg_path.write_text(
        "model:\n  provider: baseline\n  model_id: fake\n"
        "repo:\n  repo_id: testrepo\n  repo_commit: abc\n"
        "run:\n  seed: 42\n  deterministic: true\n",
        encoding="utf-8",
    )
    return cfg_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImportSmoke:
    def test_module_imports(self) -> None:
        module = _load_module()
        assert hasattr(module, "main")
        assert hasattr(module, "run_real")
        assert hasattr(module, "run_dry")


class TestSeedPropagation:
    """``run_real`` resolves cfg.run.seed and forwards it to every query."""

    def test_run_real_propagates_seed_from_cfg(self, tmp_path: Path) -> None:
        """seed=42 in cfg.run → every pipeline.query call receives seed=42."""
        module = _load_module()
        cfg_path = _make_temp_yaml(tmp_path)

        sessions = [
            {
                "session_id": "s1",
                "turns": [
                    {"turn_id": 1, "question": "Q1"},
                    {"turn_id": 2, "question": "Q2"},
                ],
            }
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _mk_result()

        # Patch ``build_pipeline`` (lazy-imported inside run_real) so the
        # heavy MLX/index stack is never touched.
        with patch(
            "baseline_reporag.pipeline_factory.build_pipeline",
            return_value=mock_pipeline,
        ):
            module.run_real(sessions, str(cfg_path), concurrency=1)

        # Both turns must have been called with seed=42.
        assert mock_pipeline.query.call_count == 2
        for call in mock_pipeline.query.call_args_list:
            assert call.kwargs.get("seed") == 42, (
                f"seed=42 did not reach pipeline.query: kwargs={call.kwargs}"
            )

    def test_run_real_seed_none_when_deterministic_false(self, tmp_path: Path) -> None:
        """deterministic=False → pipeline.query receives seed=None."""
        module = _load_module()
        cfg_path = tmp_path / "nondet.yaml"
        cfg_path.write_text(
            "model:\n  provider: baseline\n  model_id: fake\n"
            "repo:\n  repo_id: testrepo\n  repo_commit: abc\n"
            "run:\n  seed: 42\n  deterministic: false\n",
            encoding="utf-8",
        )

        sessions = [{"session_id": "s1", "turns": [{"turn_id": 1, "question": "Q1"}]}]

        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _mk_result()
        with patch(
            "baseline_reporag.pipeline_factory.build_pipeline",
            return_value=mock_pipeline,
        ):
            module.run_real(sessions, str(cfg_path), concurrency=1)

        call = mock_pipeline.query.call_args
        assert call.kwargs.get("seed") is None
