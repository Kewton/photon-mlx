"""Smoke + unit tests for ``scripts/compare_generators.py``.

The heavy Qwen / PHOTON E2E is exercised by the manual acceptance smoke
(§12.4); CI only needs to confirm that the module imports cleanly and
that the pure helpers behave as documented.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# CB-004 (codex-fix): the mlx stub hack was only needed because
# ``compare_generators.py`` transitively pulled in ``baseline_reporag.
# photon_pipeline`` at import time (which imports ``mlx.core``). With the
# ``pipeline_factory`` lightweight surface, no MLX import happens at
# script load, so the stub is no longer required.


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "compare_generators.py"
)


def _load_module():
    """Dynamically import ``scripts.compare_generators`` by path.

    The ``scripts/`` directory is not a package; importing by path
    keeps CI independent of sys.path tweaks.
    """
    spec = importlib.util.spec_from_file_location("compare_generators", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_generators"] = module
    spec.loader.exec_module(module)
    return module


class TestImportSmoke:
    def test_import_script_module(self) -> None:
        """The script module must import cleanly (no top-level side effects)."""
        module = _load_module()
        assert hasattr(module, "main")
        assert hasattr(module, "run_variant")
        assert hasattr(module, "load_questions")
        assert hasattr(module, "build_output_path")
        assert hasattr(module, "override_photon_generation")


class TestLoadQuestions:
    def test_load_questions_parses_jsonl(self, tmp_path: Path) -> None:
        module = _load_module()
        path = tmp_path / "q.jsonl"
        path.write_text(
            '{"id": "q1", "question": "Q1?"}\n{"id": "q2", "question": "Q2?"}\n',
            encoding="utf-8",
        )
        questions = module.load_questions(path)
        assert len(questions) == 2
        assert questions[0]["id"] == "q1"
        assert questions[1]["question"] == "Q2?"

    def test_load_questions_skips_blank_lines(self, tmp_path: Path) -> None:
        module = _load_module()
        path = tmp_path / "q.jsonl"
        path.write_text(
            '\n{"id": "q1", "question": "Q1?"}\n\n',
            encoding="utf-8",
        )
        questions = module.load_questions(path)
        assert len(questions) == 1
        assert questions[0]["id"] == "q1"

    def test_load_questions_raises_on_invalid_json(self, tmp_path: Path) -> None:
        module = _load_module()
        path = tmp_path / "q.jsonl"
        path.write_text("not-json\n", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            module.load_questions(path)


class TestBuildOutputPath:
    def test_build_output_path_uses_timestamp_and_dir(self, tmp_path: Path) -> None:
        module = _load_module()
        out = module.build_output_path(tmp_path)
        assert out.parent == tmp_path
        assert out.name.startswith("compare_generators_")
        assert out.name.endswith(".jsonl")

    def test_build_output_path_defaults_to_reports(self) -> None:
        module = _load_module()
        out = module.build_output_path(None)
        assert out.parent.name == "reports"


class TestOverridePhotonGeneration:
    def test_override_sets_flag_to_true_and_false(self) -> None:
        """override_photon_generation must flip the flag on a Config-like object."""
        module = _load_module()
        from baseline_reporag.config import Config

        cfg = Config({"inference": {"photon_generation_enabled": False}})
        module.override_photon_generation(cfg, True)
        assert cfg.inference.photon_generation_enabled is True

        module.override_photon_generation(cfg, False)
        assert cfg.inference.photon_generation_enabled is False

    def test_override_adds_inference_section_when_missing(self) -> None:
        module = _load_module()
        from baseline_reporag.config import Config

        cfg = Config({"model": {"provider": "photon"}})
        module.override_photon_generation(cfg, True)
        assert cfg.inference.photon_generation_enabled is True


class TestWriteRows:
    def test_write_rows_emits_jsonl(self, tmp_path: Path) -> None:
        module = _load_module()
        path = tmp_path / "out.jsonl"
        rows = [
            {"variant_requested": "qwen", "latency_ms": 1.0},
            {"variant_requested": "photon", "latency_ms": 2.0},
        ]
        module.write_rows(rows, path)
        text = path.read_text(encoding="utf-8")
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
        assert parsed == rows


class TestRunVariantAttachesGeneratorUsed:
    """CB-003 (codex-fix): ``run_variant`` must surface the real
    ``generator_used`` value from the pipeline's ``QueryResult`` — not
    silently record ``null`` because the field is missing.
    """

    def test_run_variant_attaches_generator_used_from_result(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from unittest.mock import MagicMock

        from baseline_reporag.config import Config

        # CB2-001 (codex-fix): import ``QueryResult`` from the MLX-free
        # ``baseline_reporag.contracts`` module rather than from
        # ``baseline_reporag.pipeline`` (which transitively imports
        # ``mlx_lm`` via ``.generation.generator``).  This keeps the test
        # meaningful on baseline-only environments.
        from baseline_reporag.contracts import QueryResult
        from baseline_reporag.profiler import LatencyBreakdown, MemorySnapshot

        module = _load_module()

        fake_latency = LatencyBreakdown(
            retrieval_ms=5.0,
            generation_ms=10.0,
            citation_ms=0.0,
            total_ms=15.0,
        )
        fake_memory = MemorySnapshot(peak_mb=42.0, current_mb=10.0)

        fake_result = QueryResult(
            answer="hello",
            session_id="compare-row1",
            turn_id=1,
            cited_chunk_ids=["c1"],
            wrong_citation_indices=[],
            no_citation=False,
            latency=fake_latency,
            memory=fake_memory,
            citation_postprocessed=False,
            generator_used="photon",
            generator_fallback_reason=None,
        )

        fake_pipeline = MagicMock()
        fake_pipeline.query.return_value = fake_result

        def _fake_build_pipeline(_cfg):
            return fake_pipeline

        # CB-004 (codex-fix): ``compare_generators.run_variant`` now
        # lazy-imports ``build_pipeline`` from
        # ``baseline_reporag.pipeline_factory``. Patch that target so the
        # test bypasses real config / filesystem wiring.
        import baseline_reporag.pipeline_factory as pipeline_factory_module

        monkeypatch.setattr(
            pipeline_factory_module, "build_pipeline", _fake_build_pipeline
        )

        cfg = Config({"inference": {}, "model": {"provider": "photon"}})
        questions = [{"id": "q1", "question": "Q?"}]

        rows = module.run_variant(
            cfg,
            questions,
            repo_id="test-repo",
            photon_generation_enabled=True,
        )

        assert len(rows) == 1
        assert rows[0]["generator_used"] == "photon"
        assert rows[0]["variant_requested"] == "photon"
        assert rows[0]["answer"] == "hello"


class TestRunVariantSeedPropagation:
    """Issue #143 / Step 7: ``run_variant`` forwards ``seed`` into ``pipeline.query``.

    The acceptance criterion is byte-identical Qwen output across the
    two variants (Qwen-only / PHOTON-with-Qwen-fallback) when
    ``cfg.run.seed`` is pinned. That guarantee starts with this kwarg
    actually reaching ``pipeline.query``.
    """

    def _drive_run_variant(self, monkeypatch, *, seed):
        from unittest.mock import MagicMock

        from baseline_reporag.config import Config
        from baseline_reporag.contracts import QueryResult
        from baseline_reporag.profiler import LatencyBreakdown, MemorySnapshot

        module = _load_module()

        fake_latency = LatencyBreakdown(
            retrieval_ms=1.0, generation_ms=1.0, citation_ms=0.0, total_ms=2.0
        )
        fake_memory = MemorySnapshot(peak_mb=10.0, current_mb=5.0)
        fake_result = QueryResult(
            answer="x",
            session_id="compare-row1",
            turn_id=1,
            cited_chunk_ids=["c1"],
            wrong_citation_indices=[],
            no_citation=False,
            latency=fake_latency,
            memory=fake_memory,
            citation_postprocessed=False,
            generator_used="qwen",
            generator_fallback_reason=None,
        )

        fake_pipeline = MagicMock()
        fake_pipeline.query.return_value = fake_result

        import baseline_reporag.pipeline_factory as pipeline_factory_module

        monkeypatch.setattr(
            pipeline_factory_module,
            "build_pipeline",
            lambda _cfg: fake_pipeline,
        )

        cfg = Config({"inference": {}, "model": {"provider": "baseline"}})
        questions = [{"id": "q1", "question": "Q?"}]

        if seed is _SENTINEL:
            module.run_variant(
                cfg, questions, repo_id="r", photon_generation_enabled=False
            )
        else:
            module.run_variant(
                cfg,
                questions,
                repo_id="r",
                photon_generation_enabled=False,
                seed=seed,
            )
        return fake_pipeline

    def test_default_seed_none_forwards_none(self, monkeypatch) -> None:
        """Default invocation forwards ``seed=None`` (no MLX seeding)."""
        fake_pipeline = self._drive_run_variant(monkeypatch, seed=_SENTINEL)
        call = fake_pipeline.query.call_args
        assert call.kwargs.get("seed") is None

    def test_explicit_seed_propagates(self, monkeypatch) -> None:
        """``seed=42`` reaches ``pipeline.query``."""
        fake_pipeline = self._drive_run_variant(monkeypatch, seed=42)
        call = fake_pipeline.query.call_args
        assert call.kwargs.get("seed") == 42

    def test_seed_zero_propagates(self, monkeypatch) -> None:
        """``seed=0`` MUST propagate (DR3-002 silent-bug guard)."""
        fake_pipeline = self._drive_run_variant(monkeypatch, seed=0)
        call = fake_pipeline.query.call_args
        assert call.kwargs.get("seed") == 0


_SENTINEL = object()
