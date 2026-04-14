"""Tests for PHOTON-RAG pipeline integration (Issue #3)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# TDD Cycle 1: Config._config_path and build_pipeline factory
# ---------------------------------------------------------------------------


class TestConfigPath:
    """load_config must remember the source YAML path."""

    def test_load_config_stores_config_path(self, tmp_path):
        from baseline_reporag.config import load_config

        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("model:\n  provider: baseline\n")
        cfg = load_config(str(cfg_file))
        assert cfg._config_path == str(cfg_file)

    def test_load_config_provider_field_accessible(self, tmp_path):
        from baseline_reporag.config import load_config

        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("model:\n  provider: photon\n")
        cfg = load_config(str(cfg_file))
        assert cfg.model.provider == "photon"


class TestBuildPipeline:
    """build_pipeline(cfg) factory routing based on cfg.model.provider."""

    def test_baseline_provider_returns_reporag_pipeline(self, tmp_path):
        """provider != 'photon' → RepoRAGPipeline."""
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import build_pipeline

        cfg_file = tmp_path / "baseline.yaml"
        cfg_file.write_text(
            "model:\n  provider: mlx_lm\n  model_id: test\n"
            "repo:\n  repo_id: test\n  repo_commit: abc\n"
            "retrieval:\n  lexical_top_k: 5\n"
            "memory:\n  log_dir: null\n"
        )
        cfg = load_config(str(cfg_file))
        # build_pipeline should not fail for baseline (mock heavy deps)
        with patch(
            "baseline_reporag.photon_pipeline._build_baseline_deps"
        ) as mock_deps:
            mock_deps.return_value = _make_mock_deps()
            pipeline = build_pipeline(cfg)
        from baseline_reporag.pipeline import RepoRAGPipeline

        assert isinstance(pipeline, RepoRAGPipeline)

    def test_photon_provider_returns_photon_rag_pipeline(self, tmp_path):
        """provider == 'photon' → PhotonRAGPipeline."""
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import build_pipeline

        cfg_file = tmp_path / "photon.yaml"
        cfg_file.write_text(
            "model:\n  provider: photon\n  model_id: test\n"
            "repo:\n  repo_id: test\n  repo_commit: abc\n"
            "retrieval:\n  lexical_top_k: 5\n"
            "memory:\n  log_dir: null\n"
        )
        cfg = load_config(str(cfg_file))
        with (
            patch("baseline_reporag.photon_pipeline._build_baseline_deps") as mock_deps,
            patch("baseline_reporag.photon_pipeline._build_photon_deps") as mock_photon,
        ):
            mock_deps.return_value = _make_mock_deps()
            mock_photon.return_value = _make_mock_photon_deps()
            pipeline = build_pipeline(cfg)
        from baseline_reporag.photon_pipeline import PhotonRAGPipeline

        assert isinstance(pipeline, PhotonRAGPipeline)

    def test_missing_provider_defaults_to_baseline(self, tmp_path):
        """No model.provider field → baseline pipeline."""
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import build_pipeline

        cfg_file = tmp_path / "no_provider.yaml"
        cfg_file.write_text(
            "model:\n  model_id: test\n"
            "repo:\n  repo_id: test\n  repo_commit: abc\n"
            "memory:\n  log_dir: null\n"
        )
        cfg = load_config(str(cfg_file))
        with patch(
            "baseline_reporag.photon_pipeline._build_baseline_deps"
        ) as mock_deps:
            mock_deps.return_value = _make_mock_deps()
            pipeline = build_pipeline(cfg)
        from baseline_reporag.pipeline import RepoRAGPipeline

        assert isinstance(pipeline, RepoRAGPipeline)


# ---------------------------------------------------------------------------
# TDD Cycle 2: tokenize_evidence_pack
# ---------------------------------------------------------------------------


class TestTokenizeEvidencePack:
    """tokenize_evidence_pack encodes text and applies chunk-aligned padding."""

    def test_basic_tokenization(self):
        from baseline_reporag.photon_pipeline import tokenize_evidence_pack

        import mlx.core as mx

        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(20))  # 20 tokens
        tokenizer.pad_token_id = 0

        cfg = MagicMock()
        cfg.hierarchy.chunk_sizes = [4, 4]  # prod = 16

        result = tokenize_evidence_pack("hello world", tokenizer, cfg)
        assert isinstance(result, mx.array)
        assert result.shape[0] % 16 == 0  # chunk-aligned
        assert result.shape[0] == 32  # 20 → pad to 32 (next multiple of 16)

    def test_truncation_to_max_tokens(self):
        from baseline_reporag.photon_pipeline import tokenize_evidence_pack

        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(3000))  # exceeds 2048
        tokenizer.pad_token_id = 0

        cfg = MagicMock()
        cfg.hierarchy.chunk_sizes = [4, 4]

        result = tokenize_evidence_pack("long text", tokenizer, cfg, max_tokens=2048)
        assert result.shape[0] == 2048  # 2048 is already multiple of 16

    def test_empty_text(self):
        from baseline_reporag.photon_pipeline import tokenize_evidence_pack

        tokenizer = MagicMock()
        tokenizer.encode.return_value = []
        tokenizer.pad_token_id = 0

        cfg = MagicMock()
        cfg.hierarchy.chunk_sizes = [4, 4]

        result = tokenize_evidence_pack("", tokenizer, cfg)
        # Empty → pad to padding_multiple (16)
        assert result.shape[0] == 0 or result.shape[0] % 16 == 0

    def test_exact_multiple_no_extra_padding(self):
        from baseline_reporag.photon_pipeline import tokenize_evidence_pack

        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(32))  # already 32 = 2*16
        tokenizer.pad_token_id = 0

        cfg = MagicMock()
        cfg.hierarchy.chunk_sizes = [4, 4]

        result = tokenize_evidence_pack("text", tokenizer, cfg)
        assert result.shape[0] == 32  # no extra padding needed


# ---------------------------------------------------------------------------
# TDD Cycle 3: compute_confidence
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    """compute_confidence extracts max softmax probability from logits."""

    def test_returns_float(self):
        from baseline_reporag.photon_pipeline import compute_confidence

        import mlx.core as mx

        logits = mx.random.normal((1, 10, 100))
        result = compute_confidence(logits)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_high_confidence_for_peaked_logits(self):
        from baseline_reporag.photon_pipeline import compute_confidence

        import mlx.core as mx

        # Very peaked logits → high confidence
        logits = mx.zeros((1, 5, 50))
        logits = logits.at[:, :, 0].add(100.0)  # one dominant logit
        result = compute_confidence(logits)
        assert result > 0.9

    def test_low_confidence_for_uniform_logits(self):
        from baseline_reporag.photon_pipeline import compute_confidence

        import mlx.core as mx

        # Uniform logits → low confidence (1/vocab_size)
        logits = mx.zeros((1, 5, 1000))
        result = compute_confidence(logits)
        assert result < 0.1


# ---------------------------------------------------------------------------
# TDD Cycle 4: QueryResult PHOTON fields
# ---------------------------------------------------------------------------


class TestQueryResultExtension:
    """QueryResult supports optional PHOTON fields."""

    def test_baseline_query_result_unchanged(self):
        from baseline_reporag.pipeline import QueryResult
        from baseline_reporag.profiler import LatencyBreakdown, MemorySnapshot

        result = QueryResult(
            answer="test",
            session_id="s1",
            turn_id=1,
            cited_chunk_ids=[],
            wrong_citation_indices=[],
            no_citation=False,
            latency=LatencyBreakdown(0, 0, 0, 0, 0),
            memory=MemorySnapshot(0, 0),
        )
        assert result.answer == "test"
        # PHOTON fields should default to None
        assert result.drift_metrics is None
        assert result.confidence is None
        assert result.fallback_decision is None

    def test_photon_query_result_with_extras(self):
        from baseline_reporag.pipeline import QueryResult
        from baseline_reporag.profiler import LatencyBreakdown, MemorySnapshot

        result = QueryResult(
            answer="photon answer",
            session_id="s2",
            turn_id=2,
            cited_chunk_ids=["c1"],
            wrong_citation_indices=[],
            no_citation=False,
            latency=LatencyBreakdown(0, 0, 0, 0, 0),
            memory=MemorySnapshot(0, 0),
            drift_metrics={"turn_id": 2, "cosine_drift": 0.1},
            confidence=0.85,
            fallback_decision={"should_fallback": False},
        )
        assert result.drift_metrics == {"turn_id": 2, "cosine_drift": 0.1}
        assert result.confidence == 0.85
        assert result.fallback_decision == {"should_fallback": False}

    def test_query_result_has_citation_postprocessed_field(self):
        """citation_postprocessed defaults to False (backward compatible)."""
        from baseline_reporag.pipeline import QueryResult
        from baseline_reporag.profiler import LatencyBreakdown, MemorySnapshot

        # Existing call sites do not pass citation_postprocessed — must still work.
        result = QueryResult(
            answer="test",
            session_id="s1",
            turn_id=1,
            cited_chunk_ids=[],
            wrong_citation_indices=[],
            no_citation=False,
            latency=LatencyBreakdown(0, 0, 0, 0, 0),
            memory=MemorySnapshot(0, 0),
        )
        assert hasattr(result, "citation_postprocessed")
        assert result.citation_postprocessed is False

    def test_query_result_accepts_citation_postprocessed_true(self):
        """citation_postprocessed can be set to True via kwarg."""
        from baseline_reporag.pipeline import QueryResult
        from baseline_reporag.profiler import LatencyBreakdown, MemorySnapshot

        result = QueryResult(
            answer="test [C:1]",
            session_id="s1",
            turn_id=1,
            cited_chunk_ids=["c1"],
            wrong_citation_indices=[],
            no_citation=False,
            latency=LatencyBreakdown(0, 0, 0, 0, 0),
            memory=MemorySnapshot(0, 0),
            citation_postprocessed=True,
        )
        assert result.citation_postprocessed is True


# ---------------------------------------------------------------------------
# TDD Cycle 5: LatencyBreakdown PHOTON fields
# ---------------------------------------------------------------------------


class TestLatencyBreakdownExtension:
    """LatencyBreakdown supports PHOTON timing fields."""

    def test_baseline_latency_unchanged(self):
        from baseline_reporag.profiler import LatencyBreakdown

        lb = LatencyBreakdown(10, 20, 30, 40, 50)
        d = lb.as_dict()
        assert d["retrieval_ms"] == 10
        assert d["generation_ms"] == 40
        # PHOTON fields default to 0
        assert lb.photon_prefill_ms == 0.0
        assert lb.drift_eval_ms == 0.0
        assert lb.safe_recgen_ms == 0.0

    def test_photon_latency_with_extras(self):
        from baseline_reporag.profiler import LatencyBreakdown

        lb = LatencyBreakdown(10, 20, 30, 40, 50)
        lb.photon_prefill_ms = 15.0
        lb.drift_eval_ms = 5.0
        lb.safe_recgen_ms = 3.0
        assert lb.photon_prefill_ms == 15.0


# ---------------------------------------------------------------------------
# TDD Cycle 6: build_messages session_summary
# ---------------------------------------------------------------------------


class TestBuildMessagesSessionSummary:
    """build_messages supports optional session_summary parameter."""

    def test_without_session_summary(self):
        from baseline_reporag.generation.prompt import build_messages

        msgs = build_messages(
            question="What is X?",
            evidence_text="Evidence here",
            history_text="",
        )
        # Should work without session_summary (backward compatible)
        assert len(msgs) > 0

    def test_with_session_summary(self):
        from baseline_reporag.generation.prompt import build_messages

        msgs = build_messages(
            question="What is X?",
            evidence_text="Evidence here",
            history_text="",
            session_summary="[PHOTON] Topic shift detected",
        )
        # session_summary should appear in the messages
        full_text = str(msgs)
        assert "[PHOTON]" in full_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_deps():
    """Create mocked baseline pipeline dependencies."""
    return {
        "store": MagicMock(),
        "lexical": MagicMock(),
        "embedding": MagicMock(),
        "graph": MagicMock(),
        "sessions": MagicMock(),
        "generator": MagicMock(),
        "logger": MagicMock(),
    }


def _make_mock_photon_deps():
    """Create mocked PHOTON pipeline dependencies."""
    return {
        "photon_inference": MagicMock(),
        "safe_recgen": MagicMock(),
        "photon_cfg": MagicMock(),
        "tokenizer": MagicMock(),
    }


# ---------------------------------------------------------------------------
# TDD Cycle 7: _build_photon_deps wires real PHOTON components
# ---------------------------------------------------------------------------


class TestBuildPhotonDeps:
    """_build_photon_deps constructs PHOTON components from config."""

    def test_returns_required_keys(self, tmp_path):
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps

        cfg_file = tmp_path / "photon.yaml"
        cfg_file.write_text(
            "model:\n"
            "  provider: photon\n"
            "  architecture: photon_decoder\n"
            "  base_embed_dim: 64\n"
            "  hidden_size: 128\n"
            "  intermediate_size: 256\n"
            "  num_heads: 4\n"
            "  vocab_size: 1000\n"
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: true\n"
            "safe_recgen:\n"
            "  enabled: true\n"
            "  thresholds:\n"
            "    confidence_floor: 0.40\n"
        )
        cfg = load_config(str(cfg_file))
        deps = _build_photon_deps(cfg)
        assert "photon_inference" in deps
        assert "safe_recgen" in deps
        assert "photon_cfg" in deps
        assert "tokenizer" in deps

    def test_safe_recgen_disabled(self, tmp_path):
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps

        cfg_file = tmp_path / "photon.yaml"
        cfg_file.write_text(
            "model:\n"
            "  provider: photon\n"
            "  architecture: photon_decoder\n"
            "  base_embed_dim: 64\n"
            "  hidden_size: 128\n"
            "  intermediate_size: 256\n"
            "  num_heads: 4\n"
            "  vocab_size: 1000\n"
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
        )
        cfg = load_config(str(cfg_file))
        deps = _build_photon_deps(cfg)
        assert deps["safe_recgen"] is None


# ---------------------------------------------------------------------------
# TDD Cycle 8: PhotonRAGPipeline.query with PHOTON inference path
# ---------------------------------------------------------------------------


class TestPhotonQueryFlow:
    """PhotonRAGPipeline.query runs PHOTON prefill, drift, and fallback."""

    def test_query_populates_drift_metrics(self):
        from baseline_reporag.photon_pipeline import PhotonRAGPipeline
        from baseline_reporag.profiler import LatencyBreakdown, MemorySnapshot
        from baseline_reporag.pipeline import QueryResult

        import mlx.core as mx

        baseline_deps = _make_mock_deps()
        photon_deps = _make_mock_photon_deps()

        # Setup mock baseline query result
        mock_baseline_result = QueryResult(
            answer="baseline answer",
            session_id="s1",
            turn_id=1,
            cited_chunk_ids=["c1"],
            wrong_citation_indices=[],
            no_citation=False,
            latency=LatencyBreakdown(10, 20, 5, 30, 65),
            memory=MemorySnapshot(100, 50),
        )
        baseline_deps["sessions"].get_or_create.return_value = MagicMock()

        cfg = MagicMock()
        cfg.model.provider = "photon"
        cfg.hierarchy.chunk_sizes = [4, 4]

        pipeline = PhotonRAGPipeline(
            cfg=cfg, baseline_deps=baseline_deps, photon_deps=photon_deps
        )
        # Mock baseline.query
        pipeline.baseline.query = MagicMock(return_value=mock_baseline_result)

        # Mock photon_inference.session_forward
        mock_drift = MagicMock()
        mock_drift.as_dict.return_value = {"latent_cosine_drift": 0.05}
        photon_deps["photon_inference"].session_forward.return_value = (
            mx.zeros((1, 10, 1000)),
            mock_drift,
        )

        # Mock safe_recgen.evaluate
        mock_decision = MagicMock()
        mock_decision.should_fallback = False
        mock_decision.as_dict.return_value = {"should_fallback": False}
        photon_deps["safe_recgen"].evaluate.return_value = mock_decision

        result = pipeline.query("test question", session_id="s1", repo_id="r1")
        assert isinstance(result, QueryResult)
        assert result.drift_metrics is not None
        assert result.confidence is not None

    def test_query_fallback_to_baseline_on_safe_recgen(self):
        from baseline_reporag.photon_pipeline import PhotonRAGPipeline
        from baseline_reporag.profiler import LatencyBreakdown, MemorySnapshot
        from baseline_reporag.pipeline import QueryResult

        import mlx.core as mx

        baseline_deps = _make_mock_deps()
        photon_deps = _make_mock_photon_deps()

        mock_baseline_result = QueryResult(
            answer="fallback answer",
            session_id="s1",
            turn_id=1,
            cited_chunk_ids=[],
            wrong_citation_indices=[],
            no_citation=True,
            latency=LatencyBreakdown(10, 20, 5, 30, 65),
            memory=MemorySnapshot(100, 50),
        )

        cfg = MagicMock()
        cfg.model.provider = "photon"
        cfg.hierarchy.chunk_sizes = [4, 4]

        pipeline = PhotonRAGPipeline(
            cfg=cfg, baseline_deps=baseline_deps, photon_deps=photon_deps
        )
        pipeline.baseline.query = MagicMock(return_value=mock_baseline_result)

        # Mock: safe_recgen triggers fallback
        mock_drift = MagicMock()
        mock_drift.as_dict.return_value = {"latent_cosine_drift": 0.5}
        photon_deps["photon_inference"].session_forward.return_value = (
            mx.zeros((1, 10, 1000)),
            mock_drift,
        )

        mock_decision = MagicMock()
        mock_decision.should_fallback = True
        mock_decision.actions = ["fallback_to_baseline_path"]
        mock_decision.as_dict.return_value = {
            "should_fallback": True,
            "actions": ["fallback_to_baseline_path"],
        }
        photon_deps["safe_recgen"].evaluate.return_value = mock_decision

        result = pipeline.query("security auth question", session_id="s1", repo_id="r1")
        assert result.fallback_decision is not None
        assert result.fallback_decision["should_fallback"] is True
