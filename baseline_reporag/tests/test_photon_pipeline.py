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
        cfg.model.max_position_embeddings = 2048

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
        cfg.model.max_position_embeddings = 2048

        result = tokenize_evidence_pack("long text", tokenizer, cfg, max_tokens=2048)
        assert result.shape[0] == 2048  # 2048 is already multiple of 16

    def test_empty_text(self):
        from baseline_reporag.photon_pipeline import tokenize_evidence_pack

        tokenizer = MagicMock()
        tokenizer.encode.return_value = []
        tokenizer.pad_token_id = 0

        cfg = MagicMock()
        cfg.hierarchy.chunk_sizes = [4, 4]
        cfg.model.max_position_embeddings = 2048

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
        cfg.model.max_position_embeddings = 2048

        result = tokenize_evidence_pack("text", tokenizer, cfg)
        assert result.shape[0] == 32  # no extra padding needed

    def test_raises_on_non_positive_max_tokens(self):
        """max_tokens <= 0 must raise ValueError (DR1-001)."""
        import pytest

        from baseline_reporag.photon_pipeline import tokenize_evidence_pack

        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(10))
        tokenizer.pad_token_id = 0

        cfg = MagicMock()
        cfg.hierarchy.chunk_sizes = [4, 4]
        cfg.model.max_position_embeddings = 2048

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            tokenize_evidence_pack("text", tokenizer, cfg, max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            tokenize_evidence_pack("text", tokenizer, cfg, max_tokens=-1)


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
        "reranker": None,
    }


def _make_mock_photon_deps():
    """Create mocked PHOTON pipeline dependencies."""
    photon_cfg = MagicMock()
    # tokenize_evidence_pack reads cfg.model.max_position_embeddings and
    # cfg.hierarchy.chunk_sizes; make both deterministic so the PHOTON
    # prefill branch exercises real arithmetic.
    photon_cfg.model.max_position_embeddings = 2048
    photon_cfg.hierarchy.chunk_sizes = [4, 4]
    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(range(32))
    tokenizer.pad_token_id = 0
    return {
        "photon_inference": MagicMock(),
        "safe_recgen": MagicMock(),
        "photon_cfg": photon_cfg,
        "tokenizer": tokenizer,
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
# Shared helpers for pipeline query tests (Issue #37+)
# ---------------------------------------------------------------------------


def _make_pruning_cfg():
    """Create a minimal Config for evidence pruning tests."""
    from baseline_reporag.config import Config

    return Config(
        {
            "model": {
                "provider": "photon",
                "model_id": "test-model",
            },
            "repo": {
                "repo_id": "test-repo",
                "repo_commit": "abc123",
            },
            "hierarchy": {
                "chunk_sizes": [4, 4],
            },
            "retrieval": {
                "lexical_top_k": 20,
                "embedding_top_k": 20,
                "fused_top_k": 16,
                "rerank_top_k": 12,
                "weights": {
                    "lexical": 0.45,
                    "embedding": 0.45,
                },
                "query_expansion": {"enabled": False},
                "graph_expansion": {"max_hops": 1, "max_nodes": 24},
                "neighborhood_expansion": {"before": 1, "after": 1},
                "file_type_boost": 0.0,
            },
            "evidence_pack": {
                "max_chunks": 16,
                "max_tokens": 16000,
            },
            "inference": {
                "evidence_pruning_enabled": True,
                "pruned_max_chunks": 8,
            },
        }
    )


def _make_pruning_cfg_disabled():
    """Config with evidence pruning disabled."""
    from baseline_reporag.config import Config

    return Config(
        {
            "model": {
                "provider": "photon",
                "model_id": "test-model",
            },
            "repo": {
                "repo_id": "test-repo",
                "repo_commit": "abc123",
            },
            "hierarchy": {
                "chunk_sizes": [4, 4],
            },
            "retrieval": {
                "lexical_top_k": 20,
                "embedding_top_k": 20,
                "fused_top_k": 16,
                "rerank_top_k": 12,
                "weights": {
                    "lexical": 0.45,
                    "embedding": 0.45,
                },
                "query_expansion": {"enabled": False},
                "graph_expansion": {"max_hops": 1, "max_nodes": 24},
                "neighborhood_expansion": {"before": 1, "after": 1},
                "file_type_boost": 0.0,
            },
            "evidence_pack": {
                "max_chunks": 16,
                "max_tokens": 16000,
            },
            "inference": {
                "evidence_pruning_enabled": False,
                "pruned_max_chunks": 8,
            },
        }
    )


def _setup_pipeline_for_pruning(cfg, *, session_turns=0):
    """Build a PhotonRAGPipeline with mocked internals for pruning tests.

    Args:
        cfg: Config object.
        session_turns: number of pre-existing turns in the session (0 = turn 1).

    Returns:
        (pipeline, baseline_deps, photon_deps, mock_session, mock_results)
    """
    import mlx.core as mx
    from baseline_reporag.photon_pipeline import PhotonRAGPipeline
    from baseline_reporag.memory.session import SessionState

    baseline_deps = _make_mock_deps()
    photon_deps = _make_mock_photon_deps()

    # Build a real SessionState to track turns
    mock_session = SessionState(
        session_id="s1",
        repo_id="test-repo",
        repo_commit="abc123",
    )
    # Add pre-existing turns
    for i in range(session_turns):
        mock_session.add_turn(f"q{i + 1}", f"a{i + 1}", [])

    baseline_deps["sessions"].get_or_create.return_value = mock_session

    # Mock hybrid_search to return some results
    mock_results = []
    for i in range(16):
        r = MagicMock()
        r.chunk_id = f"chunk_{i}"
        r.score = 1.0 - i * 0.05
        mock_results.append(r)

    # Mock PHOTON inference
    mock_drift = MagicMock()
    mock_drift.as_dict.return_value = {"latent_cosine_drift": 0.05}
    photon_deps["photon_inference"].session_forward.return_value = (
        mx.zeros((1, 16, 1000)),
        mock_drift,
    )

    # Mock safe_recgen
    mock_decision = MagicMock()
    mock_decision.should_fallback = False
    mock_decision.as_dict.return_value = {"should_fallback": False}
    photon_deps["safe_recgen"].evaluate.return_value = mock_decision

    pipeline = PhotonRAGPipeline(
        cfg=cfg, baseline_deps=baseline_deps, photon_deps=photon_deps
    )

    return pipeline, baseline_deps, photon_deps, mock_session, mock_results


# ---------------------------------------------------------------------------
# TDD Cycle 8: PhotonRAGPipeline.query with PHOTON inference path
# ---------------------------------------------------------------------------


class TestPhotonQueryFlow:
    """PhotonRAGPipeline.query runs PHOTON prefill, drift, and fallback."""

    def test_query_populates_drift_metrics(self):
        from baseline_reporag.pipeline import QueryResult
        from baseline_reporag.ingestion.chunker import Chunk

        cfg = _make_pruning_cfg_disabled()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=0)
        )

        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                repo_id="test-repo",
                repo_commit="abc123",
                rel_path=f"file{i}.py",
                language="python",
                start_line=1,
                end_line=10,
                content=f"def func_{i}(): pass",
                symbols=[f"func_{i}"],
                section_header="",
                file_header="",
            )
            for i in range(16)
        ]

        def mock_get_many(ids):
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many

        expanded_ids = [f"chunk_{i}" for i in range(16)]

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            result = pipeline.query(
                "test question", session_id="s1", repo_id="test-repo"
            )

        assert isinstance(result, QueryResult)
        assert result.drift_metrics is not None
        assert result.confidence is not None

    def test_query_fallback_to_baseline_on_safe_recgen(self):
        from baseline_reporag.ingestion.chunker import Chunk

        import mlx.core as mx

        cfg = _make_pruning_cfg_disabled()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=0)
        )

        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                repo_id="test-repo",
                repo_commit="abc123",
                rel_path=f"file{i}.py",
                language="python",
                start_line=1,
                end_line=10,
                content=f"def func_{i}(): pass",
                symbols=[f"func_{i}"],
                section_header="",
                file_header="",
            )
            for i in range(16)
        ]

        def mock_get_many(ids):
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many

        expanded_ids = [f"chunk_{i}" for i in range(16)]

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

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "fallback answer [C:1]"
            result = pipeline.query(
                "security auth question", session_id="s1", repo_id="test-repo"
            )

        assert result.fallback_decision is not None
        assert result.fallback_decision["should_fallback"] is True


# ---------------------------------------------------------------------------
# Issue #58: PHOTON prefill receives question + evidence text
# ---------------------------------------------------------------------------


class TestPhotonInputContainsQuestionPlusEvidence:
    """session_forward must receive more tokens than the question alone.

    After Issue #58, PhotonRAGPipeline.query concatenates the question and
    the evidence pack text before running the PHOTON prefill.  The resulting
    ``input_ids`` passed to ``session_forward`` therefore must be strictly
    larger than what the question alone would produce; this test exercises
    that contract with a controlled, injectable tokenizer.
    """

    def test_query_passes_question_plus_evidence_to_photon_prefill(self):
        from baseline_reporag.ingestion.chunker import Chunk
        import mlx.core as mx

        cfg = _make_pruning_cfg_disabled()
        pipeline, baseline_deps, photon_deps, _mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=0)
        )

        # Install a deterministic byte-level tokenizer so we can compare
        # "question only" vs "question + evidence" token counts reliably.
        class _ByteTokenizer:
            vocab_size = 256
            pad_token_id = 0

            def encode(self, text):
                return list(text.encode("utf-8"))

        pipeline.tokenizer = _ByteTokenizer()
        pipeline.photon_cfg = MagicMock()
        pipeline.photon_cfg.model.max_position_embeddings = 4096
        pipeline.photon_cfg.hierarchy.chunk_sizes = [4, 4]

        question = "What does func_0 return?"

        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                repo_id="test-repo",
                repo_commit="abc123",
                rel_path=f"file{i}.py",
                language="python",
                start_line=1,
                end_line=10,
                content=(
                    f"def func_{i}(x):\n    # evidence body with some bytes\n"
                    f"    return x + {i}"
                ),
                symbols=[f"func_{i}"],
                section_header="",
                file_header="",
            )
            for i in range(4)
        ]

        def mock_get_many(ids):
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many
        expanded_ids = [f"chunk_{i}" for i in range(4)]

        mock_drift = MagicMock()
        mock_drift.as_dict.return_value = {"latent_cosine_drift": 0.02}
        photon_deps["photon_inference"].session_forward.return_value = (
            mx.zeros((1, 16, 256)),
            mock_drift,
        )

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "answer [C:1]"
            pipeline.query(question, session_id="s1", repo_id="test-repo")

        # session_forward must have been called with the concatenated
        # question + evidence input.  Raw byte-length comparisons are unsafe
        # because ``tokenize_evidence_pack`` pads to a chunk-aligned length
        # (CB-003); a question-only regression would still pad up (e.g. a
        # 24-byte question with chunk_sizes=[4,4] pads to 32 tokens) and slip
        # through such an assertion.  Instead compare against the actual
        # chunk-aligned length that ``tokenize_evidence_pack(question, ...)``
        # would produce under the same config.
        from baseline_reporag.photon_pipeline import tokenize_evidence_pack

        call_args = photon_deps["photon_inference"].session_forward.call_args
        assert call_args is not None, "session_forward was not invoked"
        input_ids = (
            call_args.args[0] if call_args.args else call_args.kwargs["input_ids"]
        )
        total_tokens = int(input_ids.shape[-1])

        question_only_tokens = int(
            tokenize_evidence_pack(
                question,
                pipeline.tokenizer,
                pipeline.photon_cfg,
            ).shape[-1]
        )
        assert total_tokens > question_only_tokens, (
            f"input_ids ({total_tokens} tokens) must exceed the chunk-aligned "
            f"question-only length ({question_only_tokens} tokens) because "
            "the evidence pack must be concatenated in before PHOTON prefill."
        )
        # The total length must also be a multiple of prod(chunk_sizes)=16,
        # which is a structural invariant of tokenize_evidence_pack.
        assert total_tokens % 16 == 0, (
            f"input_ids length {total_tokens} must be aligned to prod(chunk_sizes)=16."
        )


# ---------------------------------------------------------------------------
# TDD Cycle 9: Evidence pruning (Issue #37)
# ---------------------------------------------------------------------------


class TestEvidencePruningTurn1:
    """Turn 1 uses full evidence — no pruning applied."""

    def test_turn1_no_pruning(self):
        """On turn 1 (no prior turns), all chunks should be used."""
        from baseline_reporag.ingestion.chunker import Chunk

        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=0)
        )

        # Create chunk objects for the store
        chunks = []
        for i in range(16):
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{i}",
                    repo_id="test-repo",
                    repo_commit="abc123",
                    rel_path=f"file{i}.py",
                    language="python",
                    start_line=1,
                    end_line=10,
                    content=f"def func_{i}(): pass",
                    symbols=[f"func_{i}"],
                    section_header="",
                    file_header="",
                )
            )

        def mock_get_many(ids):
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many

        # Mock the retrieval + graph expansion to return chunk IDs
        expanded_ids = [f"chunk_{i}" for i in range(16)]

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            # Generator returns a simple answer
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"

            result = pipeline.query("What is X?", session_id="s1", repo_id="test-repo")

        assert result.answer == "Answer [C:1]"
        # Turn 1 → no pruning, prune_evidence should NOT be called
        photon_deps["photon_inference"].prune_evidence.assert_not_called()

    def test_turn1_pruning_disabled_no_prune(self):
        """With pruning disabled, no pruning even on follow-up turns."""
        from baseline_reporag.ingestion.chunker import Chunk

        cfg = _make_pruning_cfg_disabled()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        chunks = []
        for i in range(16):
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{i}",
                    repo_id="test-repo",
                    repo_commit="abc123",
                    rel_path=f"file{i}.py",
                    language="python",
                    start_line=1,
                    end_line=10,
                    content=f"def func_{i}(): pass",
                    symbols=[f"func_{i}"],
                    section_header="",
                    file_header="",
                )
            )

        def mock_get_many(ids):
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many

        expanded_ids = [f"chunk_{i}" for i in range(16)]

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("Follow-up question?", session_id="s1", repo_id="test-repo")

        # Pruning disabled → prune_evidence should NOT be called
        photon_deps["photon_inference"].prune_evidence.assert_not_called()


class TestEvidencePruningTurn2:
    """Turn 2+ uses pruned evidence — max_chunks reduced."""

    def test_turn2_calls_prune_evidence(self):
        """On turn 2 with pruning enabled, prune_evidence should be called."""
        from baseline_reporag.ingestion.chunker import Chunk

        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        chunks = []
        for i in range(16):
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{i}",
                    repo_id="test-repo",
                    repo_commit="abc123",
                    rel_path=f"file{i}.py",
                    language="python",
                    start_line=1,
                    end_line=10,
                    content=f"def func_{i}(): pass",
                    symbols=[f"func_{i}"],
                    section_header="",
                    file_header="",
                )
            )

        call_log = {"get_many_calls": []}

        def mock_get_many(ids):
            call_log["get_many_calls"].append(list(ids))
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many

        expanded_ids = [f"chunk_{i}" for i in range(16)]

        # prune_evidence returns top 8 indices
        photon_deps["photon_inference"].prune_evidence.return_value = list(range(8))

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Pruned answer [C:1]"
            pipeline.query("Follow-up question?", session_id="s1", repo_id="test-repo")

        # prune_evidence should have been called
        photon_deps["photon_inference"].prune_evidence.assert_called_once()
        call_kwargs = photon_deps["photon_inference"].prune_evidence.call_args
        assert call_kwargs.kwargs.get("max_chunks") == 8 or (
            len(call_kwargs.args) >= 4 and call_kwargs.args[3] == 8
        )

        # The build_evidence_pack call should use only the 8 pruned chunk IDs
        # (the second get_many call is from build_evidence_pack with pruned IDs)
        assert len(call_log["get_many_calls"]) >= 2
        pruned_ids_to_build = call_log["get_many_calls"][-1]
        assert len(pruned_ids_to_build) <= 8

    def test_turn2_result_has_correct_answer(self):
        """Turn 2 with pruning still returns a valid QueryResult."""
        from baseline_reporag.ingestion.chunker import Chunk

        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        chunks = []
        for i in range(16):
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{i}",
                    repo_id="test-repo",
                    repo_commit="abc123",
                    rel_path=f"file{i}.py",
                    language="python",
                    start_line=1,
                    end_line=10,
                    content=f"def func_{i}(): pass",
                    symbols=[f"func_{i}"],
                    section_header="",
                    file_header="",
                )
            )

        def mock_get_many(ids):
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many

        expanded_ids = [f"chunk_{i}" for i in range(16)]
        photon_deps["photon_inference"].prune_evidence.return_value = [0, 2, 4, 6]

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Pruned [C:1]"
            result = pipeline.query("Follow-up?", session_id="s1", repo_id="test-repo")

        from baseline_reporag.pipeline import QueryResult

        assert isinstance(result, QueryResult)
        assert result.drift_metrics is not None
        assert result.confidence is not None


# ---------------------------------------------------------------------------
# TDD Cycle 10: Safe RecGen fallback integration (Issue #57)
# ---------------------------------------------------------------------------


def _make_fallback_chunks_and_store(baseline_deps, n: int = 16):
    """Populate the mock store with n chunks and return expanded_ids."""
    from baseline_reporag.ingestion.chunker import Chunk

    chunks = [
        Chunk(
            chunk_id=f"chunk_{i}",
            repo_id="test-repo",
            repo_commit="abc123",
            rel_path=f"file{i}.py",
            language="python",
            start_line=1,
            end_line=10,
            content=f"def func_{i}(): pass",
            symbols=[f"func_{i}"],
            section_header="",
            file_header="",
        )
        for i in range(n)
    ]

    def mock_get_many(ids):
        by_id = {c.chunk_id: c for c in chunks}
        return [by_id[cid] for cid in ids if cid in by_id]

    baseline_deps["store"].get_many.side_effect = mock_get_many
    return [f"chunk_{i}" for i in range(n)]


def _configure_fallback_decision(photon_deps, *, should_fallback: bool, actions=None):
    """Configure safe_recgen.evaluate to return a fallback decision."""
    mock_decision = MagicMock()
    mock_decision.should_fallback = should_fallback
    mock_decision.actions = list(actions or [])
    mock_decision.as_dict.return_value = {
        "should_fallback": should_fallback,
        "actions": list(actions or []),
        "reasons": [],
    }
    photon_deps["safe_recgen"].evaluate.return_value = mock_decision


class TestSafeRecGenFallbackIntegration:
    """fallback_dict from safe_recgen must steer reranker / pruning / session state."""

    def test_fallback_does_not_force_reranker_on_follow_up(self):
        """After Issue #58 the Safe RecGen decision is computed after the
        evidence pack is built, so a current-turn fallback can no longer gate
        reranking within the same turn.  Reranker therefore does NOT run on
        follow-up turns, regardless of the fallback decision; reranking
        quality on the fallback path is covered by the baseline pipeline."""
        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        reranker = MagicMock()
        reranker.rerank.return_value = mock_results
        pipeline.baseline.reranker = reranker

        expanded_ids = _make_fallback_chunks_and_store(baseline_deps)
        _configure_fallback_decision(
            photon_deps,
            should_fallback=True,
            actions=["re_retrieve", "strengthen_local_refresh"],
        )

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("security question", session_id="s1", repo_id="test-repo")

        # New contract: follow-up turns skip the reranker; fallback cannot
        # influence that decision from within the same turn.
        reranker.rerank.assert_not_called()

    def test_fallback_does_not_skip_pruning_on_follow_up(self):
        """After Issue #58 prune_evidence runs before the PHOTON prefill and
        therefore before Safe RecGen evaluation; the current-turn fallback
        decision can no longer skip pruning.  Pruning always runs on
        follow-up turns when enabled."""
        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )
        photon_deps["photon_inference"].prune_evidence.return_value = list(range(8))

        expanded_ids = _make_fallback_chunks_and_store(baseline_deps)
        _configure_fallback_decision(
            photon_deps,
            should_fallback=True,
            actions=["fallback_to_baseline_path"],
        )

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("delete user account", session_id="s1", repo_id="test-repo")

        photon_deps["photon_inference"].prune_evidence.assert_called_once()

    def test_reprefill_hierarchy_resets_photon_session_state(self):
        """reprefill_hierarchy action must clear current_state/prev_state
        and prev_logits (Codex CB-004: stale logits leak drift otherwise)."""
        from photon_mlx.session import HierarchicalState, PhotonSessionState

        import mlx.core as mx

        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        real_state = PhotonSessionState("s1", "test-repo", "abc123")
        real_state.current_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.prev_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.prev_logits = mx.zeros((1, 4, 16))
        photon_deps["photon_inference"]._sessions = {"s1": real_state}

        expanded_ids = _make_fallback_chunks_and_store(baseline_deps)
        _configure_fallback_decision(
            photon_deps,
            should_fallback=True,
            actions=["reprefill_hierarchy", "re_retrieve"],
        )

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("topic shifted", session_id="s1", repo_id="test-repo")

        assert real_state.current_state is None
        assert real_state.prev_state is None
        assert real_state.prev_logits is None

    def test_fallback_to_baseline_path_resets_photon_session_state(self):
        """fallback_to_baseline_path action must clear PHOTON session state
        including prev_logits (Codex CB-004: stale logits leak drift)."""
        from photon_mlx.session import HierarchicalState, PhotonSessionState

        import mlx.core as mx

        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        real_state = PhotonSessionState("s1", "test-repo", "abc123")
        real_state.current_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.prev_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.prev_logits = mx.zeros((1, 4, 16))
        photon_deps["photon_inference"]._sessions = {"s1": real_state}

        expanded_ids = _make_fallback_chunks_and_store(baseline_deps)
        _configure_fallback_decision(
            photon_deps,
            should_fallback=True,
            actions=["fallback_to_baseline_path", "strengthen_local_refresh"],
        )

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("security audit", session_id="s1", repo_id="test-repo")

        assert real_state.current_state is None
        assert real_state.prev_state is None
        assert real_state.prev_logits is None

    def test_normal_follow_up_still_runs_pruning(self):
        """should_fallback=False on follow-up must still prune (regression guard)."""
        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        expanded_ids = _make_fallback_chunks_and_store(baseline_deps)
        _configure_fallback_decision(photon_deps, should_fallback=False, actions=[])
        photon_deps["photon_inference"].prune_evidence.return_value = list(range(8))

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("follow-up", session_id="s1", repo_id="test-repo")

        photon_deps["photon_inference"].prune_evidence.assert_called_once()

    def test_normal_follow_up_does_not_reset_photon_session_state(self):
        """Without fallback, PHOTON session state must be preserved across turns."""
        from photon_mlx.session import HierarchicalState, PhotonSessionState

        import mlx.core as mx

        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        real_state = PhotonSessionState("s1", "test-repo", "abc123")
        real_state.current_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.prev_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        photon_deps["photon_inference"]._sessions = {"s1": real_state}

        expanded_ids = _make_fallback_chunks_and_store(baseline_deps)
        _configure_fallback_decision(photon_deps, should_fallback=False, actions=[])
        photon_deps["photon_inference"].prune_evidence.return_value = list(range(8))

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("follow-up", session_id="s1", repo_id="test-repo")

        assert real_state.current_state is not None
        assert real_state.prev_state is not None


# ---------------------------------------------------------------------------
# Issue #58 Codex review fixes: fail-closed regression tests
# ---------------------------------------------------------------------------


class TestTokenizeEvidencePackFailureFailsClosed:
    """CB-001: when tokenize_evidence_pack raises, PHOTON session state must
    be cleared and generation must still complete via the baseline path."""

    def test_tokenization_failure_clears_photon_session_state(self):
        """A tokenizer.encode exception at question+evidence time must (a)
        continue generation via the baseline path, (b) leave drift_metrics
        unset (None), (c) drop any prior coarse state so the next turn is
        not polluted, and (d) surface the failure in the turn log."""
        import mlx.core as mx
        from baseline_reporag.ingestion.chunker import Chunk
        from photon_mlx.session import HierarchicalState, PhotonSessionState

        cfg = _make_pruning_cfg_disabled()
        pipeline, baseline_deps, photon_deps, _mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        # Seed a stale PHOTON session state that must be cleared when
        # tokenization fails.
        real_state = PhotonSessionState("s1", "test-repo", "abc123")
        real_state.current_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.prev_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        photon_deps["photon_inference"]._sessions = {"s1": real_state}

        # Install a tokenizer whose encode() always raises.
        class _BrokenTokenizer:
            vocab_size = 256
            pad_token_id = 0

            def encode(self, text):
                raise RuntimeError("simulated tokenizer failure")

        pipeline.tokenizer = _BrokenTokenizer()
        pipeline.photon_cfg = MagicMock()
        pipeline.photon_cfg.model.max_position_embeddings = 4096
        pipeline.photon_cfg.hierarchy.chunk_sizes = [4, 4]

        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                repo_id="test-repo",
                repo_commit="abc123",
                rel_path=f"file{i}.py",
                language="python",
                start_line=1,
                end_line=10,
                content=f"def func_{i}(): pass",
                symbols=[f"func_{i}"],
                section_header="",
                file_header="",
            )
            for i in range(4)
        ]

        def mock_get_many(ids):
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many
        expanded_ids = [f"chunk_{i}" for i in range(4)]

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "answer [C:1]"
            result = pipeline.query("follow-up?", session_id="s1", repo_id="test-repo")

        # (a) Generation still produced an answer via baseline path.
        assert result.answer == "answer [C:1]"
        # (b) PHOTON drift/confidence were not applied.
        assert result.drift_metrics is None
        # (c) Session state was explicitly cleared (fail-closed).
        assert real_state.current_state is None
        assert real_state.prev_state is None
        assert real_state.prev_logits is None
        # session_forward must not run on the broken tokens.
        photon_deps["photon_inference"].session_forward.assert_not_called()
        # (d) Failure surfaces in the turn log for observability.
        log_call = baseline_deps["logger"].log_turn.call_args
        log_payload = log_call.args[0] if log_call.args else log_call.kwargs["entry"]
        assert log_payload["photon_tokenization_failed"] is True


class TestPruneEvidenceFailureFailsClosed:
    """CB-002: when prune_evidence's tokenizer.encode raises, the call must
    fail-closed by returning ALL chunk indices (pruning disabled) rather
    than silently surfacing an arbitrary prefix of the input list."""

    def test_prune_evidence_encode_exception_returns_all_indices(self):
        """If tokenizer.encode raises inside prune_evidence, the function
        must abandon ranking and return every index so the caller retains
        all evidence chunks."""
        import mlx.core as mx
        from photon_mlx.inference import PhotonInference
        from photon_mlx.session import HierarchicalState, PhotonSessionState
        from torch_ref.config import PhotonConfig

        photon_cfg = PhotonConfig()
        photon_cfg.hierarchy.chunk_sizes = [4, 4]
        photon_cfg.hierarchy.levels = 2

        class _BrokenTokenizer:
            vocab_size = 256
            pad_token_id = 0

            def encode(self, text):
                raise RuntimeError("simulated encode failure")

        # Build an inference that has a prior coarse state so pruning would
        # otherwise be attempted (rather than short-circuited by "turn 1").
        mock_model = MagicMock()
        inference = PhotonInference(mock_model, photon_cfg, _BrokenTokenizer())
        session = PhotonSessionState("s1", "test-repo", "abc")
        session.current_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        inference._sessions["s1"] = session

        chunk_texts = [f"def func_{i}(): pass" for i in range(10)]
        chunk_ids = [f"chunk_{i}" for i in range(10)]
        max_chunks = 4

        indices = inference.prune_evidence(
            chunk_texts=chunk_texts,
            chunk_ids=chunk_ids,
            session_id="s1",
            max_chunks=max_chunks,
        )

        # Fail-closed: return ALL indices (not just the first max_chunks).
        # The pre-fix behaviour would have returned indices [0, 1, 2, 3]
        # (arbitrary prefix); the post-fix behaviour returns the full list.
        assert indices == list(range(10)), (
            "prune_evidence must return all chunk indices when encode fails "
            "(fail-closed, CB-002); returning a ranked prefix of the input "
            "list is a bug."
        )


# ---------------------------------------------------------------------------
# Issue #55: RoPE scaling propagates through _build_photon_deps
# ---------------------------------------------------------------------------


class TestBuildPhotonDepsWiresRopeScaling:
    """_build_photon_deps must wire rope_scaling / rope_scale_factor from the
    baseline config into the ModelConfig passed to PhotonModel (Issue #55).
    """

    def test_build_deps_wires_rope_scaling(self, tmp_path):
        """When baseline YAML specifies rope_scaling='ntk', the constructed
        ModelConfig.rope_scaling must equal 'ntk'."""
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps

        cfg_file = tmp_path / "photon_long.yaml"
        cfg_file.write_text(
            "model:\n"
            "  provider: photon\n"
            "  architecture: photon_decoder\n"
            "  base_embed_dim: 64\n"
            "  hidden_size: 128\n"
            "  intermediate_size: 256\n"
            "  num_heads: 4\n"
            "  vocab_size: 1000\n"
            "  head_dim: 32\n"
            "  max_position_embeddings: 8192\n"
            "  rope_theta: 10000000.0\n"
            "  rope_scaling: ntk\n"
            "  rope_scale_factor: 4.0\n"
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
        photon_cfg = deps["photon_cfg"]
        assert photon_cfg.model.rope_scaling == "ntk"
        assert photon_cfg.model.rope_scale_factor == 4.0
        assert photon_cfg.model.max_position_embeddings == 8192
        assert photon_cfg.model.rope_theta == 10000000.0

    def test_build_deps_defaults_when_rope_scaling_missing(self, tmp_path):
        """Without rope_scaling in YAML, ModelConfig must fall back to 'none'."""
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps

        cfg_file = tmp_path / "photon_vanilla.yaml"
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
        photon_cfg = deps["photon_cfg"]
        # Legacy fallback: ModelConfig defaults apply.
        assert photon_cfg.model.rope_scaling == "none"
        assert photon_cfg.model.rope_scale_factor == 1.0
        assert photon_cfg.model.max_position_embeddings == 2048


# ---------------------------------------------------------------------------
# Issue #56: two-pass search configuration and integration
# ---------------------------------------------------------------------------


def _make_two_pass_cfg():
    """Config with two-pass search enabled on Turn 1 (Issue #56)."""
    from baseline_reporag.config import Config

    return Config(
        {
            "model": {
                "provider": "photon",
                "model_id": "test-model",
            },
            "repo": {
                "repo_id": "test-repo",
                "repo_commit": "abc123",
            },
            "hierarchy": {
                "chunk_sizes": [4, 4],
            },
            "retrieval": {
                "lexical_top_k": 20,
                "embedding_top_k": 20,
                "fused_top_k": 16,
                "rerank_top_k": 12,
                "weights": {
                    "lexical": 0.45,
                    "embedding": 0.45,
                },
                "query_expansion": {"enabled": False},
                "graph_expansion": {"max_hops": 1, "max_nodes": 24},
                "neighborhood_expansion": {"before": 1, "after": 1},
                "file_type_boost": 0.0,
                "two_pass_search": {
                    "enabled": True,
                    "pass1_top_k": 64,
                    "pass2_top_k": 16,
                },
            },
            "evidence_pack": {
                "max_chunks": 16,
                "max_tokens": 16000,
            },
            "inference": {
                "evidence_pruning_enabled": False,
                "pruned_max_chunks": 8,
            },
        }
    )


class TestTwoPassSearchPipeline:
    """pipeline.query integration with retrieval.two_pass_search (Issue #56)."""

    def test_two_pass_enabled_runs_pass1(self):
        """Turn 1 with two_pass enabled must call prune_evidence with the
        question kwarg and record a pass1_scoring profiler phase."""
        from baseline_reporag.ingestion.chunker import Chunk

        cfg = _make_two_pass_cfg()
        pipeline, baseline_deps, photon_deps, _mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=0)
        )

        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                repo_id="test-repo",
                repo_commit="abc123",
                rel_path=f"file{i}.py",
                language="python",
                start_line=1,
                end_line=10,
                content=f"def func_{i}(): pass",
                symbols=[f"func_{i}"],
                section_header="",
                file_header="",
            )
            for i in range(64)
        ]

        def mock_get_many(ids):
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many
        # hybrid_search returns 64 candidates when Pass 1 enables effective_fused_top_k=64
        mock_results_64 = []
        for i in range(64):
            r = MagicMock()
            r.chunk_id = f"chunk_{i}"
            r.score = 1.0 - i * 0.01
            mock_results_64.append(r)
        expanded_ids = [f"chunk_{i}" for i in range(64)]

        photon_deps["photon_inference"].prune_evidence.return_value = list(range(16))

        captured_fused_top_k: dict[str, int] = {}

        def capture_hybrid(*args, **kwargs):
            captured_fused_top_k["value"] = kwargs.get("fused_top_k")
            return mock_results_64

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                side_effect=capture_hybrid,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("What is func 3?", session_id="s1", repo_id="test-repo")

        # effective_fused_top_k must be at least pass1_top_k=64 on Turn 1 Pass 1.
        assert captured_fused_top_k["value"] == 64

        # prune_evidence must be called with question kwarg and max_chunks=pass2_top_k.
        photon_deps["photon_inference"].prune_evidence.assert_called_once()
        call = photon_deps["photon_inference"].prune_evidence.call_args
        assert call.kwargs.get("question") == "What is func 3?"
        assert call.kwargs.get("max_chunks") == 16

        # pass1_scoring phase must be recorded.
        # (We can read it via photon_inference session_forward's argument profiler
        # or via inspection of the profiler; simplest is to re-introspect by
        # mocking TurnProfiler below — but since prof is created internally, we
        # instead exercise the indirect effect: prune_evidence was called, which
        # is guarded by the pass1_scoring block.)

    def test_two_pass_enabled_records_pass1_phase(self):
        """pass1_scoring phase name is registered on the profiler."""
        from baseline_reporag.ingestion.chunker import Chunk
        from baseline_reporag import photon_pipeline as pp

        cfg = _make_two_pass_cfg()
        pipeline, baseline_deps, photon_deps, _mock_session, _mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=0)
        )

        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                repo_id="test-repo",
                repo_commit="abc123",
                rel_path=f"file{i}.py",
                language="python",
                start_line=1,
                end_line=10,
                content=f"def func_{i}(): pass",
                symbols=[f"func_{i}"],
                section_header="",
                file_header="",
            )
            for i in range(64)
        ]

        def mock_get_many(ids):
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many
        mock_results_64 = []
        for i in range(64):
            r = MagicMock()
            r.chunk_id = f"chunk_{i}"
            r.score = 1.0 - i * 0.01
            mock_results_64.append(r)
        expanded_ids = [f"chunk_{i}" for i in range(64)]
        photon_deps["photon_inference"].prune_evidence.return_value = list(range(16))

        captured_profilers: list = []
        original_profiler = pp.TurnProfiler

        class _SpyProfiler(original_profiler):
            def __init__(self):
                super().__init__()
                captured_profilers.append(self)

        with (
            patch("baseline_reporag.photon_pipeline.TurnProfiler", _SpyProfiler),
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results_64,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("query?", session_id="s1", repo_id="test-repo")

        assert len(captured_profilers) == 1
        prof = captured_profilers[0]
        # pass1_scoring must be registered; evidence_pruning must NOT be on
        # a Turn 1 Pass 1 run.
        assert "pass1_scoring" in prof._watches
        assert "evidence_pruning" not in prof._watches

    def test_two_pass_disabled_uses_full_evidence(self):
        """enabled=false → Turn 1 behaves as before (no prune_evidence call)."""
        from baseline_reporag.ingestion.chunker import Chunk

        # _make_pruning_cfg has no two_pass_search section → enabled=False.
        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, _mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=0)
        )

        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                repo_id="test-repo",
                repo_commit="abc123",
                rel_path=f"file{i}.py",
                language="python",
                start_line=1,
                end_line=10,
                content=f"def func_{i}(): pass",
                symbols=[f"func_{i}"],
                section_header="",
                file_header="",
            )
            for i in range(16)
        ]

        def mock_get_many(ids):
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many
        expanded_ids = [f"chunk_{i}" for i in range(16)]

        captured_fused_top_k: dict[str, int] = {}

        def capture_hybrid(*args, **kwargs):
            captured_fused_top_k["value"] = kwargs.get("fused_top_k")
            return mock_results

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                side_effect=capture_hybrid,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("query?", session_id="s1", repo_id="test-repo")

        # fused_top_k is unchanged (=16) when two_pass is disabled.
        assert captured_fused_top_k["value"] == 16
        # prune_evidence must NOT be called on Turn 1 with two_pass disabled.
        photon_deps["photon_inference"].prune_evidence.assert_not_called()

    def test_two_pass_search_cfg_missing_section_in_pipeline(self):
        """Pipeline must work with configs that omit retrieval.two_pass_search."""
        from baseline_reporag.ingestion.chunker import Chunk

        cfg = _make_pruning_cfg()  # no two_pass_search section
        pipeline, baseline_deps, photon_deps, _mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=0)
        )

        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                repo_id="test-repo",
                repo_commit="abc123",
                rel_path=f"file{i}.py",
                language="python",
                start_line=1,
                end_line=10,
                content=f"def func_{i}(): pass",
                symbols=[f"func_{i}"],
                section_header="",
                file_header="",
            )
            for i in range(16)
        ]

        def mock_get_many(ids):
            by_id = {c.chunk_id: c for c in chunks}
            return [by_id[cid] for cid in ids if cid in by_id]

        baseline_deps["store"].get_many.side_effect = mock_get_many
        expanded_ids = [f"chunk_{i}" for i in range(16)]

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=expanded_ids,
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            result = pipeline.query("query?", session_id="s1", repo_id="test-repo")

        # Query completes without raising for configs that omit the section.
        assert result.answer == "Answer [C:1]"


class TestTwoPassSearchConfig:
    """_resolve_two_pass_search_cfg validates and defaults two-pass settings."""

    def test_two_pass_search_config_validation(self):
        """pass1_top_k < pass2_top_k must raise ValueError."""
        import pytest

        from baseline_reporag.config import Config
        from baseline_reporag.photon_pipeline import _resolve_two_pass_search_cfg

        retrieval = Config(
            {
                "two_pass_search": {
                    "enabled": True,
                    "pass1_top_k": 8,
                    "pass2_top_k": 16,
                }
            }
        )
        with pytest.raises(ValueError, match="pass1_top_k must be >= pass2_top_k"):
            _resolve_two_pass_search_cfg(
                retrieval, fused_top_k=16, evidence_max_chunks=16
            )

        # pass2_top_k < 1 also rejected
        retrieval2 = Config(
            {
                "two_pass_search": {
                    "enabled": False,
                    "pass1_top_k": 64,
                    "pass2_top_k": 0,
                }
            }
        )
        with pytest.raises(ValueError, match="pass2_top_k must be >= 1"):
            _resolve_two_pass_search_cfg(
                retrieval2, fused_top_k=16, evidence_max_chunks=16
            )

    def test_two_pass_search_cfg_missing_section(self):
        """Missing two_pass_search section defaults to disabled without errors."""
        from baseline_reporag.config import Config
        from baseline_reporag.photon_pipeline import _resolve_two_pass_search_cfg

        retrieval = Config({"fused_top_k": 16})
        enabled, p1, p2 = _resolve_two_pass_search_cfg(
            retrieval, fused_top_k=16, evidence_max_chunks=16
        )
        assert enabled is False
        assert p1 == 16
        assert p2 == 16

    def test_two_pass_search_cfg_warn_and_clamp_pass1_below_fused(self, caplog):
        """pass1_top_k < fused_top_k must warn and clamp up to fused_top_k."""
        import logging

        from baseline_reporag.config import Config
        from baseline_reporag.photon_pipeline import _resolve_two_pass_search_cfg

        retrieval = Config(
            {
                "two_pass_search": {
                    "enabled": True,
                    "pass1_top_k": 8,
                    "pass2_top_k": 4,
                }
            }
        )
        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            enabled, p1, p2 = _resolve_two_pass_search_cfg(
                retrieval, fused_top_k=16, evidence_max_chunks=8
            )
        assert enabled is True
        assert p1 == 16  # clamped up
        assert p2 == 4
        assert any("clamping pass1_top_k" in rec.message for rec in caplog.records)
