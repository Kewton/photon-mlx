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
        "reranker": None,
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

    def test_fallback_forces_reranker_on_follow_up(self):
        """should_fallback=True must run reranker even on follow-up turns."""
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

        reranker.rerank.assert_called_once()

    def test_fallback_skips_pruning_on_follow_up(self):
        """should_fallback=True must skip PHOTON pruning on follow-up turns."""
        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

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

        photon_deps["photon_inference"].prune_evidence.assert_not_called()

    def test_reprefill_hierarchy_resets_photon_session_state(self):
        """reprefill_hierarchy action must clear current_state/prev_state."""
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

    def test_fallback_to_baseline_path_resets_photon_session_state(self):
        """fallback_to_baseline_path action must clear PHOTON session state."""
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
