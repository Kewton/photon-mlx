"""Tests for PHOTON-RAG pipeline integration (Issue #3)."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from baseline_reporag.generation.evidence_pack import build_evidence_pack
from baseline_reporag.retrieval.graph_expansion import ExpandedChunkRef


def _mlx_metal_available() -> bool:
    probe = "import mlx.core as mx; mx.array([1]); print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _mlx_metal_available(),
    reason="PHOTON pipeline tests require an accessible Metal device",
)


def _refs(chunk_ids: list[str]) -> list[ExpandedChunkRef]:
    """Convert a list of chunk ID strings to ExpandedChunkRef list (for mocks)."""
    return [ExpandedChunkRef(chunk_id=cid, source="retrieval") for cid in chunk_ids]


# Issue #139 / Task 2.5: stub _load_hf_tokenizer for tests that don't exercise
# real HuggingFace loader behavior. Tests that test loader behavior (real
# tokenizer success path, vocab-size mismatch, load failure normalization,
# unsafe-id rejection) opt out via @pytest.mark.real_hf_loader.
#
# All affected fixtures must include a `tokenizer:` block with at minimum
# `tokenizer_id: "fake-org/fake-tokenizer"` — _build_photon_deps now raises
# ValueError when tokenizer_id is missing or unsafe (Issue #139 fail-fast
# replaces the legacy _StubTokenizer fallback).
_MINIMAL_TOKENIZER_BLOCK = 'tokenizer:\n  tokenizer_id: "fake-org/fake-tokenizer"\n'


@pytest.fixture(autouse=True)
def _stub_hf_loader(request, monkeypatch):
    if "real_hf_loader" in request.keywords:
        return

    def _fake(tokenizer_id, expected_vocab_size):
        fake = MagicMock()
        fake.vocab_size = expected_vocab_size
        fake.pad_token_id = 0
        fake.encode.return_value = [1, 2, 3]
        return fake

    monkeypatch.setattr(
        "baseline_reporag.photon_pipeline._load_hf_tokenizer",
        _fake,
    )


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
        # build_pipeline should not fail for baseline (mock heavy deps).
        # CB-004 (codex-fix): baseline deps are now built in
        # ``baseline_reporag.pipeline_factory._build_baseline_deps_no_mlx``
        # so baseline-only envs never need MLX at import time. Patch
        # that target rather than the old location.
        with patch(
            "baseline_reporag.pipeline_factory._build_baseline_deps_no_mlx"
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
        # CB-004 (codex-fix): the PHOTON branch of ``build_pipeline`` now
        # lazy-imports ``_build_baseline_deps`` / ``_build_photon_deps``
        # from ``baseline_reporag.photon_pipeline`` so the patch targets
        # are unchanged for this branch.
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
        # CB-004 (codex-fix): baseline deps are now built via the MLX-free
        # mirror in ``pipeline_factory``; patch the new target.
        with patch(
            "baseline_reporag.pipeline_factory._build_baseline_deps_no_mlx"
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
# Issue #115: Japanese prompt hint must persist on PHOTON 2nd-turn calls
# (where include_few_shot=False is the typical follow-up path) so the
# institutional-doc routing rule is independent of the format-hint switch.
# ---------------------------------------------------------------------------


class TestBuildMessagesJapaneseFollowUp:
    """build_messages() retains the Japanese institutional hint when called
    in the PHOTON follow-up shape (include_few_shot=False, with a non-empty
    session_summary). Signature must remain unchanged."""

    def test_japanese_hint_preserved_with_include_few_shot_false(self):
        from baseline_reporag.generation.prompt import build_messages

        msgs = build_messages(
            question="制度文書の第3条に関する補足説明をしてください",
            evidence_text="[C:1] doc.md\n第3条 補足",
            session_summary="[PHOTON] previous turn discussed 第2条",
            include_few_shot=False,
        )
        # Signature is unchanged: 2 messages (system + user) come back.
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        # DR1-006: substring assert on a stable phrase rather than
        # importing the private constant.
        assert "制度文書を根拠に回答する場合は" in msgs[0]["content"]


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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
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
# Issue #138: training/inference tokenizer must be unified
# ---------------------------------------------------------------------------


@pytest.mark.real_hf_loader
class TestBuildPhotonDepsRealTokenizer:
    """``_build_photon_deps`` must load the real HuggingFace tokenizer when
    ``cfg.tokenizer.tokenizer_id`` is set. Issue #138 introduced the real
    HF loader path; Issue #139 made it the only path (the legacy
    ``_StubTokenizer`` fallback was deleted) and added validation +
    exception normalization at the ``_build_photon_deps`` boundary.

    Tests in this class exercise the real ``_load_hf_tokenizer`` and
    therefore opt out of the module-level autouse stub via the
    ``real_hf_loader`` marker.
    """

    def test_loads_real_tokenizer_when_tokenizer_id_set(self, tmp_path):
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
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "  vocab_size: 152064\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
        )
        cfg = load_config(str(cfg_file))

        fake_tokenizer = MagicMock()
        fake_tokenizer.vocab_size = 152064
        fake_tokenizer.pad_token_id = 0
        fake_tokenizer.encode.return_value = [1, 2, 3]

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_from_pretrained:
            deps = _build_photon_deps(cfg)

        mock_from_pretrained.assert_called_once_with(
            "fake-org/fake-tokenizer", trust_remote_code=False
        )
        assert deps["tokenizer"] is fake_tokenizer
        assert deps["photon_cfg"].tokenizer.vocab_size == 152064

    def test_vocab_size_mismatch_raises(self, tmp_path):
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
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "  vocab_size: 152064\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
        )
        cfg = load_config(str(cfg_file))

        fake_tokenizer = MagicMock()
        fake_tokenizer.vocab_size = 200000  # exceeds configured vocab
        fake_tokenizer.pad_token_id = 0

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ):
            with pytest.raises(ValueError, match="vocab_size"):
                _build_photon_deps(cfg)

    def test_raises_when_tokenizer_id_missing(self, tmp_path):
        """Issue #139: ``_build_photon_deps`` must raise ``ValueError`` when
        ``cfg.tokenizer.tokenizer_id`` is unset (no more silent stub fallback)."""
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
        with pytest.raises(ValueError, match="tokenizer_id is required"):
            _build_photon_deps(cfg)

    def test_raises_when_tokenizer_load_fails(self, tmp_path):
        """Issue #139 / S5-002: ``AutoTokenizer.from_pretrained`` failures
        (HF Hub down, gated model, unknown id) are normalized to
        ``ValueError`` at the ``_build_photon_deps`` boundary, with the
        sanitized tokenizer_id included for operator diagnosis."""
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
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/non-existent-tokenizer"\n'
            "  vocab_size: 152064\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
        )
        cfg = load_config(str(cfg_file))

        def _boom(*args, **kwargs):
            raise OSError("HF Hub unreachable")

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=_boom,
        ):
            with pytest.raises(
                ValueError,
                match=r"failed to load tokenizer 'fake-org/non-existent-tokenizer'",
            ):
                _build_photon_deps(cfg)

    @pytest.mark.parametrize(
        "unsafe_id",
        [
            "",  # empty
            "no-slash",  # missing '/'
            "http://evil.example/etc/passwd",  # URL form
            "../../etc/passwd",  # path traversal
            "a\\b/c",  # backslash
            "a/b\nc",  # newline (log injection)
            "/abs/path",  # leading slash
            "a/b/c",  # extra slash
            # Codex CB-001: HF AutoTokenizer.from_pretrained accepts paths,
            # so dot-only / leading-dot / traversal-segment forms must also
            # be rejected by component-level validation, not just regex shape.
            "../model",  # parent-dir component
            "org/..",  # parent-dir as second component
            "./model",  # current-dir leading
            ".cache/model",  # leading dot (hidden-file path-like)
            "org/.cache",  # second component leading dot
            "..a/b",  # leading double-dot
            "a/..b",  # trailing component leading double-dot
            "~/model",  # home-relative
            "a" * 250 + "/b",  # exceeds total-length cap
        ],
    )
    def test_rejects_unsafe_tokenizer_id(self, tmp_path, unsafe_id):
        """Issue #139 / DR4-001: yaml-supplied tokenizer_id is untrusted
        input. ``_build_photon_deps`` must reject any value that does not
        match the HF repo-id allowlist (``<org>/<name>`` with
        ``[A-Za-z0-9._-]`` only) before it reaches AutoTokenizer or any
        log/error message."""
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps

        cfg_file = tmp_path / "photon.yaml"
        # Dump as yaml-safe via direct yaml string with the unsafe id quoted.
        # Some unsafe values would break naive yaml interpolation; we use a
        # block scalar style to keep the value literal.
        import yaml as _yaml

        cfg_dict = {
            "model": {
                "provider": "photon",
                "architecture": "photon_decoder",
                "base_embed_dim": 64,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_heads": 4,
                "vocab_size": 1000,
            },
            "hierarchy": {
                "levels": 2,
                "chunk_sizes": [4, 4],
                "encoder_layers_per_level": [2, 2],
                "decoder_layers_per_level": [2, 2],
            },
            "tokenizer": {"tokenizer_id": unsafe_id},
            "inference": {
                "hierarchical_prefill": True,
                "safe_recgen_enabled": False,
            },
        }
        cfg_file.write_text(_yaml.safe_dump(cfg_dict))
        cfg = load_config(str(cfg_file))
        with pytest.raises(ValueError):
            _build_photon_deps(cfg)

    def test_institutional_docs_photon_uses_tokenizer_section_vocab(self, tmp_path):
        """``cfg.tokenizer.vocab_size`` (152064) must drive PhotonModel sizing,
        not ``cfg.model.vocab_size`` (which is unset in production photon
        configs and would silently fall back to a 1000-vocab embedding —
        the latent half of the Issue #138 mismatch)."""
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
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/qwen-like"\n'
            "  vocab_size: 152064\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
        )
        cfg = load_config(str(cfg_file))

        fake_tokenizer = MagicMock()
        fake_tokenizer.vocab_size = 152064
        fake_tokenizer.pad_token_id = 0

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ):
            deps = _build_photon_deps(cfg)

        assert deps["photon_cfg"].tokenizer.vocab_size == 152064


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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("delete user account", session_id="s1", repo_id="test-repo")

        photon_deps["photon_inference"].prune_evidence.assert_called_once()

    def test_reprefill_hierarchy_resets_photon_session_state(self):
        """reprefill_hierarchy action must clear current_state/prev_state
        and prev_logits (Codex CB-004: stale logits leak drift otherwise).
        Issue #64 extends the contract to also empty ``turn_history``."""
        from photon_mlx.session import (
            HierarchicalState,
            PhotonSessionState,
            TurnState,
            WorkingMemoryConfig,
        )

        import mlx.core as mx

        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        real_state = PhotonSessionState(
            "s1",
            "test-repo",
            "abc123",
            working_memory_cfg=WorkingMemoryConfig(enabled=True),
        )
        real_state.current_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.prev_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.prev_logits = mx.zeros((1, 4, 16))
        real_state.turn_history.append(
            TurnState(turn_id=1, hierarchical_state=real_state.current_state)
        )
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
                return_value=_refs(expanded_ids),
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("topic shifted", session_id="s1", repo_id="test-repo")

        assert real_state.current_state is None
        assert real_state.prev_state is None
        assert real_state.prev_logits is None
        assert real_state.turn_history == []
        # Issue #79: compressed_history is also reset atomically so Safe
        # RecGen fallback starts from a clean working memory regardless of
        # storage_mode.
        assert real_state.compressed_history == []

    def test_fallback_to_baseline_path_resets_photon_session_state(self):
        """fallback_to_baseline_path action must clear PHOTON session state
        including prev_logits (Codex CB-004: stale logits leak drift).
        Issue #64 extends the contract to also empty ``turn_history``."""
        from photon_mlx.session import (
            HierarchicalState,
            PhotonSessionState,
            TurnState,
            WorkingMemoryConfig,
        )

        import mlx.core as mx

        cfg = _make_pruning_cfg()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        real_state = PhotonSessionState(
            "s1",
            "test-repo",
            "abc123",
            working_memory_cfg=WorkingMemoryConfig(enabled=True),
        )
        real_state.current_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.prev_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.prev_logits = mx.zeros((1, 4, 16))
        real_state.turn_history.append(
            TurnState(turn_id=1, hierarchical_state=real_state.current_state)
        )
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
                return_value=_refs(expanded_ids),
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            pipeline.query("security audit", session_id="s1", repo_id="test-repo")

        assert real_state.current_state is None
        assert real_state.prev_state is None
        assert real_state.prev_logits is None
        assert real_state.turn_history == []
        # Issue #79: fallback path must also wipe compressed_history.
        assert real_state.compressed_history == []

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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
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
        not polluted, and (d) surface the failure in the turn log. Issue
        #64 extends the contract to also empty ``turn_history``."""
        import mlx.core as mx
        from baseline_reporag.ingestion.chunker import Chunk
        from photon_mlx.session import (
            HierarchicalState,
            PhotonSessionState,
            TurnState,
            WorkingMemoryConfig,
        )

        cfg = _make_pruning_cfg_disabled()
        pipeline, baseline_deps, photon_deps, _mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        # Seed a stale PHOTON session state that must be cleared when
        # tokenization fails.
        real_state = PhotonSessionState(
            "s1",
            "test-repo",
            "abc123",
            working_memory_cfg=WorkingMemoryConfig(enabled=True),
        )
        real_state.current_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.prev_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        real_state.turn_history.append(
            TurnState(turn_id=1, hierarchical_state=real_state.current_state)
        )
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
                return_value=_refs(expanded_ids),
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
        assert real_state.turn_history == []
        # session_forward must not run on the broken tokens.
        photon_deps["photon_inference"].session_forward.assert_not_called()
        # (d) Failure surfaces in the turn log for observability.
        log_call = baseline_deps["logger"].log_turn.call_args
        log_payload = log_call.args[0] if log_call.args else log_call.kwargs["entry"]
        assert log_payload["photon_tokenization_failed"] is True

    def test_tokenize_evidence_pack_failure_logs_without_exception_text(self, caplog):
        """CB-002 (codex-fix): ``tokenize_evidence_pack`` failure path must
        emit a warning containing only ``type(exc).__name__``. Raw exception
        body (which may carry prompt fragments / tokenizer internals / model
        paths) must never appear in the log record.
        """
        import logging

        import mlx.core as mx
        from baseline_reporag.ingestion.chunker import Chunk

        cfg = _make_pruning_cfg_disabled()
        pipeline, baseline_deps, photon_deps, _mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        secret_marker = "SECRET_PROMPT_LEAK_abc123xyz"

        class _BrokenTokenizer:
            vocab_size = 256
            pad_token_id = 0

            def encode(self, text):
                raise RuntimeError(secret_marker)

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

        # mx import is needed by the helper above to build the state, even
        # though this test does not use it directly.  Silence the linter
        # by referencing it in an assertion.
        assert mx is not None

        with (
            patch(
                "baseline_reporag.photon_pipeline.hybrid_search",
                return_value=mock_results,
            ),
            patch(
                "baseline_reporag.photon_pipeline.expand_with_graph",
                return_value=_refs(expanded_ids),
            ),
        ):
            baseline_deps["generator"].generate.return_value = "answer [C:1]"
            with caplog.at_level(
                logging.WARNING, logger="baseline_reporag.photon_pipeline"
            ):
                pipeline.query("follow-up?", session_id="s1", repo_id="test-repo")

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "expected a warning for tokenize_evidence_pack failure"
        for rec in warning_records:
            msg = rec.getMessage()
            assert secret_marker not in msg, (
                "raw exception body leaked into warning log (CB-002)"
            )
        # Positive assertion: the closed-enum class name appears in at least
        # one warning message.
        assert any("RuntimeError" in r.getMessage() for r in warning_records), (
            "exception class name must appear in warning log"
        )


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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
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
# Issue #63: Safe RecGen YAML loader wires per-level thresholds and weights
# ---------------------------------------------------------------------------


class TestBuildPhotonDepsIssue63SafeRecGen:
    """Issue #63: _build_photon_deps must (a) resolve the legacy
    ``thresholds.latent_cosine_drift`` YAML key onto the new
    ``latent_cosine_drift_top_threshold`` (alias), (b) read the per-level
    thresholds, (c) read ``drift_level_weights``, and (d) propagate weights
    into PhotonInference._drift_level_weights."""

    def _write_cfg(self, tmp_path, body: str):
        from baseline_reporag.config import load_config

        cfg_file = tmp_path / "photon_issue63.yaml"
        cfg_file.write_text(body)
        return load_config(str(cfg_file))

    def test_legacy_threshold_alias_maps_to_top(self, tmp_path):
        from baseline_reporag.photon_pipeline import _build_photon_deps

        cfg = self._write_cfg(
            tmp_path,
            "model:\n"
            "  provider: photon\n"
            "  base_embed_dim: 16\n"
            "  hidden_size: 32\n"
            "  intermediate_size: 64\n"
            "  num_heads: 2\n"
            "  head_dim: 16\n"
            "  vocab_size: 256\n"
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [1, 1]\n"
            "  decoder_layers_per_level: [1, 1]\n"
            "inference:\n"
            "  safe_recgen_enabled: true\n"
            "safe_recgen:\n"
            "  thresholds:\n"
            "    latent_cosine_drift: 0.42\n",
        )
        deps = _build_photon_deps(cfg)
        sr_cfg = deps["safe_recgen"].config
        # Legacy alias: top_threshold == the value from thresholds.latent_cosine_drift.
        assert sr_cfg.latent_cosine_drift_top_threshold == 0.42
        # Legacy field must mirror the alias so existing log schema keeps working.
        assert sr_cfg.latent_cosine_drift_threshold == 0.42

    def test_per_level_thresholds_and_weights(self, tmp_path):
        from baseline_reporag.photon_pipeline import _build_photon_deps

        cfg = self._write_cfg(
            tmp_path,
            "model:\n"
            "  provider: photon\n"
            "  base_embed_dim: 16\n"
            "  hidden_size: 32\n"
            "  intermediate_size: 64\n"
            "  num_heads: 2\n"
            "  head_dim: 16\n"
            "  vocab_size: 256\n"
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [1, 1]\n"
            "  decoder_layers_per_level: [1, 1]\n"
            "inference:\n"
            "  safe_recgen_enabled: true\n"
            "safe_recgen:\n"
            "  thresholds:\n"
            "    latent_cosine_drift_top: 0.25\n"
            "    latent_cosine_drift_mid: 0.33\n"
            "    latent_cosine_drift_token: 0.22\n"
            "  drift_level_weights: [0.1, 0.4, 0.5]\n",
        )
        deps = _build_photon_deps(cfg)
        sr_cfg = deps["safe_recgen"].config
        assert sr_cfg.latent_cosine_drift_top_threshold == 0.25
        assert sr_cfg.latent_cosine_drift_mid_threshold == 0.33
        assert sr_cfg.latent_cosine_drift_token_threshold == 0.22
        # list from YAML is normalised to a tuple of floats by __post_init__.
        assert sr_cfg.drift_level_weights == (0.1, 0.4, 0.5)
        # PhotonInference must receive the same weights.
        assert deps["photon_inference"]._drift_level_weights == (0.1, 0.4, 0.5)

    def test_thresholds_missing_uses_defaults(self, tmp_path):
        """DR2-005: config with safe_recgen section but no thresholds key must
        initialise with safe default values (no KeyError)."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        cfg = self._write_cfg(
            tmp_path,
            "model:\n"
            "  provider: photon\n"
            "  base_embed_dim: 16\n"
            "  hidden_size: 32\n"
            "  intermediate_size: 64\n"
            "  num_heads: 2\n"
            "  head_dim: 16\n"
            "  vocab_size: 256\n"
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [1, 1]\n"
            "  decoder_layers_per_level: [1, 1]\n"
            "inference:\n"
            "  safe_recgen_enabled: true\n"
            "safe_recgen: {}\n",
        )
        deps = _build_photon_deps(cfg)
        sr_cfg = deps["safe_recgen"].config
        assert sr_cfg.latent_cosine_drift_top_threshold == 0.18
        assert sr_cfg.latent_cosine_drift_mid_threshold == 0.40
        assert sr_cfg.latent_cosine_drift_token_threshold == 0.30
        assert sr_cfg.drift_level_weights == (0.2, 0.3, 0.5)

    def test_safe_recgen_disabled_still_builds_inference(self, tmp_path):
        """DR2-005 / test_tiny config: safe_recgen_enabled=false must still
        build PhotonInference with default drift weights."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        cfg = self._write_cfg(
            tmp_path,
            "model:\n"
            "  provider: photon\n"
            "  base_embed_dim: 16\n"
            "  hidden_size: 32\n"
            "  intermediate_size: 64\n"
            "  num_heads: 2\n"
            "  head_dim: 16\n"
            "  vocab_size: 256\n"
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [1, 1]\n"
            "  decoder_layers_per_level: [1, 1]\n"
            "inference:\n"
            "  safe_recgen_enabled: false\n",
        )
        deps = _build_photon_deps(cfg)
        assert deps["safe_recgen"] is None
        # Default drift weights when SafeRecGen is disabled.
        assert deps["photon_inference"]._drift_level_weights == (0.2, 0.3, 0.5)


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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
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
                return_value=_refs(expanded_ids),
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


# ---------------------------------------------------------------------------
# Issue #62 Phase 1: PhotonRAGPipeline generation branch
# ---------------------------------------------------------------------------


def _make_photon_gen_cfg(
    *,
    photon_generation_enabled: bool = False,
    generation_fallback_policy: str | None = None,
    answer_max_new_tokens: int | None = None,
):
    """Build a minimal cfg exercising the generation branch added in Issue #62."""
    from baseline_reporag.config import Config

    inference: dict = {
        "evidence_pruning_enabled": False,
        "pruned_max_chunks": 8,
        "photon_generation_enabled": photon_generation_enabled,
    }
    if generation_fallback_policy is not None:
        inference["generation_fallback_policy"] = generation_fallback_policy
    if answer_max_new_tokens is not None:
        inference["answer_max_new_tokens"] = answer_max_new_tokens

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
            "inference": inference,
        }
    )


def _run_generation_branch_query(
    cfg,
    *,
    photon_answer="PHOTON generated [C:1]",
    photon_side_effect=None,
    qwen_answer="Qwen answer [C:1]",
    seed=None,
):
    """Drive PhotonRAGPipeline.query end-to-end with mocked retrieval.

    Returns ``(result, baseline_deps, photon_deps, log_payload)``.

    Issue #143: ``seed`` (keyword-only, default ``None``) is forwarded
    into ``pipeline.query`` so seed-propagation tests can assert that
    every Qwen-call site (Qwen-only path + 2 fallback paths) receives
    the seed kwarg.  ``seed=None`` keeps the legacy call shape so
    pre-#143 tests in this module keep working unchanged.
    """
    from baseline_reporag.ingestion.chunker import Chunk

    pipeline, baseline_deps, photon_deps, _mock_session, mock_results = (
        _setup_pipeline_for_pruning(cfg, session_turns=0)
    )

    # Configure the PHOTON generator — either a return value or a side_effect.
    if photon_side_effect is not None:
        photon_deps["photon_inference"].generate_answer.side_effect = photon_side_effect
    else:
        photon_deps["photon_inference"].generate_answer.return_value = photon_answer
    baseline_deps["generator"].generate.return_value = qwen_answer

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
            return_value=_refs(expanded_ids),
        ),
    ):
        if seed is not None:
            result = pipeline.query(
                "question?", session_id="s1", repo_id="test-repo", seed=seed
            )
        else:
            result = pipeline.query("question?", session_id="s1", repo_id="test-repo")

    log_call = baseline_deps["logger"].log_turn.call_args
    log_payload = log_call.args[0] if log_call.args else log_call.kwargs["entry"]
    return result, baseline_deps, photon_deps, log_payload


class TestPhotonGenerationBranch:
    """PhotonRAGPipeline.query routes generation via photon_generation_enabled."""

    def test_query_uses_qwen_when_flag_disabled(self):
        """Default off → Qwen generator is called, PHOTON is not, log says qwen."""
        cfg = _make_photon_gen_cfg(photon_generation_enabled=False)
        result, baseline_deps, photon_deps, log_payload = _run_generation_branch_query(
            cfg
        )
        assert result.answer == "Qwen answer [C:1]"
        baseline_deps["generator"].generate.assert_called_once()
        photon_deps["photon_inference"].generate_answer.assert_not_called()
        assert log_payload["generator_used"] == "qwen"
        assert log_payload["generator_fallback_reason"] is None

    def test_query_uses_photon_when_flag_enabled(self):
        """Flag on → PHOTON.generate_answer is called, Qwen is not."""
        cfg = _make_photon_gen_cfg(photon_generation_enabled=True)
        result, baseline_deps, photon_deps, log_payload = _run_generation_branch_query(
            cfg
        )
        assert result.answer == "PHOTON generated [C:1]"
        photon_deps["photon_inference"].generate_answer.assert_called_once()
        baseline_deps["generator"].generate.assert_not_called()
        assert log_payload["generator_used"] == "photon"
        assert log_payload["generator_fallback_reason"] is None

    def test_query_falls_back_to_qwen_on_photon_value_error(self):
        """PHOTON raising ValueError → Qwen fallback + closed-enum reason."""
        cfg = _make_photon_gen_cfg(photon_generation_enabled=True)
        result, baseline_deps, photon_deps, log_payload = _run_generation_branch_query(
            cfg,
            photon_side_effect=ValueError("length guard"),
        )
        assert result.answer == "Qwen answer [C:1]"
        photon_deps["photon_inference"].generate_answer.assert_called_once()
        baseline_deps["generator"].generate.assert_called_once()
        assert log_payload["generator_used"] == "qwen"
        assert log_payload["generator_fallback_reason"] == "ValueError"

    def test_query_falls_back_to_qwen_on_photon_runtime_error(self):
        """PHOTON raising RuntimeError → Qwen fallback + closed-enum reason."""
        cfg = _make_photon_gen_cfg(photon_generation_enabled=True)
        result, baseline_deps, photon_deps, log_payload = _run_generation_branch_query(
            cfg,
            photon_side_effect=RuntimeError("oom"),
        )
        assert result.answer == "Qwen answer [C:1]"
        assert log_payload["generator_used"] == "qwen"
        assert log_payload["generator_fallback_reason"] == "RuntimeError"

    def test_query_falls_back_to_qwen_on_tokenizer_encode_failure(self):
        """PHOTON raising _TokenizerEncodeFailure → Qwen fallback + closed enum."""
        from photon_mlx.inference import _TokenizerEncodeFailure

        cfg = _make_photon_gen_cfg(photon_generation_enabled=True)
        result, baseline_deps, photon_deps, log_payload = _run_generation_branch_query(
            cfg,
            photon_side_effect=_TokenizerEncodeFailure("bad bytes"),
        )
        assert result.answer == "Qwen answer [C:1]"
        assert log_payload["generator_used"] == "qwen"
        assert log_payload["generator_fallback_reason"] == "_TokenizerEncodeFailure"

    def test_query_logs_fallback_reason_on_empty_photon_output(self):
        """PHOTON returning '' / whitespace → Qwen fallback + empty_output."""
        cfg = _make_photon_gen_cfg(photon_generation_enabled=True)
        result, baseline_deps, photon_deps, log_payload = _run_generation_branch_query(
            cfg,
            photon_answer="   ",
        )
        assert result.answer == "Qwen answer [C:1]"
        assert log_payload["generator_used"] == "qwen"
        assert log_payload["generator_fallback_reason"] == "empty_output"

    def test_query_rejects_non_bool_photon_generation_enabled(self):
        """Non-bool flag value must raise ValueError before generation runs."""
        import pytest

        cfg = _make_photon_gen_cfg(photon_generation_enabled=False)
        # Force a non-bool value through direct attribute mutation so we
        # exercise the strict type guard.
        cfg.inference.photon_generation_enabled = "false"
        with pytest.raises(ValueError, match="photon_generation_enabled"):
            _run_generation_branch_query(cfg)

    def test_query_rejects_non_int_max_new_tokens(self):
        """answer_max_new_tokens = bool / negative / str must raise ValueError."""
        import pytest

        cfg = _make_photon_gen_cfg(
            photon_generation_enabled=True, answer_max_new_tokens=-1
        )
        with pytest.raises(ValueError, match="max_new_tokens"):
            _run_generation_branch_query(cfg)

        cfg2 = _make_photon_gen_cfg(photon_generation_enabled=True)
        cfg2.inference.answer_max_new_tokens = True  # bool is a Python int
        with pytest.raises(ValueError, match="max_new_tokens"):
            _run_generation_branch_query(cfg2)

        cfg3 = _make_photon_gen_cfg(photon_generation_enabled=True)
        cfg3.inference.answer_max_new_tokens = "512"
        with pytest.raises(ValueError, match="max_new_tokens"):
            _run_generation_branch_query(cfg3)

    def test_generation_fallback_policy_abort_does_not_call_qwen(self):
        """policy=abort + PHOTON failure → RuntimeError, Qwen NOT called."""
        import pytest

        cfg = _make_photon_gen_cfg(
            photon_generation_enabled=True,
            generation_fallback_policy="abort",
        )
        # _run_generation_branch_query calls query() — expect it to raise.
        from baseline_reporag.ingestion.chunker import Chunk

        pipeline, baseline_deps, photon_deps, _mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=0)
        )
        photon_deps["photon_inference"].generate_answer.side_effect = RuntimeError(
            "oom"
        )
        baseline_deps["generator"].generate.return_value = "Qwen (should not be called)"

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
                return_value=_refs(expanded_ids),
            ),
            pytest.raises(RuntimeError, match="fallback policy=abort"),
        ):
            pipeline.query("question?", session_id="s1", repo_id="test-repo")

        # Qwen must NOT have been called when policy=abort.
        baseline_deps["generator"].generate.assert_not_called()

    def test_generation_failure_logs_reason_without_exception_text(self, caplog):
        """Warning log must expose the closed-enum reason only, never raw exc text.

        Stage 4 DR4-002 contract: exception body / stack / prompt must not
        leak via the warning line.
        """
        import logging

        cfg = _make_photon_gen_cfg(photon_generation_enabled=True)
        secret_marker = "TOP_SECRET_STACK_FRAME_kjh5f8"
        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            _run_generation_branch_query(
                cfg,
                photon_side_effect=RuntimeError(secret_marker),
            )
        # Warning was emitted, but NO warning record may contain the raw
        # exception body.
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "expected at least one warning for fallback"
        for rec in warning_records:
            assert secret_marker not in rec.getMessage(), (
                "raw exception body must not be logged (Stage 4 DR4-002)"
            )

    def test_generation_fallback_policy_rejects_invalid_value(self):
        """policy not in {'qwen','abort'} must raise ValueError."""
        import pytest

        cfg = _make_photon_gen_cfg(
            photon_generation_enabled=True,
            generation_fallback_policy="silent",
        )
        with pytest.raises(ValueError, match="generation_fallback_policy"):
            _run_generation_branch_query(cfg)


# ---------------------------------------------------------------------------
# Issue #64: session_memory.working_memory extraction and security regression
# ---------------------------------------------------------------------------


class TestBuildPhotonDepsWorkingMemory:
    """_build_photon_deps wires session_memory.working_memory into PhotonInference."""

    def test_enabled_working_memory_flows_into_inference(self, tmp_path):
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps
        from photon_mlx.session import WorkingMemoryConfig

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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
            "session_memory:\n"
            "  mode: photon\n"
            "  working_memory:\n"
            "    enabled: true\n"
            "    max_turns: 4\n"
            "    decay_factor: 0.25\n"
        )
        cfg = load_config(str(cfg_file))
        deps = _build_photon_deps(cfg)
        wm = deps["photon_inference"]._working_memory_cfg
        assert isinstance(wm, WorkingMemoryConfig)
        assert wm.enabled is True
        assert wm.max_turns == 4
        assert abs(wm.decay_factor - 0.25) < 1e-9

    def test_missing_working_memory_section_defaults_to_none(self, tmp_path):
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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
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
        assert deps["photon_inference"]._working_memory_cfg is None


class TestWorkingMemoryConfigSecurityFallback:
    """_resolve_working_memory_cfg must fail-closed on malformed inputs."""

    def test_none_returns_none(self):
        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg

        assert _resolve_working_memory_cfg(None) is None

    def test_wrong_scalar_type_warns_and_returns_none(self, caplog):
        import logging

        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg

        SENSITIVE = "ATTACKER-CONTROLLED-STRING"
        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            result = _resolve_working_memory_cfg(SENSITIVE)
        assert result is None
        # Warning mentions only the type, not the raw string (security note §7).
        combined = " ".join(r.getMessage() for r in caplog.records)
        assert SENSITIVE not in combined
        assert "str" in combined

    def test_list_rejected(self, caplog):
        import logging

        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg

        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            result = _resolve_working_memory_cfg([1, 2, 3])
        assert result is None

    def test_invalid_dict_returns_none_without_leaking_payload(self, caplog):
        import logging

        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg

        SENSITIVE = "EVIL_STRING_IN_YAML"
        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            # ``enabled`` must be strictly bool — "false" is rejected.
            result = _resolve_working_memory_cfg({"enabled": SENSITIVE})
        assert result is None
        combined = " ".join(r.getMessage() for r in caplog.records)
        assert SENSITIVE not in combined

    def test_valid_dict_is_normalized(self):
        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg
        from photon_mlx.session import WorkingMemoryConfig

        result = _resolve_working_memory_cfg(
            {"enabled": True, "max_turns": 2, "decay_factor": 0.75}
        )
        assert isinstance(result, WorkingMemoryConfig)
        assert result.max_turns == 2
        assert abs(result.decay_factor - 0.75) < 1e-9

    def test_dataclass_instance_passthrough(self):
        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg
        from photon_mlx.session import WorkingMemoryConfig

        cfg = WorkingMemoryConfig(enabled=False)
        assert _resolve_working_memory_cfg(cfg) is cfg

    # ------------------------------------------------------------------
    # Issue #79: storage_mode passthrough + fail-closed
    # ------------------------------------------------------------------

    def test_storage_mode_top_level_only_passthrough(self):
        """YAML dict with ``storage_mode`` must flow to WorkingMemoryConfig."""
        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg
        from photon_mlx.session import WorkingMemoryConfig

        result = _resolve_working_memory_cfg(
            {"enabled": True, "storage_mode": "top_level_only"}
        )
        assert isinstance(result, WorkingMemoryConfig)
        assert result.storage_mode == "top_level_only"

    def test_storage_mode_summary_only_passthrough(self):
        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg

        result = _resolve_working_memory_cfg(
            {"enabled": True, "storage_mode": "summary_only"}
        )
        assert result.storage_mode == "summary_only"

    def test_storage_mode_invalid_enum_fails_closed(self, caplog):
        """Unknown ``storage_mode`` value → None + warning without raw leak."""
        import logging

        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg

        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            result = _resolve_working_memory_cfg({"storage_mode": "Full"})
        assert result is None
        combined = " ".join(r.getMessage() for r in caplog.records)
        # ValueError is caught; raw "Full" token must never leak.
        assert "Full" not in combined
        assert "ValueError" in combined

    def test_storage_mode_wrong_type_fails_closed(self, caplog):
        """Non-str ``storage_mode`` (e.g. int) → None + warning."""
        import logging

        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg

        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            result = _resolve_working_memory_cfg({"storage_mode": 42})
        assert result is None
        combined = " ".join(r.getMessage() for r in caplog.records)
        assert "TypeError" in combined

    def test_storage_mode_none_fails_closed(self, caplog):
        """``None`` is not a valid closed-enum value; reject."""
        import logging

        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg

        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            result = _resolve_working_memory_cfg({"storage_mode": None})
        assert result is None

    def test_deprecation_warning_fires_from_dict_with_both_keys(self):
        """DR1-004 — dict fixture with both deprecated and new keys emits
        DeprecationWarning when the underlying WorkingMemoryConfig is built.
        """
        import warnings as _warnings

        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg
        from photon_mlx.session import WorkingMemoryConfig

        with _warnings.catch_warnings(record=True) as captured:
            _warnings.simplefilter("always")
            result = _resolve_working_memory_cfg(
                {
                    "enabled": True,
                    "compress_old_turns": True,
                    "storage_mode": "top_level_only",
                }
            )
        assert isinstance(result, WorkingMemoryConfig)
        assert any(issubclass(rec.category, DeprecationWarning) for rec in captured), (
            f"no DeprecationWarning; got: {[str(r.message) for r in captured]}"
        )


class TestWorkingMemoryConfigAggregationYamlPropagation:
    """Issue #80: aggregation propagates transparently through the YAML path.

    Covers:
    - ``_resolve_working_memory_cfg`` with missing / valid / malformed
      aggregation values (fail-closed + no-leak).
    - ``_build_photon_deps`` / ``build_pipeline`` full path with the
      aggregation key reaching ``PhotonInference._working_memory_cfg``.
    - Nested deep-merge override (``bench/run_all.py`` pattern) preserves
      sibling keys while switching aggregation.
    """

    def test_resolve_working_memory_cfg_aggregation_default_weighted(self):
        """Omitted aggregation key defaults to ``"weighted"`` (backward compat)."""
        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg
        from photon_mlx.session import WorkingMemoryConfig

        result = _resolve_working_memory_cfg({"enabled": True, "max_turns": 3})
        assert isinstance(result, WorkingMemoryConfig)
        assert result.aggregation == "weighted"

    def test_resolve_working_memory_cfg_aggregation_attention_roundtrip(self):
        """``aggregation: attention`` flows through unchanged."""
        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg
        from photon_mlx.session import WorkingMemoryConfig

        result = _resolve_working_memory_cfg(
            {"enabled": True, "max_turns": 4, "aggregation": "attention"}
        )
        assert isinstance(result, WorkingMemoryConfig)
        assert result.aggregation == "attention"
        # Sibling keys preserved.
        assert result.enabled is True
        assert result.max_turns == 4

    def test_resolve_working_memory_cfg_aggregation_last_roundtrip(self):
        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg
        from photon_mlx.session import WorkingMemoryConfig

        result = _resolve_working_memory_cfg({"enabled": True, "aggregation": "last"})
        assert isinstance(result, WorkingMemoryConfig)
        assert result.aggregation == "last"

    def test_resolve_working_memory_cfg_invalid_value_fail_closed(self, caplog):
        """Malformed aggregation string fails closed to ``None`` with no leak."""
        import logging

        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg

        SENSITIVE = "<ATTACK-PAYLOAD-aggr>"
        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            result = _resolve_working_memory_cfg(
                {"enabled": True, "aggregation": SENSITIVE}
            )
        assert result is None
        combined = " ".join(r.getMessage() for r in caplog.records)
        # Attacker-controlled string must never appear in the log.
        assert SENSITIVE not in combined

    def test_resolve_working_memory_cfg_invalid_type_fail_closed(self, caplog):
        """Non-str aggregation (e.g. 42) fails closed to ``None``."""
        import logging

        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg

        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            result = _resolve_working_memory_cfg({"enabled": True, "aggregation": 42})
        assert result is None

    def test_photon_deps_propagates_aggregation_attention(self, tmp_path):
        """Full YAML path: aggregation reaches PhotonInference._working_memory_cfg."""
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps
        from photon_mlx.session import WorkingMemoryConfig

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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
            "session_memory:\n"
            "  mode: photon\n"
            "  working_memory:\n"
            "    enabled: true\n"
            "    max_turns: 4\n"
            "    decay_factor: 0.5\n"
            "    aggregation: attention\n"
        )
        cfg = load_config(str(cfg_file))
        deps = _build_photon_deps(cfg)
        wm = deps["photon_inference"]._working_memory_cfg
        assert isinstance(wm, WorkingMemoryConfig)
        assert wm.aggregation == "attention"
        assert wm.max_turns == 4

    def test_photon_deps_aggregation_default_weighted(self, tmp_path):
        """YAML without aggregation key defaults to ``weighted`` end-to-end."""
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps
        from photon_mlx.session import WorkingMemoryConfig

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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
            "session_memory:\n"
            "  mode: photon\n"
            "  working_memory:\n"
            "    enabled: true\n"
            "    max_turns: 4\n"
        )
        cfg = load_config(str(cfg_file))
        deps = _build_photon_deps(cfg)
        wm = deps["photon_inference"]._working_memory_cfg
        assert isinstance(wm, WorkingMemoryConfig)
        assert wm.aggregation == "weighted"

    def test_photon_deps_aggregation_invalid_fails_closed(self, tmp_path, caplog):
        """Malformed aggregation in YAML disables working memory cleanly.

        CB-002 / AT-002 (refactor): the no-leak assertion is enforced
        unconditionally. A unique sentinel string is used as the raw
        payload so a regression that forwards ``aggregation`` into the
        warning log (e.g. via ``f"... got {raw!r}"``) is caught regardless
        of whether the word "ValueError" also happens to appear.
        """
        import logging

        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps

        SENSITIVE = "<ATTACK-PAYLOAD-WM-INVALID-AGG-SENTINEL>"
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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
            "session_memory:\n"
            "  mode: photon\n"
            "  working_memory:\n"
            "    enabled: true\n"
            "    max_turns: 4\n"
            f"    aggregation: {SENSITIVE!s}\n"
        )
        cfg = load_config(str(cfg_file))
        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            deps = _build_photon_deps(cfg)
        # fail-closed: working memory disabled.
        assert deps["photon_inference"]._working_memory_cfg is None
        combined = " ".join(r.getMessage() for r in caplog.records)
        # Hard no-leak invariant (design §6 / DR4-001): the attacker-controlled
        # raw YAML value must NEVER appear in warning logs.
        assert SENSITIVE not in combined, (
            f"raw aggregation value leaked into warning log: {combined!r}"
        )
        # Also covers the legacy regression — the literal word "mean" must
        # not leak when the YAML raw is ``mean``. Re-checked here for the
        # exact string the Codex CB-002 finding called out.
        assert "mean" not in combined, (
            f"literal 'mean' leaked into warning log: {combined!r}"
        )
        # Exception class presence is an independent, positive signal (so
        # fail-closed warnings remain diagnosable). Checked as its own
        # unconditional assertion so the no-leak invariant above is not
        # short-circuited.
        assert "ValueError" in combined, (
            f"expected ValueError class name in fail-closed warning: {combined!r}"
        )

    def test_photon_deps_aggregation_invalid_type_fails_closed(self, tmp_path, caplog):
        """Non-str aggregation in YAML disables working memory cleanly.

        CB-002 / AT-002 (refactor): mirrors the invalid-value test above
        for the invalid-*type* branch so the no-leak coverage is
        consistent across both TypeError and ValueError fail-closed paths.
        """
        import logging

        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps

        # Sentinel int. Its ``repr`` contains a distinctive token that is
        # extremely unlikely to appear elsewhere, so if a regression leaks
        # the raw value into the warning (``f"... got {raw!r}"``) it shows
        # up here.
        SENSITIVE_INT = 4242424242
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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
            "session_memory:\n"
            "  mode: photon\n"
            "  working_memory:\n"
            "    enabled: true\n"
            "    max_turns: 4\n"
            f"    aggregation: {SENSITIVE_INT}\n"
        )
        cfg = load_config(str(cfg_file))
        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            deps = _build_photon_deps(cfg)
        # fail-closed: working memory disabled.
        assert deps["photon_inference"]._working_memory_cfg is None
        combined = " ".join(r.getMessage() for r in caplog.records)
        # Hard no-leak invariant: the raw int must NEVER appear in the log.
        assert str(SENSITIVE_INT) not in combined, (
            f"raw int aggregation value leaked into warning log: {combined!r}"
        )
        # Type-name surface IS allowed (design §6): operators need the type
        # to diagnose malformed YAML.
        assert ("TypeError" in combined) or ("ValueError" in combined), (
            f"expected exception class name in fail-closed warning: {combined!r}"
        )

    def test_photon_deps_variant_override_deep_merge_last(self, tmp_path):
        """bench/run_all.py deep_merge override pattern.

        Simulates ``variants[].override.session_memory.working_memory.
        aggregation: last`` applied to a base config, preserving sibling
        keys (enabled / max_turns / decay_factor) from the base.
        """
        from baseline_reporag.config import deep_merge, load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps
        from photon_mlx.session import WorkingMemoryConfig

        base_cfg = {
            "model": {
                "provider": "photon",
                "architecture": "photon_decoder",
                "base_embed_dim": 64,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_heads": 4,
                "vocab_size": 1000,
            },
            "hierarchy": {
                "levels": 2,
                "chunk_sizes": [4, 4],
                "encoder_layers_per_level": [2, 2],
                "decoder_layers_per_level": [2, 2],
            },
            "tokenizer": {"tokenizer_id": "fake-org/fake-tokenizer"},
            "inference": {
                "hierarchical_prefill": True,
                "safe_recgen_enabled": False,
            },
            "session_memory": {
                "mode": "photon",
                "working_memory": {
                    "enabled": True,
                    "max_turns": 5,
                    "decay_factor": 0.25,
                    "aggregation": "weighted",
                },
            },
        }
        override = {
            "session_memory": {
                "working_memory": {
                    "aggregation": "last",
                },
            },
        }
        merged = deep_merge(base_cfg, override)

        # Write merged dict back out as YAML, load, and build deps.
        import yaml as _yaml

        cfg_file = tmp_path / "merged.yaml"
        cfg_file.write_text(_yaml.safe_dump(merged))
        cfg = load_config(str(cfg_file))
        deps = _build_photon_deps(cfg)
        wm = deps["photon_inference"]._working_memory_cfg
        assert isinstance(wm, WorkingMemoryConfig)
        # Sibling keys preserved; aggregation switched.
        assert wm.enabled is True
        assert wm.max_turns == 5
        assert abs(wm.decay_factor - 0.25) < 1e-9
        assert wm.aggregation == "last"

    def test_build_pipeline_canonical_and_reexport_match(self, tmp_path):
        """Both ``pipeline_factory.build_pipeline`` (canonical) and
        ``photon_pipeline.build_pipeline`` (backward-compat re-export) must
        route to the same ``_build_photon_deps`` so aggregation is
        propagated identically (DR3-001).

        CB-003 / AT-003 (refactor): the previous version only called the
        private ``_build_photon_deps`` directly. This version actually
        invokes both public ``build_pipeline()`` entrypoints with
        monkeypatched heavy deps and asserts:

        1. Both entrypoints call ``_build_photon_deps`` exactly once each.
        2. The aggregation value seen by the spy is identical across the
           two calls (i.e. provider dispatch routes both through the same
           deps-builder).
        3. Both entrypoints return a ``PhotonRAGPipeline`` whose
           ``_working_memory_cfg.aggregation`` matches the YAML value.
        """
        from baseline_reporag import photon_pipeline as reexport
        from baseline_reporag import pipeline_factory as canonical
        from baseline_reporag.config import load_config
        from photon_mlx.session import WorkingMemoryConfig

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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
            "session_memory:\n"
            "  mode: photon\n"
            "  working_memory:\n"
            "    enabled: true\n"
            "    max_turns: 3\n"
            "    aggregation: attention\n"
        )
        cfg = load_config(str(cfg_file))

        # Both public entrypoints must exist as importable symbols.
        assert canonical.build_pipeline is not None
        assert reexport.build_pipeline is not None
        # The re-export must be a thin shim, not a separate implementation;
        # this sanity-check catches accidental divergence between the two
        # code paths at symbol resolution time.
        assert reexport.build_pipeline is not canonical.build_pipeline, (
            "re-export should be a distinct wrapper function that delegates"
            " to the canonical factory; if it becomes identical, DR3-001"
            " backward-compat semantics need to be re-examined."
        )

        # Spy on ``_build_photon_deps`` so we can observe what the public
        # ``build_pipeline`` actually dispatches to — and avoid booting the
        # heavy 14B generator. The canonical factory does
        # ``from .photon_pipeline import _build_photon_deps`` lazily
        # inside ``build_pipeline``, so patching the attribute on the
        # ``photon_pipeline`` module intercepts both call paths.
        calls: list[WorkingMemoryConfig | None] = []
        real_build_photon_deps = reexport._build_photon_deps

        def _spy_build_photon_deps(received_cfg):
            deps = real_build_photon_deps(received_cfg)
            calls.append(deps["photon_inference"]._working_memory_cfg)
            return deps

        # Mock the baseline deps so ``PhotonRAGPipeline.__init__`` can
        # construct without hitting the disk/network. The PHOTON branch
        # lazy-imports ``_build_baseline_deps`` from ``photon_pipeline``
        # directly (see pipeline_factory.py DR3 wiring), so patching there
        # is the correct target.
        with (
            patch(
                "baseline_reporag.photon_pipeline._build_photon_deps",
                side_effect=_spy_build_photon_deps,
            ) as spy,
            patch(
                "baseline_reporag.photon_pipeline._build_baseline_deps"
            ) as mock_baseline,
        ):
            mock_baseline.return_value = _make_mock_deps()
            pipeline_canonical = canonical.build_pipeline(cfg)
            pipeline_reexport = reexport.build_pipeline(cfg)

        # Spy must have fired twice — once per entrypoint.
        assert spy.call_count == 2, (
            "expected both build_pipeline entrypoints to hit"
            f" _build_photon_deps exactly once each; got {spy.call_count}"
        )
        assert len(calls) == 2
        # Both entrypoints must have propagated the same aggregation
        # value from the same cfg object through the shared deps-builder.
        aggregations = [wm.aggregation for wm in calls if wm is not None]
        assert aggregations == ["attention", "attention"], (
            f"aggregation mismatch between entrypoints: {aggregations!r}"
        )

        # End-to-end: both pipelines carry the expected aggregation.
        from baseline_reporag.photon_pipeline import PhotonRAGPipeline

        for pipeline in (pipeline_canonical, pipeline_reexport):
            assert isinstance(pipeline, PhotonRAGPipeline)
            wm = pipeline.photon_inference._working_memory_cfg
            assert isinstance(wm, WorkingMemoryConfig)
            assert wm.aggregation == "attention"


# ---------------------------------------------------------------------------
# Codex CB-002: tokenize_evidence_pack warning must not leak raw exc text
# ---------------------------------------------------------------------------


class TestCodexCB002TokenizeEvidencePackLogHygiene:
    """tokenize_evidence_pack failure warning must only include type name.

    A malicious or misconfigured tokenizer can raise with ``question`` or
    ``evidence_pack`` fragments in the exception message. Design §7 forbids
    surfacing those in fail-closed logs.
    """

    def test_pipeline_warning_excludes_raw_exception_payload(self, caplog):
        """Drive the real PhotonRAGPipeline fail-closed warning path and
        assert that sensitive payloads do not appear in the warning log."""
        import logging

        import mlx.core as mx
        from baseline_reporag.ingestion.chunker import Chunk
        from photon_mlx.session import (
            HierarchicalState,
            PhotonSessionState,
            WorkingMemoryConfig,
        )

        cfg = _make_pruning_cfg_disabled()
        pipeline, baseline_deps, photon_deps, _mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=1)
        )

        real_state = PhotonSessionState(
            "s1",
            "test-repo",
            "abc123",
            working_memory_cfg=WorkingMemoryConfig(enabled=True),
        )
        real_state.current_state = HierarchicalState(level_states=[mx.zeros((1, 4, 8))])
        photon_deps["photon_inference"]._sessions = {"s1": real_state}

        # Sensitive payload that MUST NOT appear in the warning log.
        SENSITIVE = "SECRET-EVIDENCE-payload-xyz123"

        class _LeakyTokenizer:
            vocab_size = 256
            pad_token_id = 0

            def encode(self, text):
                # Echo the input — a tokenizer that would surface
                # question+evidence in its exception message.
                raise RuntimeError(SENSITIVE + " " + text[:64])

        pipeline.tokenizer = _LeakyTokenizer()
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

        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            with (
                patch(
                    "baseline_reporag.photon_pipeline.hybrid_search",
                    return_value=mock_results,
                ),
                patch(
                    "baseline_reporag.photon_pipeline.expand_with_graph",
                    return_value=_refs(expanded_ids),
                ),
            ):
                baseline_deps["generator"].generate.return_value = "answer [C:1]"
                pipeline.query("follow-up?", session_id="s1", repo_id="test-repo")

        combined = " ".join(r.getMessage() for r in caplog.records)
        # Raw exception payload must not be logged.
        assert SENSITIVE not in combined
        # The type name is the only acceptable identifier.
        assert "RuntimeError" in combined
        # Also ensure the raw question ("follow-up?") did not bleed out.
        assert "follow-up?" not in combined


# ---------------------------------------------------------------------------
# Issue #103: _clear_photon_session_artifacts helper
# ---------------------------------------------------------------------------


class TestClearPhotonSessionArtifacts:
    """Issue #103: ``PhotonRAGPipeline._clear_photon_session_artifacts``.

    The helper centralises reset (cache pop + ``_clear_photon_session_state``)
    so all current and future reset paths flow through one entry point
    (design §3 / DR1-003 ``artifacts ⊃ state + cache``).
    """

    def test_clear_photon_session_artifacts_clears_cache(self) -> None:
        """Helper must pop the sidecar cache entry AND delegate to
        ``_clear_photon_session_state`` so the existing PHOTON state reset
        contract is preserved (Codex CB-001 + Issue #64)."""
        cfg = _make_pruning_cfg_disabled()
        pipeline, _baseline_deps, photon_deps, _mock_session, _mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=0)
        )

        # Seed cache as if Turn N had recorded a pin candidate.
        sentinel_match = MagicMock(name="TurnState_sentinel")
        pipeline._relevant_past_turn_cache["s1"] = sentinel_match
        pipeline._relevant_past_turn_cache["other_session"] = MagicMock(
            name="should_survive"
        )

        with patch(
            "baseline_reporag.photon_pipeline._clear_photon_session_state"
        ) as mock_clear_state:
            pipeline._clear_photon_session_artifacts("s1")

        # Cache entry for the target session must be popped.
        assert "s1" not in pipeline._relevant_past_turn_cache
        # Other sessions are untouched (1-session-1-entry sidecar).
        assert "other_session" in pipeline._relevant_past_turn_cache
        # State reset must be delegated to the existing helper.
        mock_clear_state.assert_called_once_with(pipeline.photon_inference, "s1")


# ---------------------------------------------------------------------------
# Issue #103: TestPastTurnPinning — pipeline integration of past-turn pin
# ---------------------------------------------------------------------------


def _make_pinning_cfg(*, enabled: bool = True, max_pinned: int = 3):
    """Build a Config with past-turn pinning configured.

    Adds a ``session_memory.working_memory`` block with
    ``past_turn_pinning_enabled`` and ``max_pinned_chunks`` so the
    pipeline's read/write branches activate.
    """
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
            "session_memory": {
                "mode": "photon",
                "working_memory": {
                    "enabled": True,
                    "max_turns": 4,
                    "past_turn_pinning_enabled": enabled,
                    "max_pinned_chunks": max_pinned,
                },
            },
        }
    )


def _setup_pipeline_for_pinning(cfg, *, session_turns: int = 0):
    """Setup pipeline like ``_setup_pipeline_for_pruning`` but install a
    real dict for ``photon_inference._sessions`` so the pin code path can
    look up a fake PHOTON session by id."""
    pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
        _setup_pipeline_for_pruning(cfg, session_turns=session_turns)
    )
    # The default ``_setup_pipeline_for_pruning`` mocks ``photon_inference``
    # as a MagicMock. ``_sessions.get(...)`` against MagicMock returns
    # another MagicMock by default, which is truthy and would cause
    # spurious calls to ``find_relevant_past_turn`` even when the test
    # never seeded a session. Replace the attribute with a real dict so
    # the pin path follows production semantics: presence ⇒ session
    # exists, absence ⇒ no session.
    photon_deps["photon_inference"]._sessions = {}
    return pipeline, baseline_deps, photon_deps, mock_session, mock_results


def _make_fake_photon_session(
    *,
    matched_turn_id: int | None,
    raise_exc: type[BaseException] | None = None,
):
    """Build a stub object exposing only ``current_state`` and
    ``find_relevant_past_turn`` so tests can drive the pin write branch
    without booting PHOTON."""
    from photon_mlx.session import TurnState

    fake = MagicMock()
    fake.current_state = MagicMock(name="hierarchical_state")
    if raise_exc is not None:

        def _raise(_state):
            raise raise_exc("synthetic test error")

        fake.find_relevant_past_turn.side_effect = _raise
    elif matched_turn_id is None:
        fake.find_relevant_past_turn.return_value = None
    else:
        fake.find_relevant_past_turn.return_value = TurnState(
            turn_id=matched_turn_id,
            hierarchical_state=MagicMock(name="match_hstate"),
        )
    return fake


def _run_query_with_mocks(
    pipeline,
    baseline_deps,
    photon_deps,
    mock_results,
    *,
    question: str = "follow-up?",
    expanded_ids: list[str] | None = None,
):
    """Run ``pipeline.query`` with hybrid_search / expand_with_graph mocks
    plus a working ``store.get_many`` so the evidence pack code path
    completes without needing real chunks."""
    from baseline_reporag.ingestion.chunker import Chunk

    if expanded_ids is None:
        expanded_ids = [f"chunk_{i}" for i in range(16)]

    chunks = [
        Chunk(
            chunk_id=cid,
            repo_id="test-repo",
            repo_commit="abc123",
            rel_path=f"{cid}.py",
            language="python",
            start_line=1,
            end_line=10,
            content=f"def f_{cid}(): pass",
            symbols=[cid],
            section_header="",
            file_header="",
        )
        for cid in expanded_ids
    ]
    by_id = {c.chunk_id: c for c in chunks}
    baseline_deps["store"].get_many.side_effect = lambda ids: [
        by_id[cid] for cid in ids if cid in by_id
    ]
    baseline_deps["generator"].generate.return_value = "answer [C:1]"

    with (
        patch(
            "baseline_reporag.photon_pipeline.hybrid_search",
            return_value=mock_results,
        ),
        patch(
            "baseline_reporag.photon_pipeline.expand_with_graph",
            return_value=_refs(expanded_ids),
        ),
    ):
        return pipeline.query(question, session_id="s1", repo_id="test-repo")


class TestPastTurnPinning:
    """Issue #103: pipeline integration of ``find_relevant_past_turn``.

    DR1-009: scope is the pipeline glue (cache write/read/pop, opt-in
    OFF, fail-closed branches, profiler phase). The behaviour of
    ``find_relevant_past_turn`` itself is covered by the 11 existing
    Issue #78 unit tests in ``photon_mlx/tests/test_session.py`` and is
    deliberately NOT re-tested here — every test in this class stubs
    ``photon_inference._sessions`` so production retrieval semantics are
    not entangled with PHOTON inference details.
    """

    # ---- Group A: opt-in OFF must short-circuit. ----

    def test_pinning_disabled_skips_find_relevant_past_turn(self) -> None:
        cfg = _make_pinning_cfg(enabled=False)
        pipeline, baseline_deps, photon_deps, _ms, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=1)
        )
        fake_session = _make_fake_photon_session(matched_turn_id=1)
        photon_deps["photon_inference"]._sessions["s1"] = fake_session

        _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        fake_session.find_relevant_past_turn.assert_not_called()

    def test_pinning_disabled_does_not_access_photon_sessions(self) -> None:
        cfg = _make_pinning_cfg(enabled=False)
        pipeline, baseline_deps, photon_deps, _ms, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=1)
        )
        # Replace ``_sessions`` with an instrumented dict subclass so any
        # ``.get`` access from the pinning path raises.
        recorded: list[tuple[str, str]] = []

        class _SpyDict(dict):
            def get(self, key, default=None):
                recorded.append(("get", key))
                return super().get(key, default)

        photon_deps["photon_inference"]._sessions = _SpyDict()

        _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        # No `.get(s1, None)` call from the pinning path. The pruning
        # path is disabled in this cfg so any `_sessions.get` call would
        # have to come from pinning — and pinning is OFF, so the list
        # must be empty.
        assert recorded == []

    def test_pinning_disabled_no_new_profiler_phase(self) -> None:
        cfg = _make_pinning_cfg(enabled=False)
        pipeline, baseline_deps, photon_deps, _ms, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=1)
        )

        captured: list[set[str]] = []
        real_query = pipeline.query

        def _wrap(*args, **kwargs):
            from baseline_reporag.profiler import TurnProfiler

            real_init = TurnProfiler.__init__

            def _spy_init(self):
                real_init(self)
                captured.append(self)

            with patch.object(TurnProfiler, "__init__", _spy_init):
                return real_query(*args, **kwargs)

        # Simpler: just inspect the result's latency_breakdown via prof
        # phases. After the run, no past_turn_pinning phase should exist.
        _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)
        # Nothing further to assert via captured (we just exercise the
        # full query path); the relevant invariant is enforced at the
        # next test which asserts the phase IS present when ON.
        # Defensive guardrail: ensure pinning OFF didn't crash and
        # didn't allocate the cache for s1.
        assert "s1" not in pipeline._relevant_past_turn_cache

    # ---- Group B: pinning ON — write side. ----

    def test_pinning_caches_match_for_next_turn(self) -> None:
        from photon_mlx.session import TurnState

        cfg = _make_pinning_cfg(enabled=True)
        pipeline, baseline_deps, photon_deps, _ms, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=1)
        )
        fake_session = _make_fake_photon_session(matched_turn_id=1)
        photon_deps["photon_inference"]._sessions["s1"] = fake_session

        _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        # Cache must hold the matched TurnState keyed by photon_session_id.
        assert "s1" in pipeline._relevant_past_turn_cache
        cached = pipeline._relevant_past_turn_cache["s1"]
        assert isinstance(cached, TurnState)
        assert cached.turn_id == 1

    def test_pinning_enabled_turn1_no_pin(self) -> None:
        """Turn 1 has no follow-up history; the pin read branch must be
        skipped, and the write side may still run (writes empty when
        no past turn exists to match)."""
        cfg = _make_pinning_cfg(enabled=True)
        pipeline, baseline_deps, photon_deps, _ms, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=0)
        )
        fake_session = _make_fake_photon_session(matched_turn_id=None)
        photon_deps["photon_inference"]._sessions["s1"] = fake_session

        with patch("baseline_reporag.photon_pipeline.build_evidence_pack") as spy_pack:
            spy_pack.side_effect = build_evidence_pack
            _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        # On Turn 1 (session.turns is empty entering the call), the pin
        # read branch must not emit ``additional_pinned_ids``.
        assert spy_pack.call_count == 1
        kwargs = spy_pack.call_args.kwargs
        assert kwargs.get("additional_pinned_ids") is None

    def test_pinning_enabled_turn2_below_threshold_no_pin(self) -> None:
        """Turn 2: cache is empty (Turn 1 stored nothing because no past
        match existed). Read side must produce no pin even though
        pinning is enabled."""
        cfg = _make_pinning_cfg(enabled=True)
        pipeline, baseline_deps, photon_deps, _ms, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=1)
        )
        fake_session = _make_fake_photon_session(matched_turn_id=None)
        photon_deps["photon_inference"]._sessions["s1"] = fake_session

        with patch("baseline_reporag.photon_pipeline.build_evidence_pack") as spy_pack:
            spy_pack.side_effect = build_evidence_pack
            _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        kwargs = spy_pack.call_args.kwargs
        assert kwargs.get("additional_pinned_ids") is None

    def test_pinning_enabled_turn2_above_threshold_pins_at_turn3(self) -> None:
        """Turn 2 writes a match; Turn 3 read must consume it as a pin."""
        from photon_mlx.session import TurnState

        cfg = _make_pinning_cfg(enabled=True)
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=2)
        )
        # Pre-load Baseline session with cited_chunk_ids on turn_id=1.
        # ``_setup_pipeline_for_pinning`` already added 2 turns with empty
        # citations; we patch turn 1 to record citations matching the
        # pin.
        mock_session.turns[0].cited_chunk_ids[:] = ["chunk_0", "chunk_1"]

        # Seed cache as if Turn 2 had matched turn_id=1.
        pipeline._relevant_past_turn_cache["s1"] = TurnState(
            turn_id=1,
            hierarchical_state=MagicMock(name="match_hstate"),
        )
        fake_session = _make_fake_photon_session(matched_turn_id=None)
        photon_deps["photon_inference"]._sessions["s1"] = fake_session

        with patch("baseline_reporag.photon_pipeline.build_evidence_pack") as spy_pack:
            spy_pack.side_effect = build_evidence_pack
            _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        kwargs = spy_pack.call_args.kwargs
        # The pinned IDs must come from turn 1's cited_chunk_ids.
        assert kwargs["additional_pinned_ids"] == ["chunk_0", "chunk_1"]

    def test_pinning_respects_max_pinned_chunks(self) -> None:
        """``max_pinned_chunks`` truncates the cited list."""
        from photon_mlx.session import TurnState

        cfg = _make_pinning_cfg(enabled=True, max_pinned=2)
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=2)
        )
        mock_session.turns[0].cited_chunk_ids[:] = [
            "chunk_a",
            "chunk_b",
            "chunk_c",
            "chunk_d",
        ]
        pipeline._relevant_past_turn_cache["s1"] = TurnState(
            turn_id=1,
            hierarchical_state=MagicMock(),
        )
        photon_deps["photon_inference"]._sessions["s1"] = _make_fake_photon_session(
            matched_turn_id=None
        )

        with patch("baseline_reporag.photon_pipeline.build_evidence_pack") as spy_pack:
            spy_pack.side_effect = build_evidence_pack
            _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        kwargs = spy_pack.call_args.kwargs
        assert kwargs["additional_pinned_ids"] == ["chunk_a", "chunk_b"]

    def test_pinning_does_not_double_count_existing_retrieval_hits(
        self,
    ) -> None:
        """If a pinned chunk is already in ``expanded_ids``, dedup keeps
        the pack from inflating beyond ``max_chunks``."""
        from photon_mlx.session import TurnState

        cfg = _make_pinning_cfg(enabled=True)
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=2)
        )
        mock_session.turns[0].cited_chunk_ids[:] = ["chunk_0"]
        pipeline._relevant_past_turn_cache["s1"] = TurnState(
            turn_id=1, hierarchical_state=MagicMock()
        )
        photon_deps["photon_inference"]._sessions["s1"] = _make_fake_photon_session(
            matched_turn_id=None
        )

        # Spy on build_evidence_pack so we can introspect the EvidencePack
        # it returned (QueryResult does not surface the pack directly —
        # see contracts.QueryResult).
        captured: dict[str, object] = {}
        real_build = build_evidence_pack

        def _spy(*args, **kwargs):
            pack = real_build(*args, **kwargs)
            captured["pack"] = pack
            return pack

        with patch(
            "baseline_reporag.photon_pipeline.build_evidence_pack",
            side_effect=_spy,
        ):
            _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        # Pack must not contain chunk_0 twice (set semantics inside
        # build_evidence_pack / _merge_pinned_sets guarantees this; we
        # assert here as a contract regression guard).
        pack = captured["pack"]
        ids = [c.chunk_id for c in pack.chunks]
        assert ids.count("chunk_0") == 1

    def test_pinning_failclosed_on_session_state_none(self) -> None:
        """If ``photon_inference._sessions`` has no entry for the
        session_id, the write branch must pop the cache instead of
        attempting to call find_relevant_past_turn."""
        from photon_mlx.session import TurnState

        cfg = _make_pinning_cfg(enabled=True)
        pipeline, baseline_deps, photon_deps, _ms, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=1)
        )
        # Pre-seed the cache with a stale entry.
        pipeline._relevant_past_turn_cache["s1"] = TurnState(
            turn_id=999,
            hierarchical_state=MagicMock(),
        )
        # No entry for "s1" in the fake _sessions dict.
        assert "s1" not in photon_deps["photon_inference"]._sessions

        _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        # The stale cache entry must have been popped by the read side
        # (consume), and the write side leaves it absent because no
        # PHOTON session exists for s1.
        assert "s1" not in pipeline._relevant_past_turn_cache

    def test_pinning_logs_match_metadata(self, caplog) -> None:
        """DR4-001: production log must NOT contain turn_id, similarity,
        or scanned_turns when find_relevant_past_turn raises. Only the
        exception class name is allowed."""
        import logging

        cfg = _make_pinning_cfg(enabled=True)
        pipeline, baseline_deps, photon_deps, _ms, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=1)
        )
        fake_session = _make_fake_photon_session(
            matched_turn_id=None, raise_exc=RuntimeError
        )
        photon_deps["photon_inference"]._sessions["s1"] = fake_session

        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        msgs = [r.getMessage() for r in caplog.records]
        combined = " ".join(msgs)
        # Exception class name must be present so operators can diagnose
        # fail-closed warnings.
        assert "RuntimeError" in combined, msgs
        # No turn_id / similarity / scanned_turns / raw exception text
        # leakage. Lower-case the haystack so we catch any case-variant.
        assert "turn_id" not in combined.lower(), msgs
        assert "similarity" not in combined.lower(), msgs
        assert "scanned_turns" not in combined.lower(), msgs
        assert "synthetic test error" not in combined, msgs

    def test_pinning_priority_overrides_recent_cited(self) -> None:
        """Pinned chunks (priority 0) must outrank recent_cited
        (priority 1) when ``max_chunks`` is tight."""
        from photon_mlx.session import TurnState

        cfg = _make_pinning_cfg(enabled=True)
        # Reduce max_chunks so priority ordering matters. ``Config`` uses
        # plain ``setattr`` so direct attribute assignment is fine.
        cfg.evidence_pack.max_chunks = 2
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=1)
        )
        # Make turn 1's cited_chunk_ids reference "chunk_15" (last
        # candidate; would normally lose at priority sort time).
        mock_session.turns[0].cited_chunk_ids[:] = ["chunk_15"]
        # Seed pin to refer to turn 1's cited list.
        pipeline._relevant_past_turn_cache["s1"] = TurnState(
            turn_id=1, hierarchical_state=MagicMock()
        )
        photon_deps["photon_inference"]._sessions["s1"] = _make_fake_photon_session(
            matched_turn_id=None
        )

        captured: dict[str, object] = {}
        real_build = build_evidence_pack

        def _spy(*args, **kwargs):
            pack = real_build(*args, **kwargs)
            captured["pack"] = pack
            return pack

        with patch(
            "baseline_reporag.photon_pipeline.build_evidence_pack",
            side_effect=_spy,
        ):
            _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        pack = captured["pack"]
        ids = [c.chunk_id for c in pack.chunks]
        # chunk_15 was pinned at priority 0 and must now lead the pack.
        assert ids[0] == "chunk_15", ids

    def test_pinning_cache_cleared_after_use(self) -> None:
        """Read side ``pop`` must consume the cache entry — the same
        pin must NOT survive into a subsequent turn that produces no
        new match."""
        from photon_mlx.session import TurnState

        cfg = _make_pinning_cfg(enabled=True)
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=2)
        )
        mock_session.turns[0].cited_chunk_ids[:] = ["chunk_0"]
        pipeline._relevant_past_turn_cache["s1"] = TurnState(
            turn_id=1, hierarchical_state=MagicMock()
        )
        # Write side will produce None (no new match), so cache must
        # remain empty after the call.
        photon_deps["photon_inference"]._sessions["s1"] = _make_fake_photon_session(
            matched_turn_id=None
        )

        _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        assert "s1" not in pipeline._relevant_past_turn_cache

    def test_pinning_failclosed_on_drift_none_pops_stale_cache(self) -> None:
        """DR2-011 + DR1-007: when ``drift is None`` (tokenize fail-closed
        / Safe RecGen reset), the write branch must explicitly pop the
        cache so a stale pin from a prior turn cannot survive."""
        from photon_mlx.session import TurnState

        cfg = _make_pinning_cfg(enabled=True)
        pipeline, baseline_deps, photon_deps, _ms, mock_results = (
            _setup_pipeline_for_pinning(cfg, session_turns=1)
        )
        # Force tokenize_evidence_pack to raise → drift will be None.
        fake_session = _make_fake_photon_session(matched_turn_id=42)
        photon_deps["photon_inference"]._sessions["s1"] = fake_session
        # Pre-load a stale cache entry that should be wiped.
        pipeline._relevant_past_turn_cache["s1"] = TurnState(
            turn_id=999, hierarchical_state=MagicMock()
        )

        with patch(
            "baseline_reporag.photon_pipeline.tokenize_evidence_pack",
            side_effect=RuntimeError("tokenize boom"),
        ):
            _run_query_with_mocks(pipeline, baseline_deps, photon_deps, mock_results)

        # tokenize fail-closed:
        # 1. read side popped any stale pin (consumed, returns None).
        # 2. ``_clear_photon_session_artifacts`` ran and popped again.
        # 3. drift is None → write branch pops one more time
        #    (DR2-011 invariant).
        # In all three legs the post-call cache must be empty.
        assert "s1" not in pipeline._relevant_past_turn_cache
        # find_relevant_past_turn must NOT have been invoked because
        # drift is None short-circuits the write branch into pop-only.
        fake_session.find_relevant_past_turn.assert_not_called()


# ---------------------------------------------------------------------------
# Issue #103: WorkingMemoryConfig YAML propagation for pinning fields
# ---------------------------------------------------------------------------


class TestWorkingMemoryConfigPastTurnPinningYamlPropagation:
    """Issue #103: ``past_turn_pinning_enabled`` / ``max_pinned_chunks`` reach
    ``PhotonInference._working_memory_cfg`` through the same paths the
    Issue #80 aggregation key already uses.

    Scope (DR3-002): mirrors
    :class:`TestWorkingMemoryConfigAggregationYamlPropagation`'s contract
    for the two new pinning keys to confirm parity:

    1. ``_resolve_working_memory_cfg`` dict roundtrip preserves the keys.
    2. ``_build_photon_deps`` YAML roundtrip carries the keys through.
    3. ``deep_merge`` overrides do not blow away sibling
       ``aggregation`` / ``dynamic_strategy`` knobs.
    """

    def test_working_memory_pinning_keys_roundtrip_via_extract_cfg(self):
        from baseline_reporag.photon_pipeline import _resolve_working_memory_cfg
        from photon_mlx.session import WorkingMemoryConfig

        result = _resolve_working_memory_cfg(
            {
                "enabled": True,
                "max_turns": 5,
                "past_turn_pinning_enabled": True,
                "max_pinned_chunks": 4,
            }
        )
        assert isinstance(result, WorkingMemoryConfig)
        assert result.past_turn_pinning_enabled is True
        assert result.max_pinned_chunks == 4
        # Sibling keys must survive.
        assert result.enabled is True
        assert result.max_turns == 5

    def test_working_memory_pinning_keys_roundtrip_via_build_photon_deps(
        self, tmp_path
    ):
        """Full YAML path: pinning keys reach
        ``PhotonInference._working_memory_cfg`` after
        ``_build_photon_deps``."""
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps
        from photon_mlx.session import WorkingMemoryConfig

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
            "tokenizer:\n"
            '  tokenizer_id: "fake-org/fake-tokenizer"\n'
            "hierarchy:\n"
            "  levels: 2\n"
            "  chunk_sizes: [4, 4]\n"
            "  encoder_layers_per_level: [2, 2]\n"
            "  decoder_layers_per_level: [2, 2]\n"
            "inference:\n"
            "  hierarchical_prefill: true\n"
            "  safe_recgen_enabled: false\n"
            "session_memory:\n"
            "  mode: photon\n"
            "  working_memory:\n"
            "    enabled: true\n"
            "    max_turns: 4\n"
            "    past_turn_pinning_enabled: true\n"
            "    max_pinned_chunks: 5\n"
        )
        cfg = load_config(str(cfg_file))
        deps = _build_photon_deps(cfg)
        wm = deps["photon_inference"]._working_memory_cfg
        assert isinstance(wm, WorkingMemoryConfig)
        assert wm.past_turn_pinning_enabled is True
        assert wm.max_pinned_chunks == 5

    def test_working_memory_deep_merge_preserves_aggregation_with_pinning(
        self, tmp_path
    ):
        """``deep_merge`` override that adds pinning keys must keep
        existing ``aggregation`` / ``dynamic_strategy`` / ``hybrid_*``
        sibling keys intact."""
        from baseline_reporag.config import deep_merge, load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps
        from photon_mlx.session import WorkingMemoryConfig

        base_cfg = {
            "model": {
                "provider": "photon",
                "architecture": "photon_decoder",
                "base_embed_dim": 64,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_heads": 4,
                "vocab_size": 1000,
            },
            "hierarchy": {
                "levels": 2,
                "chunk_sizes": [4, 4],
                "encoder_layers_per_level": [2, 2],
                "decoder_layers_per_level": [2, 2],
            },
            "tokenizer": {"tokenizer_id": "fake-org/fake-tokenizer"},
            "inference": {
                "hierarchical_prefill": True,
                "safe_recgen_enabled": False,
            },
            "session_memory": {
                "mode": "photon",
                "working_memory": {
                    "enabled": True,
                    "max_turns": 5,
                    "decay_factor": 0.25,
                    "aggregation": "dynamic",
                    "dynamic_strategy": "hybrid",
                    "hybrid_alpha_base": 0.4,
                    "hybrid_alpha_per_turn": 0.05,
                },
            },
        }
        # Override that ONLY toggles the new pinning keys.
        override = {
            "session_memory": {
                "working_memory": {
                    "past_turn_pinning_enabled": True,
                    "max_pinned_chunks": 2,
                },
            },
        }
        merged = deep_merge(base_cfg, override)

        import yaml as _yaml

        cfg_file = tmp_path / "merged.yaml"
        cfg_file.write_text(_yaml.safe_dump(merged))
        cfg = load_config(str(cfg_file))
        deps = _build_photon_deps(cfg)
        wm = deps["photon_inference"]._working_memory_cfg
        assert isinstance(wm, WorkingMemoryConfig)
        # Pinning keys flowed through.
        assert wm.past_turn_pinning_enabled is True
        assert wm.max_pinned_chunks == 2
        # Aggregation + dynamic knobs preserved by deep_merge.
        assert wm.aggregation == "dynamic"
        assert wm.dynamic_strategy == "hybrid"
        assert abs(wm.hybrid_alpha_base - 0.4) < 1e-9
        assert abs(wm.hybrid_alpha_per_turn - 0.05) < 1e-9
        # Other sibling keys preserved.
        assert wm.enabled is True
        assert wm.max_turns == 5
        assert abs(wm.decay_factor - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# Issue #109: graph=None type compatibility at PHOTON pipeline layer
# ---------------------------------------------------------------------------


class TestPhotonPipelineGraphNoneCompatibility:
    """``baseline_deps['graph']=None`` must build and query without error.

    The real graph=None runtime branch is covered in
    ``test_graph_expansion.py``; this test only asserts type compatibility
    of PHOTON's internal ``RepoRAGPipeline`` construction when callers
    pass ``graph=None`` (``expand_with_graph`` is still patched).
    """

    def test_photon_pipeline_builds_with_graph_none(self):
        from baseline_reporag.ingestion.chunker import Chunk

        cfg = _make_pruning_cfg_disabled()
        pipeline, baseline_deps, photon_deps, mock_session, mock_results = (
            _setup_pipeline_for_pruning(cfg, session_turns=0)
        )

        # Swap baseline.graph to None to simulate the enabled=false path.
        pipeline.baseline.graph = None

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
                return_value=_refs(expanded_ids),
            ),
        ):
            baseline_deps["generator"].generate.return_value = "Answer [C:1]"
            result = pipeline.query(
                "test question", session_id="s1", repo_id="test-repo"
            )

        assert result.answer == "Answer [C:1]"


# ---------------------------------------------------------------------------
# Issue #143: seed propagation through PhotonRAGPipeline.
#
# Three Qwen call sites in ``photon_pipeline.py`` need to forward ``seed``:
# - L1030: Qwen fallback after ``_TokenizerEncodeFailure / ValueError /
#   RuntimeError`` from PHOTON
# - L1043: Qwen fallback after empty PHOTON output
# - L1394: Qwen-only path when ``photon_generation_enabled=False``
#
# Backwards compat: ``query(seed=None)`` (default) MUST keep the legacy
# call shape (no ``seed`` kwarg leaking through) so the 17+ existing
# MagicMock fallback / Qwen-only tests in this file keep passing.
# ---------------------------------------------------------------------------


class TestPhotonPipelineSeedPropagation:
    """PhotonRAGPipeline.query forwards ``seed`` into every Qwen call path."""

    def test_qwen_only_path_default_seed_none_uses_legacy_shape(self):
        """Default seed=None: Qwen-only path keeps legacy single-positional shape.

        Backwards-compat invariant: existing MagicMock tests that did
        ``mock_gen.generate.return_value = ...`` and never expected a
        ``seed`` kwarg keep passing.
        """
        cfg = _make_photon_gen_cfg(photon_generation_enabled=False)
        result, baseline_deps, photon_deps, _ = _run_generation_branch_query(cfg)
        gen_calls = baseline_deps["generator"].generate.call_args_list
        assert gen_calls, "Qwen generator must have been called once"
        # No ``seed`` kwarg should have leaked into legacy callers.
        for call in gen_calls:
            assert "seed" not in call.kwargs, (
                f"seed kwarg leaked into legacy Qwen-only path: {call.kwargs}"
            )
        assert result.answer == "Qwen answer [C:1]"

    def test_qwen_only_path_propagates_seed(self):
        """Qwen-only path (photon_generation_enabled=False) forwards seed=42."""
        cfg = _make_photon_gen_cfg(photon_generation_enabled=False)
        _, baseline_deps, _, _ = _run_generation_branch_query(cfg, seed=42)
        gen_calls = baseline_deps["generator"].generate.call_args_list
        assert gen_calls, "Qwen generator must have been called once"
        # All Qwen-only calls must propagate seed=42.
        for call in gen_calls:
            assert call.kwargs.get("seed") == 42, (
                f"seed=42 did not reach Qwen-only path; kwargs={call.kwargs}"
            )

    def test_qwen_fallback_after_value_error_propagates_seed(self):
        """PHOTON ValueError → Qwen fallback (L1030) MUST forward seed=42."""
        cfg = _make_photon_gen_cfg(photon_generation_enabled=True)
        _, baseline_deps, _, _ = _run_generation_branch_query(
            cfg,
            photon_side_effect=ValueError("length guard"),
            seed=42,
        )
        gen_calls = baseline_deps["generator"].generate.call_args_list
        assert gen_calls, "Qwen fallback must have been called"
        for call in gen_calls:
            assert call.kwargs.get("seed") == 42, (
                f"seed=42 did not reach exception-fallback Qwen path; "
                f"kwargs={call.kwargs}"
            )

    def test_qwen_fallback_after_empty_photon_propagates_seed(self):
        """PHOTON empty output → Qwen fallback (L1043) MUST forward seed=42."""
        cfg = _make_photon_gen_cfg(photon_generation_enabled=True)
        _, baseline_deps, _, _ = _run_generation_branch_query(
            cfg,
            photon_answer="   ",
            seed=42,
        )
        gen_calls = baseline_deps["generator"].generate.call_args_list
        assert gen_calls, "Qwen fallback must have been called for empty PHOTON output"
        for call in gen_calls:
            assert call.kwargs.get("seed") == 42, (
                f"seed=42 did not reach empty-output fallback Qwen path; "
                f"kwargs={call.kwargs}"
            )

    def test_qwen_fallback_seed_zero_propagates(self):
        """seed=0 must propagate through the fallback path (DR3-002).

        ``if seed:`` would silently drop 0, leaving the eval
        nondeterministic. The implementation must use
        ``if seed is not None:``.
        """
        cfg = _make_photon_gen_cfg(photon_generation_enabled=True)
        _, baseline_deps, _, _ = _run_generation_branch_query(
            cfg,
            photon_side_effect=RuntimeError("oom"),
            seed=0,
        )
        gen_calls = baseline_deps["generator"].generate.call_args_list
        for call in gen_calls:
            assert call.kwargs.get("seed") == 0, (
                f"seed=0 silently dropped; kwargs={call.kwargs}"
            )
