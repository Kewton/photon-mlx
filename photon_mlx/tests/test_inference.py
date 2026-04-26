"""Tests for Issue #63 hierarchical scoring in PhotonInference.

Covers:
* ``HierarchicalVecs`` NamedTuple field order.
* ``_tokenize_and_prefill_chunks`` produces per-level mean-pooled tensors.
* ``_encode_chunks_to_vecs_hierarchical`` returns the full triplet while the
  legacy ``_encode_chunks_to_vecs`` still returns only the top vecs.
* The hierarchical scoring path uses ``_drift_level_weights``.
* Fail-closed behaviour when ``level_states`` is empty.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch_ref.config import (
    HierarchyConfig,
    ModelConfig,
    PhotonConfig,
    TokenizerConfig,
)
from photon_mlx.inference import (
    HierarchicalVecs,
    PhotonInference,
    _batch_cosine_similarity,
)
from photon_mlx.model import PhotonModel
from photon_mlx.session import HierarchicalState, weighted_hierarchical_score


# Issue #140 / DR4-002: tests construct fresh small PhotonModel instances whose
# random-init embedding has high σ; we suppress the start-up WARNING by setting
# a finite-but-large threshold. ``float('inf')`` is intentionally NOT used —
# production validation rejects non-finite values, and using a finite sentinel
# keeps the test path on the same code branch as production configs.
TEST_EMBEDDING_RANDOM_INIT_THRESHOLD = 1e9


def _tiny_cfg() -> PhotonConfig:
    cfg = PhotonConfig(
        model=ModelConfig(
            base_embed_dim=16,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=128,
        ),
        hierarchy=HierarchyConfig(
            levels=2,
            chunk_sizes=[4, 4],
            converter_prefix_lengths=[2, 2],
            encoder_layers_per_level=[1, 1],
            decoder_layers_per_level=[1, 1],
        ),
        tokenizer=TokenizerConfig(vocab_size=256),
    )
    cfg.model.embedding_random_init_threshold = TEST_EMBEDDING_RANDOM_INIT_THRESHOLD
    return cfg


def _photon_cfg(
    threshold: float = TEST_EMBEDDING_RANDOM_INIT_THRESHOLD,
) -> PhotonConfig:
    """Issue #140 (§7.2 / DR1-007): wrapper that lets the embedding-norm
    threshold be controlled per test. Defaults to a finite high value so
    existing tests do not emit WARNING logs.
    """
    cfg = _tiny_cfg()
    cfg.model.embedding_random_init_threshold = threshold
    return cfg


@pytest.fixture
def engine(stub_tokenizer_for_cfg) -> PhotonInference:
    mx.random.seed(42)
    cfg = _tiny_cfg()
    model = PhotonModel(cfg)
    tokenizer = stub_tokenizer_for_cfg(cfg)
    return PhotonInference(model, cfg, tokenizer)


# ---------------------------------------------------------------
# HierarchicalVecs
# ---------------------------------------------------------------


class TestHierarchicalVecs:
    def test_namedtuple_field_order(self) -> None:
        """Field order must be (token, mid, top) so it matches drift_level_weights
        (DR1-003)."""
        assert HierarchicalVecs._fields == ("token", "mid", "top")

    def test_namedtuple_round_trip(self) -> None:
        t = mx.zeros((2, 4))
        m = mx.zeros((2, 4))
        top = mx.zeros((2, 4))
        vecs = HierarchicalVecs(token=t, mid=m, top=top)
        # Positional access also yields (token, mid, top).
        assert vecs[0] is t
        assert vecs[1] is m
        assert vecs[2] is top


# ---------------------------------------------------------------
# PhotonInference constructor — drift_level_weights
# ---------------------------------------------------------------


class TestPhotonInferenceConstructor:
    def test_default_drift_level_weights(self, stub_tokenizer_for_cfg) -> None:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        engine = PhotonInference(PhotonModel(cfg), cfg, stub_tokenizer_for_cfg(cfg))
        assert engine._drift_level_weights == (0.2, 0.3, 0.5)

    def test_custom_drift_level_weights_tuple(self, stub_tokenizer_for_cfg) -> None:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        engine = PhotonInference(
            PhotonModel(cfg),
            cfg,
            stub_tokenizer_for_cfg(cfg),
            drift_level_weights=(0.1, 0.4, 0.5),
        )
        assert engine._drift_level_weights == (0.1, 0.4, 0.5)

    def test_custom_drift_level_weights_list_normalised(
        self, stub_tokenizer_for_cfg
    ) -> None:
        """list input is normalised to tuple[float, ...]."""
        mx.random.seed(42)
        cfg = _tiny_cfg()
        engine = PhotonInference(
            PhotonModel(cfg),
            cfg,
            stub_tokenizer_for_cfg(cfg),
            drift_level_weights=[0.3, 0.3, 0.4],
        )
        assert engine._drift_level_weights == (0.3, 0.3, 0.4)
        assert isinstance(engine._drift_level_weights, tuple)

    def test_get_session_propagates_weights(self, stub_tokenizer_for_cfg) -> None:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        engine = PhotonInference(
            PhotonModel(cfg),
            cfg,
            stub_tokenizer_for_cfg(cfg),
            drift_level_weights=(0.1, 0.4, 0.5),
        )
        session = engine.get_session("s1", "repo", "abc")
        assert session.drift_level_weights == (0.1, 0.4, 0.5)


# ---------------------------------------------------------------
# Hierarchical chunk encoding
# ---------------------------------------------------------------


class TestEncodeChunksHierarchical:
    def test_hierarchical_vecs_shapes(self, engine: PhotonInference) -> None:
        """3 per-level vecs with matching (N, D) shape and finite values."""
        texts = ["alpha beta gamma delta", "epsilon zeta"]
        valid_indices, vecs = engine._encode_chunks_to_vecs_hierarchical(texts)
        assert valid_indices == [0, 1]
        assert vecs is not None
        assert vecs.token.shape[0] == 2
        assert vecs.mid.shape[0] == 2
        assert vecs.top.shape[0] == 2
        # All three share the hidden dimension.
        d = vecs.top.shape[1]
        assert vecs.token.shape[1] == d
        assert vecs.mid.shape[1] == d

    def test_hierarchical_vecs_empty_chunks_returns_none(
        self, engine: PhotonInference
    ) -> None:
        valid_indices, vecs = engine._encode_chunks_to_vecs_hierarchical(
            ["", "   ", ""]
        )
        assert valid_indices == []
        assert vecs is None

    def test_legacy_top_only_wrapper_matches_hierarchical(
        self, engine: PhotonInference
    ) -> None:
        """_encode_chunks_to_vecs returns exactly HierarchicalVecs.top —
        bit-for-bit under the shared helper."""
        texts = ["alpha beta gamma", "another chunk"]
        vi_a, top_only = engine._encode_chunks_to_vecs(texts)
        vi_b, full = engine._encode_chunks_to_vecs_hierarchical(texts)
        assert vi_a == vi_b
        assert top_only is not None and full is not None
        diff = mx.max(mx.abs(top_only - full.top))
        mx.eval(diff)
        assert diff.item() < 1e-6


# ---------------------------------------------------------------
# Hierarchical scoring
# ---------------------------------------------------------------


class TestHierarchicalScoring:
    def test_score_prune_candidates_uses_weights(self, engine: PhotonInference) -> None:
        """Scoring via _score_prune_candidates respects drift_level_weights
        (weighted sum of per-level cosines matches the batched output)."""
        setup_ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(setup_ids, "s1", "repo", "abc")

        texts = [
            "alpha beta gamma delta",
            "epsilon zeta eta theta",
            "iota kappa lambda mu",
        ]
        batched = engine._score_prune_candidates(texts, "s1")

        # Hand-built reference using the helpers.
        session = engine._sessions["s1"]
        q_token, q_mid, q_top = engine._build_query_hierarchical_vecs(
            session.current_state
        )
        _, vecs = engine._encode_chunks_to_vecs_hierarchical(texts)
        sim_token = _batch_cosine_similarity(q_token, vecs.token)
        sim_mid = _batch_cosine_similarity(q_mid, vecs.mid)
        sim_top = _batch_cosine_similarity(q_top, vecs.top)
        stack = mx.stack([sim_token, sim_mid, sim_top], axis=-1)
        expected = weighted_hierarchical_score(stack, engine._drift_level_weights)
        mx.eval(expected)
        expected_list = expected.tolist()

        for (idx, s_batched), s_expected in zip(batched, expected_list):
            assert abs(s_batched - s_expected) < 1e-5, f"idx={idx}"

    def test_custom_weights_change_score(self, stub_tokenizer_for_cfg) -> None:
        """Different drift_level_weights → different scores on the same
        inputs (guards against weights being dropped somewhere)."""
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)

        engine_a = PhotonInference(
            model,
            cfg,
            stub_tokenizer_for_cfg(cfg),
            drift_level_weights=(0.2, 0.3, 0.5),
        )
        engine_b = PhotonInference(
            model,
            cfg,
            stub_tokenizer_for_cfg(cfg),
            drift_level_weights=(0.9, 0.05, 0.05),
        )

        setup_ids = mx.random.randint(0, 256, (1, 16))
        engine_a.session_forward(setup_ids, "s1", "repo", "abc")
        engine_b.session_forward(setup_ids, "s1", "repo", "abc")

        texts = [
            "alpha beta gamma",
            "long long long content here " * 3,
            "tiny",
        ]
        scores_a = engine_a._score_prune_candidates(texts, "s1")
        scores_b = engine_b._score_prune_candidates(texts, "s1")

        # At least one score must differ between the two weight schemes.
        differs = any(abs(a[1] - b[1]) > 1e-4 for a, b in zip(scores_a, scores_b))
        assert differs


# ---------------------------------------------------------------
# Fail-closed on partial prefill
# ---------------------------------------------------------------


class TestFailClosed:
    def test_tokenize_and_prefill_returns_none_when_level_states_empty(
        self, stub_tokenizer_for_cfg, monkeypatch
    ) -> None:
        """When hierarchical_prefill returns an empty level_states list,
        the helper must fail-closed with ``None`` so the caller drops to
        no-prune (DR4-002)."""
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        engine = PhotonInference(model, cfg, stub_tokenizer_for_cfg(cfg))

        def _broken_prefill(input_ids):
            # Pretend the bottom-up encoder collapsed: no level_states.
            return mx.zeros((1, 1, 1)), HierarchicalState(level_states=[])

        monkeypatch.setattr(engine, "hierarchical_prefill", _broken_prefill)

        valid_indices, vecs = engine._tokenize_and_prefill_chunks(["alpha beta gamma"])
        assert valid_indices == [0]
        assert vecs is None

    def test_prune_evidence_handles_prefill_runtime_error(
        self, engine: PhotonInference, monkeypatch
    ) -> None:
        """CB-003: non-tokenizer runtime errors must fail closed.

        When ``hierarchical_prefill`` raises RuntimeError during the scoring
        path, ``prune_evidence`` must return all indices instead of
        propagating the exception.
        """
        setup_ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(setup_ids, "s1", "repo", "abc")

        def _raise_runtime(input_ids):
            raise RuntimeError("simulated mlx runtime error")

        monkeypatch.setattr(engine, "hierarchical_prefill", _raise_runtime)

        texts = [f"chunk {i}" for i in range(10)]
        ids = [f"c{i}" for i in range(10)]
        # Must NOT raise; must return all indices as fail-closed fallback.
        result = engine.prune_evidence(texts, ids, "s1", max_chunks=4)
        assert result == list(range(10))

    def test_prune_evidence_handles_score_value_error(
        self, engine: PhotonInference, monkeypatch
    ) -> None:
        """CB-003: ValueError from any scoring helper must fail closed.

        When ``_batch_cosine_similarity`` raises ValueError (shape mismatch,
        NaN, etc), ``prune_evidence`` must return all indices.
        """
        setup_ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(setup_ids, "s1", "repo", "abc")

        import photon_mlx.inference as inf_mod

        def _raise_value(query, keys, eps=1e-8):
            raise ValueError("simulated shape mismatch")

        monkeypatch.setattr(inf_mod, "_batch_cosine_similarity", _raise_value)

        texts = [f"chunk {i}" for i in range(10)]
        ids = [f"c{i}" for i in range(10)]
        result = engine.prune_evidence(texts, ids, "s1", max_chunks=4)
        assert result == list(range(10))

    def test_prune_evidence_handles_turn1_prefill_error(
        self, engine: PhotonInference, monkeypatch
    ) -> None:
        """CB-003: same fail-closed semantics for the Turn-1 question path."""

        def _raise_runtime(input_ids):
            raise RuntimeError("simulated mlx runtime error")

        monkeypatch.setattr(engine, "hierarchical_prefill", _raise_runtime)

        texts = [f"chunk {i}" for i in range(10)]
        ids = [f"c{i}" for i in range(10)]
        result = engine.prune_evidence(
            texts, ids, "no-session-turn1", max_chunks=4, question="what is this?"
        )
        assert result == list(range(10))

    def test_prune_evidence_with_mock_model_does_not_crash(self) -> None:
        """MagicMock model → prune_evidence still works because
        _score_prune_candidates only reads the session.current_state
        hierarchy vectors (no forward needed before has_state)."""

        class _BrokenTokenizer:
            vocab_size = 256
            pad_token_id = 0

            def encode(self, text: str) -> list[int]:
                # Make encode return tokens so scoring runs.
                return [b % 256 for b in text.encode("utf-8")]

        # Drive through the public API: with no session state, Turn 1
        # + question=None returns all indices (no hierarchical_prefill call).
        photon_cfg = PhotonConfig()
        photon_cfg.hierarchy.chunk_sizes = [4, 4]
        photon_cfg.hierarchy.levels = 2
        inference = PhotonInference(MagicMock(), photon_cfg, _BrokenTokenizer())
        texts = [f"chunk {i}" for i in range(10)]
        ids = [f"c{i}" for i in range(10)]
        result = inference.prune_evidence(texts, ids, "no-session", max_chunks=4)
        assert result == list(range(10))


# ---------------------------------------------------------------
# Issue #79: scoring path tolerates all three storage_mode values
# ---------------------------------------------------------------


class TestScoringPathAcrossStorageModes:
    """The ``q_top ← session.get_session_coarse_state()`` override at
    photon_mlx/inference.py:550 must tolerate all three storage_mode
    values. Each mode drives a different coarse_vec path:

    * full            — turn_history + compressed_history mixed
    * top_level_only  — turn_history only, compressed_history untouched
    * summary_only    — turn_history empty, compressed_history only

    With a single turn the batched scoring simply needs to produce a
    finite score per chunk (no shape/dtype crash). We do NOT assert
    numerical equivalence across modes because each mode pools a
    different vector.
    """

    @pytest.fixture
    def stub_tokenizer_factory(self, stub_tokenizer_for_cfg):
        return stub_tokenizer_for_cfg

    def _make_engine(self, stub_tokenizer_factory, mode: str) -> PhotonInference:
        from photon_mlx.session import WorkingMemoryConfig

        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        return PhotonInference(
            model,
            cfg,
            stub_tokenizer_factory(cfg),
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, storage_mode=mode
            ),
        )

    @pytest.mark.parametrize("mode", ["full", "top_level_only", "summary_only"])
    def test_score_prune_candidates_returns_finite_scores(
        self, stub_tokenizer_factory, mode: str
    ) -> None:
        engine = self._make_engine(stub_tokenizer_factory, mode)
        setup_ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(setup_ids, "s1", "repo", "abc")

        texts = ["alpha beta", "gamma delta epsilon", "zeta eta theta iota"]
        scored = engine._score_prune_candidates(texts, "s1")
        assert len(scored) == 3
        # CB-003: the previous assertion was only ``isinstance(score,
        # float)``, which trivially passes for the ``-1.0`` fail-closed
        # sentinel that ``_score_prune_candidates`` returns when scoring
        # bails out. Without rejecting that sentinel the test had no
        # regression power against the storage_mode-specific coarse-state
        # paths it was supposed to exercise. Tighten the assertions to:
        #   (a) finite float (no ``NaN`` / ``inf``), and
        #   (b) NOT equal to the ``-1.0`` sentinel.
        # If scoring falls through for every candidate under any mode the
        # assertion now fails loudly.
        for idx, score in scored:
            assert 0 <= idx < 3
            assert isinstance(score, float)
            assert math.isfinite(score), (
                f"storage_mode={mode} produced non-finite score {score!r}"
            )
            assert score != -1.0, (
                f"storage_mode={mode} returned -1.0 fail-closed sentinel "
                "(scoring path did not actually run)"
            )
        # Sanity: at least one candidate must have been scored (i.e. we
        # never want all three to be the sentinel even though we now
        # reject individual sentinels above).
        assert any(score != -1.0 for _, score in scored), (
            f"storage_mode={mode}: every candidate returned the sentinel"
        )

    @pytest.mark.parametrize("mode", ["full", "top_level_only", "summary_only"])
    def test_prune_evidence_full_path(self, stub_tokenizer_factory, mode: str) -> None:
        engine = self._make_engine(stub_tokenizer_factory, mode)
        setup_ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(setup_ids, "s1", "repo", "abc")

        texts = [f"chunk text number {i}" for i in range(10)]
        cids = [f"c{i}" for i in range(10)]
        result = engine.prune_evidence(texts, cids, "s1", max_chunks=4)
        assert len(result) == 4
        assert result == sorted(result)

    def test_summary_only_coarse_none_tolerated_when_compressed_empty(
        self, stub_tokenizer_factory
    ) -> None:
        """``summary_only`` + fresh session → get_session_coarse_state() is
        None (no coarse override), scoring falls back to pure Issue #63
        hierarchical path without crashing."""
        from photon_mlx.session import WorkingMemoryConfig

        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        engine = PhotonInference(
            model,
            cfg,
            stub_tokenizer_factory(cfg),
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, storage_mode="summary_only"
            ),
        )
        # No session_forward yet → no state. But we still exercise the
        # scoring path with Turn 1 + question (pure-question path).
        texts = [f"chunk {i}" for i in range(8)]
        cids = [f"c{i}" for i in range(8)]
        result = engine.prune_evidence(
            texts, cids, "fresh", max_chunks=3, question="what is this?"
        )
        assert len(result) == 3

    def test_full_mode_token_and_mid_still_come_from_current_state(
        self, stub_tokenizer_factory
    ) -> None:
        """Only ``q_top`` is overridden by the coarse aggregate; ``q_token``
        and ``q_mid`` remain keyed on ``session.current_state``. Verified by
        the fact that scoring with working memory enabled on a single turn
        gives a bit-for-bit match with the pure-#63 reference (the coarse
        aggregate of one turn equals the pooled current_state top level).
        """
        from photon_mlx.session import WorkingMemoryConfig, weighted_hierarchical_score

        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        engine = PhotonInference(
            model,
            cfg,
            stub_tokenizer_factory(cfg),
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, storage_mode="full"
            ),
        )
        setup_ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(setup_ids, "s1", "repo", "abc")

        texts = ["alpha", "beta gamma", "delta epsilon"]
        batched = engine._score_prune_candidates(texts, "s1")

        # Build pure-#63 reference: no coarse override (single turn, coarse
        # aggregate equals current_state top after mean_pool).
        session = engine._sessions["s1"]
        q_token, q_mid, q_top = engine._build_query_hierarchical_vecs(
            session.current_state
        )
        _, vecs = engine._encode_chunks_to_vecs_hierarchical(texts)
        sim_token = _batch_cosine_similarity(q_token, vecs.token)
        sim_mid = _batch_cosine_similarity(q_mid, vecs.mid)
        sim_top = _batch_cosine_similarity(q_top, vecs.top)
        stack = mx.stack([sim_token, sim_mid, sim_top], axis=-1)
        expected = weighted_hierarchical_score(stack, engine._drift_level_weights)
        mx.eval(expected)
        expected_list = expected.tolist()

        for (_, s_batched), s_expected in zip(batched, expected_list):
            assert abs(s_batched - s_expected) < 1e-4


# ---------------------------------------------------------------
# Issue #140 / S7-001: PhotonInference start-up embedding-norm WARNING
# ---------------------------------------------------------------


class _BrokenTokenizerForInit:
    """Stand-in tokenizer for the MagicMock-model start-up test.

    ``PhotonInference`` only stores the tokenizer; it isn't called in
    ``__init__``, so any object with ``vocab_size`` / ``pad_token_id`` will
    do. ``encode`` is provided in case future code paths exercise it during
    construction.
    """

    vocab_size = 256
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:  # pragma: no cover - defensive
        raise RuntimeError("broken tokenizer (test stub)")


class TestCheckWeightInitialization:
    """Behaviour of :func:`_check_weight_initialization` invoked by
    ``PhotonInference.__init__`` (Issue #140 / S7-001).

    The threshold absolute value is intentionally NOT asserted: we only
    verify the WARNING fires above threshold and stays silent below.
    """

    def test_check_weight_initialization_warns_on_high_variance(
        self, stub_tokenizer_for_cfg, caplog: pytest.LogCaptureFixture
    ) -> None:
        mx.random.seed(42)
        cfg = _photon_cfg(threshold=0.1)  # low threshold → WARNING expected
        model = PhotonModel(cfg)
        tokenizer = stub_tokenizer_for_cfg(cfg)
        with caplog.at_level(logging.WARNING, logger="photon_mlx.inference"):
            PhotonInference(model, cfg, tokenizer)
        assert any(
            "high variance" in rec.getMessage()
            and rec.name == "photon_mlx.inference"
            and rec.levelno == logging.WARNING
            for rec in caplog.records
        )

    def test_check_weight_initialization_silent_on_low_variance(
        self, stub_tokenizer_for_cfg, caplog: pytest.LogCaptureFixture
    ) -> None:
        mx.random.seed(42)
        cfg = _photon_cfg(threshold=10.0)  # high threshold → silent
        model = PhotonModel(cfg)
        tokenizer = stub_tokenizer_for_cfg(cfg)
        with caplog.at_level(logging.WARNING, logger="photon_mlx.inference"):
            PhotonInference(model, cfg, tokenizer)
        assert not any("high variance" in rec.getMessage() for rec in caplog.records)

    def test_check_weight_initialization_silent_on_magicmock_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MagicMock model → silent skip via ``isinstance(weight, mx.array)``
        guard. No exception, no WARNING."""
        cfg = _photon_cfg()
        with caplog.at_level(logging.WARNING, logger="photon_mlx.inference"):
            PhotonInference(MagicMock(), cfg, _BrokenTokenizerForInit())
        assert not any("high variance" in rec.getMessage() for rec in caplog.records)
