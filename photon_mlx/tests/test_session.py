"""Tests for PHOTON session inference and drift tracking."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch_ref.config import (
    HierarchyConfig,
    ModelConfig,
    PhotonConfig,
    TokenizerConfig,
)
from photon_mlx.model import PhotonModel
from photon_mlx.inference import (
    MICRO_BATCH_SIZE,
    PAD_TOKEN_ID,
    PhotonInference,
    _batch_cosine_similarity,
)
from photon_mlx.session import (
    DriftMetrics,
    HierarchicalState,
    PhotonSessionState,
    TurnState,
    WorkingMemoryConfig,
    cosine_distance,
    kl_divergence,
    mean_pool,
    token_agreement_rate,
    weighted_hierarchical_score,
)


def _tiny_cfg() -> PhotonConfig:
    return PhotonConfig(
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


# ---------------------------------------------------------------
# Drift metric unit tests
# ---------------------------------------------------------------


class TestDriftMetrics:
    def test_cosine_distance_identical(self) -> None:
        a = mx.ones((4, 64))
        assert cosine_distance(a, a) < 1e-5

    def test_cosine_distance_orthogonal(self) -> None:
        a = mx.array([[1.0, 0.0]])
        b = mx.array([[0.0, 1.0]])
        assert abs(cosine_distance(a, b) - 1.0) < 1e-5

    def test_kl_divergence_identical(self) -> None:
        logits = mx.random.normal((1, 8, 256))
        assert kl_divergence(logits, logits) < 1e-4

    def test_kl_divergence_different(self) -> None:
        a = mx.random.normal((1, 8, 256))
        b = mx.random.normal((1, 8, 256))
        assert kl_divergence(a, b) > 0.0

    def test_token_agreement_identical(self) -> None:
        logits = mx.random.normal((1, 8, 256))
        assert token_agreement_rate(logits, logits) == 1.0

    def test_token_agreement_different(self) -> None:
        a = mx.random.normal((1, 8, 256))
        b = mx.random.normal((1, 8, 256))
        rate = token_agreement_rate(a, b)
        assert 0.0 <= rate <= 1.0

    def test_different_seq_lengths_no_crash(self) -> None:
        """Regression: different sequence lengths must not crash (Issue #36)."""
        short = mx.random.normal((1, 4, 64))
        long = mx.random.normal((1, 12, 64))
        # cosine_distance uses mean-pooling, so shapes always reduce to (64,)
        dist = cosine_distance(short, long)
        assert 0.0 <= dist <= 2.0

        # kl_divergence and token_agreement_rate truncate to min length
        logits_short = mx.random.normal((1, 4, 256))
        logits_long = mx.random.normal((1, 12, 256))
        kl = kl_divergence(logits_short, logits_long)
        assert kl >= 0.0
        rate = token_agreement_rate(logits_short, logits_long)
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------
# Issue #63: weighted_hierarchical_score (shared helper)
# ---------------------------------------------------------------


class TestWeightedHierarchicalScore:
    def test_scalar_tuple_input(self) -> None:
        """Scalar-tuple path returns a Python float equal to sum(w*v)."""
        out = weighted_hierarchical_score((0.1, 0.2, 0.3), (0.2, 0.3, 0.5))
        expected = 0.2 * 0.1 + 0.3 * 0.2 + 0.5 * 0.3
        assert isinstance(out, float)
        assert abs(out - expected) < 1e-9

    def test_mx_array_last_axis_reduction(self) -> None:
        """mx.array path reduces along the last axis and returns (N,)."""
        values = mx.array([[0.1, 0.2, 0.3], [1.0, 0.0, 0.0]])
        out = weighted_hierarchical_score(values, (0.2, 0.3, 0.5))
        assert isinstance(out, mx.array)
        mx.eval(out)
        out_list = out.tolist()
        expected0 = 0.2 * 0.1 + 0.3 * 0.2 + 0.5 * 0.3
        expected1 = 0.2
        assert abs(out_list[0] - expected0) < 1e-5
        assert abs(out_list[1] - expected1) < 1e-5

    def test_mx_array_shape_assertion(self) -> None:
        """Mismatched trailing dim vs len(weights) must assert-fail."""
        values = mx.array([[0.1, 0.2]])  # last dim = 2
        with pytest.raises(AssertionError):
            weighted_hierarchical_score(values, (0.2, 0.3, 0.5))

    def test_mx_array_dtype_cast(self) -> None:
        """Weights are cast to the values dtype so no dtype promotion."""
        values = mx.array([[0.1, 0.2, 0.3]], dtype=mx.float32)
        out = weighted_hierarchical_score(values, (0.2, 0.3, 0.5))
        assert out.dtype == mx.float32


# ---------------------------------------------------------------
# Session state tests
# ---------------------------------------------------------------


class TestSessionState:
    def test_initial_drift_is_zero(self) -> None:
        session = PhotonSessionState("s1", "repo", "abc123")
        state = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)), mx.ones((1, 1, 64))],
        )
        logits = mx.random.normal((1, 16, 256))
        drift = session.update(state, logits)
        assert drift.latent_cosine_drift == 0.0
        assert drift.turn_id == 1

    def test_second_turn_has_drift(self) -> None:
        session = PhotonSessionState("s1", "repo", "abc123")

        s1 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)), mx.ones((1, 1, 64))],
        )
        l1 = mx.random.normal((1, 16, 256))
        session.update(s1, l1)

        s2 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)) * 2, mx.ones((1, 1, 64)) * -1],
        )
        l2 = mx.random.normal((1, 16, 256))
        drift = session.update(s2, l2)

        assert drift.turn_id == 2
        assert drift.latent_cosine_drift > 0.0
        assert 0.0 <= drift.token_agreement <= 1.0
        assert drift.logit_kl >= 0.0

    def test_drift_metrics_hierarchical(self) -> None:
        """Issue #63: with ``level_states`` length 2 and ``token_proj`` provided,
        all three drift fields are computed independently and the weighted
        ``topic_shift_score`` is ``sum(w_i * drift_i)`` (DR1-004)."""
        session = PhotonSessionState("s1", "repo", "abc123")

        s1 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)), mx.ones((1, 1, 64))],
            token_proj=mx.ones((1, 16, 64)),
        )
        session.update(s1)

        s2 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)) * -1, mx.ones((1, 1, 64)) * -1],
            token_proj=mx.ones((1, 16, 64)) * -1,
        )
        drift = session.update(s2)

        # 3 per-level drifts must all be non-zero (opposite-signed vectors).
        assert drift.latent_cosine_drift_top > 0.0
        assert drift.latent_cosine_drift_mid > 0.0
        assert drift.latent_cosine_drift_token > 0.0
        # Backward-compat: latent_cosine_drift == top (property).
        assert drift.latent_cosine_drift == drift.latent_cosine_drift_top
        # topic_shift_score must equal the weighted sum (default weights).
        expected = (
            0.2 * drift.latent_cosine_drift_token
            + 0.3 * drift.latent_cosine_drift_mid
            + 0.5 * drift.latent_cosine_drift_top
        )
        assert abs(drift.topic_shift_score - expected) < 1e-6

    def test_drift_metrics_token_proj_none(self) -> None:
        """token_proj=None on either side → drift_token fallback 0.0."""
        session = PhotonSessionState("s1", "repo", "abc123")
        s1 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)), mx.ones((1, 1, 64))],
            token_proj=None,
        )
        session.update(s1)

        # Both level_states[0] (mid) and level_states[-1] (top) differ from
        # s1, and token_proj is None on both sides.
        s2 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)) * -1, mx.ones((1, 1, 64)) * -1],
            token_proj=None,
        )
        drift = session.update(s2)
        # token_proj=None → fallback 0.0.
        assert drift.latent_cosine_drift_token == 0.0
        # mid / top still computed normally.
        assert drift.latent_cosine_drift_top > 0.0
        assert drift.latent_cosine_drift_mid > 0.0

    def test_drift_metrics_levels_one(self) -> None:
        """len(level_states)==1 → drift_mid fallback 0.0, drift_top from
        the sole (last) level_states entry."""
        session = PhotonSessionState("s1", "repo", "abc123")
        s1 = HierarchicalState(level_states=[mx.ones((1, 1, 64))])
        session.update(s1)

        s2 = HierarchicalState(level_states=[mx.ones((1, 1, 64)) * -1])
        drift = session.update(s2)
        assert drift.latent_cosine_drift_mid == 0.0
        assert drift.latent_cosine_drift_top > 0.0

    def test_drift_metrics_identical_state(self) -> None:
        """Identical states turn-over-turn → all three drifts are 0.0."""
        session = PhotonSessionState("s1", "repo", "abc123")
        s1 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)), mx.ones((1, 1, 64))],
            token_proj=mx.ones((1, 16, 64)),
        )
        session.update(s1)
        # Same-shaped / same-valued state → drift == 0 on every level.
        s2 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)), mx.ones((1, 1, 64))],
            token_proj=mx.ones((1, 16, 64)),
        )
        drift = session.update(s2)
        assert drift.latent_cosine_drift_top < 1e-5
        assert drift.latent_cosine_drift_mid < 1e-5
        assert drift.latent_cosine_drift_token < 1e-5
        assert drift.topic_shift_score < 1e-5

    def test_drift_metrics_as_dict_superset(self) -> None:
        """DriftMetrics.as_dict() superset contract (DR3-002 + Issue #92 T-3):

        * Legacy keys stay present (numeric drift fields — finite-checked).
        * Three per-level keys added in Issue #63 stay present.
        * Two new Issue #92 keys are present with their default values
          (``selected_aggregation_mode`` / ``selected_aggregation_alpha``).
        * Finite-check for the numeric drift fields is separate from the
          type check for the mode/alpha fields (Issue body (g)).
        """
        import math as _math

        session = PhotonSessionState("s1", "repo", "abc123")
        s1 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)), mx.ones((1, 1, 64))],
            token_proj=mx.ones((1, 16, 64)),
        )
        session.update(s1)
        s2 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)) * -1, mx.ones((1, 1, 64)) * -1],
            token_proj=mx.ones((1, 16, 64)) * -1,
        )
        drift = session.update(s2)
        d = drift.as_dict()
        # Legacy numeric keys preserved — all must be finite.
        for key in (
            "turn_id",
            "latent_cosine_drift",
            "token_agreement",
            "logit_kl",
            "topic_shift_score",
        ):
            assert key in d
        # New per-level keys present.
        for key in (
            "latent_cosine_drift_top",
            "latent_cosine_drift_mid",
            "latent_cosine_drift_token",
        ):
            assert key in d
        # Finite-check (separated from mode/alpha in Issue #92 T-3).
        numeric_keys = (
            "latent_cosine_drift",
            "latent_cosine_drift_top",
            "latent_cosine_drift_mid",
            "latent_cosine_drift_token",
            "token_agreement",
            "logit_kl",
            "topic_shift_score",
        )
        for key in numeric_keys:
            assert _math.isfinite(d[key]), f"{key} must be finite"
        # Alias still returns top.
        assert d["latent_cosine_drift"] == d["latent_cosine_drift_top"]

        # Issue #92: new telemetry fields present by default as ``None``.
        assert "selected_aggregation_mode" in d
        assert "selected_aggregation_alpha" in d
        assert d["selected_aggregation_mode"] is None
        assert d["selected_aggregation_alpha"] is None

    def test_drift_metrics_as_dict_mode_alpha_types(self) -> None:
        """Issue #92 T-3 (g): mode/alpha fields carry correct types when set.

        Separated from the legacy numeric ``as_dict`` test so that the
        type-check for mode (str) and alpha (float) does NOT run through
        the numeric finite-check path.
        """
        m = DriftMetrics(turn_id=2)
        m.selected_aggregation_mode = "hybrid"
        m.selected_aggregation_alpha = 0.375
        d = m.as_dict()
        assert d["selected_aggregation_mode"] == "hybrid"
        assert isinstance(d["selected_aggregation_mode"], str)
        assert d["selected_aggregation_alpha"] == pytest.approx(0.375)
        assert isinstance(d["selected_aggregation_alpha"], float)

        # When only mode is set (weighted/attention/last), alpha stays None.
        m2 = DriftMetrics(turn_id=3)
        m2.selected_aggregation_mode = "weighted"
        d2 = m2.as_dict()
        assert d2["selected_aggregation_mode"] == "weighted"
        assert d2["selected_aggregation_alpha"] is None

    def test_custom_drift_level_weights(self) -> None:
        """Custom weights propagate into topic_shift_score."""
        weights = (0.5, 0.25, 0.25)
        session = PhotonSessionState(
            "s1", "repo", "abc123", drift_level_weights=weights
        )
        # Weights are normalised to a tuple of Python floats.
        assert session.drift_level_weights == weights

        s1 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)), mx.ones((1, 1, 64))],
            token_proj=mx.ones((1, 16, 64)),
        )
        session.update(s1)
        s2 = HierarchicalState(
            level_states=[mx.ones((1, 4, 64)) * -1, mx.ones((1, 1, 64)) * -1],
            token_proj=mx.ones((1, 16, 64)) * -1,
        )
        drift = session.update(s2)
        expected = (
            0.5 * drift.latent_cosine_drift_token
            + 0.25 * drift.latent_cosine_drift_mid
            + 0.25 * drift.latent_cosine_drift_top
        )
        assert abs(drift.topic_shift_score - expected) < 1e-6

    def test_drift_history_accumulates(self) -> None:
        """Length-1 ``level_states`` (no mid) and ``token_proj=None`` exercise
        the Issue #63 fallback path: ``drift_mid=0.0`` and
        ``drift_token=0.0`` while ``drift_top`` is still computed. The legacy
        assertion on ``turn_count`` / history length is preserved (DR2-004)."""
        session = PhotonSessionState("s1", "repo", "abc123")
        for i in range(5):
            state = HierarchicalState(
                level_states=[mx.ones((1, 4, 64)) * (i + 1)],
            )
            session.update(state)
        assert len(session.drift_history) == 5
        assert session.turn_count == 5
        # Fallback path: mid and token drift must stay at the fallback 0.0
        # throughout the history because this fixture omits level_states[0]
        # (length 1) and never provides token_proj.
        for metrics in session.drift_history:
            assert metrics.latent_cosine_drift_mid == 0.0
            assert metrics.latent_cosine_drift_token == 0.0


# ---------------------------------------------------------------
# End-to-end inference tests
# ---------------------------------------------------------------


class TestInference:
    @pytest.fixture
    def engine(self, stub_tokenizer_for_cfg) -> PhotonInference:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        tokenizer = stub_tokenizer_for_cfg(cfg)
        return PhotonInference(model, cfg, tokenizer)

    def test_hierarchical_prefill_shape(self, engine: PhotonInference) -> None:
        ids = mx.random.randint(0, 256, (1, 16))
        logits, state = engine.hierarchical_prefill(ids)
        assert logits.shape == (1, 16, 256)
        assert len(state.level_states) == 2
        assert state.token_proj is not None

    def test_session_forward_returns_drift(self, engine: PhotonInference) -> None:
        ids = mx.random.randint(0, 256, (1, 16))
        logits, drift = engine.session_forward(ids, "s1", "repo", "abc")
        assert logits.shape == (1, 16, 256)
        assert drift.turn_id == 1

    def test_multi_turn_drift_tracking(self, engine: PhotonInference) -> None:
        for i in range(3):
            ids = mx.random.randint(0, 256, (1, 16))
            _, drift = engine.session_forward(ids, "s1", "repo", "abc")
            assert drift.turn_id == i + 1

        history = engine.get_drift_history("s1")
        assert len(history) == 3
        # Turn 2+ should have nonzero drift (different random inputs)
        assert history[1]["latent_cosine_drift"] > 0.0

    def test_separate_sessions_independent(self, engine: PhotonInference) -> None:
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")
        engine.session_forward(ids, "s2", "repo", "abc")

        h1 = engine.get_drift_history("s1")
        h2 = engine.get_drift_history("s2")
        assert len(h1) == 1
        assert len(h2) == 1


# ---------------------------------------------------------------
# Evidence pruning tests (Issue #37)
# ---------------------------------------------------------------


class TestPruneEvidence:
    """prune_evidence selects top-K chunks using PHOTON coarse state."""

    @pytest.fixture
    def engine(self, stub_tokenizer_for_cfg) -> PhotonInference:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        tokenizer = stub_tokenizer_for_cfg(cfg)
        return PhotonInference(model, cfg, tokenizer)

    def test_turn1_no_pruning(self, engine: PhotonInference) -> None:
        """On turn 1 (no session state), all indices are returned."""
        texts = [f"chunk text {i}" for i in range(12)]
        ids = [f"c{i}" for i in range(12)]
        result = engine.prune_evidence(texts, ids, "no_session", max_chunks=8)
        # No session state → return all indices
        assert result == list(range(12))

    def test_turn1_existing_session_no_state(self, engine: PhotonInference) -> None:
        """Session exists but has no state yet → no pruning."""
        engine.get_session("s1", "repo", "abc")
        texts = [f"chunk {i}" for i in range(10)]
        ids = [f"c{i}" for i in range(10)]
        result = engine.prune_evidence(texts, ids, "s1", max_chunks=8)
        assert result == list(range(10))

    def test_turn2_prunes_to_max_chunks(self, engine: PhotonInference) -> None:
        """After session_forward, prune_evidence returns max_chunks indices."""
        # Run a forward pass to establish session state
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")

        texts = [f"chunk text number {i}" for i in range(12)]
        chunk_ids = [f"c{i}" for i in range(12)]
        result = engine.prune_evidence(texts, chunk_ids, "s1", max_chunks=6)

        assert len(result) == 6
        # Indices should be sorted
        assert result == sorted(result)
        # All indices should be valid
        assert all(0 <= idx < 12 for idx in result)

    def test_fewer_chunks_than_max_no_pruning(self, engine: PhotonInference) -> None:
        """If chunks <= max_chunks, all are returned even on turn 2."""
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")

        texts = [f"chunk {i}" for i in range(5)]
        chunk_ids = [f"c{i}" for i in range(5)]
        result = engine.prune_evidence(texts, chunk_ids, "s1", max_chunks=8)
        assert result == list(range(5))

    def test_empty_chunks_handled(self, engine: PhotonInference) -> None:
        """Empty chunk list returns empty."""
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")

        result = engine.prune_evidence([], [], "s1", max_chunks=8)
        assert result == []

    # ───────────── Issue #61: batched-prune external behaviour ──────────────

    def test_hierarchical_prefill_batch_shape(self, engine: PhotonInference) -> None:
        """B>1 input must produce per-batch level_states (B, T_top, D)."""
        ids = mx.random.randint(0, 256, (2, 16))
        logits, state = engine.hierarchical_prefill(ids)
        assert logits.shape[0] == 2
        # Top-level encoder output: leading dim is batch.
        assert state.level_states[-1].shape[0] == 2

    def test_all_empty_chunks_returns_first_max_chunks(
        self, engine: PhotonInference
    ) -> None:
        """All-empty chunks → return [0..max_chunks-1] (no scoring possible)."""
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")

        texts = ["" for _ in range(12)]
        chunk_ids = [f"c{i}" for i in range(12)]
        result = engine.prune_evidence(texts, chunk_ids, "s1", max_chunks=8)
        # All scores stay at -1.0; the sorted top-K selection must return the
        # first max_chunks indices in ascending order (CB-003: pin tie-break).
        assert result == list(range(8))

    def test_single_valid_chunk_returns_valid_first(
        self, engine: PhotonInference
    ) -> None:
        """One valid chunk + many empty: the valid index must be selected."""
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")

        texts = ["" for _ in range(12)]
        texts[7] = "real content here for the only valid chunk"
        chunk_ids = [f"c{i}" for i in range(12)]
        result = engine.prune_evidence(texts, chunk_ids, "s1", max_chunks=4)
        assert 7 in result
        assert len(result) == 4

    def test_boundary_len_equals_max_chunks_returns_all(
        self, engine: PhotonInference
    ) -> None:
        """len(chunks) == max_chunks → early return (b) returns all."""
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")

        texts = [f"chunk text {i}" for i in range(8)]
        chunk_ids = [f"c{i}" for i in range(8)]
        result = engine.prune_evidence(texts, chunk_ids, "s1", max_chunks=8)
        assert result == list(range(8))

    def test_oversized_chunk_truncated_to_max_position_embeddings(
        self, engine: PhotonInference
    ) -> None:
        """Chunks longer than cfg.model.max_position_embeddings must be capped."""
        # _tiny_cfg sets max_position_embeddings=128.
        long_text = "x" * 5000  # 5000 bytes -> 5000 raw token ids
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")

        token_ids = engine._tokenize_chunk(long_text)
        assert len(token_ids) <= engine.cfg.model.max_position_embeddings
        # Length must remain chunk-aligned after cap.
        assert len(token_ids) % engine._chunk_alignment == 0

        # End-to-end pruning still completes.
        texts = [long_text] + [f"text {i}" for i in range(11)]
        cids = [f"c{i}" for i in range(12)]
        result = engine.prune_evidence(texts, cids, "s1", max_chunks=4)
        assert len(result) == 4

    def test_micro_batch_equivalence_2_vs_64(self, engine: PhotonInference) -> None:
        """micro_batch_size=2 must yield the same selection as =64."""
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")

        texts = [f"chunk number {i} body content" for i in range(16)]
        cids = [f"c{i}" for i in range(16)]
        result_small = engine.prune_evidence(
            texts, cids, "s1", max_chunks=8, micro_batch_size=2
        )
        result_large = engine.prune_evidence(
            texts, cids, "s1", max_chunks=8, micro_batch_size=64
        )
        assert result_small == result_large

    def test_micro_batch_none_equals_default(self, engine: PhotonInference) -> None:
        """micro_batch_size=None must behave identically to MICRO_BATCH_SIZE."""
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")

        texts = [f"chunk content {i}" for i in range(16)]
        cids = [f"c{i}" for i in range(16)]
        result_none = engine.prune_evidence(
            texts, cids, "s1", max_chunks=6, micro_batch_size=None
        )
        result_default = engine.prune_evidence(
            texts, cids, "s1", max_chunks=6, micro_batch_size=MICRO_BATCH_SIZE
        )
        assert result_none == result_default

    def test_invalid_micro_batch_size_raises(self, engine: PhotonInference) -> None:
        """Invalid micro_batch_size must raise ValueError."""
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")
        texts = [f"x{i}" for i in range(12)]
        cids = [f"c{i}" for i in range(12)]
        for bad in (0, -1, True, False, 1.5, "64"):
            with pytest.raises(ValueError):
                engine.prune_evidence(
                    texts, cids, "s1", max_chunks=8, micro_batch_size=bad
                )

    # ───────────── Issue #56: Pass 1 scoring (Turn 1 + question) ──────────────

    def test_pass1_returns_topk(self, engine: PhotonInference) -> None:
        """Turn 1 + question (no session state) must return max_chunks indices
        ranked by question↔chunk similarity, sorted ascending."""
        texts = [f"chunk number {i} body content" for i in range(12)]
        cids = [f"c{i}" for i in range(12)]
        result = engine.prune_evidence(
            texts,
            cids,
            "no_session_pass1",
            max_chunks=6,
            question="what is chunk 3 about?",
        )
        assert len(result) == 6
        assert result == sorted(result)
        assert all(0 <= idx < 12 for idx in result)

    def test_pass1_does_not_mutate_session(self, engine: PhotonInference) -> None:
        """Pass 1 must not touch self._sessions (DR1-001 read-only guarantee)."""
        session = engine.get_session("s_pass1", "repo", "abc")
        assert session.current_state is None
        assert session.prev_state is None
        turn_count_before = session.turn_count
        drift_len_before = len(session.drift_history)

        texts = [f"chunk {i} text" for i in range(10)]
        cids = [f"c{i}" for i in range(10)]
        engine.prune_evidence(
            texts,
            cids,
            "s_pass1",
            max_chunks=4,
            question="query text",
        )

        # Session state remains untouched — Pass 1 is read-only.
        assert session.current_state is None
        assert session.prev_state is None
        assert session.turn_count == turn_count_before
        assert len(session.drift_history) == drift_len_before

    def test_pass1_no_question_returns_all(self, engine: PhotonInference) -> None:
        """question=None on Turn 1 preserves the pre-Issue-#56 behaviour
        (all_indices) so callers that have not opted into two-pass are
        unaffected (DR1-006 backward-compat spec)."""
        texts = [f"chunk {i}" for i in range(12)]
        cids = [f"c{i}" for i in range(12)]
        result = engine.prune_evidence(
            texts, cids, "no_session_pass1_none", max_chunks=4, question=None
        )
        assert result == list(range(12))

        # Empty-string question behaves the same (early return).
        result2 = engine.prune_evidence(
            texts, cids, "no_session_pass1_empty", max_chunks=4, question="   "
        )
        assert result2 == list(range(12))

    def test_pass1_tokenize_failure_returns_all(self, stub_tokenizer_for_cfg) -> None:
        """tokenizer.encode raising inside Pass 1 must fail-closed to
        all_indices (CB-002 / DR1-002)."""
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)

        class _BrokenTokenizer:
            vocab_size = cfg.tokenizer.vocab_size
            pad_token_id = 0

            def encode(self, text: str) -> list[int]:
                raise RuntimeError("simulated encode failure")

        engine = PhotonInference(model, cfg, _BrokenTokenizer())
        texts = [f"chunk {i}" for i in range(10)]
        cids = [f"c{i}" for i in range(10)]
        result = engine.prune_evidence(
            texts,
            cids,
            "no_session_broken",
            max_chunks=4,
            question="some question",
        )
        assert result == list(range(10))

    @pytest.mark.parametrize("working_memory_enabled", [True, False])
    def test_mixed_length_batch_topk_matches_sequential(
        self,
        stub_tokenizer_for_cfg,
        working_memory_enabled: bool,
    ) -> None:
        """Mixed-length non-empty chunks: batched top-K and raw scores must
        match a sequential hierarchical reference within ε=1e-3.

        Post-Issue-#63: ``_score_prune_candidates`` returns the
        ``weighted_hierarchical_score(sim_token, sim_mid, sim_top)`` combo
        rather than top-only cosine, so the sequential reference is built
        the same way.

        Run with both working_memory enabled and disabled (Issue #64) to
        ensure the coarse-state aggregation does not change the prune
        ranking when only a single turn has been recorded (S3-004). With
        a single turn, ``get_session_coarse_state()`` returns a vector
        that is a decayed mean of one entry — equal to that turn's
        ``mean_pool(level_states[-1])`` — so the overridden q_top stays
        bit-for-bit equivalent to the pure-#63 path.
        """
        from photon_mlx.session import weighted_hierarchical_score

        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        tokenizer = stub_tokenizer_for_cfg(cfg)
        engine = PhotonInference(
            model,
            cfg,
            tokenizer,
            working_memory_cfg=WorkingMemoryConfig(enabled=working_memory_enabled),
        )

        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")

        # 10 chunks of 3 distinct lengths, all non-empty (per DR2-005).
        texts = [
            "short",
            "another short text",
            "medium length chunk content here",
            "longer chunk text " * 4,
            "some words",
            "yet more medium length chunk content",
            "longer chunk text " * 6,
            "tiny",
            "medium chunk again with words",
            "longer chunk text " * 5,
        ]
        cids = [f"c{i}" for i in range(len(texts))]

        # Batched scoring via the helper (hierarchical).
        batched_scores = engine._score_prune_candidates(texts, "s1")

        # Sequential hierarchical reference: per-chunk prefill, three
        # masked-means (token/mid/top), three cosines, weighted sum.
        session = engine._sessions["s1"]
        state = session.current_state
        top_state = state.level_states[-1].astype(mx.float32)
        q_top = mx.mean(top_state, axis=tuple(range(top_state.ndim - 1)))
        if len(state.level_states) >= 2:
            mid_state = state.level_states[0].astype(mx.float32)
            q_mid = mx.mean(mid_state, axis=tuple(range(mid_state.ndim - 1)))
        else:
            q_mid = q_top
        if state.token_proj is not None:
            tok_state = state.token_proj.astype(mx.float32)
            q_token = mx.mean(tok_state, axis=tuple(range(tok_state.ndim - 1)))
        else:
            q_token = q_top

        weights = engine._drift_level_weights

        seq_scores: list[tuple[int, float]] = []
        for idx, text in enumerate(texts):
            token_ids = engine._tokenize_chunk(text)
            assert token_ids, "test fixture must have only non-empty chunks"
            inp = mx.array(token_ids, dtype=mx.int32).reshape(1, -1)
            _, h = engine.hierarchical_prefill(inp)
            ct = h.level_states[-1].astype(mx.float32)
            c_top = mx.mean(ct, axis=tuple(range(ct.ndim - 1)))
            cm_raw = (
                h.level_states[0].astype(mx.float32) if len(h.level_states) >= 2 else ct
            )
            c_mid = mx.mean(cm_raw, axis=tuple(range(cm_raw.ndim - 1)))
            if h.token_proj is not None:
                cto_raw = h.token_proj.astype(mx.float32)
                c_token = mx.mean(cto_raw, axis=tuple(range(cto_raw.ndim - 1)))
            else:
                c_token = c_top

            sim_top = _batch_cosine_similarity(q_top, c_top[None, :])
            sim_mid = _batch_cosine_similarity(q_mid, c_mid[None, :])
            sim_token = _batch_cosine_similarity(q_token, c_token[None, :])
            stacked = mx.stack([sim_token, sim_mid, sim_top], axis=-1)
            combined = weighted_hierarchical_score(stacked, weights)
            mx.eval(combined)
            seq_scores.append((idx, float(combined.tolist()[0])))

        # Raw-score check: batched hierarchical matches sequential within 1e-3.
        for (i_b, s_b), (i_s, s_s) in zip(batched_scores, seq_scores):
            assert i_b == i_s
            assert abs(s_b - s_s) <= 1e-3, (
                f"mixed-length raw score mismatch at idx {i_b}: "
                f"batched={s_b} sequential={s_s} delta={abs(s_b - s_s)}"
            )

        # Top-K selection match.
        result_batched = engine.prune_evidence(texts, cids, "s1", max_chunks=4)
        seq_top = sorted(i for i, _ in sorted(seq_scores, key=lambda x: -x[1])[:4])
        assert result_batched == seq_top


# ---------------------------------------------------------------
# Issue #61: pure-function and helper unit tests
# ---------------------------------------------------------------


class TestBatchCosineSimilarity:
    """Unit tests for _batch_cosine_similarity (pure function)."""

    def test_orthogonal_vectors_zero_similarity(self) -> None:
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        keys = mx.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        sims = _batch_cosine_similarity(q, keys)
        for s in sims.tolist():
            assert abs(s) < 1e-6

    def test_identical_vectors_one_similarity(self) -> None:
        q = mx.array([0.5, 0.25, -0.75, 1.0])
        keys = mx.stack([q, q * 2.0, q * -1.0])  # parallel/antiparallel
        sims_list = _batch_cosine_similarity(q, keys).tolist()
        # 0: identical → +1
        assert abs(sims_list[0] - 1.0) < 1e-5
        # 1: positive scalar multiple → +1 (cosine ignores magnitude)
        assert abs(sims_list[1] - 1.0) < 1e-5
        # 2: anti-parallel → -1
        assert abs(sims_list[2] + 1.0) < 1e-5

    def test_numerical_stability_zero_norm(self) -> None:
        """eps must keep results finite when one of the vectors is zero."""
        q = mx.array([0.0, 0.0, 0.0])
        keys = mx.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        sims = _batch_cosine_similarity(q, keys, eps=1e-8)
        for s in sims.tolist():
            # Value must be finite (not NaN / inf).
            assert s == s  # not NaN
            assert -1.0 <= s <= 1.0

    def test_pad_token_id_constant(self) -> None:
        """PAD_TOKEN_ID is 0 — a regression guard for the SSOT contract."""
        assert PAD_TOKEN_ID == 0

    def test_micro_batch_size_default(self) -> None:
        """MICRO_BATCH_SIZE default matches the production cap."""
        assert MICRO_BATCH_SIZE == 64


# ---------------------------------------------------------------
# Issue #64: Cross-turn working memory (Phase 1)
# ---------------------------------------------------------------


def _mk_state(value: float, hidden: int = 4) -> HierarchicalState:
    """Build a HierarchicalState whose top-level state is a constant vector."""
    top = mx.ones((1, 2, hidden)) * value
    return HierarchicalState(level_states=[top])


class TestWorkingMemory:
    """Issue #64 — PhotonSessionState cross-turn working memory."""

    def test_turn_history_accumulates(self) -> None:
        """update() must append one TurnState per turn up to max_turns."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(enabled=True, max_turns=3),
        )
        for i in range(5):
            session.update(_mk_state(float(i + 1)), question_text=f"q{i + 1}")

        # max_turns=3 → oldest two turns dropped
        assert len(session.turn_history) == 3
        assert [t.turn_id for t in session.turn_history] == [3, 4, 5]
        assert all(isinstance(t, TurnState) for t in session.turn_history)
        assert session.turn_history[-1].question_text == "q5"

    def test_get_session_coarse_state_weighted_average(self) -> None:
        """Weighted average uses decay_factor ** (N-i-1)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, decay_factor=0.5
            ),
        )
        # Three turns with scalar-encoded top states: values 1, 2, 4.
        session.update(_mk_state(1.0))
        session.update(_mk_state(2.0))
        session.update(_mk_state(4.0))

        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        # Analytical expected value:
        #   weights = [0.25, 0.5, 1.0], sum = 1.75
        #   weighted sum = 1*0.25 + 2*0.5 + 4*1.0 = 5.25
        #   result = 5.25 / 1.75 = 3.0 (broadcast across hidden dim)
        vals = coarse.tolist()
        for v in vals:
            assert abs(v - 3.0) < 1e-5

    def test_working_memory_enabled_false_preserves_legacy(self) -> None:
        """enabled=False must leave turn_history empty and coarse state None."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(enabled=False),
        )
        session.update(_mk_state(1.0), question_text="q")
        session.update(_mk_state(2.0), question_text="q2")

        assert session.turn_history == []
        assert session.get_session_coarse_state() is None
        # Legacy current_state path still functional
        assert session.current_state is not None

    def test_prev_logits_preserved_across_update(self) -> None:
        """Second-turn update retains prior logits to compute logit_kl."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(enabled=True),
        )
        logits_1 = mx.random.normal((1, 8, 64))
        session.update(_mk_state(1.0), logits_1)
        # After first update(), prev_logits tracks the most recent logits so
        # the next turn can compute drift.
        assert session.prev_logits is not None
        logits_2 = mx.random.normal((1, 8, 64))
        drift = session.update(_mk_state(2.0), logits_2)
        assert drift.logit_kl >= 0.0
        # prev_logits rolled forward to logits_2 for the next turn
        assert session.prev_logits is logits_2

    def test_drift_invariance_across_working_memory_toggle(self) -> None:
        """latent_cosine_drift / logit_kl must be identical with WM on vs off."""
        mx.random.seed(123)
        s1_logits = [mx.random.normal((1, 8, 64)) for _ in range(2)]
        mx.random.seed(123)
        s2_logits = [mx.random.normal((1, 8, 64)) for _ in range(2)]

        session_on = PhotonSessionState(
            "on",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(enabled=True),
        )
        session_off = PhotonSessionState(
            "off",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(enabled=False),
        )
        for v, l_on, l_off in zip((1.0, 2.0), s1_logits, s2_logits, strict=True):
            d_on = session_on.update(_mk_state(v), l_on)
            d_off = session_off.update(_mk_state(v), l_off)
            assert abs(d_on.latent_cosine_drift - d_off.latent_cosine_drift) < 1e-6
            assert abs(d_on.logit_kl - d_off.logit_kl) < 1e-6

    # ----------------------------------------------------------------
    # Issue #78 — find_relevant_past_turn()
    # ----------------------------------------------------------------
    #
    # T1..T11 cover the reference implementation in the design doc §4.5.
    # ``_mk_state(v)`` (defined above) yields a constant top-level vector and
    # therefore only produces cosine similarities of sign(v1)*sign(v2); tests
    # that need fractional similarities or non-finite values build the
    # ``HierarchicalState`` directly via ``_mk_state_vec`` / ``_mk_state_nan``.

    @staticmethod
    def _mk_state_vec(vec: list[float]) -> HierarchicalState:
        """Build a HierarchicalState whose top-level state is the given 1-D vector.

        The vector is wrapped to ``(1, 1, D)`` so ``mean_pool`` produces the
        same ``(D,)`` values back — giving tests deterministic control over the
        cosine similarity between states.
        """
        arr = mx.array([[vec]], dtype=mx.float32)
        return HierarchicalState(level_states=[arr])

    @staticmethod
    def _mk_state_nan() -> HierarchicalState:
        """Build a HierarchicalState guaranteed to yield a non-finite similarity."""
        import math as _math

        arr = mx.array([[[_math.nan, 0.0, 0.0, 0.0]]], dtype=mx.float32)
        return HierarchicalState(level_states=[arr])

    def test_find_relevant_past_turn_empty_history(self) -> None:
        """T1: turn_history is empty → None (no update() has been called)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(enabled=True),
        )
        assert session.turn_history == []
        assert session.find_relevant_past_turn(_mk_state(1.0)) is None

    def test_find_relevant_past_turn_single_turn(self) -> None:
        """T2: only the current turn is recorded (len == 1) → None."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(enabled=True),
        )
        session.update(_mk_state(1.0), question_text="q1")
        assert len(session.turn_history) == 1
        # current_state is the just-updated state, so this emulates the
        # production call pattern (update() then find_relevant_past_turn()).
        assert session.find_relevant_past_turn(session.current_state) is None

    def test_find_relevant_past_turn_returns_best_match(self) -> None:
        """T3: multiple past turns above threshold → highest similarity wins."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, relevant_turn_threshold=0.7
            ),
        )
        # past turn 1: identical direction to current → sim = 1.0
        session.update(self._mk_state_vec([1.0, 0.0, 0.0, 0.0]), question_text="past1")
        # past turn 2: cos_sim = 4/5 = 0.8 against current (above threshold)
        session.update(self._mk_state_vec([4.0, 3.0, 0.0, 0.0]), question_text="past2")
        # current turn (turn_history[-1])
        current = self._mk_state_vec([1.0, 0.0, 0.0, 0.0])
        session.update(current, question_text="current")

        result = session.find_relevant_past_turn(current)
        assert result is not None
        # past1 (turn_id=1) has sim=1.0, past2 (turn_id=2) has sim=0.8.
        assert result.turn_id == 1
        assert result.question_text == "past1"

    def test_find_relevant_past_turn_below_threshold(self) -> None:
        """T4: no past turn clears the threshold → None."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, relevant_turn_threshold=0.7
            ),
        )
        # past1: sim = 3/5 = 0.6 vs current [1, 0, 0, 0] (below 0.7)
        session.update(self._mk_state_vec([3.0, 4.0, 0.0, 0.0]), question_text="p1")
        # past2: sim = 0.0 (orthogonal)
        session.update(self._mk_state_vec([0.0, 1.0, 0.0, 0.0]), question_text="p2")
        current = self._mk_state_vec([1.0, 0.0, 0.0, 0.0])
        session.update(current, question_text="current")

        assert session.find_relevant_past_turn(current) is None

    def test_find_relevant_past_turn_at_threshold_boundary(self) -> None:
        """T5: sim exactly equal to threshold → returned (>= admits equality)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, relevant_turn_threshold=1.0
            ),
        )
        # past1 and current share the same unit direction → sim == 1.0 == threshold
        session.update(_mk_state(1.0), question_text="past1")
        current = _mk_state(1.0)
        session.update(current, question_text="current")

        result = session.find_relevant_past_turn(current)
        assert result is not None
        assert result.turn_id == 1
        assert result.question_text == "past1"

    def test_find_relevant_past_turn_disabled_returns_none(self) -> None:
        """T6: enabled=False → None, without mutating session state."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(enabled=False),
        )
        # Real update() won't populate turn_history when disabled. We manually
        # append two TurnStates to prove the enabled-gate fires independently
        # of the len(turn_history) guard.
        session.turn_history.append(
            TurnState(turn_id=1, hierarchical_state=_mk_state(1.0), question_text="p1")
        )
        session.turn_history.append(
            TurnState(turn_id=2, hierarchical_state=_mk_state(1.0), question_text="cur")
        )
        before_len = len(session.turn_history)

        assert session.find_relevant_past_turn(_mk_state(1.0)) is None
        # No side effects on turn_history.
        assert len(session.turn_history) == before_len
        assert [t.turn_id for t in session.turn_history] == [1, 2]

    def test_find_relevant_past_turn_skips_non_finite(self) -> None:
        """T7: non-finite sim on one past turn is skipped, best finite wins."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, relevant_turn_threshold=0.5
            ),
        )
        # past1: contains NaN → cos_sim produces NaN → skipped
        session.update(self._mk_state_nan(), question_text="nan_turn")
        # past2: sim = 1.0 with current (same direction)
        session.update(_mk_state(1.0), question_text="finite_turn")
        current = _mk_state(1.0)
        session.update(current, question_text="current")

        result = session.find_relevant_past_turn(current)
        assert result is not None
        # Only the finite past turn (turn_id=2) is eligible.
        assert result.turn_id == 2
        assert result.question_text == "finite_turn"

    def test_find_relevant_past_turn_tiebreak_by_latest_turn_id(self) -> None:
        """T8: identical similarities → latest turn_id wins."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, relevant_turn_threshold=0.7
            ),
        )
        # Two past turns with the same direction as the current turn.
        session.update(_mk_state(1.0), question_text="older")
        session.update(_mk_state(1.0), question_text="newer")
        current = _mk_state(1.0)
        session.update(current, question_text="current")

        result = session.find_relevant_past_turn(current)
        assert result is not None
        # Latest past turn (turn_id=2) wins over older past (turn_id=1).
        assert result.turn_id == 2
        assert result.question_text == "newer"

    def test_find_relevant_past_turn_skips_empty_level_states(self) -> None:
        """T9: one past turn with empty level_states is skipped, others evaluated."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, relevant_turn_threshold=0.7
            ),
        )
        # past1: normal turn with sim = 1.0
        session.update(_mk_state(1.0), question_text="normal_past")
        # past2 with empty level_states (hand-crafted to bypass update()'s own
        # invariant — we replace the HierarchicalState on an existing turn).
        session.update(_mk_state(1.0), question_text="empty_past")
        session.turn_history[-1].hierarchical_state = HierarchicalState(level_states=[])
        current = _mk_state(1.0)
        session.update(current, question_text="current")

        result = session.find_relevant_past_turn(current)
        assert result is not None
        # The empty-level_states turn must be skipped.
        assert result.turn_id == 1
        assert result.question_text == "normal_past"

    def test_find_relevant_past_turn_all_past_level_states_empty(self) -> None:
        """T10: every past turn has empty level_states → None (scores empty)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, relevant_turn_threshold=0.7
            ),
        )
        session.update(_mk_state(1.0), question_text="p1")
        session.update(_mk_state(1.0), question_text="p2")
        # Blank out the level_states of every *past* turn (leave current alone).
        current = _mk_state(1.0)
        session.update(current, question_text="current")
        for past in session.turn_history[:-1]:
            past.hierarchical_state = HierarchicalState(level_states=[])

        assert session.find_relevant_past_turn(current) is None

    def test_find_relevant_past_turn_all_non_finite_returns_none(self) -> None:
        """T11: every past turn yields non-finite sim → None (fail-closed)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, relevant_turn_threshold=0.5
            ),
        )
        session.update(self._mk_state_nan(), question_text="nan_a")
        session.update(self._mk_state_nan(), question_text="nan_b")
        current = _mk_state(1.0)
        session.update(current, question_text="current")

        assert session.find_relevant_past_turn(current) is None


class TestSessionStateReset:
    """Issue #64 — reset_working_memory() preserves telemetry only."""

    def test_reset_clears_latents_and_history_preserves_drift(self) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(enabled=True),
        )
        logits_1 = mx.random.normal((1, 8, 64))
        session.update(_mk_state(1.0), logits_1, question_text="q1")
        logits_2 = mx.random.normal((1, 8, 64))
        session.update(_mk_state(2.0), logits_2, question_text="q2")

        assert session.current_state is not None
        assert session.prev_state is not None
        assert session.prev_logits is not None
        assert len(session.turn_history) == 2
        drift_len_before = len(session.drift_history)
        turn_count_before = session.turn_count

        session.reset_working_memory()

        # Cleared
        assert session.current_state is None
        assert session.prev_state is None
        assert session.prev_logits is None
        assert session.turn_history == []
        # Issue #79 DR1-010: compressed_history also cleared atomically.
        assert session.compressed_history == []
        # Preserved
        assert len(session.drift_history) == drift_len_before
        assert session.turn_count == turn_count_before


class TestWorkingMemorySecurityRegression:
    """Issue #64 — DR4 security regression coverage."""

    def test_working_memory_config_rejects_non_bool_enabled(self) -> None:
        with pytest.raises(TypeError, match="enabled must be bool"):
            WorkingMemoryConfig(enabled="true")  # type: ignore[arg-type]

    def test_working_memory_config_rejects_bad_max_turns(self) -> None:
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            WorkingMemoryConfig(max_turns=0)
        with pytest.raises(TypeError):
            WorkingMemoryConfig(max_turns=1.5)  # type: ignore[arg-type]

    def test_working_memory_config_rejects_non_finite_decay(self) -> None:
        import math as _math

        with pytest.raises(ValueError):
            WorkingMemoryConfig(decay_factor=_math.nan)
        with pytest.raises(ValueError):
            WorkingMemoryConfig(decay_factor=1.5)
        with pytest.raises(ValueError):
            WorkingMemoryConfig(relevant_turn_threshold=_math.inf)

    def test_drift_metrics_as_dict_coerces_nan_and_inf(self) -> None:
        import math as _math
        import warnings as _warnings

        # Issue #63 made ``latent_cosine_drift`` a read-only @property alias
        # for ``latent_cosine_drift_top`` (DR1-009), so to drive a NaN through
        # the legacy alias we have to set the underlying _top field directly.
        m = DriftMetrics(
            turn_id=3,
            latent_cosine_drift_top=_math.nan,
            token_agreement=_math.inf,
            logit_kl=-_math.inf,
            topic_shift_score=0.25,
        )
        with _warnings.catch_warnings(record=True) as captured:
            _warnings.simplefilter("always")
            out = m.as_dict()
        # Non-finite values coerced to 0.0, finite value preserved.
        assert out["latent_cosine_drift"] == 0.0
        assert out["latent_cosine_drift_top"] == 0.0
        assert out["token_agreement"] == 0.0
        assert out["logit_kl"] == 0.0
        assert abs(out["topic_shift_score"] - 0.25) < 1e-9
        # JSON-safety: no NaN / Inf tokens in the NUMERIC fields. Issue #92
        # adds mode/alpha fields of types str | None and float | None so we
        # finite-check only the drift numeric keys here.
        numeric_keys = {
            "turn_id",
            "latent_cosine_drift",
            "latent_cosine_drift_top",
            "latent_cosine_drift_mid",
            "latent_cosine_drift_token",
            "token_agreement",
            "logit_kl",
            "topic_shift_score",
        }
        for key in numeric_keys:
            assert _math.isfinite(out[key])
        # A warning was surfaced for each non-finite coercion. The alias
        # ``latent_cosine_drift`` and the authoritative ``_top`` field both
        # resolve to NaN, so at minimum the 4 coercions (drift + alias +
        # token_agreement + logit_kl) are reported.
        assert len(captured) >= 3
        # No leaked payload: warning text must reference the field name only.
        for rec in captured:
            msg = str(rec.message)
            assert "latent" in msg or "agreement" in msg or "logit_kl" in msg
            assert "q1" not in msg

    def test_reset_warning_does_not_leak_question_text(self) -> None:
        """reset_working_memory() runs silently; no question_text is logged."""
        import io
        import logging

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("photon_mlx.session")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            session = PhotonSessionState(
                "s1",
                "repo",
                "abc",
                working_memory_cfg=WorkingMemoryConfig(enabled=True),
            )
            session.update(_mk_state(1.0), question_text="SECRET-USER-QUESTION")
            session.reset_working_memory()
        finally:
            logger.removeHandler(handler)

        assert "SECRET-USER-QUESTION" not in stream.getvalue()


class TestCodexCB001NoneDisablesWorkingMemory:
    """Codex CB-001: explicit ``None`` must be treated as disabled.

    ``_build_photon_deps()`` returns ``None`` when the YAML is malformed or
    the section is missing (fail-closed, design §7). The session layer must
    honour that disable signal instead of silently re-enabling working memory.
    """

    def test_explicit_none_yields_disabled_working_memory(self) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=None,
        )
        assert session.working_memory_cfg.enabled is False

    def test_explicit_none_skips_turn_history(self) -> None:
        """When None is passed, update() must not accumulate turn_history."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=None,
        )
        session.update(_mk_state(1.0), question_text="SECRET-Q")
        session.update(_mk_state(2.0), question_text="SECRET-Q2")
        assert session.turn_history == []
        assert session.get_session_coarse_state() is None

    def test_unset_default_keeps_working_memory_enabled(self) -> None:
        """Callers that omit working_memory_cfg keep the legacy default on."""
        # No ``working_memory_cfg`` argument at all → default sentinel path.
        session = PhotonSessionState("s1", "repo", "abc")
        assert session.working_memory_cfg.enabled is True

    def test_inference_propagates_none_as_disabled(
        self, stub_tokenizer_for_cfg
    ) -> None:
        """PhotonInference(working_memory_cfg=None).get_session must be disabled."""
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        tokenizer = stub_tokenizer_for_cfg(cfg)
        engine = PhotonInference(
            model,
            cfg,
            tokenizer,
            working_memory_cfg=None,
        )
        session = engine.get_session("s1", "repo", "abc")
        assert session.working_memory_cfg.enabled is False


class TestCodexCB003PruneTokenizerLogHygiene:
    """Codex CB-003: _tokenize_chunk warning must not leak raw exception text.

    The tokenizer can raise with input fragments in the message (question
    text on Pass 1, repo chunk text on Turn 2+). Warning must expose only
    ``type(exc).__name__`` (design §7).
    """

    def test_tokenize_chunk_warning_contains_only_type_name(
        self, caplog, stub_tokenizer_for_cfg
    ) -> None:
        import logging

        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        tokenizer = stub_tokenizer_for_cfg(cfg)
        engine = PhotonInference(model, cfg, tokenizer)

        SENSITIVE = "LEAKED-CHUNK-TEXT-abcdef123"

        class _EvilTokenizer:
            pad_token_id = 0

            def encode(self, text: str):
                raise RuntimeError(SENSITIVE)

        engine.tokenizer = _EvilTokenizer()

        with caplog.at_level(logging.WARNING, logger="photon_mlx.inference"):
            with pytest.raises(Exception):
                # propagates as _TokenizerEncodeFailure
                engine._tokenize_chunk("some repo chunk text")

        combined = " ".join(r.getMessage() for r in caplog.records)
        assert SENSITIVE not in combined
        assert "RuntimeError" in combined


class TestWorkingMemoryConfigAggregation:
    """Issue #80: aggregation mode field on WorkingMemoryConfig."""

    def test_working_memory_config_aggregation_default_weighted(self) -> None:
        """Default aggregation must be ``weighted`` for backward-compat."""
        cfg = WorkingMemoryConfig()
        assert cfg.aggregation == "weighted"

    def test_working_memory_config_aggregation_accepts_valid_values(self) -> None:
        """All four documented modes must construct successfully.

        Issue #92 adds ``"dynamic"`` to the closed-enum. The three legacy
        modes (weighted/attention/last) continue to construct unchanged.
        """
        for mode in ("weighted", "attention", "last", "dynamic"):
            cfg = WorkingMemoryConfig(aggregation=mode)
            assert cfg.aggregation == mode

    def test_working_memory_config_aggregation_rejects_non_str(self) -> None:
        """Non-str raises TypeError with type name only (no raw value leak).

        CB-001 / AT-001 (refactor): both invariants are enforced as
        *unconditional* asserts so a regression that starts leaking the
        raw payload into the error message is guaranteed to fail this
        test. Each non-str payload embeds a unique sentinel token whose
        ``repr`` cannot be an incidental substring of the type name or
        anything else session.py produces.
        """
        # Sentinel payloads: each has a ``repr`` containing a rare, unique
        # token that cannot incidentally appear in the type-name-only
        # error message. If session.py ever surfaces the raw value (e.g.
        # via ``{self.aggregation!r}``), the sentinel shows up in ``msg``
        # and the unconditional ``assert repr(bad) not in msg`` below
        # fails.
        #
        # NOTE: ``None`` is intentionally excluded because ``repr(None) ==
        # "None"`` is a substring of the type name ``"NoneType"``, which
        # would give a false positive. The empty-case (``None`` passed as
        # cfg) is covered at the WorkingMemoryConfig level by
        # ``test_working_memory_config_aggregation_rejects_unknown_value``
        # via distinct sentinels; the same type-branch logic runs for
        # ``None`` regardless.
        sentinels: tuple[object, ...] = (
            4242424242,  # rare int far from any known constant
            1.5,
            ["WM_RAW_LEAK_SENTINEL_DO_NOT_APPEAR"],
            {"WM_RAW_LEAK_DICT_SENTINEL_DO_NOT_APPEAR": 1},
            ("WM_RAW_LEAK_TUPLE_SENTINEL_DO_NOT_APPEAR",),
            b"WM_RAW_LEAK_BYTES_SENTINEL_DO_NOT_APPEAR",
        )

        for bad in sentinels:
            with pytest.raises(TypeError) as excinfo:
                WorkingMemoryConfig(aggregation=bad)  # type: ignore[arg-type]
            msg = str(excinfo.value)
            # Hard no-leak invariant (design §6 / DR4-001): the raw value's
            # ``repr`` must NEVER appear in the surfaced error.
            assert repr(bad) not in msg, (
                f"raw value leaked into TypeError for {type(bad).__name__}: {msg!r}"
            )
            # Separately, the type name IS required so operators can
            # diagnose the malformed YAML without seeing attacker-controlled
            # content.
            assert type(bad).__name__ in msg, (
                f"type name missing from TypeError for {type(bad).__name__}: {msg!r}"
            )

        # ``None`` case: verify the TypeError still fires with type-name
        # surface only (no assertion on repr(None) because it is a
        # substring of "NoneType" and would incidentally appear).
        with pytest.raises(TypeError) as excinfo:
            WorkingMemoryConfig(aggregation=None)  # type: ignore[arg-type]
        assert "NoneType" in str(excinfo.value)

    def test_working_memory_config_aggregation_rejects_unknown_value(self) -> None:
        """Unknown modes raise ValueError without leaking the raw value.

        CB-001 / AT-001 (refactor): no-leak invariant is enforced
        unconditionally. The only exempted payload is the empty string,
        which is handled by a dedicated branch (``""`` is trivially a
        substring of every string so ``assert "" not in msg`` would be
        vacuous; empty case is covered by the ValueError raise itself).
        """
        SENSITIVE = "<ATTACK-PAYLOAD-WM-AGGR-SENTINEL>"
        with pytest.raises(ValueError) as excinfo:
            WorkingMemoryConfig(aggregation=SENSITIVE)
        assert SENSITIVE not in str(excinfo.value)

        for bad in ("mean", "WEIGHTED", " weighted ", "WM_UNKNOWN_SENTINEL"):
            with pytest.raises(ValueError) as excinfo:
                WorkingMemoryConfig(aggregation=bad)
            # The raw attacker-controlled string must not appear in the error.
            assert bad not in str(excinfo.value), (
                f"raw value {bad!r} leaked into ValueError: {excinfo.value!s}"
            )

        # Empty string: verify the raise still happens (no-leak check is
        # vacuous for the empty string).
        with pytest.raises(ValueError):
            WorkingMemoryConfig(aggregation="")


class TestDynamicStrategyConfigValidation:
    """Issue #92 T-2: validation rules for the five new dynamic fields.

    Closed-enum and finite-float invariants follow the existing DR4-001
    no-leak pattern — raw attacker-controlled values must never surface in
    the error messages.
    """

    # ---- dynamic_strategy (closed enum) ----
    @pytest.mark.parametrize("strategy", ["turn_position", "drift_based", "hybrid"])
    def test_dynamic_strategy_accepts_valid_values(self, strategy: str) -> None:
        cfg = WorkingMemoryConfig(aggregation="dynamic", dynamic_strategy=strategy)
        assert cfg.dynamic_strategy == strategy

    def test_dynamic_strategy_default_is_turn_position(self) -> None:
        cfg = WorkingMemoryConfig()
        assert cfg.dynamic_strategy == "turn_position"

    def test_dynamic_strategy_rejects_non_str(self) -> None:
        for bad in (1, 1.5, ["hybrid"], None):
            with pytest.raises(TypeError) as excinfo:
                WorkingMemoryConfig(dynamic_strategy=bad)  # type: ignore[arg-type]
            # Type name is surfaced; raw repr of the payload is not
            # (except when ``None`` where ``repr(None)="None"`` is a
            # substring of ``NoneType`` — we skip that assert for None).
            assert type(bad).__name__ in str(excinfo.value)
            if bad is not None:
                assert repr(bad) not in str(excinfo.value)

    def test_dynamic_strategy_rejects_unknown_value_no_leak(self) -> None:
        SENSITIVE = "<ATTACK-DYN-STRATEGY-SENTINEL>"
        with pytest.raises(ValueError) as excinfo:
            WorkingMemoryConfig(dynamic_strategy=SENSITIVE)
        assert SENSITIVE not in str(excinfo.value)

        for bad in ("turnposition", "hybrid_plus", "TURN_POSITION"):
            with pytest.raises(ValueError) as excinfo:
                WorkingMemoryConfig(dynamic_strategy=bad)
            assert bad not in str(excinfo.value)

    # ---- weighted_until_turn ----
    def test_weighted_until_turn_default_is_three(self) -> None:
        cfg = WorkingMemoryConfig()
        assert cfg.weighted_until_turn == 3

    @pytest.mark.parametrize("value", [0, 1, 3, 8])
    def test_weighted_until_turn_accepts_non_negative(self, value: int) -> None:
        cfg = WorkingMemoryConfig(weighted_until_turn=value)
        assert cfg.weighted_until_turn == value

    def test_weighted_until_turn_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="weighted_until_turn"):
            WorkingMemoryConfig(weighted_until_turn=-1)

    def test_weighted_until_turn_rejects_non_int(self) -> None:
        for bad in (1.5, "3", True):
            with pytest.raises(TypeError):
                WorkingMemoryConfig(weighted_until_turn=bad)  # type: ignore[arg-type]

    def test_weighted_until_turn_warns_when_above_max_turns(self) -> None:
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as captured:
            _warnings.simplefilter("always")
            cfg = WorkingMemoryConfig(max_turns=3, weighted_until_turn=10)
        assert cfg.weighted_until_turn == 10  # stored even though above cap
        assert any("weighted_until_turn" in str(w.message) for w in captured)

    # ---- attention_drift_threshold (finite float in [0, 1]) ----
    def test_attention_drift_threshold_default(self) -> None:
        cfg = WorkingMemoryConfig()
        assert cfg.attention_drift_threshold == 0.5

    @pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_attention_drift_threshold_accepts_in_range(self, value: float) -> None:
        cfg = WorkingMemoryConfig(attention_drift_threshold=value)
        assert cfg.attention_drift_threshold == value

    def test_attention_drift_threshold_rejects_out_of_range(self) -> None:
        import math as _math

        for bad in (-0.1, 1.1, 2.0, _math.nan, _math.inf, -_math.inf):
            with pytest.raises(ValueError, match="attention_drift_threshold"):
                WorkingMemoryConfig(attention_drift_threshold=bad)

    def test_attention_drift_threshold_rejects_non_float(self) -> None:
        for bad in ("0.5", [0.5], None, True):
            with pytest.raises(TypeError):
                WorkingMemoryConfig(attention_drift_threshold=bad)  # type: ignore[arg-type]

    # ---- hybrid_alpha_base ----
    def test_hybrid_alpha_base_default(self) -> None:
        cfg = WorkingMemoryConfig()
        assert cfg.hybrid_alpha_base == 0.5

    @pytest.mark.parametrize("value", [-5.0, 0.0, 0.25, 1.0, 10.0])
    def test_hybrid_alpha_base_accepts_finite(self, value: float) -> None:
        # Finite floats are accepted; clamp happens at dispatch time.
        cfg = WorkingMemoryConfig(hybrid_alpha_base=value)
        assert cfg.hybrid_alpha_base == value

    def test_hybrid_alpha_base_rejects_non_finite(self) -> None:
        import math as _math

        for bad in (_math.nan, _math.inf, -_math.inf):
            with pytest.raises(ValueError, match="hybrid_alpha_base"):
                WorkingMemoryConfig(hybrid_alpha_base=bad)

    def test_hybrid_alpha_base_rejects_non_float(self) -> None:
        for bad in ("0.5", [0.5], None, True):
            with pytest.raises(TypeError):
                WorkingMemoryConfig(hybrid_alpha_base=bad)  # type: ignore[arg-type]

    # ---- hybrid_alpha_per_turn ----
    def test_hybrid_alpha_per_turn_default(self) -> None:
        cfg = WorkingMemoryConfig()
        assert cfg.hybrid_alpha_per_turn == 0.1

    @pytest.mark.parametrize("value", [-1.0, 0.0, 0.1, 5.0])
    def test_hybrid_alpha_per_turn_accepts_finite(self, value: float) -> None:
        cfg = WorkingMemoryConfig(hybrid_alpha_per_turn=value)
        assert cfg.hybrid_alpha_per_turn == value

    def test_hybrid_alpha_per_turn_rejects_non_finite(self) -> None:
        import math as _math

        for bad in (_math.nan, _math.inf, -_math.inf):
            with pytest.raises(ValueError, match="hybrid_alpha_per_turn"):
                WorkingMemoryConfig(hybrid_alpha_per_turn=bad)


def _mk_mode_cfg(aggregation_param: str, **kwargs: object) -> WorkingMemoryConfig:
    """Build a :class:`WorkingMemoryConfig` for parametrized mode tests.

    Accepts either a static mode (``"weighted"``/``"attention"``/``"last"``)
    or a ``"dynamic-<strategy>"`` tag (Issue #92 T-6). This collapses the
    Issue #80 3-mode matrix and the Issue #92 3-strategy matrix into a
    single 6-value parametrize for the (a) common-contract tests.
    """
    if aggregation_param.startswith("dynamic-"):
        strategy = aggregation_param.split("-", 1)[1]
        return WorkingMemoryConfig(
            aggregation="dynamic",
            dynamic_strategy=strategy,
            **kwargs,  # type: ignore[arg-type]
        )
    return WorkingMemoryConfig(aggregation=aggregation_param, **kwargs)  # type: ignore[arg-type]


# Parametrize ids used by TestGetSessionCoarseStateModes (a) common
# contract: legacy 3 static + 3 dynamic strategies (Issue #92 T-6 (b)).
_MODE_IDS_ALL: tuple[str, ...] = (
    "weighted",
    "attention",
    "last",
    "dynamic-turn_position",
    "dynamic-drift_based",
    "dynamic-hybrid",
)


class TestGetSessionCoarseStateModes:
    """Issue #80 + #92: mode dispatch for ``get_session_coarse_state()``.

    Four categories (design §8 DR1-009):
    - (a) common contract across modes (shape, dtype, None)
    - (b) mode-specific edge cases
    - (c) validation / defensive raise
    - (d) prune parity (covered in TestPruneParityAcrossAggregationModes)

    Issue #92 T-6 (b) extends the (a) common-contract parametrize to also
    cover the 3 new dynamic strategies.
    """

    # ---- (a) common contract across modes ----
    @pytest.mark.parametrize("aggregation", list(_MODE_IDS_ALL))
    def test_all_modes_shape_dtype(self, aggregation: str) -> None:
        """All modes (static 3 + dynamic 3) must return (D,) float32 vectors."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=_mk_mode_cfg(aggregation, max_turns=4),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(2.0))
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        # _mk_state(...) uses hidden=4 by default.
        assert coarse.shape == (4,)
        assert coarse.dtype == mx.float32

    @pytest.mark.parametrize("aggregation", list(_MODE_IDS_ALL))
    def test_all_modes_empty_turn_history_returns_none(self, aggregation: str) -> None:
        """All modes return None on empty turn_history."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=_mk_mode_cfg(aggregation),
        )
        assert session.get_session_coarse_state() is None

    @pytest.mark.parametrize("aggregation", list(_MODE_IDS_ALL))
    def test_all_modes_disabled_returns_none(self, aggregation: str) -> None:
        """All modes return None when working memory is disabled."""
        # Even if aggregation is set, enabled=False short-circuits.
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=_mk_mode_cfg(aggregation, enabled=False),
        )
        session.update(_mk_state(1.0))
        assert session.get_session_coarse_state() is None

    @pytest.mark.parametrize("aggregation", list(_MODE_IDS_ALL))
    def test_all_modes_all_empty_level_states_returns_none(
        self, aggregation: str
    ) -> None:
        """All modes return None when every turn has empty level_states."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=_mk_mode_cfg(aggregation),
        )
        # Inject a turn whose hierarchical_state has empty level_states
        # (simulating the skip invariant).
        session.update(HierarchicalState(level_states=[]))
        session.update(HierarchicalState(level_states=[]))
        assert session.get_session_coarse_state() is None

    @pytest.mark.parametrize("aggregation", list(_MODE_IDS_ALL))
    def test_all_modes_single_turn_equivalent(self, aggregation: str) -> None:
        """Single-turn history: all modes must produce the same coarse vec."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=_mk_mode_cfg(aggregation),
        )
        session.update(_mk_state(2.5))
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        for v in coarse.tolist():
            assert abs(v - 2.5) < 1e-5

    # ---- (b) mode-specific edge cases ----
    def test_mode_last_returns_most_recent_valid_turn(self) -> None:
        """``last`` returns the mean-pooled coarse vec of the most recent turn."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, aggregation="last"
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(2.0))
        session.update(_mk_state(4.0))
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        for v in coarse.tolist():
            assert abs(v - 4.0) < 1e-5

    def test_mode_weighted_default_backward_compat(self) -> None:
        """Default (aggregation omitted) stays weighted — existing contract."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, decay_factor=0.5
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(2.0))
        session.update(_mk_state(4.0))
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        # Same analytical expected value as the legacy test (=3.0).
        for v in coarse.tolist():
            assert abs(v - 3.0) < 1e-5

    def test_mode_weighted_decay_zero_fallback(self) -> None:
        """``weighted`` with decay=0 on >1 turn falls back to ``vecs[-1]``."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, decay_factor=0.0, aggregation="weighted"
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(7.0))
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        # weight_sum == 0 → fallback_vec = vecs[-1] == 7.0.
        for v in coarse.tolist():
            assert abs(v - 7.0) < 1e-5

    def test_mode_attention_prefers_similar_past_turn(self) -> None:
        """attention softmax should weight past turns by similarity to current."""
        # Build a session where the current turn is close to turn 1 (value 1.0)
        # and far from turn 2 (value -1.0). The attention weighted sum should
        # be closer to 1.0 than a naive mean (0.0).
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, aggregation="attention"
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(-1.0))
        session.update(_mk_state(1.0))  # current — similar to turn 1
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        vals = coarse.tolist()
        # attention excludes the current turn, so candidates are {+1, -1}.
        # The current vec is +1 so softmax leans toward +1. All dims > 0.
        for v in vals:
            assert v > 0.0
            # Must be finite.
            assert not (v != v)  # not NaN

    def test_mode_attention_excludes_current_turn(self) -> None:
        """Current turn id must be excluded — no self-match domination."""
        # If current turn is NOT excluded, attention will dominate with cos=1
        # self-match → output equals current vec (-10.0). We verify exclusion
        # by making the current vec very large and confirming the output stays
        # within the past-turn range.
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, aggregation="attention"
            ),
        )
        session.update(_mk_state(2.0))
        session.update(_mk_state(3.0))
        session.update(_mk_state(-10.0))  # current — very different
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        vals = coarse.tolist()
        # Past-turn values are {2.0, 3.0}; after softmax over similarity to
        # current (-10.0), the result should be a convex combination and
        # definitely between 2 and 3.
        for v in vals:
            assert 2.0 - 1e-5 <= v <= 3.0 + 1e-5

    def test_mode_attention_fallback_none_current_state(self) -> None:
        """``current_state = None`` → attention fallback to vecs[-1]."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, aggregation="attention"
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(5.0))
        # Force current_state = None to simulate a partial state.
        session.current_state = None
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        for v in coarse.tolist():
            assert abs(v - 5.0) < 1e-5

    def test_mode_attention_fallback_empty_level_states(self) -> None:
        """``current_state.level_states = []`` → fallback to vecs[-1]."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, aggregation="attention"
            ),
        )
        session.update(_mk_state(2.0))
        session.update(_mk_state(6.0))
        # Replace current_state with a separate HierarchicalState that has
        # empty level_states — leaves turn_history intact so vecs[-1] still
        # has a valid candidate (simulating a partial/corrupt current state).
        assert session.current_state is not None
        session.current_state = HierarchicalState(level_states=[])
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        for v in coarse.tolist():
            assert abs(v - 6.0) < 1e-5

    def test_mode_attention_fallback_only_current_turn(self) -> None:
        """Attention with only current turn (0 past candidates) → vecs[-1]."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, aggregation="attention"
            ),
        )
        session.update(_mk_state(2.5))
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        for v in coarse.tolist():
            assert abs(v - 2.5) < 1e-5

    def test_mode_attention_hard_cap_numerical_stability(self) -> None:
        """attention at hard cap (max_turns=32) must not produce NaN/Inf."""
        from photon_mlx.session import WORKING_MEMORY_MAX_TURNS_HARD_CAP

        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=WORKING_MEMORY_MAX_TURNS_HARD_CAP,
                aggregation="attention",
            ),
        )
        # Tiny magnitudes stress the epsilon in norm denominators.
        for i in range(WORKING_MEMORY_MAX_TURNS_HARD_CAP):
            session.update(_mk_state(1e-10 * float(i + 1)))
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        import math as _math

        for v in coarse.tolist():
            assert _math.isfinite(v)

    # ---- (c) validation / defensive raise ----
    def test_unknown_mode_defensive_raise(self) -> None:
        """Post-init mutation to an unknown mode must fail-fast at dispatch."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, aggregation="weighted"
            ),
        )
        session.update(_mk_state(1.0))
        # Bypass __post_init__ via direct attribute assignment to simulate a
        # corrupted/deserialized object (design §8 decision #1 safety net).
        object.__setattr__(session.working_memory_cfg, "aggregation", "unknown_xxx")
        with pytest.raises(ValueError, match="Unknown aggregation mode"):
            session.get_session_coarse_state()


class TestPruneParityAcrossAggregationModes:
    """Issue #80 (DR3-002): ``_score_prune_candidates`` must be bit-for-bit
    equivalent across ``weighted`` / ``attention`` / ``last`` when only a
    single turn has been recorded.

    For single-turn sessions:

    - ``weighted``: ``vecs = [v]``, ``weights = [1.0]`` → returns ``v``.
    - ``last``: ``vecs[-1] == v``.
    - ``attention``: the current turn is the only candidate and is excluded,
      so helper returns ``None`` and the dispatcher falls back to ``vecs[-1] ==
      v``.

    All three produce the same overridden ``q_top`` and therefore the same
    scores (ε=1e-5).
    """

    @pytest.mark.parametrize(
        "aggregation",
        [
            "weighted",
            "attention",
            "last",
            # Issue #92 T-6 (c): dynamic 3 strategies with storage_mode="full"
            # — scope-bounded per plan (no 3x3 matrix, that lives in
            # TestDynamicAggregationStorageModeMatrix).
            "dynamic-turn_position",
            "dynamic-drift_based",
            "dynamic-hybrid",
        ],
    )
    def test_single_turn_prune_scores_match_across_modes(
        self, stub_tokenizer_for_cfg, aggregation: str
    ) -> None:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        tokenizer = stub_tokenizer_for_cfg(cfg)
        engine = PhotonInference(
            model,
            cfg,
            tokenizer,
            working_memory_cfg=_mk_mode_cfg(aggregation, storage_mode="full"),
        )
        ids = mx.random.randint(0, 256, (1, 16))
        engine.session_forward(ids, "s1", "repo", "abc")

        texts = [
            "short",
            "another short text",
            "medium length chunk content here",
            "longer chunk text " * 4,
            "some words",
        ]
        scores = engine._score_prune_candidates(texts, "s1")
        # Stash for cross-test comparison.
        setattr(
            self,
            f"_scores_{aggregation}",
            [s for _, s in scores],
        )

    def test_all_modes_identical_single_turn(self, stub_tokenizer_for_cfg) -> None:
        """Full cross-mode comparison: gather the 3 score lists and compare."""
        all_scores: dict[str, list[float]] = {}
        texts = [
            "short",
            "another short text",
            "medium length chunk content here",
            "longer chunk text " * 4,
            "some words",
        ]
        for aggregation in ("weighted", "attention", "last"):
            mx.random.seed(42)
            cfg = _tiny_cfg()
            model = PhotonModel(cfg)
            tokenizer = stub_tokenizer_for_cfg(cfg)
            engine = PhotonInference(
                model,
                cfg,
                tokenizer,
                working_memory_cfg=WorkingMemoryConfig(
                    enabled=True, aggregation=aggregation
                ),
            )
            ids = mx.random.randint(0, 256, (1, 16))
            engine.session_forward(ids, "s1", "repo", "abc")
            raw = engine._score_prune_candidates(texts, "s1")
            all_scores[aggregation] = [s for _, s in raw]

        # ε=1e-5 equivalence across the 3 modes on single-turn sessions.
        weighted_scores = all_scores["weighted"]
        for mode in ("attention", "last"):
            other = all_scores[mode]
            assert len(other) == len(weighted_scores)
            for i, (a, b) in enumerate(zip(weighted_scores, other)):
                assert abs(a - b) <= 1e-5, (
                    f"prune score mismatch at idx {i}: "
                    f"weighted={a} {mode}={b} delta={abs(a - b)}"
                )


class TestDynamicAggregation:
    """Issue #92 T-4/T-5/T-7: dynamic aggregation strategies + dispatcher.

    Exercises the three strategy helpers, the hybrid aggregator, and the
    dispatcher in :meth:`PhotonSessionState.get_session_coarse_state`,
    plus the per-turn mode recording on :class:`DriftMetrics`.
    """

    # ---------- (T-4) _dynamic_turn_position ----------
    def test_turn_position_weighted_at_threshold(self) -> None:
        """turn_count == weighted_until_turn → 'weighted' (inclusive)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=8,
                aggregation="dynamic",
                dynamic_strategy="turn_position",
                weighted_until_turn=3,
            ),
        )
        for i in range(3):
            session.update(_mk_state(float(i + 1)))
        mode, alpha = session._dynamic_turn_position()
        assert mode == "weighted"
        assert alpha is None

    def test_turn_position_attention_above_threshold(self) -> None:
        """turn_count > weighted_until_turn → 'attention'."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=8,
                aggregation="dynamic",
                dynamic_strategy="turn_position",
                weighted_until_turn=2,
            ),
        )
        for i in range(4):
            session.update(_mk_state(float(i + 1)))
        mode, alpha = session._dynamic_turn_position()
        assert mode == "attention"
        assert alpha is None

    def test_turn_position_weighted_zero_turns(self) -> None:
        """turn_count == 0 → 'weighted' (no turns yet)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                aggregation="dynamic",
                dynamic_strategy="turn_position",
                weighted_until_turn=3,
            ),
        )
        mode, alpha = session._dynamic_turn_position()
        assert mode == "weighted"
        assert alpha is None

    # ---------- (T-4) _dynamic_drift_based ----------
    def test_drift_based_empty_drift_history_returns_weighted(self) -> None:
        """drift_history empty (Turn 0 / reset) → ('weighted', None)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                aggregation="dynamic",
                dynamic_strategy="drift_based",
                attention_drift_threshold=0.5,
            ),
        )
        assert session.drift_history == []
        mode, alpha = session._dynamic_drift_based()
        assert mode == "weighted"
        assert alpha is None

    def test_drift_based_below_threshold_weighted(self) -> None:
        """drift <= threshold → 'weighted'."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                aggregation="dynamic",
                dynamic_strategy="drift_based",
                attention_drift_threshold=0.5,
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(1.0))  # identical → drift ~ 0
        mode, alpha = session._dynamic_drift_based()
        assert mode == "weighted"
        assert alpha is None

    def test_drift_based_above_threshold_attention(self) -> None:
        """drift > threshold → 'attention'."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                aggregation="dynamic",
                dynamic_strategy="drift_based",
                attention_drift_threshold=0.5,
            ),
        )
        # Force a drift above threshold by mutating the latest DriftMetrics.
        session.update(_mk_state(1.0))
        session.update(_mk_state(1.0))
        session.drift_history[-1].latent_cosine_drift_top = 0.9
        mode, alpha = session._dynamic_drift_based()
        assert mode == "attention"
        assert alpha is None

    def test_drift_based_at_threshold_boundary(self) -> None:
        """drift == threshold → 'weighted' (strict '>' per design)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                aggregation="dynamic",
                dynamic_strategy="drift_based",
                attention_drift_threshold=0.5,
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(1.0))
        session.drift_history[-1].latent_cosine_drift_top = 0.5  # == threshold
        mode, _alpha = session._dynamic_drift_based()
        assert mode == "weighted"

    # ---------- (T-4) _dynamic_hybrid_select ----------
    def test_hybrid_select_alpha_before_ramp(self) -> None:
        """turn_count <= weighted_until_turn → alpha = base (clamped)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=8,
                aggregation="dynamic",
                dynamic_strategy="hybrid",
                weighted_until_turn=3,
                hybrid_alpha_base=0.5,
                hybrid_alpha_per_turn=0.1,
            ),
        )
        for i in range(3):
            session.update(_mk_state(float(i + 1)))
        mode, alpha = session._dynamic_hybrid_select()
        assert mode == "hybrid"
        assert alpha == pytest.approx(0.5)

    def test_hybrid_select_alpha_after_ramp(self) -> None:
        """alpha = clamp(base + per_turn * (turn_count - until), 0, 1)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=8,
                aggregation="dynamic",
                dynamic_strategy="hybrid",
                weighted_until_turn=3,
                hybrid_alpha_base=0.5,
                hybrid_alpha_per_turn=0.1,
            ),
        )
        for i in range(5):  # turn_count = 5 → alpha = 0.5 + 0.1*2 = 0.7
            session.update(_mk_state(float(i + 1)))
        mode, alpha = session._dynamic_hybrid_select()
        assert mode == "hybrid"
        assert alpha == pytest.approx(0.7)

    def test_hybrid_select_alpha_clamped_above_one(self) -> None:
        """Huge ramp → alpha clamped to 1.0."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=8,
                aggregation="dynamic",
                dynamic_strategy="hybrid",
                weighted_until_turn=0,
                hybrid_alpha_base=10.0,
                hybrid_alpha_per_turn=5.0,
            ),
        )
        session.update(_mk_state(1.0))
        _mode, alpha = session._dynamic_hybrid_select()
        assert alpha == pytest.approx(1.0)

    def test_hybrid_select_alpha_clamped_below_zero(self) -> None:
        """Negative base → alpha clamped to 0.0."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=8,
                aggregation="dynamic",
                dynamic_strategy="hybrid",
                weighted_until_turn=3,
                hybrid_alpha_base=-5.0,
                hybrid_alpha_per_turn=0.1,
            ),
        )
        session.update(_mk_state(1.0))
        _mode, alpha = session._dynamic_hybrid_select()
        assert alpha == pytest.approx(0.0)

    # ---------- (T-5) get_session_coarse_state dispatcher ----------
    @pytest.mark.parametrize("strategy", ["turn_position", "drift_based", "hybrid"])
    def test_dispatcher_returns_shape_d_dtype_float32(self, strategy: str) -> None:
        """dynamic dispatcher: all 3 strategies produce (D,) float32 vec."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=4,
                aggregation="dynamic",
                dynamic_strategy=strategy,
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(2.0))
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        assert coarse.shape == (4,)
        assert coarse.dtype == mx.float32

    @pytest.mark.parametrize("strategy", ["turn_position", "drift_based", "hybrid"])
    def test_dispatcher_empty_turn_history_returns_none(self, strategy: str) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                aggregation="dynamic",
                dynamic_strategy=strategy,
            ),
        )
        assert session.get_session_coarse_state() is None

    def test_selected_mode_recorded_per_turn_weighted(self) -> None:
        """static ``weighted`` mode is recorded on the latest DriftMetrics."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=4, aggregation="weighted"
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(2.0))
        session.get_session_coarse_state()
        assert session.drift_history[-1].selected_aggregation_mode == "weighted"
        assert session.drift_history[-1].selected_aggregation_alpha is None

    def test_selected_mode_recorded_per_turn_dynamic_hybrid(self) -> None:
        """hybrid: mode='hybrid' AND alpha is a float on DriftMetrics."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=4,
                aggregation="dynamic",
                dynamic_strategy="hybrid",
                weighted_until_turn=0,
                hybrid_alpha_base=0.5,
                hybrid_alpha_per_turn=0.1,
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(2.0))
        session.get_session_coarse_state()
        last = session.drift_history[-1]
        assert last.selected_aggregation_mode == "hybrid"
        assert isinstance(last.selected_aggregation_alpha, float)

    def test_selected_mode_recorded_per_turn_dynamic_turn_position(self) -> None:
        """turn_position records 'weighted' or 'attention' with alpha=None."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=8,
                aggregation="dynamic",
                dynamic_strategy="turn_position",
                weighted_until_turn=1,
            ),
        )
        # Turn 1 → mode=weighted (turn_count=1 <= until=1).
        session.update(_mk_state(1.0))
        session.get_session_coarse_state()
        assert session.drift_history[-1].selected_aggregation_mode == "weighted"
        assert session.drift_history[-1].selected_aggregation_alpha is None

        # Turn 3 → mode=attention (turn_count=3 > until=1).
        session.update(_mk_state(2.0))
        session.update(_mk_state(3.0))
        session.get_session_coarse_state()
        assert session.drift_history[-1].selected_aggregation_mode == "attention"
        assert session.drift_history[-1].selected_aggregation_alpha is None

    def test_record_selected_mode_noop_when_drift_history_empty(self) -> None:
        """Empty drift_history → _record_selected_mode is a no-op (no raise)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                aggregation="dynamic",
                dynamic_strategy="turn_position",
            ),
        )
        # With no turn_history, get_session_coarse_state short-circuits to
        # None. _record_selected_mode is not reached — but a direct call
        # must also be a no-op.
        session._record_selected_mode("weighted", alpha=None)
        assert session.drift_history == []

    def test_hybrid_aggregate_combines_weighted_and_attention(self) -> None:
        """_aggregate_hybrid mixes weighted and attention results."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=4,
                aggregation="dynamic",
                dynamic_strategy="hybrid",
                weighted_until_turn=0,
                hybrid_alpha_base=0.5,
                hybrid_alpha_per_turn=0.0,  # keep alpha at 0.5
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(-1.0))
        session.update(_mk_state(1.0))  # current similar to turn 1
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)
        import math as _math

        for v in coarse.tolist():
            assert _math.isfinite(v)


class TestDynamicAggregationStorageModeMatrix:
    """Issue #92 T-7 (f): storage_mode × dynamic_strategy 3×3 = 9 combos."""

    @pytest.mark.parametrize("storage_mode", ["full", "top_level_only", "summary_only"])
    @pytest.mark.parametrize("strategy", ["turn_position", "drift_based", "hybrid"])
    def test_shape_dtype_across_storage_modes(
        self, storage_mode: str, strategy: str
    ) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=3,
                storage_mode=storage_mode,
                aggregation="dynamic",
                dynamic_strategy=strategy,
            ),
        )
        for v in (1.0, 2.0):
            session.update(_mk_state(v))
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        assert coarse.shape == (4,)
        assert coarse.dtype == mx.float32

    @pytest.mark.parametrize("storage_mode", ["full", "top_level_only", "summary_only"])
    @pytest.mark.parametrize("strategy", ["turn_position", "drift_based", "hybrid"])
    def test_empty_returns_none_across_storage_modes(
        self, storage_mode: str, strategy: str
    ) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=3,
                storage_mode=storage_mode,
                aggregation="dynamic",
                dynamic_strategy=strategy,
            ),
        )
        assert session.get_session_coarse_state() is None


class TestCodexCB004WorkingMemoryMaxTurnsHardCap:
    """Codex CB-004: WorkingMemoryConfig.max_turns must reject unbounded values."""

    def test_hard_cap_constant_is_reasonable(self) -> None:
        from photon_mlx.session import WORKING_MEMORY_MAX_TURNS_HARD_CAP

        assert isinstance(WORKING_MEMORY_MAX_TURNS_HARD_CAP, int)
        # Design §7 references max_turns=8 GA. The hard cap must be at least
        # large enough to accept that default, but small enough to bound
        # memory (photon_small hidden=640, photon_tiny hidden=1024) — 32 is
        # the documented recommended value.
        assert WORKING_MEMORY_MAX_TURNS_HARD_CAP >= 8
        assert WORKING_MEMORY_MAX_TURNS_HARD_CAP <= 64

    def test_max_turns_rejects_values_above_hard_cap(self) -> None:
        from photon_mlx.session import WORKING_MEMORY_MAX_TURNS_HARD_CAP

        # Exactly the cap is allowed.
        WorkingMemoryConfig(max_turns=WORKING_MEMORY_MAX_TURNS_HARD_CAP)
        # One above the cap must raise ValueError.
        with pytest.raises(ValueError, match="max_turns"):
            WorkingMemoryConfig(max_turns=WORKING_MEMORY_MAX_TURNS_HARD_CAP + 1)
        # A huge value (DoS attempt) must raise.
        with pytest.raises(ValueError, match="max_turns"):
            WorkingMemoryConfig(max_turns=10**9)


class TestMeanPool:
    """``mean_pool`` is the shared leading-dims reducer (DR1-002)."""

    def test_mean_pool_returns_1d_for_3d(self) -> None:
        x = mx.ones((2, 3, 4))
        v = mean_pool(x)
        assert v.shape == (4,)

    def test_mean_pool_passthrough_for_1d(self) -> None:
        x = mx.array([1.0, 2.0, 3.0])
        v = mean_pool(x)
        mx.eval(v)
        assert v.shape == (3,)
        assert v.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------
# Issue #79: storage_mode (full / top_level_only / summary_only)
# ---------------------------------------------------------------


class TestWorkingMemoryStorageMode:
    """Issue #79 — ``WorkingMemoryConfig.storage_mode`` closed enum."""

    def test_storage_mode_default_full(self) -> None:
        cfg = WorkingMemoryConfig()
        assert cfg.storage_mode == "full"

    @pytest.mark.parametrize("bad", [42, None, True, 1.5])
    def test_storage_mode_validation_type(self, bad: object) -> None:
        with pytest.raises(TypeError, match="storage_mode must be str"):
            WorkingMemoryConfig(storage_mode=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["Full", "FULL", "invalid", "full ", "", "Summary"])
    def test_storage_mode_validation_enum(self, bad: str) -> None:
        """Enum rejection does NOT leak the raw value into the exception text.

        Design §4 / DR4 / DR6 — the raw value must never appear in the
        ValueError message so attacker-controlled YAML cannot leak via
        exception traces.
        """
        with pytest.raises(ValueError) as exc_info:
            WorkingMemoryConfig(storage_mode=bad)
        # Closed-enum message text only — no echo of the raw value.
        msg = str(exc_info.value)
        assert "storage_mode must be one of" in msg
        assert "'full'" in msg and "'top_level_only'" in msg and "'summary_only'" in msg
        # Skip the empty-string case (vacuously contained in any string).
        if bad:
            assert bad not in msg

    @pytest.mark.parametrize("mode", ["full", "top_level_only", "summary_only"])
    def test_storage_mode_accepts_enum_values(self, mode: str) -> None:
        cfg = WorkingMemoryConfig(storage_mode=mode)
        assert cfg.storage_mode == mode


class TestDeprecationWarning:
    """Issue #79 DR1-004 — ``compress_old_turns`` DeprecationWarning policy."""

    def test_silent_when_only_compress_old_turns_specified(self) -> None:
        """Existing YAML (compress_old_turns=True, storage_mode omitted) must
        stay silent — otherwise every ``_build_photon_deps()`` call would
        spam DeprecationWarnings (design §3.1 DR1-004 repo YAML exemption).
        """
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as captured:
            _warnings.simplefilter("always")
            WorkingMemoryConfig(compress_old_turns=True)
        for rec in captured:
            assert not issubclass(rec.category, DeprecationWarning)

    def test_silent_when_only_storage_mode_specified(self) -> None:
        """storage_mode alone (the forward-looking form) must stay silent."""
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as captured:
            _warnings.simplefilter("always")
            WorkingMemoryConfig(storage_mode="top_level_only")
        for rec in captured:
            assert not issubclass(rec.category, DeprecationWarning)

    def test_silent_when_storage_mode_is_full_even_if_compress_specified(self) -> None:
        """Mixing compress_old_turns with storage_mode="full" is unambiguous
        (``full`` is the default) so no warning fires."""
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as captured:
            _warnings.simplefilter("always")
            WorkingMemoryConfig(compress_old_turns=True, storage_mode="full")
        for rec in captured:
            assert not issubclass(rec.category, DeprecationWarning)

    def test_warns_when_storage_mode_and_compress_old_turns_both_specified(
        self,
    ) -> None:
        with pytest.warns(DeprecationWarning, match="compress_old_turns is deprecated"):
            WorkingMemoryConfig(
                compress_old_turns=True,
                storage_mode="top_level_only",
            )

    def test_warns_for_summary_only_mode(self) -> None:
        with pytest.warns(DeprecationWarning, match="compress_old_turns is deprecated"):
            WorkingMemoryConfig(
                compress_old_turns=False,
                storage_mode="summary_only",
            )


def _mk_two_level_state(value: float, hidden: int = 4) -> HierarchicalState:
    """Two-level state with a distinct top/mid so the DR1-007 invariant
    (``top_level_only`` must not mutate) can be checked on ``level_states``.
    """
    mid = mx.ones((1, 4, hidden)) * value
    top = mx.ones((1, 2, hidden)) * (value + 0.5)
    return HierarchicalState(
        level_states=[mid, top],
        token_proj=mx.ones((1, 4, hidden)) * (value - 0.25),
    )


class TestWorkingMemoryFullMode:
    """Issue #79 — ``storage_mode="full"``: oldest turn is compressed."""

    def test_compress_oldest_turn_on_overflow(self) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="full"
            ),
        )
        for i in range(4):
            session.update(_mk_state(float(i + 1)), question_text=f"q{i + 1}")

        # max_turns=3 → one turn compressed (turn_id=1).
        assert len(session.turn_history) == 3
        assert [t.turn_id for t in session.turn_history] == [2, 3, 4]
        assert len(session.compressed_history) == 1
        assert session.compressed_history[0].turn_id == 1
        # summary_vec is a (hidden,) float32 vector.
        assert session.compressed_history[0].summary_vec.shape == (4,)
        assert session.compressed_history[0].summary_vec.dtype == mx.float32

    def test_compressed_history_upper_bound(self) -> None:
        """``max_turns * 4`` cap: silent pop(0) per DR1-009 D5."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=2, storage_mode="full"
            ),
        )
        # max_turns=2 → cap = 8. Push 12 turns → cap capped at 8, each
        # turn beyond max_turns compresses one entry.
        for i in range(12):
            session.update(_mk_state(float(i + 1)))
        assert len(session.turn_history) == 2
        assert len(session.compressed_history) == 8  # max_turns * 4

    def test_full_mode_compresses_not_drops(self) -> None:
        """Contrast with Phase 1 (pop(0) without retention)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=2, storage_mode="full"
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(2.0))
        session.update(_mk_state(3.0))  # turn 1 compressed
        assert len(session.compressed_history) == 1
        assert session.compressed_history[0].turn_id == 1

    def test_full_mode_coarse_state_mixes_compressed_history(self) -> None:
        """D7 — full-mode coarse state concatenates compressed + live."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=2,
                decay_factor=0.5,
                storage_mode="full",
            ),
        )
        # 4 turns with values 1, 2, 4, 8 → after max_turns=2:
        #   compressed: turn1(1) + turn2(2), turn_history: turn3(4) + turn4(8)
        for v in (1.0, 2.0, 4.0, 8.0):
            session.update(_mk_state(v))
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        mx.eval(coarse)

        # Derive the same aggregated value with 4 decayed entries chronologically:
        # weights = [0.5^3, 0.5^2, 0.5^1, 0.5^0] = [0.125, 0.25, 0.5, 1.0]
        # values  = [1, 2, 4, 8]
        # weighted sum = 0.125 + 0.5 + 2.0 + 8.0 = 10.625
        # weight_sum   = 1.875
        # expected     = 10.625 / 1.875 ≈ 5.666666...
        vals = coarse.tolist()
        for v in vals:
            assert abs(v - 10.625 / 1.875) < 1e-4


class TestWorkingMemoryTopLevelOnly:
    """Issue #79 — ``storage_mode="top_level_only"``: keeps only level_states[-1]."""

    def test_level_states_single_element(self) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="top_level_only"
            ),
        )
        input_state = _mk_two_level_state(1.0)
        session.update(input_state)
        stored = session.turn_history[-1].hierarchical_state
        assert len(stored.level_states) == 1

    def test_token_proj_none(self) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="top_level_only"
            ),
        )
        session.update(_mk_two_level_state(1.0))
        stored = session.turn_history[-1].hierarchical_state
        assert stored.token_proj is None

    def test_does_not_mutate_input_state(self) -> None:
        """DR1-007 invariant — update() must leave ``new_state`` unchanged."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="top_level_only"
            ),
        )
        input_state = _mk_two_level_state(1.0)
        n_levels_before = len(input_state.level_states)
        token_proj_before = input_state.token_proj
        level_states_obj_id = id(input_state.level_states)

        session.update(input_state)

        # Reference identity + length preserved; token_proj still populated.
        assert len(input_state.level_states) == n_levels_before
        assert input_state.token_proj is token_proj_before
        assert id(input_state.level_states) == level_states_obj_id
        # Stored state is a different HierarchicalState instance.
        stored = session.turn_history[-1].hierarchical_state
        assert stored is not input_state

    def test_sanitizes_question_text(self) -> None:
        """DR4-002 — sanitize contract must fire in top_level_only too."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="top_level_only"
            ),
        )
        # Control char should be stripped; NUL / bell.
        session.update(_mk_two_level_state(1.0), question_text="abc\x07def\x00ghi")
        stored_q = session.turn_history[-1].question_text
        # No raw control chars retained.
        assert "\x07" not in stored_q and "\x00" not in stored_q
        assert stored_q == "abcdefghi"

    def test_pops_on_overflow(self) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=2, storage_mode="top_level_only"
            ),
        )
        for i in range(5):
            session.update(_mk_two_level_state(float(i + 1)))
        # Oldest turns popped, compressed_history not populated (top_level_only).
        assert len(session.turn_history) == 2
        assert session.compressed_history == []


class TestWorkingMemorySummaryOnly:
    """Issue #79 — ``storage_mode="summary_only"``: compressed_history only."""

    def test_turn_history_stays_empty(self) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="summary_only"
            ),
        )
        for v in (1.0, 2.0, 3.0):
            session.update(_mk_state(v))
        assert session.turn_history == []
        assert len(session.compressed_history) == 3

    def test_compressed_history_populated(self) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="summary_only"
            ),
        )
        session.update(_mk_state(2.0))
        assert len(session.compressed_history) == 1
        entry = session.compressed_history[0]
        assert entry.turn_id == 1
        assert entry.summary_vec.shape == (4,)
        assert entry.summary_vec.dtype == mx.float32

    def test_sanitizes_question_text(self) -> None:
        """DR1-003 — sanitize contract fires even though value is discarded."""
        # Use a question that exceeds MAX_LEN and contains control chars.
        long_q = "x" * 4096 + "\x07malicious"
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="summary_only"
            ),
        )
        # Must not raise and must not leak question_text anywhere visible.
        session.update(_mk_state(1.0), question_text=long_q)
        assert session.compressed_history and session.turn_history == []
        # CompressedTurnState does not expose question_text (DR1-005).
        assert not hasattr(session.compressed_history[0], "question_text")

    def test_compressed_history_upper_bound(self) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=2, storage_mode="summary_only"
            ),
        )
        for i in range(20):
            session.update(_mk_state(float(i + 1)))
        assert len(session.compressed_history) == 8  # max_turns * 4


class TestGetSessionCoarseStateAcrossModes:
    """Issue #79 — parametric same-API contract for all three modes."""

    @pytest.mark.parametrize("mode", ["full", "top_level_only", "summary_only"])
    def test_returns_shape_d(self, mode: str) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode=mode
            ),
        )
        for v in (1.0, 2.0):
            session.update(_mk_state(v))
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        assert coarse.shape == (4,)
        assert coarse.dtype == mx.float32

    @pytest.mark.parametrize("mode", ["full", "top_level_only", "summary_only"])
    def test_returns_none_when_empty(self, mode: str) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode=mode
            ),
        )
        assert session.get_session_coarse_state() is None

    def test_summary_only_uses_compressed_history(self) -> None:
        """turn_history empty but compressed_history populated → non-None."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="summary_only"
            ),
        )
        session.update(_mk_state(1.0))
        assert session.turn_history == []
        assert session.compressed_history
        assert session.get_session_coarse_state() is not None

    def test_top_level_only_ignores_compressed_history(self) -> None:
        """DR1-008 — top_level_only never reads compressed_history."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="top_level_only"
            ),
        )
        # Manually seed compressed_history to prove it is not consulted.
        from photon_mlx.session import CompressedTurnState

        session.compressed_history.append(
            CompressedTurnState(turn_id=0, summary_vec=mx.ones((4,)) * 999.0)
        )
        assert session.get_session_coarse_state() is None

    def test_full_mode_uses_compressed_history_when_non_empty(self) -> None:
        """D7 — full-mode coarse reflects both buckets."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True,
                max_turns=2,
                decay_factor=0.5,
                storage_mode="full",
            ),
        )
        # Two turns fit exactly in turn_history.
        session.update(_mk_state(10.0))
        session.update(_mk_state(20.0))
        coarse_before = session.get_session_coarse_state()
        assert coarse_before is not None

        # Extra turn bumps turn 1 (value=10) into compressed_history.
        session.update(_mk_state(30.0))
        coarse_after = session.get_session_coarse_state()
        assert coarse_after is not None
        # Values differ: compressed_history entry changes the aggregation.
        mx.eval(coarse_before)
        mx.eval(coarse_after)
        assert coarse_before.tolist() != coarse_after.tolist()


class TestMakeTurnSummary:
    """Issue #79 A-3 — ``_make_turn_summary`` helper contract."""

    def test_returns_shape_d(self) -> None:
        session = PhotonSessionState("s1", "repo", "abc")
        state = _mk_state(1.0, hidden=8)
        summary = session._make_turn_summary(state)
        assert summary.shape == (8,)

    def test_dtype_float32(self) -> None:
        session = PhotonSessionState("s1", "repo", "abc")
        state = _mk_state(1.0)
        summary = session._make_turn_summary(state)
        assert summary.dtype == mx.float32

    def test_empty_level_states_returns_zero_length(self) -> None:
        session = PhotonSessionState("s1", "repo", "abc")
        summary = session._make_turn_summary(HierarchicalState(level_states=[]))
        assert summary.shape == (0,)


class TestResetClearsCompressedHistory:
    """Issue #79 — reset_working_memory() clears compressed_history atomically."""

    def test_reset_working_memory_clears_compressed_history(self) -> None:
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=2, storage_mode="full"
            ),
        )
        for i in range(4):
            session.update(_mk_state(float(i + 1)))
        assert len(session.compressed_history) >= 1
        session.reset_working_memory()
        assert session.compressed_history == []
        assert session.turn_history == []

    def test_reset_working_memory_clears_summary_only_compressed_history(self) -> None:
        """summary_only: reset leaves compressed_history empty even when
        turn_history was always empty."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="summary_only"
            ),
        )
        session.update(_mk_state(1.0))
        session.update(_mk_state(2.0))
        assert len(session.compressed_history) == 2
        session.reset_working_memory()
        assert session.compressed_history == []


class TestCompressedHistoryRejectsZeroLengthSummaries:
    """CB-001 regression — ``_append_summary_only`` /
    ``_compress_oldest_turn`` must not store a ``shape=(0,)`` summary from
    an empty ``HierarchicalState.level_states``, and
    ``get_session_coarse_state()`` must not return a zero-length vector.

    ``_make_turn_summary()`` still returns ``mx.zeros((0,))`` for empty
    ``level_states`` (existing API contract, see
    ``TestMakeTurnSummary.test_empty_level_states_returns_zero_length``);
    the fix is to filter at the save / aggregate sites (design §3.8,
    defense-in-depth).
    """

    def test_summary_only_skips_empty_level_states(self) -> None:
        """update() with an empty HierarchicalState must NOT push a
        zero-length summary onto ``compressed_history``."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="summary_only"
            ),
        )
        # Valid update populates one entry.
        session.update(_mk_state(1.0))
        assert len(session.compressed_history) == 1

        # Pathological update with empty level_states must be a no-op for
        # compressed_history (save skipped — design §3.8 option A).
        session.update(HierarchicalState(level_states=[]))
        assert len(session.compressed_history) == 1
        # And the stored entry is the valid one, unchanged.
        assert session.compressed_history[0].summary_vec.shape == (4,)

    def test_full_mode_compress_oldest_skips_empty_level_states(self) -> None:
        """``_compress_oldest_turn`` must not leak a zero-length summary
        into ``compressed_history`` when the oldest turn's
        ``level_states`` is empty (belt-and-braces — ``_append_full``
        normally pushes populated states, but hand-constructed fixtures
        or forward-pass oddities could expose this)."""
        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=1, storage_mode="full"
            ),
        )
        # Seed turn_history directly with an empty state so
        # _compress_oldest_turn is exercised in isolation.
        session.turn_history.append(
            TurnState(
                turn_id=0,
                hierarchical_state=HierarchicalState(level_states=[]),
                question_text=None,
            )
        )
        session._compress_oldest_turn()
        # Oldest was popped from turn_history (existing contract).
        assert session.turn_history == []
        # But the zero-length summary must NOT have been stored.
        assert session.compressed_history == []

    def test_get_session_coarse_state_excludes_zero_length_summary_only(
        self,
    ) -> None:
        """``summary_only`` path must filter zero-length entries; a
        non-empty valid entry still produces a ``(D,)`` coarse state."""
        from photon_mlx.session import CompressedTurnState

        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="summary_only"
            ),
        )
        # Seed a zero-length entry directly to simulate a legacy state
        # that might have slipped through.
        session.compressed_history.append(
            CompressedTurnState(
                turn_id=0,
                summary_vec=mx.zeros((0,), dtype=mx.float32),
                timestamp=0.0,
            )
        )
        # Also add a valid entry.
        session.compressed_history.append(
            CompressedTurnState(
                turn_id=1,
                summary_vec=mx.ones((4,), dtype=mx.float32),
                timestamp=1.0,
            )
        )
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        assert coarse.shape == (4,)
        assert coarse.dtype == mx.float32

    def test_get_session_coarse_state_excludes_zero_length_full_mode(
        self,
    ) -> None:
        """``full`` path must also filter zero-length compressed entries
        when mixing with ``turn_history`` (design §3.7 D7)."""
        from photon_mlx.session import CompressedTurnState

        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="full"
            ),
        )
        # Populate turn_history with a valid turn.
        session.update(_mk_state(2.0))
        # Seed a zero-length entry into compressed_history (legacy state).
        session.compressed_history.append(
            CompressedTurnState(
                turn_id=99,
                summary_vec=mx.zeros((0,), dtype=mx.float32),
                timestamp=0.0,
            )
        )
        coarse = session.get_session_coarse_state()
        assert coarse is not None
        assert coarse.shape == (4,)
        assert coarse.dtype == mx.float32

    def test_compressed_history_only_zero_length_returns_none(self) -> None:
        """If compressed_history contains ONLY zero-length entries
        (summary_only) and turn_history is empty, get_session_coarse_state
        must return ``None`` rather than a ``shape=(0,)`` array (API
        contract: mx.array or None)."""
        from photon_mlx.session import CompressedTurnState

        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="summary_only"
            ),
        )
        session.compressed_history.append(
            CompressedTurnState(
                turn_id=0,
                summary_vec=mx.zeros((0,), dtype=mx.float32),
                timestamp=0.0,
            )
        )
        assert session.get_session_coarse_state() is None

    def test_full_mode_all_zero_length_returns_none(self) -> None:
        """``full`` mode: if turn_history is empty and compressed_history
        has only zero-length entries, coarse must be ``None``."""
        from photon_mlx.session import CompressedTurnState

        session = PhotonSessionState(
            "s1",
            "repo",
            "abc",
            working_memory_cfg=WorkingMemoryConfig(
                enabled=True, max_turns=3, storage_mode="full"
            ),
        )
        session.compressed_history.append(
            CompressedTurnState(
                turn_id=0,
                summary_vec=mx.zeros((0,), dtype=mx.float32),
                timestamp=0.0,
            )
        )
        assert session.get_session_coarse_state() is None
