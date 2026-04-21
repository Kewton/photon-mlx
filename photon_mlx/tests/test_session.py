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
        """DriftMetrics.as_dict() superset contract (DR3-002): legacy keys
        stay present, three new keys are added."""
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
        # Legacy keys preserved.
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
        # Alias still returns top.
        assert d["latent_cosine_drift"] == d["latent_cosine_drift_top"]

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
        # JSON-safety: no NaN / Inf tokens in the dict.
        for v in out.values():
            assert _math.isfinite(v)
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
