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
    HierarchicalState,
    PhotonSessionState,
    cosine_distance,
    kl_divergence,
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

    def test_mixed_length_batch_topk_matches_sequential(
        self, engine: PhotonInference
    ) -> None:
        """Mixed-length non-empty chunks: batched top-K and raw scores must
        match a sequential hierarchical reference within ε=1e-3.

        Post-Issue-#63: ``_score_prune_candidates`` returns the
        ``weighted_hierarchical_score(sim_token, sim_mid, sim_top)`` combo
        rather than top-only cosine, so the sequential reference is built
        the same way.
        """
        from photon_mlx.session import weighted_hierarchical_score

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
