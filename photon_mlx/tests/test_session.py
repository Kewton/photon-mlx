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
from photon_mlx.inference import PhotonInference
from photon_mlx.session import (
    HierarchicalState,
    PhotonSessionState,
    cosine_distance,
    kl_divergence,
    token_agreement_rate,
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

    def test_drift_history_accumulates(self) -> None:
        session = PhotonSessionState("s1", "repo", "abc123")
        for i in range(5):
            state = HierarchicalState(
                level_states=[mx.ones((1, 4, 64)) * (i + 1)],
            )
            session.update(state)
        assert len(session.drift_history) == 5
        assert session.turn_count == 5


# ---------------------------------------------------------------
# End-to-end inference tests
# ---------------------------------------------------------------


class TestInference:
    @pytest.fixture
    def engine(self) -> PhotonInference:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        return PhotonInference(model, cfg)

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
