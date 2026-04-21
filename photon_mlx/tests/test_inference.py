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
