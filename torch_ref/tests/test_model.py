"""Correctness tests for the minimal LM."""

from __future__ import annotations

import torch
import pytest

from torch_ref.config import PhotonConfig, ModelConfig, HierarchyConfig, TokenizerConfig
from torch_ref.model import MinimalLM


def _tiny_config() -> PhotonConfig:
    return PhotonConfig(
        model=ModelConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=128,
            base_embed_dim=16,
        ),
        hierarchy=HierarchyConfig(
            encoder_layers_per_level=[2, 2],
            decoder_layers_per_level=[2, 2],
        ),
        tokenizer=TokenizerConfig(vocab_size=256),
    )


@pytest.fixture
def tiny_model() -> MinimalLM:
    cfg = _tiny_config()
    model = MinimalLM(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------


class TestShapes:
    def test_forward_logits_shape(self, tiny_model: MinimalLM) -> None:
        ids = torch.randint(0, 256, (2, 16))
        logits, loss = tiny_model(ids)
        assert logits.shape == (2, 16, 256)
        assert loss is None

    def test_forward_with_labels(self, tiny_model: MinimalLM) -> None:
        ids = torch.randint(0, 256, (2, 16))
        logits, loss = tiny_model(ids, labels=ids)
        assert logits.shape == (2, 16, 256)
        assert loss is not None
        assert loss.ndim == 0  # scalar

    def test_greedy_decode_shape(self, tiny_model: MinimalLM) -> None:
        ids = torch.randint(0, 256, (1, 4))
        out = tiny_model.greedy_decode(ids, max_new_tokens=8)
        assert out.shape == (1, 12)  # 4 + 8


# ---------------------------------------------------------------
# Mask test: causal mask prevents attending to future tokens
# ---------------------------------------------------------------


class TestCausalMask:
    def test_causal_mask_shape(self, tiny_model: MinimalLM) -> None:
        mask = tiny_model.causal_mask
        pos = tiny_model.cfg.model.max_position_embeddings
        assert mask.shape == (pos, pos)

    def test_causal_mask_upper_triangular(self, tiny_model: MinimalLM) -> None:
        mask = tiny_model.causal_mask
        # Lower triangle + diagonal should be 0 (no masking)
        assert (mask[torch.tril(torch.ones_like(mask)).bool()] == 0).all()
        # Strictly upper triangle should be -inf
        upper_tri = torch.triu(torch.ones_like(mask), diagonal=1).bool()
        assert (mask[upper_tri] == float("-inf")).all()

    def test_different_prefix_same_logits(self, tiny_model: MinimalLM) -> None:
        """Logits at position t must not change when tokens after t change."""
        ids = torch.randint(0, 256, (1, 8))
        logits_full, _ = tiny_model(ids)

        ids_alt = ids.clone()
        ids_alt[0, 4:] = torch.randint(0, 256, (4,))
        logits_alt, _ = tiny_model(ids_alt)

        # First 4 positions should produce identical logits
        torch.testing.assert_close(logits_full[:, :4], logits_alt[:, :4])


# ---------------------------------------------------------------
# Overfit test: model can memorize 1 batch
# ---------------------------------------------------------------


class TestOverfit:
    def test_1batch_overfit(self) -> None:
        cfg = _tiny_config()
        model = MinimalLM(cfg)
        model.train()

        torch.manual_seed(42)
        ids = torch.randint(0, 256, (2, 16))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_loss = None
        for step in range(200):
            _, loss = model(ids, labels=ids)
            if initial_loss is None:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.1, (
            f"Failed to overfit: initial={initial_loss:.4f} final={final_loss:.4f}"
        )


# ---------------------------------------------------------------
# Reproducibility test
# ---------------------------------------------------------------


class TestReproducibility:
    def test_seed_determinism(self) -> None:
        cfg = _tiny_config()

        def run(seed: int) -> torch.Tensor:
            torch.manual_seed(seed)
            model = MinimalLM(cfg)
            model.eval()
            ids = torch.randint(0, 256, (1, 8))
            logits, _ = model(ids)
            return logits

        logits_a = run(123)
        logits_b = run(123)
        torch.testing.assert_close(logits_a, logits_b)


# ---------------------------------------------------------------
# Logits sanity check
# ---------------------------------------------------------------


class TestLogitsSanity:
    def test_logits_finite(self, tiny_model: MinimalLM) -> None:
        ids = torch.randint(0, 256, (1, 32))
        logits, _ = tiny_model(ids)
        assert torch.isfinite(logits).all()

    def test_logits_not_constant(self, tiny_model: MinimalLM) -> None:
        ids = torch.randint(0, 256, (1, 8))
        logits, _ = tiny_model(ids)
        # Different positions should produce different logit distributions
        assert not torch.allclose(logits[0, 0], logits[0, -1])

    def test_parameter_count(self) -> None:
        cfg = _tiny_config()
        model = MinimalLM(cfg)
        n = model.count_parameters()
        assert n > 0
        # Tiny config should be well under 1M params
        assert n < 1_000_000, f"Tiny model has {n} params, expected < 1M"
