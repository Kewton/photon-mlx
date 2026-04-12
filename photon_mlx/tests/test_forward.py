"""Shape and smoke tests for the PHOTON MLX model."""
from __future__ import annotations

import sys
from pathlib import Path

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch_ref.config import (
    HierarchyConfig,
    ModelConfig,
    PhotonConfig,
    TokenizerConfig,
)
from photon_mlx.model import PhotonModel


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
            encoder_layers_per_level=[2, 2],
            decoder_layers_per_level=[2, 2],
        ),
        tokenizer=TokenizerConfig(vocab_size=256),
    )


@pytest.fixture
def tiny_model() -> PhotonModel:
    mx.random.seed(42)
    return PhotonModel(_tiny_cfg())


# ---------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------

class TestShapes:
    def test_forward_logits_shape(self, tiny_model: PhotonModel) -> None:
        # T=64 divisible by 4*4=16
        ids = mx.random.randint(0, 256, (2, 64))
        logits, loss = tiny_model(ids)
        assert logits.shape == (2, 64, 256)
        assert loss is None

    def test_forward_with_labels(self, tiny_model: PhotonModel) -> None:
        ids = mx.random.randint(0, 256, (2, 64))
        logits, loss = tiny_model(ids, labels=ids)
        assert logits.shape == (2, 64, 256)
        assert loss.ndim == 0

    def test_batch_size_1(self, tiny_model: PhotonModel) -> None:
        ids = mx.random.randint(0, 256, (1, 16))
        logits, _ = tiny_model(ids)
        assert logits.shape == (1, 16, 256)


# ---------------------------------------------------------------
# Chunk boundary tests
# ---------------------------------------------------------------

class TestChunkBoundary:
    def test_minimum_length(self, tiny_model: PhotonModel) -> None:
        """Minimum T = product of chunk_sizes = 4*4 = 16."""
        ids = mx.random.randint(0, 256, (1, 16))
        logits, _ = tiny_model(ids)
        assert logits.shape == (1, 16, 256)

    def test_longer_sequence(self, tiny_model: PhotonModel) -> None:
        ids = mx.random.randint(0, 256, (1, 128))
        logits, _ = tiny_model(ids)
        assert logits.shape == (1, 128, 256)


# ---------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------

class TestSmoke:
    def test_logits_finite(self, tiny_model: PhotonModel) -> None:
        ids = mx.random.randint(0, 256, (1, 32))
        logits, _ = tiny_model(ids)
        mx.eval(logits)
        assert mx.isfinite(logits).all().item()

    def test_loss_positive(self, tiny_model: PhotonModel) -> None:
        ids = mx.random.randint(0, 256, (2, 32))
        _, loss = tiny_model(ids, labels=ids)
        mx.eval(loss)
        assert loss.item() > 0

    def test_loss_decreases_on_overfit(self) -> None:
        """Verify gradient flow: loss should decrease over a few steps."""
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        ids = mx.random.randint(0, 256, (2, 16))

        def loss_fn(model, ids):
            _, loss = model(ids, labels=ids)
            return loss

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        optimizer = mlx.optimizers.Adam(learning_rate=1e-3)

        initial_loss = None
        for _ in range(30):
            loss, grads = loss_and_grad(model, ids)
            mx.eval(loss)
            if initial_loss is None:
                initial_loss = loss.item()
            optimizer.update(model, grads)
            mx.eval(model.parameters())

        final_loss = loss.item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_parameter_count(self) -> None:
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        n = model.count_parameters()
        assert n > 0
        assert n < 5_000_000, f"Tiny model has {n:,} params, expected < 5M"
