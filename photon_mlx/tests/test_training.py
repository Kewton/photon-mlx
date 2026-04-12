"""Training pipeline tests for PHOTON."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

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
from photon_mlx.loss import photon_loss, next_token_loss
from photon_mlx.data import pack_sequences, create_batches, load_jsonl
from photon_mlx.trainer import save_checkpoint, load_checkpoint, TrainState


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
# Loss tests
# ---------------------------------------------------------------

class TestLoss:
    def test_next_token_loss_positive(self) -> None:
        logits = mx.random.normal((2, 16, 256))
        labels = mx.random.randint(0, 256, (2, 16))
        loss = next_token_loss(logits, labels)
        mx.eval(loss)
        assert loss.item() > 0

    def test_photon_loss_breakdown(self) -> None:
        logits = mx.random.normal((2, 16, 256))
        labels = mx.random.randint(0, 256, (2, 16))
        total, breakdown = photon_loss(logits, labels, recursive_loss_weight=0.0)
        mx.eval(total)
        assert "next_token_loss" in breakdown
        assert "total_loss" in breakdown

    def test_photon_loss_with_recursive(self) -> None:
        logits = mx.random.normal((2, 16, 256))
        labels = mx.random.randint(0, 256, (2, 16))
        total, breakdown = photon_loss(logits, labels, recursive_loss_weight=0.1)
        mx.eval(total)
        assert "recursive_loss" in breakdown


# ---------------------------------------------------------------
# Data pipeline tests
# ---------------------------------------------------------------

class TestData:
    def test_pack_sequences(self) -> None:
        docs = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12]]
        packed = pack_sequences(docs, context_length=4)
        assert len(packed) == 3  # 12 tokens → 3 × 4
        assert all(len(s) == 4 for s in packed)

    def test_create_batches(self) -> None:
        seqs = [[i] * 8 for i in range(10)]
        batches = create_batches(seqs, batch_size=3, shuffle=False)
        assert len(batches) == 3  # 10 // 3 = 3
        assert batches[0].shape == (3, 8)

    def test_load_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.jsonl"
        p.write_text(
            '{"tokens": [1,2,3]}\n{"tokens": [4,5,6,7]}\n',
            encoding="utf-8",
        )
        docs = load_jsonl(p)
        assert len(docs) == 2
        assert docs[0] == [1, 2, 3]


# ---------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------

class TestCheckpoint:
    def test_save_and_load(self, tmp_path: Path) -> None:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        state = TrainState(step=100, best_val_loss=2.5)

        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(model, state, ckpt_dir)

        # Verify files exist
        assert (ckpt_dir / "weights.npz").exists()
        assert (ckpt_dir / "state.json").exists()

        # Load into fresh model
        model2 = PhotonModel(cfg)
        state2 = load_checkpoint(model2, ckpt_dir)
        assert state2.step == 100
        assert state2.best_val_loss == 2.5


# ---------------------------------------------------------------
# Overfit test (end-to-end training sanity)
# ---------------------------------------------------------------

class TestOverfit:
    def test_tiny_overfit(self) -> None:
        """Train for a few steps on a fixed batch and verify loss drops."""
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)

        batch = mx.random.randint(0, 256, (2, 16))

        def loss_fn(model, batch):
            logits, _ = model(batch, labels=batch)
            total, _ = photon_loss(logits, batch, 0.0)
            return total

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        optimizer = mlx.optimizers.Adam(learning_rate=1e-3)

        initial_loss = None
        for _ in range(50):
            loss, grads = loss_and_grad(model, batch)
            mx.eval(loss)
            if initial_loss is None:
                initial_loss = loss.item()
            optimizer.update(model, grads)
            mx.eval(model.parameters())

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, (
            f"Loss did not drop enough: {initial_loss:.4f} → {final_loss:.4f}"
        )
