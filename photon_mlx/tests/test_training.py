"""Training pipeline tests for PHOTON."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch_ref.config import (
    HierarchyConfig,
    ModelConfig,
    PhotonConfig,
    TokenizerConfig,
    TrainingConfig,
)
from photon_mlx.model import PhotonModel
from photon_mlx.loss import photon_loss, next_token_loss
from photon_mlx.data import pack_sequences, create_batches, load_jsonl
from photon_mlx.trainer import (
    save_checkpoint,
    load_checkpoint,
    load_model,
    train,
    TrainState,
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


class TestLoadModel:
    def test_load_model(self, tmp_path: Path) -> None:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        state = TrainState(step=100, best_val_loss=2.0)

        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(model, state, ckpt_dir)

        # Write a minimal config YAML
        config_path = tmp_path / "config.yaml"
        import yaml

        cfg_dict = {
            "model": {
                "base_embed_dim": 16,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 16,
                "max_position_embeddings": 128,
            },
            "hierarchy": {
                "levels": 2,
                "chunk_sizes": [4, 4],
                "converter_prefix_lengths": [2, 2],
                "encoder_layers_per_level": [1, 1],
                "decoder_layers_per_level": [1, 1],
            },
            "tokenizer": {"vocab_size": 256},
        }
        config_path.write_text(yaml.dump(cfg_dict), encoding="utf-8")

        loaded = load_model(config_path, ckpt_dir)
        assert isinstance(loaded, PhotonModel)

    def test_load_model_forward_consistency(self, tmp_path: Path) -> None:
        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)

        ids = mx.random.randint(0, 256, (1, 16))
        logits_before, _ = model(ids)
        mx.eval(logits_before)

        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(model, TrainState(), ckpt_dir)

        import yaml

        config_path = tmp_path / "config.yaml"
        cfg_dict = {
            "model": {
                "base_embed_dim": 16,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 16,
                "max_position_embeddings": 128,
            },
            "hierarchy": {
                "levels": 2,
                "chunk_sizes": [4, 4],
                "converter_prefix_lengths": [2, 2],
                "encoder_layers_per_level": [1, 1],
                "decoder_layers_per_level": [1, 1],
            },
            "tokenizer": {"vocab_size": 256},
        }
        config_path.write_text(yaml.dump(cfg_dict), encoding="utf-8")

        loaded = load_model(config_path, ckpt_dir)
        logits_after, _ = loaded(ids)
        mx.eval(logits_after)

        assert mx.allclose(logits_before, logits_after, atol=1e-5).item()


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


# ---------------------------------------------------------------
# Helper: dummy corpus
# ---------------------------------------------------------------


def _write_dummy_corpus(path: Path, n_docs: int = 10, seq_len: int = 32) -> None:
    import json
    import random

    random.seed(42)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n_docs):
            tokens = [random.randint(1, 255) for _ in range(seq_len)]
            f.write(json.dumps({"tokens": tokens}) + "\n")


def _tiny_train_cfg(tmp_path: Path, **overrides) -> PhotonConfig:
    """Create a tiny config with TrainingConfig for testing train()."""
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_dummy_corpus(train_path, n_docs=10, seq_len=32)
    _write_dummy_corpus(val_path, n_docs=3, seq_len=32)

    defaults = dict(
        learning_rate=1e-3,
        micro_batch_size=2,
        max_steps=10,
        eval_every_steps=5,
        save_every_steps=5,
        log_every_steps=5,
        gradient_accumulation_steps=1,
        context_length=16,
        train_corpus=str(train_path),
        val_corpus=str(val_path),
        weight_decay=0.0,
        max_grad_norm=1.0,
        warmup_ratio=0.0,
        min_learning_rate=0.0,
    )
    defaults.update(overrides)
    tc = TrainingConfig(**defaults)

    cfg = _tiny_cfg()
    cfg.training = tc
    return cfg


# ---------------------------------------------------------------
# train() integration tests
# ---------------------------------------------------------------


class TestTrainFunction:
    def test_train_uses_config_max_steps(self, tmp_path: Path) -> None:
        """train() should run for cfg.training.max_steps."""
        cfg = _tiny_train_cfg(tmp_path, max_steps=10)
        state = train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        assert state.step == 10

    def test_train_saves_checkpoint_at_configured_interval(
        self, tmp_path: Path
    ) -> None:
        """Checkpoints should be saved at save_every_steps intervals."""
        cfg = _tiny_train_cfg(tmp_path, max_steps=10, save_every_steps=5)
        train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        assert (tmp_path / "ckpt" / "step_000005").exists()
        assert (tmp_path / "ckpt" / "final").exists()

    def test_train_gradient_accumulation(self, tmp_path: Path) -> None:
        """train() with gradient_accumulation_steps > 1 should complete."""
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=8,
            gradient_accumulation_steps=4,
            eval_every_steps=100,
            save_every_steps=100,
            log_every_steps=4,
        )
        state = train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        assert state.step == 8

    def test_train_adamw_weight_decay(self, tmp_path: Path) -> None:
        """train() with weight_decay > 0 should complete without error."""
        cfg = _tiny_train_cfg(tmp_path, max_steps=5, weight_decay=0.1)
        state = train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        assert state.step == 5

    def test_train_lr_warmup_cosine(self, tmp_path: Path) -> None:
        """train() with warmup + cosine decay should complete."""
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=20,
            warmup_ratio=0.1,
            min_learning_rate=1e-5,
            eval_every_steps=100,
            save_every_steps=100,
        )
        state = train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        assert state.step == 20

    def test_train_grad_clipping(self, tmp_path: Path) -> None:
        """train() with max_grad_norm should complete without error."""
        cfg = _tiny_train_cfg(tmp_path, max_steps=5, max_grad_norm=0.5)
        state = train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        assert state.step == 5

    def test_loss_breakdown_in_log(self, tmp_path: Path) -> None:
        """Loss breakdown (next_token_loss) should appear in train log."""
        import json

        cfg = _tiny_train_cfg(tmp_path, max_steps=5, log_every_steps=5)
        train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")

        log_path = tmp_path / "logs" / "train_log.jsonl"
        assert log_path.exists()
        found_breakdown = False
        for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
            entry = json.loads(line)
            if "next_token_loss" in entry:
                found_breakdown = True
                break
        assert found_breakdown, "Loss breakdown not found in train log"

    def test_train_loss_decreases(self, tmp_path: Path) -> None:
        """Loss should decrease during training."""
        cfg = _tiny_train_cfg(tmp_path, max_steps=30, log_every_steps=10)
        state = train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        assert len(state.train_losses) > 1
        assert state.train_losses[-1] < state.train_losses[0]
