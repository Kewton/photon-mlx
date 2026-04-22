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
        state = TrainState(
            step=100, best_val_loss=2.5, best_step=50, patience_counter=2
        )

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
        assert state2.best_step == 50
        assert state2.patience_counter == 2

    def test_load_checkpoint_ignores_unknown_state_keys(self, tmp_path: Path) -> None:
        """load_checkpoint should drop unknown keys from state.json (forward compat)."""
        import json as _json

        mx.random.seed(42)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        state = TrainState(step=42, best_val_loss=1.2)

        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(model, state, ckpt_dir)

        # Inject an unknown key (simulating a future schema extension)
        state_path = ckpt_dir / "state.json"
        data = _json.loads(state_path.read_text(encoding="utf-8"))
        data["future_unknown_key"] = "ignored"
        state_path.write_text(_json.dumps(data), encoding="utf-8")

        model2 = PhotonModel(cfg)
        state2 = load_checkpoint(model2, ckpt_dir)
        assert state2.step == 42
        assert state2.best_val_loss == 1.2


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
    """Create a tiny config with TrainingConfig for testing train().

    Note: this helper does not set ``early_stopping``; the default
    ``EarlyStoppingConfig()`` (enabled=False) is inherited via
    ``TrainingConfig.early_stopping``'s default_factory so existing tests
    keep their pre-Issue#60 behaviour (runs full max_steps).
    """
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
        # Early stopping is disabled by default, so best/ should NOT exist.
        assert not (tmp_path / "ckpt" / "best").exists()

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


# ---------------------------------------------------------------
# Early stopping tests (Issue #60)
# ---------------------------------------------------------------


def _patch_evaluate_with_series(monkeypatch, losses: list[float]) -> None:
    """Patch photon_mlx.trainer.evaluate to yield the given val_loss series."""
    import photon_mlx.trainer as _trainer_mod

    seq = iter(losses)

    def fake_evaluate(model, val_batches, recursive_loss_weight=0.0):
        try:
            return next(seq)
        except StopIteration:
            # Keep returning the last value so patient tests don't crash.
            return losses[-1]

    monkeypatch.setattr(_trainer_mod, "evaluate", fake_evaluate)


class TestEarlyStopping:
    def test_early_stopping_disabled_runs_full_max_steps(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """enabled=false => loop runs for the full max_steps even if val_loss never improves."""
        _patch_evaluate_with_series(monkeypatch, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=5,
            eval_every_steps=1,
            log_every_steps=100,
            save_every_steps=100,
        )
        # early_stopping defaults to enabled=False
        state = train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        assert state.step == 5

    def test_early_stopping_triggers_on_patience(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """After `patience` evals without improvement, training must stop early."""
        from torch_ref.config import EarlyStoppingConfig

        # val_losses: improve at step 1 (2.0 -> 1.5), then plateau.
        # With patience=2, step 2 counter=1, step 3 counter=2 => stop after step 3.
        _patch_evaluate_with_series(monkeypatch, [2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=10,
            eval_every_steps=1,
            log_every_steps=100,
            save_every_steps=100,
        )
        cfg.training.early_stopping = EarlyStoppingConfig(
            enabled=True, patience=2, min_delta=0.0, restore_best=False
        )
        state = train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        # Must stop before reaching max_steps
        assert state.step < 10
        assert state.best_step == 2  # step where val_loss first improved to 1.5
        assert state.patience_counter >= 2

    def test_early_stopping_min_delta_ignores_tiny_improvement(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Improvements smaller than min_delta should not reset patience."""
        from torch_ref.config import EarlyStoppingConfig

        # Tiny improvements well under min_delta=0.01
        _patch_evaluate_with_series(
            monkeypatch, [2.0, 1.995, 1.990, 1.985, 1.980, 1.975]
        )
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=10,
            eval_every_steps=1,
            log_every_steps=100,
            save_every_steps=100,
        )
        cfg.training.early_stopping = EarlyStoppingConfig(
            enabled=True, patience=2, min_delta=0.01, restore_best=False
        )
        state = train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        # patience should fire quickly since each improvement is < min_delta
        assert state.step < 10

    def test_early_stopping_saves_best_checkpoint(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """`best/` dir is created once an improvement happens."""
        from torch_ref.config import EarlyStoppingConfig

        _patch_evaluate_with_series(monkeypatch, [2.0, 1.5, 1.6, 1.7, 1.8])
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=5,
            eval_every_steps=1,
            log_every_steps=100,
            save_every_steps=100,
        )
        cfg.training.early_stopping = EarlyStoppingConfig(
            enabled=True, patience=5, min_delta=0.0, restore_best=False
        )
        train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        best_dir = tmp_path / "ckpt" / "best"
        assert best_dir.exists()
        assert (best_dir / "weights.npz").exists()
        assert (best_dir / "state.json").exists()

    def test_early_stopping_restore_best(self, tmp_path: Path, monkeypatch) -> None:
        """restore_best=true => final/state.json.step == best_step."""
        import json as _json
        from torch_ref.config import EarlyStoppingConfig

        _patch_evaluate_with_series(monkeypatch, [2.0, 1.0, 1.5, 1.5, 1.5, 1.5])
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=10,
            eval_every_steps=1,
            log_every_steps=100,
            save_every_steps=100,
        )
        cfg.training.early_stopping = EarlyStoppingConfig(
            enabled=True, patience=2, min_delta=0.0, restore_best=True
        )
        train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        final_state_path = tmp_path / "ckpt" / "final" / "state.json"
        best_state_path = tmp_path / "ckpt" / "best" / "state.json"
        assert final_state_path.exists()
        final_data = _json.loads(final_state_path.read_text(encoding="utf-8"))
        best_data = _json.loads(best_state_path.read_text(encoding="utf-8"))
        # final/ should match best/ on the critical fields
        assert final_data["step"] == best_data["step"]
        assert final_data["best_step"] == best_data["best_step"]

    def test_early_stopping_no_restore_best(self, tmp_path: Path, monkeypatch) -> None:
        """restore_best=false => final/ reflects stop-time step, not best_step."""
        import json as _json
        from torch_ref.config import EarlyStoppingConfig

        _patch_evaluate_with_series(monkeypatch, [2.0, 1.0, 1.5, 1.5, 1.5, 1.5])
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=10,
            eval_every_steps=1,
            log_every_steps=100,
            save_every_steps=100,
        )
        cfg.training.early_stopping = EarlyStoppingConfig(
            enabled=True, patience=2, min_delta=0.0, restore_best=False
        )
        state = train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        final_state_path = tmp_path / "ckpt" / "final" / "state.json"
        assert final_state_path.exists()
        final_data = _json.loads(final_state_path.read_text(encoding="utf-8"))
        # stop-time step is strictly after best_step (1)
        assert final_data["step"] == state.step
        assert final_data["step"] > state.best_step

    def test_early_stopping_best_dir_missing_graceful_fallback(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """If best/ disappears mid-run, train() still exits and writes final/."""
        import shutil as _shutil
        from torch_ref.config import EarlyStoppingConfig

        # Small loss series that triggers patience quickly.
        _patch_evaluate_with_series(monkeypatch, [2.0, 1.0, 1.5, 1.5, 1.5, 1.5])
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=10,
            eval_every_steps=1,
            log_every_steps=100,
            save_every_steps=100,
        )
        cfg.training.early_stopping = EarlyStoppingConfig(
            enabled=True, patience=2, min_delta=0.0, restore_best=True
        )

        # Monkeypatch load_checkpoint to fail (simulating corrupt/missing best/)
        import photon_mlx.trainer as _trainer_mod

        original_load = _trainer_mod.load_checkpoint

        def raising_load(model, path):
            # Remove best/ before calling original to simulate race.
            p = Path(path)
            if p.name == "best" and p.exists():
                _shutil.rmtree(p)
                raise FileNotFoundError(f"best dir gone: {p}")
            return original_load(model, path)

        monkeypatch.setattr(_trainer_mod, "load_checkpoint", raising_load)

        train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        # final/ must still be present even after the fake failure.
        assert (tmp_path / "ckpt" / "final" / "weights.npz").exists()
        assert (tmp_path / "ckpt" / "final" / "state.json").exists()

    def test_early_stopping_best_state_json_malformed_graceful_fallback(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """If best/state.json is JSON-parsable but not a dict (e.g. "oops"),
        train() should still finish and write final/ with stop-time weights,
        rather than crashing on ValueError from load_checkpoint.
        """
        from torch_ref.config import EarlyStoppingConfig

        _patch_evaluate_with_series(monkeypatch, [2.0, 1.0, 1.5, 1.5, 1.5, 1.5])
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=10,
            eval_every_steps=1,
            log_every_steps=100,
            save_every_steps=100,
        )
        cfg.training.early_stopping = EarlyStoppingConfig(
            enabled=True, patience=2, min_delta=0.0, restore_best=True
        )

        import photon_mlx.trainer as _trainer_mod

        original_load = _trainer_mod.load_checkpoint

        def corrupt_then_load(model, path):
            p = Path(path)
            # Let save_checkpoint happen as usual, but right before the
            # end-of-train restore_best load, replace state.json with a
            # malformed-but-parsable payload ("oops" decodes to a str).
            if p.name == "best" and (p / "state.json").exists():
                (p / "state.json").write_text('"oops"', encoding="utf-8")
            return original_load(model, p)

        monkeypatch.setattr(_trainer_mod, "load_checkpoint", corrupt_then_load)

        train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        # final/ must still be present with stop-time weights, even though
        # load_checkpoint(best/) raised ValueError under the hood.
        assert (tmp_path / "ckpt" / "final" / "weights.npz").exists()
        assert (tmp_path / "ckpt" / "final" / "state.json").exists()

    def test_train_log_jsonl_contains_early_stop_fields(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """eval entries in train_log.jsonl must carry early-stop fields."""
        import json as _json
        from torch_ref.config import EarlyStoppingConfig

        _patch_evaluate_with_series(monkeypatch, [2.0, 1.0, 1.0, 1.0, 1.0])
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=5,
            eval_every_steps=1,
            log_every_steps=100,
            save_every_steps=100,
        )
        cfg.training.early_stopping = EarlyStoppingConfig(
            enabled=True, patience=2, min_delta=0.0, restore_best=False
        )
        train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        log_path = tmp_path / "logs" / "train_log.jsonl"
        assert log_path.exists()
        eval_entries = [
            _json.loads(line)
            for line in log_path.read_text(encoding="utf-8").strip().split("\n")
            if "val_loss" in _json.loads(line)
        ]
        assert eval_entries, "No eval entries found"
        for entry in eval_entries:
            assert "best_step" in entry
            assert "best_val_loss" in entry
            assert "patience_counter" in entry
            assert "early_stopped" in entry

    def test_train_log_has_no_eval_only_fields(self, tmp_path: Path) -> None:
        """train (log_every_steps) entries must not carry eval-only fields."""
        import json as _json

        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=5,
            eval_every_steps=100,  # disable eval
            log_every_steps=1,
            save_every_steps=100,
        )
        train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        log_path = tmp_path / "logs" / "train_log.jsonl"
        assert log_path.exists()
        for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
            entry = _json.loads(line)
            # No eval entry should be here (val_loss never written)
            assert "val_loss" not in entry
            assert "best_step" not in entry
            assert "best_val_loss" not in entry
            assert "patience_counter" not in entry
            assert "early_stopped" not in entry

    def test_eval_log_always_has_val_loss(self, tmp_path: Path, monkeypatch) -> None:
        """Every eval entry must include val_loss (schema invariant)."""
        import json as _json

        _patch_evaluate_with_series(monkeypatch, [2.0, 1.5, 1.0, 0.8, 0.6])
        cfg = _tiny_train_cfg(
            tmp_path,
            max_steps=5,
            eval_every_steps=1,
            log_every_steps=100,
            save_every_steps=100,
        )
        train(cfg, checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
        log_path = tmp_path / "logs" / "train_log.jsonl"
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        # Find entries that look like eval (have best/best_step/etc) and confirm val_loss
        found_eval = False
        for line in lines:
            entry = _json.loads(line)
            if "best" in entry or "best_step" in entry or "patience_counter" in entry:
                assert "val_loss" in entry
                found_eval = True
        assert found_eval


# ---------------------------------------------------------------
# Issue #55: long-context + NTK RoPE integration tests
# ---------------------------------------------------------------


class TestLongContextConfig:
    """Sanity tests for configs/photon_long_context.yaml (Issue #55)."""

    def test_long_context_config_loads(self) -> None:
        """configs/photon_long_context.yaml must load without error.

        Cross-config validation is strict:
        training.context_length (32768) must be a multiple of
        prod(chunk_sizes)=16 and <= max_position_embeddings (65536).
        """
        from torch_ref.config import load_photon_config

        cfg = load_photon_config("configs/photon_long_context.yaml")
        assert cfg.model.max_position_embeddings == 65536
        assert cfg.model.rope_scaling == "ntk"
        assert cfg.model.rope_scale_factor == 32.0
        assert cfg.training is not None
        assert cfg.training.context_length == 32768


class TestCrossConfigCheckpointLoad:
    """Checkpoints trained with a small max_position_embeddings must still
    load into a PhotonModel configured with a larger max_position_embeddings
    (Issue #55 backward-compat)."""

    def test_cross_config_checkpoint_load(self, tmp_path: Path) -> None:
        """Save under tiny config (max_pos=128), load into a larger-context
        config (max_pos=512) and run generate() without error.

        Note: the plan mentions 65536; we downscale to 512 to keep the
        PhotonModel instantiation within a reasonable unit-test memory
        budget (65536 * 65536 causal-mask style buffers would OOM).
        """
        mx.random.seed(42)
        cfg_small = _tiny_cfg()
        model = PhotonModel(cfg_small)
        state = TrainState(step=10, best_val_loss=1.0)

        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(model, state, ckpt_dir)

        # Larger-context config: still tiny hidden size, but bigger
        # max_position_embeddings + NTK scaling.
        cfg_long = PhotonConfig(
            model=ModelConfig(
                base_embed_dim=16,
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=16,
                max_position_embeddings=512,
                rope_scaling="ntk",
                rope_scale_factor=4.0,  # 128 -> 512
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
        model2 = PhotonModel(cfg_long)
        load_checkpoint(model2, ckpt_dir)

        # generate() at prompt_len=64, max_new=4 must not raise.
        ids = mx.random.randint(0, 256, (1, 64))
        out_ids, _ = model2.generate(ids, max_new_tokens=4)
        mx.eval(out_ids)
        assert out_ids.shape == (1, 64 + 4)


class TestNtkRopeLongInference:
    """2048-trained PHOTON weights + NTK RoPE scaling must produce valid
    inference outputs beyond the training context (Issue #55 accept-criteria
    surrogate — scaled down to keep unit tests fast)."""

    def test_ntk_rope_long_inference(self) -> None:
        """Construct a tiny PhotonModel with rope_scaling='ntk',
        rope_scale_factor=2.0, max_position_embeddings=2048 and run
        generate(prompt_len=64, max_new=4) without raising."""
        mx.random.seed(0)
        cfg = PhotonConfig(
            model=ModelConfig(
                base_embed_dim=16,
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=16,
                max_position_embeddings=2048,
                rope_theta=1_000_000.0,
                rope_scaling="ntk",
                rope_scale_factor=2.0,
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
        model = PhotonModel(cfg)
        ids = mx.random.randint(0, 256, (1, 64))
        out_ids, _ = model.generate(ids, max_new_tokens=4)
        mx.eval(out_ids)
        assert out_ids.shape == (1, 64 + 4)


class TestLongContextModelInit:
    """PhotonModel(load_photon_config('configs/photon_long_context.yaml'))
    must not raise (Issue #55 basic-init receipt).

    Gated behind a hidden_size downscale done at PhotonConfig level so we
    don't actually allocate weights for the full 1024-dim 65536-position
    model in unit tests.
    """

    def test_long_context_model_init(self) -> None:
        from torch_ref.config import load_photon_config

        cfg = load_photon_config("configs/photon_long_context.yaml")
        # Downscale the heavy dims (hidden_size/intermediate/layers) to keep
        # this test fast; the 65536 max_position_embeddings path is what we
        # want to exercise.  Do NOT touch max_position_embeddings, rope_theta,
        # rope_scaling, rope_scale_factor — those are the actual Issue #55
        # surface under test.
        cfg.model.base_embed_dim = 16
        cfg.model.hidden_size = 64
        cfg.model.intermediate_size = 128
        cfg.model.num_attention_heads = 4
        cfg.model.num_key_value_heads = 4
        cfg.model.head_dim = 16
        cfg.hierarchy.encoder_layers_per_level = [1, 1]
        cfg.hierarchy.decoder_layers_per_level = [1, 1]
        cfg.tokenizer.vocab_size = 256

        model = PhotonModel(cfg)
        # Touch the top-level RoPE table to confirm it got allocated.
        assert model._rope_cos.shape[0] == 65536
