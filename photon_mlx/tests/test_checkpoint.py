"""Tests for ``photon_mlx.checkpoint`` (Issue #135 / DR1-002 / DR3-001).

The checkpoint module is a runtime-only I/O surface that ``baseline_reporag``
imports without pulling in training-time dependencies (``mlx.optimizers``,
``photon_mlx.loss``).  These tests exercise three properties:

1. ``CheckpointState`` is a self-contained DTO that does not depend on
   ``photon_mlx.trainer.TrainState``.
2. ``photon_mlx.checkpoint`` can save/load weights + state.json without
   importing ``mlx.optimizers``.
3. The ``photon_mlx.trainer`` re-export wrapper preserves the existing
   ``TrainState`` API so callers stay unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch_ref.config import (  # noqa: E402
    HierarchyConfig,
    ModelConfig,
    PhotonConfig,
    TokenizerConfig,
)
from photon_mlx.model import PhotonModel  # noqa: E402


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


class TestCheckpointState:
    def test_default_construction(self) -> None:
        from photon_mlx.checkpoint import CheckpointState

        s = CheckpointState()
        assert s.step == 0
        assert s.best_val_loss == float("inf")
        assert s.best_step == 0
        assert s.patience_counter == 0
        assert s.train_losses == []
        assert s.val_losses == []

    def test_explicit_construction(self) -> None:
        from photon_mlx.checkpoint import CheckpointState

        s = CheckpointState(
            step=42,
            best_val_loss=1.5,
            best_step=20,
            patience_counter=3,
            train_losses=[2.0, 1.8],
            val_losses=[1.9],
        )
        assert s.step == 42
        assert s.best_val_loss == 1.5
        assert s.train_losses == [2.0, 1.8]


class TestCheckpointIO:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        from photon_mlx.checkpoint import (
            CheckpointState,
            save_checkpoint,
            load_checkpoint,
        )

        mx.random.seed(7)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        state = CheckpointState(
            step=200, best_val_loss=1.1, best_step=180, patience_counter=1
        )

        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(model, state, ckpt_dir)

        assert (ckpt_dir / "weights.npz").exists()
        assert (ckpt_dir / "state.json").exists()

        model2 = PhotonModel(cfg)
        loaded = load_checkpoint(model2, ckpt_dir)
        assert isinstance(loaded, CheckpointState)
        assert loaded.step == 200
        assert loaded.best_val_loss == 1.1
        assert loaded.best_step == 180
        assert loaded.patience_counter == 1

    def test_load_ignores_unknown_state_keys(self, tmp_path: Path) -> None:
        from photon_mlx.checkpoint import (
            CheckpointState,
            save_checkpoint,
            load_checkpoint,
        )

        mx.random.seed(7)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        save_checkpoint(model, CheckpointState(step=5), tmp_path)

        data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        data["future_field"] = "ignored"
        (tmp_path / "state.json").write_text(json.dumps(data), encoding="utf-8")

        model2 = PhotonModel(cfg)
        loaded = load_checkpoint(model2, tmp_path)
        assert loaded.step == 5

    def test_load_corrupt_state_json_raises(self, tmp_path: Path) -> None:
        from photon_mlx.checkpoint import (
            CheckpointState,
            save_checkpoint,
            load_checkpoint,
        )

        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        save_checkpoint(model, CheckpointState(), tmp_path)

        # state.json must decode to a dict; a list is invalid.
        (tmp_path / "state.json").write_text("[1, 2, 3]", encoding="utf-8")

        try:
            load_checkpoint(PhotonModel(cfg), tmp_path)
        except ValueError:
            return
        raise AssertionError(
            "load_checkpoint should raise ValueError on non-dict state.json"
        )

    def test_module_does_not_pull_training_deps(self) -> None:
        """Importing photon_mlx.checkpoint must not pull mlx.optimizers / loss.

        Verifies the DR1-002 / DR3-001 boundary: baseline_reporag can import
        the runtime checkpoint surface without paying the training-time cost.
        """
        # Force a fresh import so we measure dependencies of the module itself,
        # not whatever earlier tests already loaded.
        for name in list(sys.modules):
            if name.startswith("mlx.optimizers") or name == "photon_mlx.loss":
                sys.modules.pop(name, None)
        sys.modules.pop("photon_mlx.checkpoint", None)

        import photon_mlx.checkpoint  # noqa: F401

        assert "mlx.optimizers" not in sys.modules, (
            "photon_mlx.checkpoint must not import mlx.optimizers (training-only)"
        )
        assert "photon_mlx.loss" not in sys.modules, (
            "photon_mlx.checkpoint must not import photon_mlx.loss (training-only)"
        )


class TestTrainerCompatWrapper:
    """The trainer re-export must keep the ``TrainState`` API unchanged."""

    def test_trainer_load_returns_train_state(self, tmp_path: Path) -> None:
        from photon_mlx.trainer import (
            TrainState,
            load_checkpoint as trainer_load,
            save_checkpoint as trainer_save,
        )

        mx.random.seed(7)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)
        ts = TrainState(step=11, best_val_loss=0.9, best_step=10, patience_counter=0)
        trainer_save(model, ts, tmp_path)

        loaded = trainer_load(PhotonModel(cfg), tmp_path)
        assert isinstance(loaded, TrainState)
        assert loaded.step == 11
        assert loaded.best_val_loss == 0.9
        assert loaded.best_step == 10
        assert loaded.patience_counter == 0

    def test_checkpoint_module_and_trainer_are_interoperable(
        self, tmp_path: Path
    ) -> None:
        """Files written by trainer.save_checkpoint must load via the
        runtime-only ``photon_mlx.checkpoint.load_checkpoint`` and vice versa.
        baseline_reporag relies on this for hot-swapping checkpoints written
        by training jobs."""
        from photon_mlx.checkpoint import (
            CheckpointState,
            load_checkpoint as ckpt_load,
            save_checkpoint as ckpt_save,
        )
        from photon_mlx.trainer import (
            TrainState,
            load_checkpoint as trainer_load,
            save_checkpoint as trainer_save,
        )

        mx.random.seed(7)
        cfg = _tiny_cfg()
        model = PhotonModel(cfg)

        # trainer-written -> ckpt-readable
        a = tmp_path / "from_trainer"
        trainer_save(model, TrainState(step=33, best_val_loss=0.5), a)
        loaded_a = ckpt_load(PhotonModel(cfg), a)
        assert isinstance(loaded_a, CheckpointState)
        assert loaded_a.step == 33

        # ckpt-written -> trainer-readable
        b = tmp_path / "from_ckpt"
        ckpt_save(model, CheckpointState(step=77, best_val_loss=0.4), b)
        loaded_b = trainer_load(PhotonModel(cfg), b)
        assert isinstance(loaded_b, TrainState)
        assert loaded_b.step == 77
