"""Training loop for PHOTON models in MLX."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers

from .data import iterate_batches
from .loss import photon_loss
from .model import PhotonModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from torch_ref.config import PhotonConfig  # noqa: E402


@dataclass
class TrainState:
    step: int = 0
    best_val_loss: float = float("inf")
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def save_checkpoint(
    model: PhotonModel,
    state: TrainState,
    path: str | Path,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # Save model weights
    weights = dict(model.parameters())
    mx.savez(str(path / "weights.npz"), **_flatten(weights))
    # Save state
    (path / "state.json").write_text(
        json.dumps({
            "step": state.step,
            "best_val_loss": state.best_val_loss,
            "train_losses": state.train_losses[-100:],
            "val_losses": state.val_losses[-100:],
        }),
        encoding="utf-8",
    )


def load_checkpoint(
    model: PhotonModel,
    path: str | Path,
) -> TrainState:
    path = Path(path)
    weights = mx.load(str(path / "weights.npz"))
    model.load_weights(list(weights.items()))
    state_data = json.loads((path / "state.json").read_text(encoding="utf-8"))
    return TrainState(**state_data)


def _flatten(tree: Any, prefix: str = "") -> dict[str, mx.array]:
    """Flatten nested dict/list to dot-separated keys."""
    flat: dict[str, mx.array] = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            flat.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            flat.update(_flatten(v, f"{prefix}{i}."))
    elif isinstance(tree, mx.array):
        flat[prefix.rstrip(".")] = tree
    return flat


def evaluate(
    model: PhotonModel,
    val_batches: list[mx.array],
    recursive_loss_weight: float = 0.0,
) -> float:
    """Run validation and return average loss."""
    total_loss = 0.0
    for batch in val_batches:
        logits, _ = model(batch, labels=batch)
        loss, _ = photon_loss(logits, batch, recursive_loss_weight)
        mx.eval(loss)
        total_loss += loss.item()
    return total_loss / max(len(val_batches), 1)


def train(
    cfg: PhotonConfig,
    train_corpus: str | Path,
    val_corpus: str | Path,
    checkpoint_dir: str | Path,
    log_dir: str | Path,
    resume_from: str | Path | None = None,
) -> TrainState:
    """Full training loop."""
    t_cfg = cfg.model
    h_cfg = cfg.hierarchy

    # Build model
    model = PhotonModel(cfg)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")

    # Optimizer
    lr = 2e-4  # default, override from training config if available
    optimizer = mlx.optimizers.Adam(learning_rate=lr)

    # State
    state = TrainState()
    if resume_from and Path(resume_from).exists():
        state = load_checkpoint(model, resume_from)
        print(f"Resumed from step {state.step}")

    # Data
    context_length = 2048  # default
    batch_size = 4
    max_steps = 5000
    eval_every = 200
    save_every = 500
    log_every = 20

    print("Loading training data...")
    train_batches = iterate_batches(train_corpus, context_length, batch_size)
    print(f"  {len(train_batches)} train batches")

    print("Loading validation data...")
    val_batches = iterate_batches(val_corpus, context_length, batch_size, shuffle=False)
    print(f"  {len(val_batches)} val batches")

    if not train_batches:
        raise ValueError("No training batches. Check corpus and context_length.")

    # Loss + grad function
    rec_w = h_cfg.recursive_loss_weight

    def loss_fn(model: PhotonModel, batch: mx.array) -> mx.array:
        logits, _ = model(batch, labels=batch)
        total, _ = photon_loss(logits, batch, rec_w)
        return total

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Log file
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.jsonl"
    checkpoint_dir = Path(checkpoint_dir)

    # Training loop
    print(f"\nTraining for {max_steps} steps...")
    t0 = time.time()
    batch_idx = 0

    while state.step < max_steps:
        batch = train_batches[batch_idx % len(train_batches)]
        batch_idx += 1

        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), loss)

        state.step += 1
        loss_val = loss.item()
        state.train_losses.append(loss_val)

        # Log
        if state.step % log_every == 0:
            elapsed = time.time() - t0
            print(f"  step {state.step:>5d}  loss {loss_val:.4f}  "
                  f"elapsed {elapsed:.0f}s")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": state.step,
                    "train_loss": loss_val,
                    "elapsed_s": round(elapsed, 1),
                }) + "\n")

        # Eval
        if state.step % eval_every == 0:
            val_loss = evaluate(model, val_batches, rec_w)
            state.val_losses.append(val_loss)
            improved = val_loss < state.best_val_loss
            if improved:
                state.best_val_loss = val_loss
            print(f"  [eval] step {state.step}  val_loss {val_loss:.4f}"
                  f"  best {state.best_val_loss:.4f}"
                  f"{'  *' if improved else ''}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": state.step,
                    "val_loss": val_loss,
                    "best": improved,
                }) + "\n")

        # Checkpoint
        if state.step % save_every == 0:
            ckpt_path = checkpoint_dir / f"step_{state.step:06d}"
            save_checkpoint(model, state, ckpt_path)
            print(f"  [ckpt] saved -> {ckpt_path}")

    # Final checkpoint
    save_checkpoint(model, state, checkpoint_dir / "final")
    print(f"\nTraining complete. Final loss: {state.train_losses[-1]:.4f}")
    return state
