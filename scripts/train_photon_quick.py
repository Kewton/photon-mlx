"""
train_photon_quick.py  –  Quick PHOTON training smoke test (100 steps).

Usage:
    python scripts/train_photon_quick.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_ref.config import load_photon_config
from photon_mlx.model import PhotonModel
from photon_mlx.data import iterate_batches
from photon_mlx.loss import photon_loss


def main() -> None:
    cfg = load_photon_config("configs/photon_tiny.yaml")

    print("Building model...")
    mx.random.seed(42)
    model = PhotonModel(cfg)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    # Load training data
    train_corpus = "data/processed/train_tiny.jsonl"
    val_corpus = "data/processed/val_tiny.jsonl"

    if not Path(train_corpus).exists():
        print(f"Training corpus not found: {train_corpus}")
        print("Run: python scripts/generate_training_corpus.py")
        return

    context_length = 64  # short for quick test (must be divisible by 4*4=16)
    batch_size = 2

    print(f"Loading data (context_length={context_length})...")
    train_batches = iterate_batches(train_corpus, context_length, batch_size, seed=42)
    val_batches = iterate_batches(val_corpus, context_length, batch_size, shuffle=False)
    print(f"  Train: {len(train_batches)} batches, Val: {len(val_batches)} batches")

    if not train_batches:
        print("No training batches. Check corpus and context_length.")
        return

    # Training
    optimizer = mlx.optimizers.Adam(learning_rate=2e-4)
    rec_w = cfg.hierarchy.recursive_loss_weight

    def loss_fn(model, batch):
        logits, _ = model(batch, labels=batch)
        total, _ = photon_loss(logits, batch, rec_w)
        return total

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    max_steps = 100
    log_every = 10
    eval_every = 50

    print(f"\nTraining for {max_steps} steps...")
    t0 = time.time()
    losses: list[float] = []

    for step in range(1, max_steps + 1):
        batch = train_batches[(step - 1) % len(train_batches)]
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), loss)
        loss_val = loss.item()
        losses.append(loss_val)

        if step % log_every == 0:
            elapsed = time.time() - t0
            print(f"  step {step:>4d}  loss {loss_val:.4f}  elapsed {elapsed:.1f}s")

        if step % eval_every == 0 and val_batches:
            val_total = 0.0
            for vb in val_batches[:5]:
                logits, _ = model(vb, labels=vb)
                vl, _ = photon_loss(logits, vb, rec_w)
                mx.eval(vl)
                val_total += vl.item()
            val_avg = val_total / min(5, len(val_batches))
            print(f"  [eval] step {step}  val_loss {val_avg:.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Reduction:    {(1 - losses[-1] / losses[0]) * 100:.1f}%")


if __name__ == "__main__":
    main()
