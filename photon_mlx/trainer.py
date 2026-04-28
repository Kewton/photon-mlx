"""Training loop for PHOTON models in MLX."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers

from .checkpoint import (
    CheckpointState,
    load_checkpoint as _ckpt_load,
    save_checkpoint as _ckpt_save,
)
from .data import iterate_batches, iterate_mixed_batches
from .loss import photon_loss
from .model import PhotonModel

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from torch_ref.config import (  # noqa: E402
    EarlyStoppingConfig,
    PhotonConfig,
    TrainingConfig,
    load_photon_config,
)

_logger = logging.getLogger(__name__)


@dataclass
class TrainState:
    step: int = 0
    best_val_loss: float = float("inf")
    best_step: int = 0
    patience_counter: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def _to_checkpoint_state(state: TrainState) -> CheckpointState:
    return CheckpointState(
        step=state.step,
        best_val_loss=state.best_val_loss,
        best_step=state.best_step,
        patience_counter=state.patience_counter,
        train_losses=list(state.train_losses),
        val_losses=list(state.val_losses),
    )


def _from_checkpoint_state(state: CheckpointState) -> TrainState:
    return TrainState(
        step=state.step,
        best_val_loss=state.best_val_loss,
        best_step=state.best_step,
        patience_counter=state.patience_counter,
        train_losses=list(state.train_losses),
        val_losses=list(state.val_losses),
    )


def save_checkpoint(
    model: PhotonModel,
    state: TrainState,
    path: str | Path,
) -> None:
    _ckpt_save(model, _to_checkpoint_state(state), path)


def load_checkpoint(
    model: PhotonModel,
    path: str | Path,
) -> TrainState:
    return _from_checkpoint_state(_ckpt_load(model, path))


def _save_named_checkpoint(
    model: PhotonModel,
    state: TrainState,
    checkpoint_dir: Path,
    name: str,
    *,
    verbose: bool = True,
) -> Path:
    """Save a checkpoint under checkpoint_dir/name.

    For name == "best", write to best.tmp/ first and then os.replace the
    directory, so an interrupted save can never leave a partially-written
    best/ around. Other names are written in-place (consistent with
    historical behavior for step_XXXXXX/ and final/).
    """
    target = Path(checkpoint_dir) / name
    if name == "best":
        tmp = Path(checkpoint_dir) / "best.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        save_checkpoint(model, state, tmp)
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp, target)
    else:
        save_checkpoint(model, state, target)
    if verbose:
        print(f"  [ckpt] saved -> {target}")
    return target


def load_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
) -> PhotonModel:
    """Build a PhotonModel from config YAML and checkpoint directory."""
    cfg = load_photon_config(str(config_path))
    model = PhotonModel(cfg)
    _state = load_checkpoint(model, checkpoint_path)
    mx.eval(model.parameters())
    return model


def _build_lr_schedule(
    t_cfg: TrainingConfig,
) -> Any:
    """Build LR schedule: optional warmup + cosine decay."""
    lr = t_cfg.learning_rate
    min_lr = t_cfg.min_learning_rate
    warmup_steps = int(t_cfg.max_steps * t_cfg.warmup_ratio)

    if warmup_steps > 0 and min_lr > 0:
        warmup = mlx.optimizers.linear_schedule(init=1e-7, end=lr, steps=warmup_steps)
        decay = mlx.optimizers.cosine_decay(
            init=lr, decay_steps=t_cfg.max_steps - warmup_steps, end=min_lr
        )
        return mlx.optimizers.join_schedules(
            schedules=[warmup, decay], boundaries=[warmup_steps]
        )
    elif warmup_steps > 0:
        warmup = mlx.optimizers.linear_schedule(init=1e-7, end=lr, steps=warmup_steps)
        constant = mlx.optimizers.cosine_decay(
            init=lr, decay_steps=t_cfg.max_steps - warmup_steps, end=lr
        )
        return mlx.optimizers.join_schedules(
            schedules=[warmup, constant], boundaries=[warmup_steps]
        )
    elif min_lr > 0:
        return mlx.optimizers.cosine_decay(
            init=lr, decay_steps=t_cfg.max_steps, end=min_lr
        )
    else:
        return lr


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


def _update_early_stopping(
    state: TrainState,
    val_loss: float,
    es_cfg: EarlyStoppingConfig,
) -> tuple[bool, bool]:
    """Update state for the given val_loss and return (improved, should_stop).

    Pure helper – no I/O – so unit tests can drive patience logic directly.
    """
    improved = val_loss < state.best_val_loss - es_cfg.min_delta
    if improved:
        state.best_val_loss = val_loss
        state.best_step = state.step
        state.patience_counter = 0
    else:
        state.patience_counter += 1
    should_stop = bool(es_cfg.enabled) and state.patience_counter >= es_cfg.patience
    return improved, should_stop


def _write_train_log(log_path: Path, entry: dict) -> None:
    """Append a train-only log record.

    Invariant: train entries never contain eval-only keys (val_loss,
    best, best_step, best_val_loss, patience_counter, early_stopped,
    early_stop_reason).
    """
    eval_only = {
        "val_loss",
        "best",
        "best_step",
        "best_val_loss",
        "patience_counter",
        "early_stopped",
        "early_stop_reason",
    }
    clean = {k: v for k, v in entry.items() if k not in eval_only}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def _write_eval_log(log_path: Path, entry: dict) -> None:
    """Append an eval-only log record.

    Invariant: eval entries must always include val_loss.
    """
    if "val_loss" not in entry:
        raise ValueError("eval log entry must include val_loss")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def train(
    cfg: PhotonConfig,
    checkpoint_dir: str | Path,
    log_dir: str | Path,
    resume_from: str | Path | None = None,
    *,
    approved_roots: list[str | Path] | None = None,
) -> TrainState:
    """Full training loop driven by cfg.training.

    ``approved_roots`` is forwarded to ``iterate_mixed_batches`` when
    ``cfg.training.train_corpora_mix`` is set; production runs leave it
    ``None`` so the loader falls back to the production allow-list
    (``data/training/`` + ``data/processed/``). Tests pass tmp paths
    here to bypass the production guard without disabling DR4-002.
    """
    h_cfg = cfg.hierarchy

    # Training config (use defaults if not provided)
    t_cfg = cfg.training if cfg.training is not None else TrainingConfig()
    es_cfg = t_cfg.early_stopping

    # Build model
    model = PhotonModel(cfg)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")

    # LR schedule
    lr_schedule = _build_lr_schedule(t_cfg)

    # Optimizer: AdamW with weight_decay
    optimizer = mlx.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=t_cfg.weight_decay
    )

    # State
    state = TrainState()
    if resume_from and Path(resume_from).exists():
        state = load_checkpoint(model, resume_from)
        print(f"Resumed from step {state.step}")

    # Training params from config
    context_length = t_cfg.context_length
    batch_size = t_cfg.micro_batch_size
    max_steps = t_cfg.max_steps
    eval_every = t_cfg.eval_every_steps
    save_every = t_cfg.save_every_steps
    log_every = t_cfg.log_every_steps
    grad_accum_steps = t_cfg.gradient_accumulation_steps
    max_grad_norm = t_cfg.max_grad_norm

    # Data
    if t_cfg.train_corpora_mix is not None:
        # Issue #135 / Phase 4: mixed-corpus path. iterate_mixed_batches
        # builds an independent sequence pool per corpus, weighted-samples
        # at sequence level, and (when val_split > 0) returns a (train, val)
        # tuple drawn from the same shuffled mixture so train/val share
        # the train_corpora_mix ratio (DR1-005). Strict validation
        # (DR1-003 / DR4-002) lives in iterate_mixed_batches; cfg-side
        # checks already ran in TrainingConfig.__post_init__.
        print("Loading mixed training data...")
        result = iterate_mixed_batches(
            t_cfg.train_corpora_mix,
            context_length=context_length,
            batch_size=batch_size,
            vocab_size=cfg.tokenizer.vocab_size,
            val_split=t_cfg.val_split,
            approved_roots=approved_roots,
        )
        if t_cfg.val_split > 0.0:
            train_batches, val_batches = result  # type: ignore[misc]
        else:
            train_batches = result  # type: ignore[assignment]
            # Legacy single val_corpus is allowed alongside a mix when
            # val_split=0 — e.g. an external val set held out from the
            # training mixture entirely.
            val_batches = (
                iterate_batches(
                    t_cfg.val_corpus, context_length, batch_size, shuffle=False
                )
                if t_cfg.val_corpus
                else []
            )
        print(f"  {len(train_batches)} train batches (mixed)")
        print(f"  {len(val_batches)} val batches")
    else:
        print("Loading training data...")
        train_batches = iterate_batches(t_cfg.train_corpus, context_length, batch_size)
        print(f"  {len(train_batches)} train batches")

        print("Loading validation data...")
        val_batches = iterate_batches(
            t_cfg.val_corpus, context_length, batch_size, shuffle=False
        )
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

    # For loss breakdown logging
    def _get_breakdown(model: PhotonModel, batch: mx.array) -> dict:
        logits, _ = model(batch, labels=batch)
        _, breakdown = photon_loss(logits, batch, rec_w)
        return {k: v.item() if hasattr(v, "item") else v for k, v in breakdown.items()}

    # Log file
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.jsonl"
    checkpoint_dir = Path(checkpoint_dir)

    # Training loop
    print(f"\nTraining for {max_steps} steps (grad_accum={grad_accum_steps})...")
    t0 = time.time()
    batch_idx = 0
    accumulated_grads = None
    micro_step = 0
    early_stopped = False

    while state.step < max_steps:
        batch = train_batches[batch_idx % len(train_batches)]
        batch_idx += 1

        loss, grads = loss_and_grad(model, batch)
        mx.eval(loss)

        # Gradient accumulation
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = _tree_add(accumulated_grads, grads)
        micro_step += 1

        if micro_step < grad_accum_steps:
            continue

        # Average gradients
        if grad_accum_steps > 1:
            accumulated_grads = _tree_scale(accumulated_grads, 1.0 / grad_accum_steps)

        # Gradient clipping
        if max_grad_norm > 0:
            accumulated_grads, _ = mlx.optimizers.clip_grad_norm(
                accumulated_grads, max_norm=max_grad_norm
            )

        # Update
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters())

        state.step += 1
        loss_val = loss.item()
        state.train_losses.append(loss_val)

        # Reset accumulation
        accumulated_grads = None
        micro_step = 0

        # Log
        if state.step % log_every == 0:
            elapsed = time.time() - t0
            breakdown = _get_breakdown(model, batch)
            print(
                f"  step {state.step:>5d}  loss {loss_val:.4f}  elapsed {elapsed:.0f}s"
            )
            log_entry = {
                "step": state.step,
                "train_loss": loss_val,
                "elapsed_s": round(elapsed, 1),
            }
            log_entry.update(breakdown)
            _write_train_log(log_path, log_entry)

        # Eval
        if state.step % eval_every == 0:
            val_loss = evaluate(model, val_batches, rec_w)
            state.val_losses.append(val_loss)
            improved, should_stop = _update_early_stopping(state, val_loss, es_cfg)
            print(
                f"  [eval] step {state.step}  val_loss {val_loss:.4f}"
                f"  best {state.best_val_loss:.4f}"
                f"{'  *' if improved else ''}"
            )
            eval_entry = {
                "step": state.step,
                "val_loss": val_loss,
                "best": improved,
                "best_step": state.best_step,
                "best_val_loss": state.best_val_loss,
                "patience_counter": state.patience_counter,
                "early_stopped": should_stop,
            }
            if should_stop:
                eval_entry["early_stop_reason"] = "patience_exhausted"
            _write_eval_log(log_path, eval_entry)

            # Save best/ on improvement (only when early stopping is enabled;
            # keeps disabled path 100% backward compatible with existing tests
            # and on-disk layouts).
            if es_cfg.enabled and improved:
                _save_named_checkpoint(model, state, checkpoint_dir, "best")

            if should_stop:
                early_stopped = True
                print(
                    f"  [early stop] patience exhausted at step {state.step} "
                    f"(best_step={state.best_step}, best_val_loss={state.best_val_loss:.4f})"
                )
                break

        # Checkpoint
        if state.step % save_every == 0:
            ckpt_path = checkpoint_dir / f"step_{state.step:06d}"
            save_checkpoint(model, state, ckpt_path)
            print(f"  [ckpt] saved -> {ckpt_path}")

    # Final checkpoint: when early stopping is enabled with restore_best,
    # reload best/ into the live model so final/ == best. Otherwise write
    # the stop-time weights.
    best_dir = checkpoint_dir / "best"
    if es_cfg.enabled and es_cfg.restore_best and best_dir.exists():
        try:
            restored = load_checkpoint(model, best_dir)
            state = restored
            print(f"  [restore_best] loaded best/ at step {state.best_step}")
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            OSError,
            ValueError,
            TypeError,
        ) as exc:
            _logger.warning("best/ restore failed: %s; keeping stop-time weights", exc)
            # Emit a trailing eval-style entry so downstream log readers can
            # surface the fallback reason.
            _write_eval_log(
                log_path,
                {
                    "step": state.step,
                    "val_loss": state.best_val_loss,
                    "best": False,
                    "best_step": state.best_step,
                    "best_val_loss": state.best_val_loss,
                    "patience_counter": state.patience_counter,
                    "early_stopped": early_stopped,
                    "early_stop_reason": "best_restore_failed",
                },
            )
    save_checkpoint(model, state, checkpoint_dir / "final")
    final_loss = state.train_losses[-1] if state.train_losses else float("nan")
    print(f"\nTraining complete. Final loss: {final_loss:.4f}")
    return state


def _tree_add(a: Any, b: Any) -> Any:
    """Element-wise add two gradient trees."""
    if isinstance(a, dict):
        return {k: _tree_add(a[k], b[k]) for k in a}
    if isinstance(a, list):
        return [_tree_add(ai, bi) for ai, bi in zip(a, b)]
    return a + b


def _tree_scale(tree: Any, scale: float) -> Any:
    """Scale all arrays in a gradient tree."""
    if isinstance(tree, dict):
        return {k: _tree_scale(v, scale) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_scale(v, scale) for v in tree]
    return tree * scale
