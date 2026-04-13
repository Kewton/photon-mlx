"""Loss functions for PHOTON training."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def next_token_loss(
    logits: mx.array,
    labels: mx.array,
) -> mx.array:
    """Standard next-token cross-entropy (shifted by 1)."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return mx.mean(
        nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
        )
    )


def photon_loss(
    logits: mx.array,
    labels: mx.array,
    recursive_loss_weight: float = 0.0,
) -> tuple[mx.array, dict[str, mx.array]]:
    """
    Combined PHOTON loss.

    Returns (total_loss, breakdown_dict).

    v1: recursive_loss_weight = 0.0 (next-token only).
    v2: add recursive consistency loss with nonzero weight.
    """
    ntp = next_token_loss(logits, labels)

    breakdown = {"next_token_loss": ntp}

    if recursive_loss_weight > 0.0:
        # Placeholder: recursive consistency loss is added in v2
        rec_loss = mx.array(0.0)
        total = ntp + recursive_loss_weight * rec_loss
        breakdown["recursive_loss"] = rec_loss
    else:
        total = ntp

    breakdown["total_loss"] = total
    return total, breakdown
