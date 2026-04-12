"""
PHOTON session-level working memory.

Maintains hierarchical latent state across turns, tracks drift,
and provides topic shift features for Safe RecGen.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class HierarchicalState:
    """Cached encoder outputs at each hierarchy level."""
    level_states: list[mx.array] = field(default_factory=list)  # per-level latents
    token_proj: mx.array | None = None                          # projected token embeddings
    turn_id: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class DriftMetrics:
    """Per-turn drift measurements."""
    turn_id: int = 0
    latent_cosine_drift: float = 0.0    # cosine distance between prev/current top latent
    token_agreement: float = 1.0        # fraction of top-1 predictions that agree
    logit_kl: float = 0.0              # KL divergence of logit distributions
    topic_shift_score: float = 0.0     # proxy for topic change magnitude

    def as_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "latent_cosine_drift": round(self.latent_cosine_drift, 6),
            "token_agreement": round(self.token_agreement, 6),
            "logit_kl": round(self.logit_kl, 6),
            "topic_shift_score": round(self.topic_shift_score, 6),
        }


def cosine_distance(a: mx.array, b: mx.array) -> float:
    """1 - cosine_similarity between two vectors (mean-pooled if multi-dim)."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
    norm_b = mx.sqrt(mx.sum(b_flat * b_flat))
    cos_sim = dot / (norm_a * norm_b + 1e-8)
    mx.eval(cos_sim)
    return 1.0 - cos_sim.item()


def kl_divergence(p_logits: mx.array, q_logits: mx.array) -> float:
    """KL(softmax(p) || softmax(q)), averaged over positions."""
    p = mx.softmax(p_logits.astype(mx.float32), axis=-1)
    q = mx.softmax(q_logits.astype(mx.float32), axis=-1)
    kl = mx.sum(p * (mx.log(p + 1e-10) - mx.log(q + 1e-10)), axis=-1)
    result = mx.mean(kl)
    mx.eval(result)
    return max(0.0, result.item())


def token_agreement_rate(logits_a: mx.array, logits_b: mx.array) -> float:
    """Fraction of positions where argmax agrees."""
    pred_a = mx.argmax(logits_a, axis=-1)
    pred_b = mx.argmax(logits_b, axis=-1)
    agree = mx.mean((pred_a == pred_b).astype(mx.float32))
    mx.eval(agree)
    return agree.item()


class PhotonSessionState:
    """
    Working memory for a single PHOTON-RAG session.

    Stores hierarchical latents from previous turns and computes
    drift metrics when the state is updated.
    """

    def __init__(self, session_id: str, repo_id: str, repo_commit: str) -> None:
        self.session_id = session_id
        self.repo_id = repo_id
        self.repo_commit = repo_commit
        self.current_state: HierarchicalState | None = None
        self.prev_state: HierarchicalState | None = None
        self.prev_logits: mx.array | None = None
        self.drift_history: list[DriftMetrics] = []
        self.turn_count: int = 0

    def update(
        self,
        new_state: HierarchicalState,
        new_logits: mx.array | None = None,
    ) -> DriftMetrics:
        """Update session state and compute drift metrics."""
        self.turn_count += 1
        self.prev_state = self.current_state
        self.current_state = new_state
        new_state.turn_id = self.turn_count

        metrics = DriftMetrics(turn_id=self.turn_count)

        if self.prev_state is not None and self.prev_state.level_states:
            # Latent cosine drift at top level
            prev_top = self.prev_state.level_states[-1]
            curr_top = new_state.level_states[-1]
            metrics.latent_cosine_drift = cosine_distance(prev_top, curr_top)

            # Topic shift = cosine drift (simple proxy; can be refined)
            metrics.topic_shift_score = metrics.latent_cosine_drift

        if self.prev_logits is not None and new_logits is not None:
            metrics.token_agreement = token_agreement_rate(
                self.prev_logits, new_logits)
            metrics.logit_kl = kl_divergence(self.prev_logits, new_logits)

        self.prev_logits = new_logits
        self.drift_history.append(metrics)
        return metrics

    def latest_drift(self) -> DriftMetrics | None:
        return self.drift_history[-1] if self.drift_history else None
