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
    token_proj: mx.array | None = None  # projected token embeddings
    turn_id: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class DriftMetrics:
    """Per-turn drift measurements.

    Issue #63: hierarchical drift decomposition.
    ``latent_cosine_drift`` is now a read-only ``@property`` alias for
    ``latent_cosine_drift_top`` (DR1-009) so that existing consumers that
    referenced the legacy name continue to work while the 3-level metrics
    become the source of truth.
    """

    turn_id: int = 0
    token_agreement: float = 1.0  # fraction of top-1 predictions that agree
    logit_kl: float = 0.0  # KL divergence of logit distributions
    topic_shift_score: float = 0.0  # weighted_hierarchical_score(drifts, weights)

    # Per-level cosine drift (Issue #63). Fallbacks when a level is missing:
    # * ``len(level_states) == 1`` → ``latent_cosine_drift_mid = 0.0``
    # * ``token_proj is None``      → ``latent_cosine_drift_token = 0.0``
    latent_cosine_drift_top: float = 0.0  # drift of level_states[-1]
    latent_cosine_drift_mid: float = 0.0  # drift of level_states[0] (≥2 levels)
    latent_cosine_drift_token: float = 0.0  # drift of token_proj

    @property
    def latent_cosine_drift(self) -> float:
        """Backward-compat alias: equals ``latent_cosine_drift_top`` (DR1-009).

        Existing consumers (SafeRecGenController, log schemas, QueryResult,
        legacy tests) that referenced ``latent_cosine_drift`` keep working
        without changes; the 3-level fields above are authoritative.
        """
        return self.latent_cosine_drift_top

    def as_dict(self) -> dict:
        """Superset schema (Issue #63, DR3-002).

        Existing keys (``latent_cosine_drift`` / ``topic_shift_score`` / ...) are
        preserved bit-for-bit for backward-compat; three new keys are added for
        per-level drift. Downstream consumers can treat missing new keys as 0.0.
        """
        return {
            "turn_id": self.turn_id,
            "latent_cosine_drift": round(self.latent_cosine_drift, 6),
            "latent_cosine_drift_top": round(self.latent_cosine_drift_top, 6),
            "latent_cosine_drift_mid": round(self.latent_cosine_drift_mid, 6),
            "latent_cosine_drift_token": round(self.latent_cosine_drift_token, 6),
            "token_agreement": round(self.token_agreement, 6),
            "logit_kl": round(self.logit_kl, 6),
            "topic_shift_score": round(self.topic_shift_score, 6),
        }


def weighted_hierarchical_score(
    values: tuple[float, ...] | mx.array,
    weights: tuple[float, ...],
) -> float | mx.array:
    """Weighted sum of hierarchical scores shared by drift and scoring paths.

    Issue #63 / DR1-004 / DR2-001: a single helper so that the weighting
    semantics live in exactly one place.

    Two input contracts:

    * ``tuple[float, ...]`` (drift path, e.g. ``(drift_token, drift_mid,
      drift_top)``): returns a Python ``float`` equal to ``sum(w * v)``.
    * ``mx.array`` (scoring path): the array is expected to have shape
      ``(..., n)`` with the last axis matching ``len(weights)``. The caller
      is responsible for building the tensor via ``mx.stack([...], axis=-1)``.
      Returns an ``mx.array`` of shape ``(...,)`` (weight sum along the last
      axis).

    The weights are cast to ``values.dtype`` on the scoring path so the
    multiplication does not force an unwanted dtype promotion.
    """
    if isinstance(values, mx.array):
        n = len(weights)
        assert values.shape[-1] == n, (
            f"weighted_hierarchical_score: values.shape[-1]={values.shape[-1]} "
            f"does not match len(weights)={n}"
        )
        w = mx.array(weights, dtype=values.dtype)
        return mx.sum(values * w, axis=-1)
    # drift path: scalar tuple / list
    return sum(float(w) * float(v) for w, v in zip(weights, values))


def cosine_distance(a: mx.array, b: mx.array) -> float:
    """1 - cosine_similarity between two vectors (mean-pooled if multi-dim).

    Multi-dim inputs are mean-pooled over all dims except the last, so
    different sequence lengths collapse to a fixed ``(hidden_size,)`` vector.
    This is the shared primitive used by :class:`PhotonSessionState` for
    per-level drift.
    """
    # Mean-pool along all dims except the last to get a fixed (hidden_size,) vector,
    # so different sequence lengths don't cause shape mismatches.
    a_vec = (
        mx.mean(a.astype(mx.float32), axis=tuple(range(a.ndim - 1)))
        if a.ndim > 1
        else a.astype(mx.float32)
    )
    b_vec = (
        mx.mean(b.astype(mx.float32), axis=tuple(range(b.ndim - 1)))
        if b.ndim > 1
        else b.astype(mx.float32)
    )
    dot = mx.sum(a_vec * b_vec)
    norm_a = mx.sqrt(mx.sum(a_vec * a_vec))
    norm_b = mx.sqrt(mx.sum(b_vec * b_vec))
    cos_sim = dot / (norm_a * norm_b + 1e-8)
    mx.eval(cos_sim)
    return 1.0 - cos_sim.item()


def kl_divergence(p_logits: mx.array, q_logits: mx.array) -> float:
    """KL(softmax(p) || softmax(q)), averaged over positions."""
    # Truncate to min sequence length so different-length turns don't crash.
    min_len = min(p_logits.shape[-2], q_logits.shape[-2])
    p_logits = p_logits[..., :min_len, :]
    q_logits = q_logits[..., :min_len, :]
    p = mx.softmax(p_logits.astype(mx.float32), axis=-1)
    q = mx.softmax(q_logits.astype(mx.float32), axis=-1)
    kl = mx.sum(p * (mx.log(p + 1e-10) - mx.log(q + 1e-10)), axis=-1)
    result = mx.mean(kl)
    mx.eval(result)
    return max(0.0, result.item())


def token_agreement_rate(logits_a: mx.array, logits_b: mx.array) -> float:
    """Fraction of positions where argmax agrees."""
    # Truncate to min sequence length so different-length turns don't crash.
    min_len = min(logits_a.shape[-2], logits_b.shape[-2])
    logits_a = logits_a[..., :min_len, :]
    logits_b = logits_b[..., :min_len, :]
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

    Issue #63 adds per-level drift aggregation. ``drift_level_weights`` is a
    session-level immutable tuple (token, mid, top) used to fold the three
    cosine drifts into ``topic_shift_score`` via
    :func:`weighted_hierarchical_score`. Kept as a constructor keyword so
    :meth:`update` retains its 2-argument signature (DR1-012).
    """

    def __init__(
        self,
        session_id: str,
        repo_id: str,
        repo_commit: str,
        *,
        drift_level_weights: tuple[float, ...] = (0.2, 0.3, 0.5),
    ) -> None:
        self.session_id = session_id
        self.repo_id = repo_id
        self.repo_commit = repo_commit
        # Normalise to an immutable tuple of Python floats so downstream
        # code (scoring, topic_shift_score, logging) always sees the same
        # concrete type regardless of caller input (DR1-001).
        self.drift_level_weights: tuple[float, ...] = tuple(
            float(w) for w in drift_level_weights
        )
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
        """Update session state and compute drift metrics.

        Signature unchanged from pre-Issue-#63 (DR1-012). Weights come from
        ``self.drift_level_weights`` so callers do not have to thread the
        configuration through every ``update()`` call.
        """
        self.turn_count += 1
        self.prev_state = self.current_state
        self.current_state = new_state
        new_state.turn_id = self.turn_count

        metrics = DriftMetrics(turn_id=self.turn_count)

        if self.prev_state is not None:
            drift_top = 0.0
            drift_mid = 0.0
            drift_token = 0.0

            prev_states = self.prev_state.level_states
            curr_states = new_state.level_states

            # top (level_states[-1]) — existing behaviour
            if prev_states and curr_states:
                drift_top = cosine_distance(prev_states[-1], curr_states[-1])
                # mid (level_states[0]) — only when both sides have ≥2 levels.
                # For levels=1 fixtures the fallback ``0.0`` preserves the
                # existing semantics (design §5 decision #4 / S3-001).
                if len(prev_states) >= 2 and len(curr_states) >= 2:
                    drift_mid = cosine_distance(prev_states[0], curr_states[0])

            # token (token_proj) — both sides must be populated; ``None`` on
            # either side falls back to 0.0 (design §5 decision #4).
            if (
                self.prev_state.token_proj is not None
                and new_state.token_proj is not None
            ):
                drift_token = cosine_distance(
                    self.prev_state.token_proj, new_state.token_proj
                )

            metrics.latent_cosine_drift_top = drift_top
            metrics.latent_cosine_drift_mid = drift_mid
            metrics.latent_cosine_drift_token = drift_token
            # latent_cosine_drift is a @property (= top), no direct assignment.
            metrics.topic_shift_score = weighted_hierarchical_score(
                (drift_token, drift_mid, drift_top),
                self.drift_level_weights,
            )

        if self.prev_logits is not None and new_logits is not None:
            metrics.token_agreement = token_agreement_rate(self.prev_logits, new_logits)
            metrics.logit_kl = kl_divergence(self.prev_logits, new_logits)

        self.prev_logits = new_logits
        self.drift_history.append(metrics)
        return metrics

    def latest_drift(self) -> DriftMetrics | None:
        return self.drift_history[-1] if self.drift_history else None
