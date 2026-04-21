"""
PHOTON session-level working memory.

Maintains hierarchical latent state across turns, tracks drift,
and provides topic shift features for Safe RecGen.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field

import mlx.core as mx


__all__ = [
    "HierarchicalState",
    "DriftMetrics",
    "TurnState",
    "WorkingMemoryConfig",
    "WORKING_MEMORY_MAX_TURNS_HARD_CAP",
    "PhotonSessionState",
    "cosine_distance",
    "kl_divergence",
    "token_agreement_rate",
    "mean_pool",
]


# Security limit: truncate question_text to this many characters when stored in
# turn_history to bound memory usage and PII exposure (design §3-1 / §7).
_QUESTION_TEXT_MAX_LEN: int = 2048


# Security hard cap on WorkingMemoryConfig.max_turns (Codex CB-004).
#
# ``turn_history`` retains one ``HierarchicalState`` per turn. Each state holds
# ``mx.array`` references whose size scales with ``hidden_size`` — at
# photon_small (hidden=640) ≈ 6.6 MiB/turn and photon_tiny (hidden=1024)
# ≈ 10.5 MiB/turn. The GA design target (design §7) is to keep working memory
# under 100 MiB for the default ``max_turns=8``. Allowing an unbounded value
# opens a memory-exhaustion DoS, so ``__post_init__`` rejects any
# ``max_turns`` above this ceiling.
#
# 32 is chosen to give ~210 MiB headroom on photon_small and ~336 MiB on
# photon_tiny — well above the GA budget yet still bounded.
WORKING_MEMORY_MAX_TURNS_HARD_CAP: int = 32


# Sentinel for ``PhotonSessionState(working_memory_cfg=...)`` (Codex CB-001).
#
# We must distinguish three call styles:
#   * omitted  (legacy callers)  → default ``WorkingMemoryConfig()`` (enabled=True)
#   * ``None`` (fail-closed)     → ``WorkingMemoryConfig(enabled=False)``
#   * instance                   → use as-is
# ``None`` used to collapse to the default, which silently re-enabled working
# memory for sessions whose YAML config was rejected by
# ``_resolve_working_memory_cfg`` (which fails closed by returning ``None``).
class _UnsetType:
    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return "<UNSET>"


_UNSET = _UnsetType()


def mean_pool(x: mx.array) -> mx.array:
    """Reduce leading dims via unweighted mean, returning a ``(D,)`` vector.

    Used to collapse a ``(B, T, D)`` or ``(T, D)`` hierarchical latent down to
    a single ``(D,)`` coarse vector for cosine comparisons and cross-turn
    aggregation (Issue #64 DR1-002). Casts to ``float32`` for stability.
    """
    x32 = x.astype(mx.float32)
    if x32.ndim > 1:
        return mx.mean(x32, axis=tuple(range(x32.ndim - 1)))
    return x32


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
        """Return a JSON-safe superset schema (Issue #63 + Issue #64).

        Includes the legacy ``latent_cosine_drift`` alias plus the three
        per-level drift keys from Issue #63 (DR3-002). Non-finite values
        (``NaN`` / ``+Inf`` / ``-Inf``) are replaced with ``0.0`` and a
        ``warnings.warn`` is emitted — only the field name is included in
        the warning text so raw latent vectors or question text are never
        surfaced through this path (design §7).
        """
        safe: dict[str, float | int] = {"turn_id": self.turn_id}
        for name in (
            "latent_cosine_drift",
            "latent_cosine_drift_top",
            "latent_cosine_drift_mid",
            "latent_cosine_drift_token",
            "token_agreement",
            "logit_kl",
            "topic_shift_score",
        ):
            raw = getattr(self, name)
            if not math.isfinite(raw):
                warnings.warn(
                    f"DriftMetrics.{name} is non-finite; coercing to 0.0",
                    RuntimeWarning,
                    stacklevel=2,
                )
                raw = 0.0
            safe[name] = round(float(raw), 6)
        return safe


@dataclass
class TurnState:
    """Public per-turn record retained in the session working memory.

    Stored in :attr:`PhotonSessionState.turn_history`. ``question_text`` is
    kept only for Phase 2 retrieval heuristics and is never fed back into
    prompts or telemetry verbatim (design §3-1 security note).
    """

    turn_id: int
    hierarchical_state: HierarchicalState
    question_text: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkingMemoryConfig:
    """Configuration for cross-turn hierarchical working memory (Issue #64).

    All fields have sensible defaults; ``__post_init__`` enforces strict type
    and range validation (DR4-001) so malformed YAML cannot silently disable
    safety invariants.
    """

    enabled: bool = True
    max_turns: int = 8
    decay_factor: float = 0.5
    relevant_turn_threshold: float = 0.7
    compress_old_turns: bool = True  # Phase 2 reservation — not consumed in Phase 1.

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be bool, got {type(self.enabled).__name__}")
        if not isinstance(self.compress_old_turns, bool):
            raise TypeError(
                "compress_old_turns must be bool, got "
                f"{type(self.compress_old_turns).__name__}"
            )
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int):
            raise TypeError(
                f"max_turns must be int, got {type(self.max_turns).__name__}"
            )
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")
        # Hard upper bound (Codex CB-004): unbounded ``max_turns`` would let
        # a malformed YAML trigger memory exhaustion via turn_history growth.
        # The ceiling is chosen from photon_small / photon_tiny per-turn
        # latent footprints (see ``WORKING_MEMORY_MAX_TURNS_HARD_CAP`` docstring).
        if self.max_turns > WORKING_MEMORY_MAX_TURNS_HARD_CAP:
            raise ValueError(
                "max_turns must be <= "
                f"{WORKING_MEMORY_MAX_TURNS_HARD_CAP} (hard cap), got "
                f"{self.max_turns}"
            )
        for name in ("decay_factor", "relevant_turn_threshold"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be float, got {type(val).__name__}")
            if not math.isfinite(float(val)):
                raise ValueError(f"{name} must be finite, got {val}")
        if not (0.0 <= float(self.decay_factor) <= 1.0):
            raise ValueError(
                f"decay_factor must be 0.0 <= float <= 1.0, got {self.decay_factor}"
            )
        if not (-1.0 <= float(self.relevant_turn_threshold) <= 1.0):
            raise ValueError(
                "relevant_turn_threshold must be -1.0 <= float <= 1.0, got "
                f"{self.relevant_turn_threshold}"
            )


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
    # so different sequence lengths don't cause shape mismatches (DR1-002:
    # shared helper with get_session_coarse_state).
    a_vec = mean_pool(a)
    b_vec = mean_pool(b)
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


def _sanitize_question_text(raw: str | None) -> str:
    """Sanitize user-supplied question text for long-term retention.

    Drops C0/C1 control characters except ``\\t \\n \\r`` (log-poisoning
    mitigation) and truncates to :data:`_QUESTION_TEXT_MAX_LEN` to bound
    memory use (design §7 security notes).
    """
    if not raw:
        return ""
    cleaned = "".join(
        ch
        for ch in raw
        if ch in ("\t", "\n", "\r") or (ord(ch) >= 0x20 and ord(ch) != 0x7F)
    )
    if len(cleaned) > _QUESTION_TEXT_MAX_LEN:
        cleaned = cleaned[:_QUESTION_TEXT_MAX_LEN]
    return cleaned


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
        working_memory_cfg: WorkingMemoryConfig | None | _UnsetType = _UNSET,
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
        # Cross-turn hierarchical working memory (Issue #64, Phase 1).
        #
        # Codex CB-001 — explicit ``None`` is the fail-closed signal
        # produced by ``_build_photon_deps()`` when YAML is malformed or
        # the section is missing. It must DISABLE working memory rather
        # than silently fall back to the default enabled config.
        # Only the ``_UNSET`` sentinel (legacy callers that omitted the
        # argument) yields the default ``WorkingMemoryConfig()``.
        resolved_cfg: WorkingMemoryConfig
        if isinstance(working_memory_cfg, _UnsetType):
            resolved_cfg = WorkingMemoryConfig()
        elif working_memory_cfg is None:
            resolved_cfg = WorkingMemoryConfig(enabled=False)
        else:
            resolved_cfg = working_memory_cfg
        self.working_memory_cfg: WorkingMemoryConfig = resolved_cfg
        self.turn_history: list[TurnState] = []

    def update(
        self,
        new_state: HierarchicalState,
        new_logits: mx.array | None = None,
        question_text: str | None = None,
    ) -> DriftMetrics:
        """Update session state and compute drift metrics.

        Signature unchanged from pre-Issue-#63 (DR1-012). Weights come from
        ``self.drift_level_weights`` so callers do not have to thread the
        configuration through every ``update()`` call.

        Args:
            new_state: the :class:`HierarchicalState` produced by the current
                turn's ``hierarchical_prefill``.
            new_logits: optional top-level logits for drift telemetry.
            question_text: optional user question associated with this turn;
                stored in :attr:`turn_history` only when
                ``working_memory_cfg.enabled`` is ``True`` (DR1-006).
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

        # Append to turn_history only when working memory is enabled (Issue #64).
        if self.working_memory_cfg.enabled:
            self.turn_history.append(
                TurnState(
                    turn_id=self.turn_count,
                    hierarchical_state=new_state,
                    question_text=_sanitize_question_text(question_text),
                )
            )
            max_turns = self.working_memory_cfg.max_turns
            while len(self.turn_history) > max_turns:
                # Phase 2 will replace this with compress_oldest_turn; for now
                # we simply drop the oldest reference so the GC can reclaim
                # latents (design §3-1 security note).
                self.turn_history.pop(0)

        return metrics

    def latest_drift(self) -> DriftMetrics | None:
        return self.drift_history[-1] if self.drift_history else None

    def get_session_coarse_state(self) -> mx.array | None:
        """Return a ``(D,)`` coarse session vector aggregated across turns.

        Aggregation is a weighted mean of each turn's top-level
        ``mean_pool(level_states[-1])`` using geometric decay
        ``decay_factor ** (N-i-1)``. Returns ``None`` when working memory
        is disabled or no turn is recorded (design §3-2 judgement #2,
        DR1-004 — callers must fall back to the legacy path).
        """
        if not self.working_memory_cfg.enabled:
            return None
        if not self.turn_history:
            return None

        decay = float(self.working_memory_cfg.decay_factor)
        n = len(self.turn_history)
        # Collect per-turn coarse vectors.
        vecs: list[mx.array] = []
        weights: list[float] = []
        for i, turn in enumerate(self.turn_history):
            if not turn.hierarchical_state.level_states:
                continue
            top = turn.hierarchical_state.level_states[-1]
            vecs.append(mean_pool(top))
            weights.append(decay ** (n - i - 1))

        if not vecs:
            return None

        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            # All weights were zero (decay=0 on history length > 1). Fall
            # back to a plain mean of the last entry.
            return vecs[-1]

        stacked = mx.stack(vecs, axis=0)  # (N, D)
        w_arr = mx.array(weights, dtype=mx.float32)[:, None]  # (N, 1)
        aggregated = mx.sum(stacked * w_arr, axis=0) / float(weight_sum)
        mx.eval(aggregated)
        return aggregated

    def reset_working_memory(self) -> None:
        """Clear stale latents and turn history for fail-closed recovery.

        Drops ``current_state``, ``prev_state``, ``prev_logits`` and the
        accumulated ``turn_history``. ``drift_history`` and ``turn_count``
        are intentionally preserved so observability APIs remain consistent
        (design judgement #4 / DR3-001).
        """
        self.current_state = None
        self.prev_state = None
        self.prev_logits = None
        self.turn_history = []
