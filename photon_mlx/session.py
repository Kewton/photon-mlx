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
    "CompressedTurnState",
    "WorkingMemoryConfig",
    "WORKING_MEMORY_MAX_TURNS_HARD_CAP",
    "STORAGE_MODES",
    "PhotonSessionState",
    "cosine_distance",
    "kl_divergence",
    "token_agreement_rate",
    "mean_pool",
]


# Closed enumeration of accepted ``storage_mode`` values (Issue #79).
#
# Kept as a module-level frozenset so validation in
# :class:`WorkingMemoryConfig.__post_init__` and
# :meth:`PhotonSessionState._append_turn_for_mode` share a single source of
# truth. Added values MUST be lowercase to match the design §3.6 D6 ruling
# that rejects ``"Full"`` / ``"FULL"`` rather than auto-normalising.
STORAGE_MODES: frozenset[str] = frozenset({"full", "top_level_only", "summary_only"})


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
class CompressedTurnState:
    """Pooled summary of a single turn retained in ``compressed_history``.

    Issue #79 DR1-003 / DR1-005: compressed entries keep only the minimum
    needed for cross-turn aggregation — a pooled ``summary_vec`` plus the
    originating turn id / timestamp. ``question_text`` is intentionally
    NOT retained (§2.2 YAGNI note); callers that want question-based
    retrieval against compressed history will revisit the field list under
    a follow-up issue.

    Invariants:
    * ``summary_vec`` is produced by
      :meth:`PhotonSessionState._make_turn_summary` and has shape
      ``(hidden_size,)`` with dtype ``float32``.
    * ``turn_id`` and ``timestamp`` are inherited from the originating
      :class:`TurnState` when compressed from ``turn_history``, or set from
      the current turn when produced directly by ``summary_only``.
    """

    turn_id: int
    summary_vec: mx.array
    timestamp: float = field(default_factory=time.time)


# Sentinel marking that the caller did not pass ``compress_old_turns`` or
# ``storage_mode`` explicitly (Issue #79 D1 / DR1-004). We cannot use ``None``
# because ``compress_old_turns`` is a strictly typed ``bool`` field, and we
# need to distinguish "user did not provide a value" from "user explicitly set
# True/False" when deciding whether to emit the ``DeprecationWarning``.
class _WMFieldSentinel:
    _instance: _WMFieldSentinel | None = None

    def __new__(cls) -> _WMFieldSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return "<WM_FIELD_UNSET>"


_WM_FIELD_UNSET = _WMFieldSentinel()


@dataclass
class WorkingMemoryConfig:
    """Configuration for cross-turn hierarchical working memory (Issue #64).

    All fields have sensible defaults; ``__post_init__`` enforces strict type
    and range validation (DR4-001) so malformed YAML cannot silently disable
    safety invariants.

    Issue #79 adds ``storage_mode`` (closed enum, see :data:`STORAGE_MODES`).
    The legacy ``compress_old_turns`` flag is kept as a parse-only deprecated
    field — ``storage_mode`` is the authoritative runtime semantics.
    ``DeprecationWarning`` is emitted only when the caller explicitly passes
    both ``compress_old_turns`` and a non-default ``storage_mode`` so existing
    YAMLs that omit ``storage_mode`` stay silent during the transition
    (design §3.1 D1 / DR1-004).
    """

    enabled: bool = True
    max_turns: int = 8
    decay_factor: float = 0.5
    relevant_turn_threshold: float = 0.7
    # ``compress_old_turns`` is DEPRECATED (Issue #79 D1). Kept as a parse-only
    # field so existing YAML configs keep loading. Runtime semantics are
    # driven by ``storage_mode``; the ``_WMFieldSentinel`` default lets
    # ``__post_init__`` tell "user omitted the key" from "user said True/False".
    compress_old_turns: bool | _WMFieldSentinel = field(
        default_factory=lambda: _WM_FIELD_UNSET
    )
    storage_mode: str | _WMFieldSentinel = field(
        default_factory=lambda: _WM_FIELD_UNSET
    )
    # Issue #80: aggregation mode selector for ``get_session_coarse_state()``.
    # ``Literal["weighted", "attention", "last"]`` — default ``weighted`` keeps
    # the pre-#80 behaviour so YAML configs without this key are backward
    # compatible.
    aggregation: str = "weighted"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be bool, got {type(self.enabled).__name__}")

        # Track explicit-vs-default for each deprecation-relevant field so the
        # DeprecationWarning fires only in the ambiguous both-specified case
        # (design §3.1 DR1-004).
        compress_explicit = not isinstance(self.compress_old_turns, _WMFieldSentinel)
        storage_explicit = not isinstance(self.storage_mode, _WMFieldSentinel)

        # Resolve ``compress_old_turns`` to its boolean default if the caller
        # omitted it, then type-validate (keeps the Issue #64 TypeError
        # contract for non-bool inputs like "true"/1).
        if not compress_explicit:
            self.compress_old_turns = True  # type: ignore[assignment]
        if not isinstance(self.compress_old_turns, bool):
            raise TypeError(
                "compress_old_turns must be bool, got "
                f"{type(self.compress_old_turns).__name__}"
            )

        # Resolve ``storage_mode`` similarly; validation below keeps the
        # closed-enum contract (design §3.6 D6).
        if not storage_explicit:
            self.storage_mode = "full"  # type: ignore[assignment]
        # Type check: strict ``str`` (rejects int/None/bool). We do NOT echo
        # the raw value back into the exception text (§4 security: raw values
        # from attacker-controlled YAML must never leak into logs / traces).
        if not isinstance(self.storage_mode, str):
            raise TypeError(
                f"storage_mode must be str, got {type(self.storage_mode).__name__}"
            )
        if self.storage_mode not in STORAGE_MODES:
            # Closed-enum message — intentionally omits the raw value per
            # DR4 security rules (fail-closed on attacker-controlled input).
            raise ValueError(
                "storage_mode must be one of {'full', 'top_level_only', 'summary_only'}"
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
        # Issue #80: aggregation mode — fail-closed on malformed YAML.
        # Error messages intentionally exclude the raw value (log-poisoning /
        # PII mitigation, design §6 and DR4-001): only the literal set and the
        # type name are surfaced.
        if not isinstance(self.aggregation, str):
            raise TypeError(
                f"aggregation must be str, got {type(self.aggregation).__name__}"
            )
        if self.aggregation not in {"weighted", "attention", "last"}:
            raise ValueError(
                "aggregation must be one of {'weighted', 'attention', 'last'}"
            )

        # DR1-004: emit DeprecationWarning only when the caller mixed the
        # deprecated ``compress_old_turns`` with a non-default ``storage_mode``.
        # Existing YAMLs (``compress_old_turns=True``, ``storage_mode`` omitted
        # ⇒ resolved to "full") stay silent so the migration can roll out
        # without noisy logs on every pipeline build.
        if compress_explicit and storage_explicit and self.storage_mode != "full":
            warnings.warn(
                "compress_old_turns is deprecated; storage_mode is the "
                "authoritative setting and compress_old_turns will be removed "
                "in a future release",
                DeprecationWarning,
                stacklevel=2,
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
        # Issue #79: pooled summary of turns that have aged out of
        # ``turn_history`` (``full`` mode overflow) or that are stored only
        # as summaries (``summary_only`` mode). Upper-bounded at
        # ``max_turns * 4`` (design §3.4 D4) with silent ``pop(0)`` once the
        # cap is hit (design §3.5 D5).
        self.compressed_history: list[CompressedTurnState] = []

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

        # Append to working memory only when enabled (Issue #64). Mode-specific
        # retention policy lives in ``_append_turn_for_mode`` so ``update()``
        # keeps a single responsibility (drift + state roll). Issue #79 D1 /
        # DR1-002.
        if self.working_memory_cfg.enabled:
            self._append_turn_for_mode(new_state, question_text)

        return metrics

    def _make_turn_summary(self, hierarchical_state: HierarchicalState) -> mx.array:
        """Produce the pooled ``(hidden_size,)`` summary of a turn.

        Single-source helper (design §2.3 / DR1-006) so all call sites —
        ``_append_full`` → ``_compress_oldest_turn``, ``_append_summary_only``,
        and ``get_session_coarse_state`` — share the same pooling policy.
        A future dtype change (e.g. bf16, OQ-2) localises here.

        Returns ``mx.zeros((0,))`` when ``level_states`` is empty so callers
        can detect "no top-level state to summarise" via ``.shape[0] == 0``
        without raising.
        """
        if not hierarchical_state.level_states:
            return mx.zeros((0,), dtype=mx.float32)
        top = hierarchical_state.level_states[-1]
        return mean_pool(top)

    def _append_turn_for_mode(
        self,
        new_state: HierarchicalState,
        question_text: str | None,
    ) -> None:
        """Dispatch turn retention to the mode-specific helper (Issue #79)."""
        mode = self.working_memory_cfg.storage_mode
        if mode == "full":
            self._append_full(new_state, question_text)
        elif mode == "top_level_only":
            self._append_top_level_only(new_state, question_text)
        elif mode == "summary_only":
            self._append_summary_only(new_state, question_text)
        # ``__post_init__`` guarantees the closed-enum invariant, so no else.

    def _append_full(
        self,
        new_state: HierarchicalState,
        question_text: str | None,
    ) -> None:
        """Retain the full :class:`HierarchicalState`; compress on overflow.

        Preserves the Issue #64 behaviour (append ``TurnState`` + drop oldest
        once ``max_turns`` is hit) but extends the drop path so the oldest
        turn is pooled into ``compressed_history`` rather than garbage-
        collected outright (Issue #79 D1 / D7). This is what lets the coarse
        state keep a faint memory of old turns even after they leave
        ``turn_history``.
        """
        self.turn_history.append(
            TurnState(
                turn_id=self.turn_count,
                hierarchical_state=new_state,
                question_text=_sanitize_question_text(question_text),
            )
        )
        max_turns = self.working_memory_cfg.max_turns
        while len(self.turn_history) > max_turns:
            self._compress_oldest_turn()

    def _append_top_level_only(
        self,
        new_state: HierarchicalState,
        question_text: str | None,
    ) -> None:
        """Retain only ``level_states[-1]`` as a length-1 ``HierarchicalState``.

        Key invariant (design §3.2 DR1-007): the incoming ``new_state``
        MUST NOT be mutated. We build a fresh ``HierarchicalState`` that
        shares the ``mx.array`` reference with ``new_state.level_states[-1]``
        but drops ``level_states[0:-1]`` and ``token_proj``, then wrap it
        in a fresh ``TurnState``. Callers in ``update()`` / drift path keep
        reading the original ``new_state``, so both references must be
        physically distinct instances (tested via
        ``test_top_level_only_does_not_mutate_input_state``).
        """
        top = new_state.level_states[-1] if new_state.level_states else None
        level_states = [top] if top is not None else []
        stored_state = HierarchicalState(
            level_states=level_states,
            token_proj=None,
            turn_id=self.turn_count,
            # Inherit timestamp so cross-mode telemetry is comparable.
            timestamp=new_state.timestamp,
        )
        self.turn_history.append(
            TurnState(
                turn_id=self.turn_count,
                hierarchical_state=stored_state,
                question_text=_sanitize_question_text(question_text),
            )
        )
        max_turns = self.working_memory_cfg.max_turns
        while len(self.turn_history) > max_turns:
            self.turn_history.pop(0)

    def _append_summary_only(
        self,
        new_state: HierarchicalState,
        question_text: str | None,
    ) -> None:
        """Skip ``turn_history`` and push a pooled summary straight to
        ``compressed_history`` (design §3.3 D3).

        ``question_text`` is sanitised (design §4 security: the sanitize
        contract must be mode-invariant — a malicious question must not
        bypass ``_sanitize_question_text`` just because the retention mode
        drops it afterwards) and then discarded; DR1-003 / DR1-005 says
        ``CompressedTurnState`` does not hold ``question_text`` in Issue #79
        scope.

        CB-001 (codex review): ``_make_turn_summary()`` returns
        ``mx.zeros((0,))`` when ``level_states`` is empty (sentinel shape).
        We must NOT persist a zero-length summary because downstream
        ``get_session_coarse_state()`` would then either (a) return a
        ``shape=(0,)`` array (breaking the ``(D,) | None`` API contract) or
        (b) trigger an ``mx.stack`` mismatch when mixed with ``(D,)``
        entries. Skip the save instead — this preserves the
        ``_make_turn_summary`` contract (§2.3) while keeping the storage
        surface clean.
        """
        # Run the sanitize pass for its side-effect contract (rejects
        # control chars, enforces MAX_LEN) even though the result is
        # immediately discarded.
        _ = _sanitize_question_text(question_text)
        summary = self._make_turn_summary(new_state)
        if summary.shape[0] == 0:
            # Empty level_states → no coarse information to retain; skip
            # the save entirely (defense-in-depth, paired with the
            # get_session_coarse_state() filter below).
            return
        self.compressed_history.append(
            CompressedTurnState(
                turn_id=self.turn_count,
                summary_vec=summary,
                timestamp=time.time(),
            )
        )
        self._truncate_compressed_history()

    def _compress_oldest_turn(self) -> None:
        """Pop the oldest ``TurnState`` and append its pooled summary.

        Called from ``_append_full`` when ``turn_history`` exceeds
        ``max_turns``. Preserves ``turn_id`` / ``timestamp`` so
        ``compressed_history`` remains chronologically consistent with the
        emptied ``turn_history`` slot (design §2.3 _compress_oldest_turn
        pseudocode).

        CB-001: the oldest turn is always popped from ``turn_history``
        (unconditional — drop semantics unchanged). But if its
        ``level_states`` is empty, ``_make_turn_summary`` returns a
        zero-length sentinel that must NOT be appended to
        ``compressed_history``; otherwise the API contract on
        ``get_session_coarse_state()`` breaks. Skip the append in that
        case.
        """
        oldest = self.turn_history.pop(0)
        summary = self._make_turn_summary(oldest.hierarchical_state)
        if summary.shape[0] == 0:
            return
        self.compressed_history.append(
            CompressedTurnState(
                turn_id=oldest.turn_id,
                summary_vec=summary,
                timestamp=oldest.timestamp,
            )
        )
        self._truncate_compressed_history()

    def _truncate_compressed_history(self) -> None:
        """Enforce the ``max_turns * 4`` upper bound with silent ``pop(0)``.

        Design §3.4 D4 + §3.5 D5: a fixed coefficient keeps ``decay_factor``
        cumulative weight >=95% and avoids a new config surface for a
        minor behaviour knob.
        """
        cap = self.working_memory_cfg.max_turns * 4
        while len(self.compressed_history) > cap:
            self.compressed_history.pop(0)

    def latest_drift(self) -> DriftMetrics | None:
        return self.drift_history[-1] if self.drift_history else None

    def _collect_turn_coarse_vecs(
        self,
    ) -> tuple[list[mx.array], list[int], list[float]]:
        """Walk ``turn_history`` and return per-turn (vecs, turn_ids, weights).

        Issue #79 D1 / DR1-008 adds a ``storage_mode`` switch so vec collection
        spans ``turn_history`` only, ``compressed_history`` only, or both:

        * ``"full"`` — concatenates compressed (older) + live turns.
        * ``"top_level_only"`` — live turns only (no compressed pool).
        * ``"summary_only"`` — compressed pool only (``turn_history`` empty
          in this mode).

        Issue #80 layers aggregation on top (see :meth:`get_session_coarse_state`)
        and needs ``turn_ids`` for the attention mode's current-turn exclusion.
        Compressed entries get ``turn_id = -1`` sentinel (impossible real
        ``TurnState.turn_id``) so attention's ``exclude_turn_id`` filter is a
        no-op against them.

        ``weights``: geometric decay ``decay_factor ** (N-i-1)`` over the
        combined vec list (so weights stay aligned with the index used for
        scoring).

        Turns whose ``hierarchical_state.level_states`` is empty and
        compressed entries with zero-length ``summary_vec`` are skipped
        (defensive invariant). Returned lists may be empty.
        """
        mode_storage = self.working_memory_cfg.storage_mode
        vecs: list[mx.array] = []
        turn_ids: list[int] = []

        if mode_storage in ("full", "summary_only"):
            for entry in self.compressed_history:
                if entry.summary_vec.shape[0] == 0:
                    continue
                vecs.append(entry.summary_vec)
                turn_ids.append(-1)  # sentinel: not a live turn

        if mode_storage in ("full", "top_level_only"):
            for turn in self.turn_history:
                if not turn.hierarchical_state.level_states:
                    continue
                vecs.append(self._make_turn_summary(turn.hierarchical_state))
                turn_ids.append(turn.turn_id)

        decay = float(self.working_memory_cfg.decay_factor)
        n = len(vecs)
        weights: list[float] = [decay ** (n - i - 1) for i in range(n)]
        return vecs, turn_ids, weights

    def _aggregate_attention(
        self,
        vecs: list[mx.array],
        turn_ids: list[int],
        curr_vec: mx.array,
        exclude_turn_id: int | None,
    ) -> mx.array | None:
        """Compute attention-weighted coarse state vs a query vector.

        Batched MLX implementation (no Python for-loop, no ``.item()``)
        following Issue #80 design §5-2 step 5:

        1. Stack candidate vecs → shape ``(M, D)`` float32.
        2. L2-norm per row / per query vec with ``+1e-8`` epsilon.
        3. Cosine similarity ``scores = (stacked @ curr_vec) / norms``.
        4. ``mx.softmax(scores, axis=-1)`` → ``(M,)`` weights.
        5. Weighted sum along rows → ``(D,)`` output.

        ``exclude_turn_id`` removes a single turn by id before stacking
        (used to drop the current turn during production path — design
        judgement #2). Compressed-history entries carry
        ``turn_id == -1`` so they are always kept regardless of
        ``exclude_turn_id``. If after exclusion ``M == 0``, returns
        ``None`` and lets the dispatcher pick ``fallback_vec`` (DR2-006).

        Pattern matches the existing ``cosine_distance`` (``mx.sqrt(mx.sum
        (x*x))`` + ``+1e-8``) so numerical behaviour is consistent.
        """
        if exclude_turn_id is not None:
            kept_vecs = [v for v, tid in zip(vecs, turn_ids) if tid != exclude_turn_id]
        else:
            kept_vecs = list(vecs)
        if not kept_vecs:
            return None

        stacked = mx.stack(kept_vecs, axis=0).astype(mx.float32)  # (M, D)
        q = curr_vec.astype(mx.float32)
        # (M,) dot products.
        dots = stacked @ q
        # Norms: per-row (M,) and scalar for q.
        k_norms = mx.sqrt(mx.sum(stacked * stacked, axis=-1)) + 1e-8  # (M,)
        q_norm = mx.sqrt(mx.sum(q * q)) + 1e-8  # scalar
        scores = dots / (k_norms * q_norm)  # (M,)
        attn_weights = mx.softmax(scores, axis=-1)  # (M,)
        result = mx.sum(attn_weights[:, None] * stacked, axis=0)  # (D,)
        return result

    def get_session_coarse_state(self) -> mx.array | None:
        """Return a ``(D,)`` coarse session vector aggregated across turns.

        Issue #79 selects **what** is aggregated via
        ``working_memory_cfg.storage_mode`` (compressed, live, or both).
        Issue #80 selects **how** those vectors are combined via
        ``working_memory_cfg.aggregation`` — three modes:

        - ``"weighted"`` (default, backward compat): weighted mean using
          geometric decay ``decay_factor ** (N-i-1)``. When
          ``weight_sum <= 0`` (e.g. ``decay_factor=0`` with history > 1)
          falls back to the most-recent valid vector.
        - ``"last"``: most-recent valid vector (still runs the common
          pre-processing so ``level_states == []`` and zero-length
          ``summary_vec`` skip invariants hold).
        - ``"attention"``: softmax-weighted sum of past-turn vecs using
          cosine similarity to the current turn's ``level_states[-1]``
          mean-pool. The current turn (``current_state.turn_id``) is
          excluded from live candidates to avoid the ``cos == 1``
          self-match that would dominate the softmax (see caller
          contract below). Compressed-history entries carry
          ``turn_id = -1`` sentinel and are therefore always kept.
          When there is no valid current vec or no past candidates,
          falls back to the most-recent valid vector.

        Shared invariants:

        - Returns ``None`` when working memory is disabled, the
          storage-specific source list is empty, or every entry is
          invalid (empty ``level_states`` / zero-length ``summary_vec``).
        - All three modes return a vector with ``shape == (D,)`` and
          ``dtype == mx.float32`` (DR1-002) — callers (notably
          :meth:`PhotonInference._score_prune_candidates`) can assume no
          additional casting is needed.
        - Errors on aggregation value (type / unknown mode) are raised
          without including the raw value in the message (Issue #80 §6,
          log-poisoning mitigation).

        Caller contract (attention / design judgement #2): on the
        production path :meth:`update` is called first, which sets
        ``current_state = new_state`` and also appends the same turn to
        ``turn_history``. The attention branch therefore excludes the
        current turn from past-turn candidates. After
        :meth:`reset_working_memory` the supported path has no live
        state and this method returns ``None`` via the early-return
        guards, so no stale ``fallback_vec`` is ever returned in reset
        recovery.
        """
        if not self.working_memory_cfg.enabled:
            return None

        # Preserve the Issue #79 ``top_level_only`` early-return before
        # collection: ``turn_history == []`` yields ``None`` without
        # consulting ``compressed_history`` (which is always empty in this
        # mode anyway but the guard is kept for explicit intent).
        mode_storage = self.working_memory_cfg.storage_mode
        if mode_storage == "top_level_only" and not self.turn_history:
            return None

        vecs, turn_ids, weights = self._collect_turn_coarse_vecs()
        if not vecs:
            return None

        # DRY: the shared fallback is always the most-recent valid vec
        # (DR1-004). Bound once and reused by last / weighted / attention.
        fallback_vec = vecs[-1]

        mode = self.working_memory_cfg.aggregation
        if mode == "last":
            result = fallback_vec
        elif mode == "weighted":
            weight_sum = sum(weights)
            if weight_sum <= 0.0:
                # All weights were zero (e.g. decay=0 with history > 1).
                result = fallback_vec
            else:
                stacked = mx.stack(vecs, axis=0)  # (N, D)
                w_arr = mx.array(weights, dtype=mx.float32)[:, None]  # (N, 1)
                result = mx.sum(stacked * w_arr, axis=0) / float(weight_sum)
        elif mode == "attention":
            # Need a valid current vec to score past turns against.
            if self.current_state is None or not self.current_state.level_states:
                result = fallback_vec
            else:
                curr_vec = mean_pool(self.current_state.level_states[-1])
                attn = self._aggregate_attention(
                    vecs,
                    turn_ids,
                    curr_vec,
                    exclude_turn_id=self.current_state.turn_id,
                )
                result = attn if attn is not None else fallback_vec
        else:
            # Defensive fail-fast for post-__post_init__ corruption
            # (Issue #80 §8 decision #1 / judgement #1 safety net). The raw
            # value is deliberately omitted from the message.
            raise ValueError("Unknown aggregation mode")

        mx.eval(result)
        return result

    def find_relevant_past_turn(
        self, current_state: HierarchicalState | None
    ) -> TurnState | None:
        """現在の質問に最も関連する過去ターンを検索.

        現在のターンを除く過去ターンそれぞれとコサイン類似度を計算し、
        閾値 ``self.working_memory_cfg.relevant_turn_threshold``
        （デフォルト ``0.7``）以上であれば最も近い ``TurnState`` を返す。
        working memory が無効、``current_state`` が ``None``、または比較対象が
        無い場合は ``None`` を fail-closed で返す（``PhotonSessionState.current_state``
        自体が ``HierarchicalState | None`` なのでこの契約が必要）。

        Contract (design §4):
            * 呼び出し元は ``session.update()`` の後で呼ぶ想定。``turn_history``
              には既に現ターンが append 済みなので、比較ループでは
              ``turn_history[:-1]`` を走査して現ターンを除外する。
            * ``current_state.level_states[-1]`` は単一サンプル前提
              （``(T, D)`` / ``(1, T, D)``）。``B > 1`` の batched state は
              本 API のサポート対象外。
            * 非有限類似度（NaN / Inf）のターンはランキング対象から除外し、
              全滅時は ``None`` を返す（fail-closed）。
            * 同点類似度は最新 ``turn_id`` 優先。
            * 戻り値 ``TurnState`` は ``turn_history`` 内部参照の借用であり、
              consumer 側が mutate すると session state を破壊するため
              read-only として扱うこと（設計 §10）。

        Refs: Issue #78, design policy §4.4 / §4.5.
        """
        if not self.working_memory_cfg.enabled:  # step 1
            return None
        if len(self.turn_history) <= 1:  # step 2
            return None
        if current_state is None or not current_state.level_states:  # step 3
            return None

        curr_top = current_state.level_states[-1]

        scores: list[tuple[TurnState, float]] = []
        for past_turn in self.turn_history[:-1]:  # step 4
            past_levels = past_turn.hierarchical_state.level_states
            if not past_levels:
                continue
            sim = 1.0 - cosine_distance(curr_top, past_levels[-1])
            if not math.isfinite(sim):
                continue
            scores.append((past_turn, sim))

        if not scores:  # step 5
            return None

        scores.sort(key=lambda x: (x[1], x[0].turn_id), reverse=True)  # step 6-1
        best_turn, best_sim = scores[0]
        if best_sim >= self.working_memory_cfg.relevant_turn_threshold:  # step 6-2
            return best_turn
        return None

    def reset_working_memory(self) -> None:
        """Clear stale latents and turn history for fail-closed recovery.

        Drops ``current_state``, ``prev_state``, ``prev_logits`` plus both
        ``turn_history`` and ``compressed_history`` (Issue #79 DR1-010). All
        five fields are cleared atomically so Safe RecGen fallback leaves a
        fully blank working memory regardless of storage mode.
        ``drift_history`` and ``turn_count`` are intentionally preserved so
        observability APIs remain consistent (Issue #64 judgement #4 /
        DR3-001).

        NOTE: if a future change pushes the clear list past ~6 fields, move
        to a ``_VOLATILE_STATE_FIELDS`` tuple + ``setattr`` loop (DR1-010).
        """
        self.current_state = None
        self.prev_state = None
        self.prev_logits = None
        self.turn_history = []
        self.compressed_history = []
