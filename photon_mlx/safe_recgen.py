"""
Safe RecGen — drift-aware fallback controller for PHOTON-RAG.

v1: fixed thresholds (spec §12, workspace/memo.md).
v2: learned calibrator (future).

Trigger conditions:
  - exact_quote request
  - diff/patch request
  - high-risk query (auth/billing/security/delete)
  - latent_cosine_drift > 0.18
  - logit_kl > 0.75
  - topic_shift_score > 0.65
  - confidence < 0.40

Fallback actions:
  - re-retrieve
  - strengthen local refresh
  - re-prefill hierarchy
  - fallback to baseline path
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto

from .session import DriftMetrics


# ================================================================
# Configuration
# ================================================================


@dataclass
class SafeRecGenConfig:
    enabled: bool = True

    # Rule-based triggers (always fire)
    trigger_exact_quote: bool = True
    trigger_diff_or_patch: bool = True
    trigger_high_risk_query: bool = True

    # Metric-based triggers (v1: fixed thresholds)
    trigger_topic_shift: bool = True
    trigger_latent_drift: bool = True
    trigger_low_confidence: bool = True

    # Thresholds (legacy; kept for backward-compat, same default as the new
    # ``latent_cosine_drift_top_threshold``)
    latent_cosine_drift_threshold: float = 0.18
    topic_shift_score_threshold: float = 0.65
    confidence_floor: float = 0.40
    logit_kl_threshold: float = 0.75

    # Issue #63: per-level drift thresholds. Defaults:
    # * top   ≈ existing single threshold (0.18)
    # * mid   = 0.40 (T/4 latents are noisier per step)
    # * token = 0.30 (token_proj is most noisy)
    latent_cosine_drift_top_threshold: float = 0.18
    latent_cosine_drift_mid_threshold: float = 0.40
    latent_cosine_drift_token_threshold: float = 0.30

    # Issue #63: weights used both by PhotonSessionState.topic_shift_score and
    # by PhotonInference hierarchical scoring. Input may be a list (YAML) but
    # is normalised to a tuple in __post_init__ (DR1-001).
    drift_level_weights: tuple[float, ...] | list[float] = field(
        default_factory=lambda: (0.2, 0.3, 0.5)
    )

    # High-risk keywords
    high_risk_keywords: list[str] = field(
        default_factory=lambda: [
            "auth",
            "authorization",
            "permission",
            "billing",
            "payment",
            "security",
            "encryption",
            "delete",
            "drop table",
            "migration",
            "compliance",
        ]
    )

    def __post_init__(self) -> None:
        """Normalise and validate ``drift_level_weights`` on construction.

        Issue #63 / DR1-007 splits this into two small steps so the
        responsibilities are easy to reason about:

        * :meth:`_normalize_drift_weights` — type invariant (list → tuple of
          ``float``).
        * :meth:`_validate_drift_weights`  — value bounds (length == 3,
          non-negative, finite).

        YAML-layer alias resolution (``thresholds.latent_cosine_drift`` →
        ``latent_cosine_drift_top_threshold``) is the loader's
        responsibility (DR1-010). However, direct constructor callers may
        still pass only the legacy ``latent_cosine_drift_threshold``; in
        that case :meth:`_alias_legacy_top_threshold` mirrors it to the
        new ``latent_cosine_drift_top_threshold`` so the controller honours
        the user's intent (Issue #63 / CB-004).
        """
        self._normalize_drift_weights()
        self._validate_drift_weights()
        self._alias_legacy_top_threshold()

    def _alias_legacy_top_threshold(self) -> None:
        """Mirror legacy ``latent_cosine_drift_threshold`` onto the new
        ``latent_cosine_drift_top_threshold`` when the caller set only the
        legacy field (Issue #63 / CB-004).

        Both fields share the same default (0.18), so we use the class-level
        attribute as a sentinel: if the instance value differs from the class
        default for the legacy field AND the new field is still at its class
        default, the caller clearly meant the legacy value to apply. When
        the caller sets the new field explicitly, the user-supplied value
        wins unconditionally.
        """
        cls = type(self)
        legacy = self.latent_cosine_drift_threshold
        top = self.latent_cosine_drift_top_threshold
        if (
            legacy != cls.latent_cosine_drift_threshold
            and top == cls.latent_cosine_drift_top_threshold
        ):
            self.latent_cosine_drift_top_threshold = legacy

    def _normalize_drift_weights(self) -> None:
        """Coerce ``drift_level_weights`` to ``tuple[float, ...]``.

        YAML deserialisation yields a ``list`` but internal code expects a
        ``tuple`` for immutability and hashability (DR1-001).
        """
        self.drift_level_weights = tuple(float(w) for w in self.drift_level_weights)

    def _validate_drift_weights(self) -> None:
        """Fail-closed validation (Issue #63 / DR4-001 + CB-002).

        Rejects invalid weights by raising ``ValueError``. Invalid
        configuration must never silently pass (security / correctness
        boundary). CB-002 tightens the rules beyond the initial DR4-001
        set: hierarchical scoring treats ``drift_level_weights`` as a
        weighted-average, so entries must be in ``[0.0, 1.0]`` and the
        sum must equal ``1.0`` (to ``1e-6``).

        Rules:

        * ``len(weights) == 3`` (token, mid, top).
        * Each ``w`` is finite (no NaN / inf).
        * ``0.0 <= w <= 1.0``.
        * ``abs(sum(weights) - 1.0) <= 1e-6``.
        """
        weights = self.drift_level_weights
        if len(weights) != 3:
            raise ValueError(
                "drift_level_weights must have length 3 (token, mid, top); "
                f"got length {len(weights)}: {weights}"
            )
        for w in weights:
            if not math.isfinite(w):
                raise ValueError(
                    f"drift_level_weights must be finite (no NaN / inf); got {weights}"
                )
            if w < 0.0 or w > 1.0:
                raise ValueError(
                    "drift_level_weights entries must be within [0.0, 1.0]; "
                    f"got {weights}"
                )
        total = math.fsum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "drift_level_weights must sum to 1.0 (± 1e-6); "
                f"got sum={total:.6f} for {weights}"
            )


class FallbackReason(Enum):
    NONE = auto()
    EXACT_QUOTE = auto()
    DIFF_OR_PATCH = auto()
    HIGH_RISK_QUERY = auto()
    LATENT_DRIFT = auto()
    TOPIC_SHIFT = auto()
    LOW_CONFIDENCE = auto()
    LOGIT_KL = auto()


@dataclass
class FallbackDecision:
    should_fallback: bool
    reasons: list[FallbackReason]
    actions: list[str]
    details: dict = field(default_factory=dict)

    def primary_reason(self) -> FallbackReason:
        return self.reasons[0] if self.reasons else FallbackReason.NONE

    def as_dict(self) -> dict:
        return {
            "should_fallback": self.should_fallback,
            "reasons": [r.name for r in self.reasons],
            "actions": self.actions,
            "details": self.details,
        }


# ================================================================
# Query classifiers
# ================================================================

_EXACT_QUOTE_PATTERNS = [
    r"exact\s*quote",
    r"そのまま(引用|出して|見せて|示して)",
    r"verbatim",
    r"原文",
    r"コード(を|の)(そのまま|正確に)",
]

_DIFF_PATCH_PATTERNS = [
    r"\bdiff\b",
    r"\bpatch\b",
    r"変更差分",
    r"修正案",
    r"コード修正",
]


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def is_exact_quote_request(question: str) -> bool:
    return _matches_any_pattern(question, _EXACT_QUOTE_PATTERNS)


def is_diff_or_patch_request(question: str) -> bool:
    return _matches_any_pattern(question, _DIFF_PATCH_PATTERNS)


def is_high_risk_query(question: str, keywords: list[str]) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in keywords)


# ================================================================
# Controller
# ================================================================


class SafeRecGenController:
    """
    Evaluate whether a fallback is needed and decide which actions to take.

    Usage:
        controller = SafeRecGenController(config)
        decision = controller.evaluate(question, drift_metrics, confidence)
        if decision.should_fallback:
            # execute decision.actions
    """

    def __init__(self, config: SafeRecGenConfig | None = None) -> None:
        self.config = config or SafeRecGenConfig()

    def evaluate(
        self,
        question: str,
        drift: DriftMetrics | None = None,
        confidence: float = 1.0,
    ) -> FallbackDecision:
        if not self.config.enabled:
            return FallbackDecision(False, [], [])

        cfg = self.config
        reasons: list[FallbackReason] = []
        details: dict = {}

        # --- Rule-based triggers ---

        if cfg.trigger_exact_quote and is_exact_quote_request(question):
            reasons.append(FallbackReason.EXACT_QUOTE)

        if cfg.trigger_diff_or_patch and is_diff_or_patch_request(question):
            reasons.append(FallbackReason.DIFF_OR_PATCH)

        if cfg.trigger_high_risk_query and is_high_risk_query(
            question, cfg.high_risk_keywords
        ):
            reasons.append(FallbackReason.HIGH_RISK_QUERY)

        # --- Metric-based triggers ---

        if drift is not None:
            if cfg.trigger_latent_drift:
                # Issue #63 / CB-001: evaluate each level against its own
                # threshold. If ANY level exceeds its threshold, append
                # ``LATENT_DRIFT`` exactly once and record the triggered
                # levels in ``details['latent_drift_triggered_levels']``.
                # When the per-level thresholds all equal the legacy
                # ``latent_cosine_drift_threshold`` (top default 0.18), the
                # behaviour on top-only drift is identical to the pre-CB-001
                # implementation.
                triggered_levels: list[str] = []
                if (
                    drift.latent_cosine_drift_top
                    > cfg.latent_cosine_drift_top_threshold
                ):
                    triggered_levels.append("top")
                if (
                    drift.latent_cosine_drift_mid
                    > cfg.latent_cosine_drift_mid_threshold
                ):
                    triggered_levels.append("mid")
                if (
                    drift.latent_cosine_drift_token
                    > cfg.latent_cosine_drift_token_threshold
                ):
                    triggered_levels.append("token")

                if triggered_levels:
                    reasons.append(FallbackReason.LATENT_DRIFT)
                    # Backward-compat: preserve the top value under the
                    # legacy key so existing log consumers keep working.
                    details["latent_cosine_drift"] = drift.latent_cosine_drift_top
                    details["latent_cosine_drift_top"] = drift.latent_cosine_drift_top
                    details["latent_cosine_drift_mid"] = drift.latent_cosine_drift_mid
                    details["latent_cosine_drift_token"] = (
                        drift.latent_cosine_drift_token
                    )
                    details["latent_drift_triggered_levels"] = triggered_levels

            if cfg.trigger_topic_shift:
                if drift.topic_shift_score > cfg.topic_shift_score_threshold:
                    reasons.append(FallbackReason.TOPIC_SHIFT)
                    details["topic_shift_score"] = drift.topic_shift_score
                    # Issue #63: optional subtype label (closed set, DR1-011).
                    # Key is only set when the classification condition
                    # matches; absence is semantically "no subtype".
                    if (
                        drift.latent_cosine_drift_top < 0.3
                        and drift.latent_cosine_drift_token > 0.5
                    ):
                        details["topic_shift_subtype"] = "expression_shift"
                    elif drift.latent_cosine_drift_top > 0.5:
                        details["topic_shift_subtype"] = "large_topic_shift"

            if drift.logit_kl > cfg.logit_kl_threshold:
                reasons.append(FallbackReason.LOGIT_KL)
                details["logit_kl"] = drift.logit_kl

        if cfg.trigger_low_confidence and confidence < cfg.confidence_floor:
            reasons.append(FallbackReason.LOW_CONFIDENCE)
            details["confidence"] = confidence

        # --- Decide actions ---

        if not reasons:
            return FallbackDecision(False, [], [])

        actions = self._select_actions(reasons)

        return FallbackDecision(
            should_fallback=True,
            reasons=reasons,
            actions=actions,
            details=details,
        )

    def _select_actions(self, reasons: list[FallbackReason]) -> list[str]:
        """Map reasons to concrete fallback actions."""
        actions: list[str] = []
        reason_set = set(reasons)

        # Always strengthen local refresh on fallback
        actions.append("strengthen_local_refresh")

        # Re-retrieve for drift / topic shift / high risk
        if reason_set & {
            FallbackReason.LATENT_DRIFT,
            FallbackReason.TOPIC_SHIFT,
            FallbackReason.HIGH_RISK_QUERY,
            FallbackReason.LOGIT_KL,
        }:
            actions.append("re_retrieve")

        # Re-prefill hierarchy on large drift
        if reason_set & {
            FallbackReason.LATENT_DRIFT,
            FallbackReason.TOPIC_SHIFT,
        }:
            actions.append("reprefill_hierarchy")

        # Exact quote / patch: always re-read source
        if reason_set & {
            FallbackReason.EXACT_QUOTE,
            FallbackReason.DIFF_OR_PATCH,
        }:
            actions.append("re_retrieve")

        # Full baseline fallback for critical cases
        if reason_set & {
            FallbackReason.HIGH_RISK_QUERY,
            FallbackReason.LOW_CONFIDENCE,
        }:
            actions.append("fallback_to_baseline_path")

        return sorted(set(actions))
