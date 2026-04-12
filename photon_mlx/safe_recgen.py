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

    # Thresholds
    latent_cosine_drift_threshold: float = 0.18
    topic_shift_score_threshold: float = 0.65
    confidence_floor: float = 0.40
    logit_kl_threshold: float = 0.75

    # High-risk keywords
    high_risk_keywords: list[str] = field(default_factory=lambda: [
        "auth", "authorization", "permission",
        "billing", "payment",
        "security", "encryption",
        "delete", "drop table", "migration",
        "compliance",
    ])


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
                if drift.latent_cosine_drift > cfg.latent_cosine_drift_threshold:
                    reasons.append(FallbackReason.LATENT_DRIFT)
                    details["latent_cosine_drift"] = drift.latent_cosine_drift

            if cfg.trigger_topic_shift:
                if drift.topic_shift_score > cfg.topic_shift_score_threshold:
                    reasons.append(FallbackReason.TOPIC_SHIFT)
                    details["topic_shift_score"] = drift.topic_shift_score

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
