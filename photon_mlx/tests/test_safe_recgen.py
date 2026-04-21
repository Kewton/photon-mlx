"""Tests for Safe RecGen controller."""

from __future__ import annotations

import pytest

from photon_mlx.safe_recgen import (
    FallbackReason,
    SafeRecGenConfig,
    SafeRecGenController,
    is_diff_or_patch_request,
    is_exact_quote_request,
    is_high_risk_query,
)
from photon_mlx.session import DriftMetrics


# ---------------------------------------------------------------
# Query classifier tests
# ---------------------------------------------------------------


class TestQueryClassifiers:
    def test_exact_quote_english(self) -> None:
        assert is_exact_quote_request("Show the exact quote from security.py")
        assert is_exact_quote_request("give me the verbatim code")

    def test_exact_quote_japanese(self) -> None:
        assert is_exact_quote_request("security.pyのコードをそのまま出して")
        assert is_exact_quote_request("原文を示してください")

    def test_not_exact_quote(self) -> None:
        assert not is_exact_quote_request("Explain the authentication flow")

    def test_diff_or_patch(self) -> None:
        assert is_diff_or_patch_request("Show me the diff")
        assert is_diff_or_patch_request("Generate a patch for this fix")
        assert is_diff_or_patch_request("変更差分を見せて")

    def test_not_diff(self) -> None:
        assert not is_diff_or_patch_request("What is different about this?")

    def test_high_risk(self) -> None:
        kw = ["auth", "billing", "security", "delete"]
        assert is_high_risk_query("How does auth work?", kw)
        assert is_high_risk_query("billing APIを変更したい", kw)
        assert is_high_risk_query("DELETE endpoint security", kw)

    def test_not_high_risk(self) -> None:
        kw = ["auth", "billing", "security"]
        assert not is_high_risk_query("How does routing work?", kw)


# ---------------------------------------------------------------
# Controller tests — rule-based triggers
# ---------------------------------------------------------------


class TestRuleBasedTriggers:
    def setup_method(self) -> None:
        self.ctrl = SafeRecGenController()

    def test_exact_quote_fires(self) -> None:
        d = self.ctrl.evaluate("Show the exact quote from main.py")
        assert d.should_fallback
        assert FallbackReason.EXACT_QUOTE in d.reasons

    def test_diff_fires(self) -> None:
        d = self.ctrl.evaluate("Generate a patch")
        assert d.should_fallback
        assert FallbackReason.DIFF_OR_PATCH in d.reasons

    def test_high_risk_fires(self) -> None:
        d = self.ctrl.evaluate("How does the auth middleware work?")
        assert d.should_fallback
        assert FallbackReason.HIGH_RISK_QUERY in d.reasons
        assert "fallback_to_baseline_path" in d.actions

    def test_benign_query_no_fallback(self) -> None:
        d = self.ctrl.evaluate("What does the main module do?")
        assert not d.should_fallback
        assert d.reasons == []


# ---------------------------------------------------------------
# Controller tests — metric-based triggers
# ---------------------------------------------------------------


class TestMetricTriggers:
    def setup_method(self) -> None:
        self.ctrl = SafeRecGenController()

    def test_latent_drift_fires(self) -> None:
        drift = DriftMetrics(turn_id=3, latent_cosine_drift_top=0.25)
        d = self.ctrl.evaluate("What is this?", drift)
        assert d.should_fallback
        assert FallbackReason.LATENT_DRIFT in d.reasons
        assert "reprefill_hierarchy" in d.actions

    def test_topic_shift_fires(self) -> None:
        drift = DriftMetrics(turn_id=3, topic_shift_score=0.70)
        d = self.ctrl.evaluate("What is this?", drift)
        assert d.should_fallback
        assert FallbackReason.TOPIC_SHIFT in d.reasons

    def test_logit_kl_fires(self) -> None:
        drift = DriftMetrics(turn_id=3, logit_kl=0.80)
        d = self.ctrl.evaluate("What is this?", drift)
        assert d.should_fallback
        assert FallbackReason.LOGIT_KL in d.reasons

    def test_low_confidence_fires(self) -> None:
        d = self.ctrl.evaluate("What is this?", confidence=0.30)
        assert d.should_fallback
        assert FallbackReason.LOW_CONFIDENCE in d.reasons
        assert "fallback_to_baseline_path" in d.actions

    def test_below_thresholds_no_fallback(self) -> None:
        drift = DriftMetrics(
            turn_id=3,
            latent_cosine_drift_top=0.10,
            topic_shift_score=0.30,
            logit_kl=0.40,
        )
        d = self.ctrl.evaluate("What is this?", drift, confidence=0.80)
        assert not d.should_fallback

    def test_multiple_triggers_combine(self) -> None:
        drift = DriftMetrics(
            turn_id=5,
            latent_cosine_drift_top=0.25,
            topic_shift_score=0.70,
            logit_kl=0.90,
        )
        d = self.ctrl.evaluate(
            "Show the exact quote of the auth handler",
            drift,
            confidence=0.35,
        )
        assert d.should_fallback
        assert (
            len(d.reasons) >= 4
        )  # exact_quote + high_risk + drift + shift + kl + low_conf


# ---------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------


class TestConfig:
    def test_disabled(self) -> None:
        cfg = SafeRecGenConfig(enabled=False)
        ctrl = SafeRecGenController(cfg)
        d = ctrl.evaluate("Show exact quote of auth code")
        assert not d.should_fallback

    def test_custom_thresholds(self) -> None:
        # Issue #63 / CB-001: ``evaluate()`` consults per-level thresholds.
        # The legacy ``latent_cosine_drift_threshold`` field is kept for
        # backward-compat callers but no longer drives the decision, so the
        # new threshold must be set on ``latent_cosine_drift_top_threshold``.
        cfg = SafeRecGenConfig(latent_cosine_drift_top_threshold=0.50)
        ctrl = SafeRecGenController(cfg)
        drift = DriftMetrics(turn_id=2, latent_cosine_drift_top=0.30)
        d = ctrl.evaluate("What is this?", drift)
        assert not d.should_fallback  # 0.30 < 0.50

    def test_as_dict(self) -> None:
        ctrl = SafeRecGenController()
        d = ctrl.evaluate("Show exact quote")
        result = d.as_dict()
        assert "should_fallback" in result
        assert "reasons" in result
        assert isinstance(result["reasons"], list)


# ---------------------------------------------------------------
# Issue #63: per-level thresholds, weights, subtype labelling
# ---------------------------------------------------------------


class TestIssue63Config:
    def test_safe_recgen_threshold_alias_top(self) -> None:
        """latent_cosine_drift_top_threshold defaults equal to the legacy
        latent_cosine_drift_threshold (0.18)."""
        cfg = SafeRecGenConfig()
        assert cfg.latent_cosine_drift_top_threshold == 0.18
        assert cfg.latent_cosine_drift_threshold == 0.18

    def test_per_level_default_thresholds(self) -> None:
        cfg = SafeRecGenConfig()
        assert cfg.latent_cosine_drift_mid_threshold == 0.40
        assert cfg.latent_cosine_drift_token_threshold == 0.30

    def test_drift_level_weights_default(self) -> None:
        cfg = SafeRecGenConfig()
        assert cfg.drift_level_weights == (0.2, 0.3, 0.5)

    def test_drift_level_weights_list_normalised_to_tuple(self) -> None:
        """list input is normalised to tuple (DR1-001)."""
        cfg = SafeRecGenConfig(drift_level_weights=[0.3, 0.3, 0.4])
        assert cfg.drift_level_weights == (0.3, 0.3, 0.4)
        assert isinstance(cfg.drift_level_weights, tuple)

    def test_drift_level_weights_fail_closed_length(self) -> None:
        """Length != 3 fails closed (DR4-001)."""
        with pytest.raises(ValueError):
            SafeRecGenConfig(drift_level_weights=(0.5, 0.5))
        with pytest.raises(ValueError):
            SafeRecGenConfig(drift_level_weights=(0.25, 0.25, 0.25, 0.25))

    def test_drift_level_weights_fail_closed_negative(self) -> None:
        """Negative entry fails closed (DR4-001)."""
        with pytest.raises(ValueError):
            SafeRecGenConfig(drift_level_weights=(-0.1, 0.3, 0.8))

    def test_drift_level_weights_fail_closed_nan(self) -> None:
        """NaN entry fails closed (DR4-001)."""
        with pytest.raises(ValueError):
            SafeRecGenConfig(drift_level_weights=(float("nan"), 0.3, 0.7))

    def test_drift_level_weights_fail_closed_inf(self) -> None:
        """inf entry fails closed (DR4-001)."""
        with pytest.raises(ValueError):
            SafeRecGenConfig(drift_level_weights=(float("inf"), 0.3, 0.7))


class TestIssue63PerLevelDriftThresholds:
    """CB-001 fix: Safe RecGen must honour per-level drift thresholds.

    Previously ``evaluate()`` only checked the single (legacy) threshold
    against ``latent_cosine_drift`` (= top alias), so configured mid/token
    thresholds were silently ignored. After the fix, ANY level exceeding its
    threshold fires ``FallbackReason.LATENT_DRIFT`` exactly once, and
    ``details['latent_drift_triggered_levels']`` enumerates which levels
    fired (subset of ``['top', 'mid', 'token']``).
    """

    def setup_method(self) -> None:
        self.ctrl = SafeRecGenController()

    def test_evaluate_latent_drift_top_threshold(self) -> None:
        """Only top exceeds → fires, triggered_levels == ['top']."""
        drift = DriftMetrics(
            turn_id=3,
            latent_cosine_drift_top=0.25,  # > 0.18
            latent_cosine_drift_mid=0.10,  # < 0.40
            latent_cosine_drift_token=0.10,  # < 0.30
        )
        d = self.ctrl.evaluate("What is this?", drift)
        assert d.should_fallback
        # Exactly one LATENT_DRIFT reason even though we added a list
        assert d.reasons.count(FallbackReason.LATENT_DRIFT) == 1
        assert d.details.get("latent_drift_triggered_levels") == ["top"]
        # Backward-compat key is still present (top value).
        assert d.details.get("latent_cosine_drift") == pytest.approx(0.25)
        assert d.details.get("latent_cosine_drift_top") == pytest.approx(0.25)

    def test_evaluate_latent_drift_mid_threshold(self) -> None:
        """Only mid exceeds → fires with ['mid']."""
        drift = DriftMetrics(
            turn_id=3,
            latent_cosine_drift_top=0.10,  # < 0.18
            latent_cosine_drift_mid=0.45,  # > 0.40
            latent_cosine_drift_token=0.10,  # < 0.30
        )
        d = self.ctrl.evaluate("What is this?", drift)
        assert d.should_fallback
        assert d.reasons.count(FallbackReason.LATENT_DRIFT) == 1
        assert d.details.get("latent_drift_triggered_levels") == ["mid"]
        assert d.details.get("latent_cosine_drift_mid") == pytest.approx(0.45)

    def test_evaluate_latent_drift_token_threshold(self) -> None:
        """Only token exceeds → fires with ['token']."""
        drift = DriftMetrics(
            turn_id=3,
            latent_cosine_drift_top=0.10,  # < 0.18
            latent_cosine_drift_mid=0.10,  # < 0.40
            latent_cosine_drift_token=0.35,  # > 0.30
        )
        d = self.ctrl.evaluate("What is this?", drift)
        assert d.should_fallback
        assert d.reasons.count(FallbackReason.LATENT_DRIFT) == 1
        assert d.details.get("latent_drift_triggered_levels") == ["token"]
        assert d.details.get("latent_cosine_drift_token") == pytest.approx(0.35)

    def test_evaluate_latent_drift_multi_level(self) -> None:
        """top + mid exceed → single LATENT_DRIFT reason with both level names."""
        drift = DriftMetrics(
            turn_id=3,
            latent_cosine_drift_top=0.25,  # > 0.18
            latent_cosine_drift_mid=0.45,  # > 0.40
            latent_cosine_drift_token=0.10,  # < 0.30
        )
        d = self.ctrl.evaluate("What is this?", drift)
        assert d.should_fallback
        assert d.reasons.count(FallbackReason.LATENT_DRIFT) == 1
        triggered = d.details.get("latent_drift_triggered_levels")
        assert triggered is not None
        assert set(triggered) == {"top", "mid"}

    def test_evaluate_latent_drift_none_triggered(self) -> None:
        """No level exceeds → no LATENT_DRIFT reason."""
        drift = DriftMetrics(
            turn_id=3,
            latent_cosine_drift_top=0.10,  # < 0.18
            latent_cosine_drift_mid=0.20,  # < 0.40
            latent_cosine_drift_token=0.10,  # < 0.30
        )
        d = self.ctrl.evaluate("What is this?", drift)
        assert FallbackReason.LATENT_DRIFT not in d.reasons
        assert "latent_drift_triggered_levels" not in d.details


class TestIssue63DriftWeightsExtraValidation:
    """CB-002 fix: drift_level_weights must also be in [0, 1] and sum to 1.0.

    The original ``_validate_drift_weights`` only checked length / finite /
    non-negative. That permitted fail-open configurations like
    ``(10.0, 0.0, 0.0)`` or ``(0.1, 0.2, 0.3)``; since hierarchical scoring
    assumes weights form a weighted average, such configs silently break
    scoring. After the fix, each weight must satisfy ``0.0 <= w <= 1.0`` and
    ``|sum - 1.0| <= 1e-6``.
    """

    def test_drift_level_weights_upper_bound(self) -> None:
        """Weights > 1.0 fail closed."""
        with pytest.raises(ValueError):
            SafeRecGenConfig(drift_level_weights=(2.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            SafeRecGenConfig(drift_level_weights=(1.5, 0.0, 0.0))

    def test_drift_level_weights_sum_not_one_low(self) -> None:
        """Weights summing to < 1.0 fail closed."""
        with pytest.raises(ValueError):
            SafeRecGenConfig(drift_level_weights=(0.1, 0.2, 0.3))  # sum=0.6

    def test_drift_level_weights_sum_not_one_high(self) -> None:
        """Weights summing to > 1.0 fail closed."""
        with pytest.raises(ValueError):
            SafeRecGenConfig(drift_level_weights=(0.5, 0.3, 0.5))  # sum=1.3

    def test_drift_level_weights_exactly_one(self) -> None:
        """Boundary case: weights in [0,1] summing to exactly 1.0 must pass."""
        cfg = SafeRecGenConfig(drift_level_weights=(0.2, 0.3, 0.5))
        assert cfg.drift_level_weights == (0.2, 0.3, 0.5)

    def test_drift_level_weights_within_eps(self) -> None:
        """Weights summing to 1.0 ± 1e-7 must still pass (floating slack)."""
        cfg = SafeRecGenConfig(drift_level_weights=(0.333333, 0.333333, 0.333334))
        # Sum is ~1.0, so no exception is raised.
        assert len(cfg.drift_level_weights) == 3


class TestIssue63LegacyThresholdAlias:
    """CB-004 fix: direct constructor callers that pass only the legacy
    ``latent_cosine_drift_threshold`` must see it reflected in the new
    ``latent_cosine_drift_top_threshold`` too. The YAML loader already
    performs this mapping (baseline_reporag.photon_pipeline); the fix
    closes the footgun for callers who skip the loader.
    """

    def test_legacy_threshold_direct_constructor_alias(self) -> None:
        """Legacy-only construction mirrors the value to the new field."""
        cfg = SafeRecGenConfig(latent_cosine_drift_threshold=0.25)
        assert cfg.latent_cosine_drift_threshold == 0.25
        assert cfg.latent_cosine_drift_top_threshold == 0.25

    def test_explicit_top_threshold_wins(self) -> None:
        """Explicit new-field value overrides the legacy alias copy."""
        cfg = SafeRecGenConfig(
            latent_cosine_drift_threshold=0.25,
            latent_cosine_drift_top_threshold=0.30,
        )
        assert cfg.latent_cosine_drift_threshold == 0.25
        # User-supplied explicit value must not be clobbered by the alias.
        assert cfg.latent_cosine_drift_top_threshold == 0.30

    def test_default_construction_unchanged(self) -> None:
        """Zero-argument construction keeps both fields at 0.18."""
        cfg = SafeRecGenConfig()
        assert cfg.latent_cosine_drift_threshold == 0.18
        assert cfg.latent_cosine_drift_top_threshold == 0.18

    def test_legacy_alias_reflected_in_evaluate(self) -> None:
        """End-to-end: legacy-only construction actually drives the
        controller's per-level check (CB-001 × CB-004)."""
        cfg = SafeRecGenConfig(latent_cosine_drift_threshold=0.50)
        ctrl = SafeRecGenController(cfg)
        drift = DriftMetrics(turn_id=2, latent_cosine_drift_top=0.30)
        d = ctrl.evaluate("What is this?", drift)
        # 0.30 < 0.50 → must not fire (pre-CB-004 this fired because the
        # top_threshold stayed at 0.18 and the legacy value was ignored).
        assert not d.should_fallback


class TestIssue63TopicShiftSubtype:
    def setup_method(self) -> None:
        self.ctrl = SafeRecGenController()

    def test_topic_shift_subtype_expression(self) -> None:
        """drift_top < 0.3 and drift_token > 0.5 → expression_shift."""
        drift = DriftMetrics(
            turn_id=3,
            topic_shift_score=0.70,
            latent_cosine_drift_top=0.2,
            latent_cosine_drift_token=0.6,
        )
        d = self.ctrl.evaluate("What is this?", drift)
        assert FallbackReason.TOPIC_SHIFT in d.reasons
        assert d.details.get("topic_shift_subtype") == "expression_shift"

    def test_topic_shift_subtype_large_topic(self) -> None:
        """drift_top > 0.5 → large_topic_shift."""
        drift = DriftMetrics(
            turn_id=3,
            topic_shift_score=0.70,
            latent_cosine_drift_top=0.6,
            latent_cosine_drift_token=0.2,
        )
        d = self.ctrl.evaluate("What is this?", drift)
        assert FallbackReason.TOPIC_SHIFT in d.reasons
        assert d.details.get("topic_shift_subtype") == "large_topic_shift"

    def test_topic_shift_subtype_absent_when_no_topic_shift(self) -> None:
        """If topic_shift_score <= threshold, no topic_shift_subtype key."""
        drift = DriftMetrics(
            turn_id=3,
            topic_shift_score=0.50,
            latent_cosine_drift_top=0.6,
            latent_cosine_drift_token=0.6,
        )
        d = self.ctrl.evaluate("What is this?", drift)
        assert "topic_shift_subtype" not in d.details

    def test_topic_shift_subtype_absent_when_no_classification_match(self) -> None:
        """TOPIC_SHIFT fires but neither subtype rule matches → key not set
        (DR1-011: None is never stored explicitly)."""
        drift = DriftMetrics(
            turn_id=3,
            topic_shift_score=0.70,
            latent_cosine_drift_top=0.35,  # between 0.3 and 0.5
            latent_cosine_drift_token=0.2,  # below 0.5
        )
        d = self.ctrl.evaluate("What is this?", drift)
        assert FallbackReason.TOPIC_SHIFT in d.reasons
        assert "topic_shift_subtype" not in d.details
