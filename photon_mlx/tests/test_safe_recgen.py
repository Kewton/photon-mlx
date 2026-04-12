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
        drift = DriftMetrics(turn_id=3, latent_cosine_drift=0.25)
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
            latent_cosine_drift=0.10,
            topic_shift_score=0.30,
            logit_kl=0.40,
        )
        d = self.ctrl.evaluate("What is this?", drift, confidence=0.80)
        assert not d.should_fallback

    def test_multiple_triggers_combine(self) -> None:
        drift = DriftMetrics(
            turn_id=5,
            latent_cosine_drift=0.25,
            topic_shift_score=0.70,
            logit_kl=0.90,
        )
        d = self.ctrl.evaluate(
            "Show the exact quote of the auth handler",
            drift,
            confidence=0.35,
        )
        assert d.should_fallback
        assert len(d.reasons) >= 4  # exact_quote + high_risk + drift + shift + kl + low_conf


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
        cfg = SafeRecGenConfig(latent_cosine_drift_threshold=0.50)
        ctrl = SafeRecGenController(cfg)
        drift = DriftMetrics(turn_id=2, latent_cosine_drift=0.30)
        d = ctrl.evaluate("What is this?", drift)
        assert not d.should_fallback  # 0.30 < 0.50

    def test_as_dict(self) -> None:
        ctrl = SafeRecGenController()
        d = ctrl.evaluate("Show exact quote")
        result = d.as_dict()
        assert "should_fallback" in result
        assert "reasons" in result
        assert isinstance(result["reasons"], list)
