"""Drift-metrics panel helpers for the Streamlit chat UI (Issue #82 Wave 3).

This module is intentionally streamlit-free (see
``app/components/__init__.py``) so the formatting logic can be unit-tested
without a Streamlit runtime. The rendering side (``st.expander`` /
``st.metric``) lives in ``app/photon_app.page_chat`` and consumes the
plain ``dict`` returned by :func:`format_drift_panel`.

Design reference: `workspace/design/issue-82-app-photon-features-design-policy.md`
§6.2 drift_panel.
"""

from __future__ import annotations

from typing import Any, Literal

DriftLevel = Literal["ok", "warn", "alert"]

# Map UI indicator names to ``DriftMetrics.as_dict()`` keys.
# Source: ``photon_mlx/session.py:113-170`` (Issue #63 three-level metrics).
# The ordering below is load-bearing: the chat UI renders rows in this
# sequence so operators always see the same top-to-bottom layout.
DRIFT_METRIC_KEYS: dict[str, str] = {
    "token_level": "latent_cosine_drift_token",
    "mid_level": "latent_cosine_drift_mid",
    "top_level": "latent_cosine_drift_top",
    "topic_shift": "topic_shift_score",
}


def classify_drift(value: float | None, threshold: float | None) -> DriftLevel:
    """Classify a drift value against its threshold.

    Rules:
        * ``value is None`` or ``threshold is None`` → ``"ok"``
          (no data or no configured threshold).
        * ``value > threshold``              → ``"alert"``.
        * ``value > threshold * 0.8``        → ``"warn"``.
        * otherwise                          → ``"ok"``.
    """
    if value is None or threshold is None:
        return "ok"
    if value > threshold:
        return "alert"
    if value > threshold * 0.8:
        return "warn"
    return "ok"


def format_drift_panel(
    drift_metrics: dict[str, Any] | None,
    thresholds: dict[str, float | None],
) -> dict[str, Any]:
    """Render-ready summary of a single turn's drift metrics.

    Args:
        drift_metrics: Output of ``DriftMetrics.as_dict()`` (or ``None``
            for baseline_rag / first-turn cases where no metric exists).
        thresholds: Map keyed by the UI indicator name
            (``token_level`` / ``mid_level`` / ``top_level`` /
            ``topic_shift``) to either a numeric threshold or ``None``.

    Returns:
        A dict with keys ``available`` (bool), ``reason`` (str),
        ``rows`` (list of per-indicator dicts) and ``safe_recgen_fired``
        (bool). When ``drift_metrics`` is empty/None, ``available=False``
        and ``reason="N/A (baseline_rag or first turn)"``; otherwise four
        rows are returned in :data:`DRIFT_METRIC_KEYS` order.
    """
    if not drift_metrics:
        return {
            "available": False,
            "reason": "N/A (baseline_rag or first turn)",
            "rows": [],
            "safe_recgen_fired": False,
        }

    badge_map: dict[DriftLevel, str] = {"alert": "⚠", "warn": "!", "ok": ""}
    rows: list[dict[str, Any]] = []
    for ui_name, dm_key in DRIFT_METRIC_KEYS.items():
        raw = drift_metrics.get(dm_key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            val_f: float | None = float(raw)
            value_str = f"{val_f:.2f}"
        else:
            val_f = None
            value_str = "—"
        th = thresholds.get(ui_name)
        level = classify_drift(val_f, th)
        rows.append(
            {
                "name": ui_name,
                "value": val_f,
                "value_str": value_str,
                "level": level,
                "badge": badge_map[level],
            }
        )

    return {
        "available": True,
        "reason": "",
        "rows": rows,
        "safe_recgen_fired": bool(drift_metrics.get("safe_recgen_fired", False)),
    }
