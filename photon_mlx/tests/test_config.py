"""Tests for ``torch_ref.config`` extensions introduced by Issue #55.

Covers:

- ``ModelConfig.__post_init__`` validation of ``rope_scaling`` and
  ``rope_scale_factor``.
- ``ModelConfig.rope_scaling_from`` classmethod (MagicMock-compatible).
- ``_set_fields`` WARNING on unknown YAML keys (DR1-007).
- ``load_photon_config`` cross-config validation between
  ``training.context_length``, ``hierarchy.chunk_sizes`` and
  ``model.max_position_embeddings``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch_ref.config import (  # noqa: E402
    HierarchyConfig,
    ModelConfig,
    TrainingConfig,
    _set_fields,
    load_photon_config,
)


# ---------------------------------------------------------------------------
# ModelConfig.__post_init__ validation
# ---------------------------------------------------------------------------


def test_modelconfig_accepts_rope_scaling_ntk() -> None:
    """rope_scaling='ntk' with scale_factor>=1 must be accepted."""
    cfg = ModelConfig(rope_scaling="ntk", rope_scale_factor=32.0)
    assert cfg.rope_scaling == "ntk"
    assert cfg.rope_scale_factor == 32.0


def test_modelconfig_rejects_invalid_rope_scaling() -> None:
    """Anything outside ROPE_SCALING_CHOICES must raise ValueError."""
    with pytest.raises(ValueError, match="invalid rope_scaling"):
        ModelConfig(rope_scaling="linear")


def test_modelconfig_rejects_rope_scale_factor_below_one() -> None:
    """rope_scale_factor < 1.0 must raise ValueError (NTK is extrapolation)."""
    with pytest.raises(ValueError, match="rope_scale_factor must be >= 1.0"):
        ModelConfig(rope_scale_factor=0.5)


def test_modelconfig_warns_on_unused_scale_factor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """rope_scaling='none' + rope_scale_factor != 1.0 must emit WARNING log."""
    with caplog.at_level(logging.WARNING, logger="torch_ref.config"):
        ModelConfig(rope_scaling="none", rope_scale_factor=32.0)

    assert any(
        rec.levelno == logging.WARNING
        and rec.name == "torch_ref.config"
        and "ignored because rope_scaling='none'" in rec.getMessage()
        for rec in caplog.records
    ), "expected warning about ignored rope_scale_factor"


# ---------------------------------------------------------------------------
# ModelConfig.rope_scaling_from
# ---------------------------------------------------------------------------


def test_rope_scaling_from_handles_magicmock() -> None:
    """rope_scaling_from must fall back to defaults when attributes missing.

    ``MagicMock()`` returns a new ``MagicMock`` for any unknown attribute
    access.  ``rope_scaling_from`` uses ``getattr(m, name, default)`` with
    concrete fallbacks, so when ``m`` is a ``MagicMock`` WITHOUT explicit
    ``rope_scaling`` / ``rope_scale_factor`` attributes set, it must return
    the pair ``("none", 1.0)``.
    """
    m = MagicMock(spec=["architecture"])  # no rope_scaling / rope_scale_factor
    scaling, factor = ModelConfig.rope_scaling_from(m)
    assert scaling == "none"
    assert factor == 1.0

    # When the attributes ARE set, they must propagate.
    m2 = MagicMock(spec=["rope_scaling", "rope_scale_factor"])
    m2.rope_scaling = "ntk"
    m2.rope_scale_factor = 8.0
    scaling, factor = ModelConfig.rope_scaling_from(m2)
    assert scaling == "ntk"
    assert factor == 8.0


# ---------------------------------------------------------------------------
# _set_fields warning on unknown YAML keys
# ---------------------------------------------------------------------------


def test_set_fields_warns_on_unknown_key(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown YAML keys must emit a WARNING but not raise (backward compat)."""
    cfg = ModelConfig()
    with caplog.at_level(logging.WARNING, logger="torch_ref.config"):
        _set_fields(cfg, {"rope_scale": "ntk"})  # typo for rope_scaling

    matched = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING
        and rec.name == "torch_ref.config"
        and "unknown config key ignored" in rec.getMessage()
        and "rope_scale" in rec.getMessage()
    ]
    assert matched, (
        "expected WARNING mentioning 'unknown config key ignored' and the key "
        f"name 'rope_scale'; got records: {[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# load_photon_config cross-config validation
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_photon_config_rejects_invalid_context_length(tmp_path: Path) -> None:
    """context_length not a multiple of prod(chunk_sizes) must raise."""
    path = _write_yaml(
        tmp_path,
        """
model:
  max_position_embeddings: 64
hierarchy:
  chunk_sizes: [4, 4]
tokenizer:
  vocab_size: 256
training:
  context_length: 17
""",
    )
    with pytest.raises(ValueError, match="must be a multiple of"):
        load_photon_config(path)


def test_load_photon_config_rejects_context_length_exceeds_max_pos(
    tmp_path: Path,
) -> None:
    """context_length greater than max_position_embeddings must raise."""
    path = _write_yaml(
        tmp_path,
        """
model:
  max_position_embeddings: 64
hierarchy:
  chunk_sizes: [4, 4]
tokenizer:
  vocab_size: 256
training:
  context_length: 128
""",
    )
    with pytest.raises(ValueError, match="must be <="):
        load_photon_config(path)


def test_load_photon_config_accepts_valid_context_length(tmp_path: Path) -> None:
    """Sanity: valid context_length loads without error."""
    path = _write_yaml(
        tmp_path,
        """
model:
  max_position_embeddings: 64
hierarchy:
  chunk_sizes: [4, 4]
tokenizer:
  vocab_size: 256
training:
  context_length: 32
""",
    )
    cfg = load_photon_config(path)
    assert cfg.training is not None
    assert cfg.training.context_length == 32


def test_load_photon_config_no_training_skips_cross_validation(tmp_path: Path) -> None:
    """Without 'training' block, cross-validation is skipped."""
    path = _write_yaml(
        tmp_path,
        """
model:
  max_position_embeddings: 64
hierarchy:
  chunk_sizes: [4, 4]
tokenizer:
  vocab_size: 256
""",
    )
    cfg = load_photon_config(path)
    assert cfg.training is None


def test_load_photon_config_rejects_invalid_rope_scaling(tmp_path: Path) -> None:
    """Setting rope_scaling to an invalid value via YAML must raise ValueError."""
    path = _write_yaml(
        tmp_path,
        """
model:
  rope_scaling: bogus
""",
    )
    with pytest.raises(ValueError, match="invalid rope_scaling"):
        load_photon_config(path)


# Regression fixture for TrainingConfig instance (indirect check that the
# cross-config helper works on bare dataclass instances too, not just YAML).
def test_validate_cross_config_on_bare_dataclasses() -> None:
    from torch_ref.config import _validate_cross_config

    m = ModelConfig(max_position_embeddings=128)
    h = HierarchyConfig(chunk_sizes=[4, 4])
    t = TrainingConfig(context_length=32)
    # Valid: should not raise.
    _validate_cross_config(m, h, t)

    t_bad = TrainingConfig(context_length=33)
    with pytest.raises(ValueError, match="multiple"):
        _validate_cross_config(m, h, t_bad)


# ---------------------------------------------------------------------------
# Issue #135 / Phase 4-1: TrainingConfig.train_corpora_mix + val_split
# ---------------------------------------------------------------------------
# DR1-005 simplification: val_corpora_mix dict was dropped in favour of a
# single ``val_split: float`` that carves val out of the same shuffled
# train pool, preserving the train_corpora_mix ratio.
# DR1-003 strict validation: empty mix dict, weight <= 0 or non-finite,
# sum(weights) outside ±1e-6 of 1.0 must all raise ValueError at config
# construction time so trainer never sees an invalid mix.


class TestTrainingConfigCorporaMix:
    """TrainingConfig.train_corpora_mix + val_split schema (DR1-003 / DR1-005)."""

    def test_default_train_corpora_mix_is_none(self) -> None:
        """Backwards compat: default config keeps the legacy single-corpus path."""
        t = TrainingConfig()
        assert t.train_corpora_mix is None
        assert t.val_split == 0.0

    def test_explicit_mix_with_valid_weights(self) -> None:
        t = TrainingConfig(
            train_corpora_mix={"a.jsonl": 0.5, "b.jsonl": 0.5},
            val_split=0.05,
        )
        assert t.train_corpora_mix == {"a.jsonl": 0.5, "b.jsonl": 0.5}
        assert t.val_split == 0.05

    def test_empty_mix_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="train_corpora_mix"):
            TrainingConfig(train_corpora_mix={})

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            TrainingConfig(train_corpora_mix={"a.jsonl": -0.1, "b.jsonl": 1.1})

    def test_zero_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            TrainingConfig(train_corpora_mix={"a.jsonl": 0.0, "b.jsonl": 1.0})

    def test_non_finite_weight_raises(self) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(train_corpora_mix={"a.jsonl": float("inf"), "b.jsonl": 0.5})

    def test_non_numeric_weight_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            TrainingConfig(
                train_corpora_mix={"a.jsonl": "0.5", "b.jsonl": 0.5}  # type: ignore[dict-item]
            )

    def test_sum_off_target_raises(self) -> None:
        with pytest.raises(ValueError, match="sum"):
            TrainingConfig(train_corpora_mix={"a.jsonl": 0.4, "b.jsonl": 0.5})  # 0.9

    def test_sum_within_tolerance_passes(self) -> None:
        # 0.5 + 0.4999999999 = 0.9999999999 — within 1e-6 of 1.0.
        TrainingConfig(train_corpora_mix={"a.jsonl": 0.5, "b.jsonl": 0.4999999999})

    def test_val_split_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="val_split"):
            TrainingConfig(val_split=-0.1)

    def test_val_split_one_or_more_raises(self) -> None:
        with pytest.raises(ValueError, match="val_split"):
            TrainingConfig(val_split=1.0)

    def test_val_split_zero_is_allowed(self) -> None:
        """val_split=0 means "no train-pool split" (legacy single-corpus path)."""
        t = TrainingConfig(val_split=0.0)
        assert t.val_split == 0.0


# ---------------------------------------------------------------------------
# Issue #135 / Phase 4-3: institutional_docs_photon_retrain.yaml shape
# ---------------------------------------------------------------------------


def test_institutional_retrain_yaml_loads_with_expected_hyperparams(
    tmp_path: Path,
) -> None:
    """Pin the retrain yaml's key knobs so silent edits surface in CI.

    The retrain yaml is the source of truth for Issue #135's training
    hyperparameters; if any of these drift, eval comparisons against
    earlier runs become invalid. We assert the values that the design
    policy nailed down (S5-001 lr / DR1-005 val_split / DR1-003 mix
    sums to 1) rather than every line, so harmless edits to comments
    or repo paths don't churn this test.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    cfg_path = repo_root / "configs" / "institutional_docs_photon_retrain.yaml"
    assert cfg_path.exists(), f"missing: {cfg_path}"

    from torch_ref.config import load_photon_config

    cfg = load_photon_config(str(cfg_path))

    t = cfg.training
    # S5-001 / DR2-009: cosine_decay 経路に乗るには min_lr > 0 必須。
    assert t.learning_rate == 3.0e-5
    assert t.min_learning_rate == 3.0e-6
    assert t.warmup_ratio == 0.0
    # DR1-005 / S5-002 reflected.
    assert t.val_split == 0.05
    assert t.max_steps >= 10000  # 10K-20K range; concrete value is per-run
    # S5-004 reflected: micro_batch * grad_accum = effective batch 32.
    assert t.micro_batch_size == 2
    assert t.gradient_accumulation_steps == 16
    # DR1-003: train_corpora_mix must be a non-empty dict summing to 1.0.
    mix = t.train_corpora_mix
    assert mix is not None and len(mix) == 2
    assert abs(sum(mix.values()) - 1.0) < 1e-6
    # Both keys must point under data/training/ (DR4-002 approved root).
    for path in mix:
        assert path.startswith("./data/training/"), (
            f"corpus path must be under ./data/training/ (DR4-002), got {path}"
        )
