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
import math
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
# Issue #140 / DR4-002: embedding_random_init_threshold field
# ---------------------------------------------------------------------------


_PHOTON_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize(
    "yml",
    sorted(_PHOTON_CONFIG_DIR.glob("photon_*.yaml")),
    ids=lambda p: p.name,
)
def test_existing_yaml_loads_without_threshold_field(yml: Path) -> None:
    """Issue #140: existing photon_*.yaml configs (5 files) must continue to
    load successfully without specifying ``embedding_random_init_threshold``.
    The default value 0.3 must be applied (DR2-008 / DR3-002).
    """
    cfg = load_photon_config(yml)
    assert cfg.model.embedding_random_init_threshold == pytest.approx(0.3)


@pytest.mark.parametrize(
    "bad_threshold",
    [-0.1, math.nan, math.inf, "0.3", True, False],
    ids=["negative", "nan", "inf", "string", "bool_true", "bool_false"],
)
def test_model_config_rejects_invalid_embedding_threshold(
    bad_threshold: object,
) -> None:
    """Issue #140 DR4-002: invalid ``embedding_random_init_threshold`` values
    must be rejected with ``ValueError`` (or ``TypeError``) at construction.

    ``bool`` (True/False) is rejected even though it is an ``int`` subclass —
    accepting it would silently swallow configuration mistakes.
    """
    with pytest.raises((TypeError, ValueError)):
        ModelConfig(embedding_random_init_threshold=bad_threshold)


def test_model_config_accepts_zero_embedding_threshold() -> None:
    """Boundary: 0.0 is valid (≥ 0 and finite)."""
    cfg = ModelConfig(embedding_random_init_threshold=0.0)
    assert cfg.embedding_random_init_threshold == 0.0


def test_model_config_accepts_int_embedding_threshold_and_coerces_to_float() -> None:
    """``int`` (non-bool) is accepted and coerced to ``float`` for type
    consistency downstream."""
    cfg = ModelConfig(embedding_random_init_threshold=2)
    assert cfg.embedding_random_init_threshold == 2.0
    assert isinstance(cfg.embedding_random_init_threshold, float)


def test_load_photon_config_rejects_invalid_embedding_threshold(
    tmp_path: Path,
) -> None:
    """YAML-supplied invalid ``embedding_random_init_threshold`` must propagate
    the ``ValueError`` from ``ModelConfig.__post_init__`` through
    ``load_photon_config``."""
    path = _write_yaml(
        tmp_path,
        """
model:
  embedding_random_init_threshold: -0.5
""",
    )
    with pytest.raises(ValueError, match="embedding_random_init_threshold"):
        load_photon_config(path)
