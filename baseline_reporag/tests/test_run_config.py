"""Tests for ``baseline_reporag.eval.run_config.resolve_eval_seed`` (Issue #143).

The helper resolves the ``cfg.run.seed`` / ``cfg.run.deterministic`` block
into the seed value that eval scripts inject into ``Generator.generate``
and ``Pipeline.query``. Because YAML happily silently coerces ``yes`` /
``true`` / ``"42"`` into truthy values that look like ints to ``bool()``,
the helper performs strict type validation (``type(x) is int``) so a
typo in an operator-authored ``configs/*.yaml`` fails fast in CI rather
than silently swapping in a nondeterministic eval run.

Contract (from work-plan §Step 1.1):
- Missing ``run`` block       -> default ``seed=42, deterministic=True``
- ``deterministic=False``     -> returns ``None`` regardless of seed
- ``run.seed=<int>``          -> returns the int (range-checked)
- ``type(seed) is bool``      -> ``TypeError`` (YAML ``true`` silent bug)
- ``type(seed) in (str, float)`` -> ``TypeError``
- ``seed`` outside ``[0, 2**32)`` -> ``ValueError``
- ``deterministic`` non-bool -> ``TypeError``
"""

from __future__ import annotations

import pytest

from baseline_reporag.config import Config
from baseline_reporag.eval.run_config import resolve_eval_seed


def _cfg_with_run(run_block: dict | None) -> Config:
    """Build a minimal Config wrapping a (possibly missing) ``run`` block."""
    data: dict = {}
    if run_block is not None:
        data["run"] = run_block
    return Config(data)


class TestResolveEvalSeedDefaults:
    """No ``run`` block / explicit defaults / deterministic toggle."""

    def test_resolve_run_block_missing_returns_default_42_true(self) -> None:
        """``cfg.run`` absent -> default seed=42, deterministic=True (=> 42)."""
        cfg = _cfg_with_run(None)
        assert resolve_eval_seed(cfg) == 42

    def test_resolve_deterministic_false_returns_none(self) -> None:
        """``run.deterministic=False`` -> ``None`` regardless of seed value."""
        cfg = _cfg_with_run({"seed": 42, "deterministic": False})
        assert resolve_eval_seed(cfg) is None

    def test_resolve_deterministic_false_skips_seed_validation(self) -> None:
        """CB-003: ``deterministic=False`` short-circuits before seed checks.

        A nondeterministic config must not fail because ``seed`` is null /
        malformed — the seed is simply ignored.
        """
        cfg = _cfg_with_run({"seed": None, "deterministic": False})
        assert resolve_eval_seed(cfg) is None
        cfg = _cfg_with_run({"seed": "stale-string", "deterministic": False})
        assert resolve_eval_seed(cfg) is None

    def test_resolve_with_int_seed_returns_42(self) -> None:
        """``run.seed=42`` -> returns 42."""
        cfg = _cfg_with_run({"seed": 42})
        assert resolve_eval_seed(cfg) == 42

    def test_resolve_with_explicit_deterministic_true(self) -> None:
        """Explicit ``deterministic=True`` together with ``seed=42`` -> 42."""
        cfg = _cfg_with_run({"seed": 42, "deterministic": True})
        assert resolve_eval_seed(cfg) == 42


class TestResolveEvalSeedTypeErrors:
    """Strict type validation rejects YAML silent-coercion footguns."""

    def test_seed_yaml_bool_true_raises_typeerror(self) -> None:
        """``run.seed: true`` (YAML bool) must raise TypeError, not pass as 1."""
        cfg = _cfg_with_run({"seed": True})
        with pytest.raises(TypeError, match="seed"):
            resolve_eval_seed(cfg)

    def test_seed_yaml_bool_false_raises_typeerror(self) -> None:
        """``run.seed: false`` must also raise TypeError (would pass as 0)."""
        cfg = _cfg_with_run({"seed": False})
        with pytest.raises(TypeError, match="seed"):
            resolve_eval_seed(cfg)

    def test_seed_str_raises_typeerror(self) -> None:
        """``run.seed: "42"`` (quoted YAML string) must raise TypeError."""
        cfg = _cfg_with_run({"seed": "42"})
        with pytest.raises(TypeError, match="seed"):
            resolve_eval_seed(cfg)

    def test_seed_float_raises_typeerror(self) -> None:
        """``run.seed: 3.14`` must raise TypeError (no implicit truncation)."""
        cfg = _cfg_with_run({"seed": 3.14})
        with pytest.raises(TypeError, match="seed"):
            resolve_eval_seed(cfg)

    def test_deterministic_str_raises_typeerror(self) -> None:
        """``run.deterministic: "false"`` must raise TypeError (silent truthy)."""
        cfg = _cfg_with_run({"seed": 42, "deterministic": "false"})
        with pytest.raises(TypeError, match="deterministic"):
            resolve_eval_seed(cfg)


class TestResolveEvalSeedRangeErrors:
    """``[0, 2**32)`` is the supported MLX/numpy seed range."""

    def test_seed_negative_raises_valueerror(self) -> None:
        """``run.seed=-1`` is out of range."""
        cfg = _cfg_with_run({"seed": -1})
        with pytest.raises(ValueError, match="seed"):
            resolve_eval_seed(cfg)

    def test_seed_too_large_raises_valueerror(self) -> None:
        """``run.seed=2**32`` is out of range (upper bound exclusive)."""
        cfg = _cfg_with_run({"seed": 2**32})
        with pytest.raises(ValueError, match="seed"):
            resolve_eval_seed(cfg)
