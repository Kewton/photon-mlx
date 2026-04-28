"""``cfg.run`` block resolver for reproducible eval scripts (Issue #143).

Eval scripts (``run_baseline_eval``, ``run_multi_turn_eval``,
``retrieval_grid_search``, ``run_stress_eval``, ``compare_generators``)
read ``cfg.run.seed`` / ``cfg.run.deterministic`` and forward the
resolved seed into ``Pipeline.query(*, seed=...)`` so MLX-LM sampling is
deterministic across runs. This module is the single source of truth for
that resolution.

Why strict type checks
----------------------
YAML coerces silently between ``true`` / ``yes`` / ``"42"`` and Python
truthy values: ``cfg.run.seed: true`` would pass an ``isinstance(seed, int)``
guard because ``bool`` is a subclass of ``int``. To prevent operator
typos in ``configs/*.yaml`` from silently swapping a deterministic eval
for a nondeterministic one, this resolver requires ``type(value) is int``
(not ``isinstance``) for ``seed`` and ``type(value) is bool`` for
``deterministic`` — a mismatch fails fast in CI / at script start.

Range
-----
MLX (``mlx.core.random.seed``) and numpy both accept seeds in
``[0, 2**32)``. Out-of-range values raise ``ValueError`` here rather
than surfacing as opaque MLX errors deep inside ``Generator.generate``.
"""

from __future__ import annotations

from typing import Any

# MLX / numpy seed range. Inclusive lower, exclusive upper.
_SEED_MIN = 0
_SEED_MAX = 2**32  # exclusive

# Default values applied when ``cfg.run`` is missing entirely. The defaults
# match the design policy: deterministic eval with a fixed seed of 42.
_DEFAULT_SEED = 42
_DEFAULT_DETERMINISTIC = True


def _validate_deterministic(run_dict: dict[str, Any]) -> bool:
    """Return ``run.deterministic`` after strict bool validation."""
    determ_raw = run_dict.get("deterministic", _DEFAULT_DETERMINISTIC)
    if type(determ_raw) is not bool:  # noqa: E721 (intentional strict)
        raise TypeError(
            "cfg.run.deterministic must be a YAML bool (true/false); "
            f"got {type(determ_raw).__name__}={determ_raw!r}. "
            'Quoted strings like "false" are silently truthy in Python — '
            "use the unquoted YAML keyword instead."
        )
    return determ_raw


def _validate_seed(run_dict: dict[str, Any]) -> int:
    """Return ``run.seed`` after strict int / range validation."""
    seed_raw = run_dict.get("seed", _DEFAULT_SEED)
    if type(seed_raw) is not int:  # noqa: E721 (intentional strict)
        raise TypeError(
            "cfg.run.seed must be a plain int (not bool/float/str). "
            f"got {type(seed_raw).__name__}={seed_raw!r}. "
            "YAML 'true'/'yes' silently coerce to bool which is a Python "
            "int subclass — use 42 (no quotes, no bool keywords)."
        )
    if not (_SEED_MIN <= seed_raw < _SEED_MAX):
        raise ValueError(
            f"cfg.run.seed must be in [{_SEED_MIN}, {_SEED_MAX}); got {seed_raw}."
        )
    return seed_raw


def resolve_eval_seed(cfg: Any) -> int | None:
    """Return the seed eval scripts should pass into ``pipeline.query``.

    Behavior:
    - ``cfg.run`` missing                 -> ``42`` (default deterministic)
    - ``cfg.run.deterministic=False``     -> ``None`` (skip seeding)
    - ``cfg.run.seed`` plus ``deterministic=True`` -> the int seed

    ``cfg`` is the dot-access :class:`baseline_reporag.config.Config` (or
    any object with a ``.get(key, default)`` shim). The function never
    mutates ``cfg``.

    See module docstring for the rationale behind strict type / range
    checks. Callers MUST treat the return value as opaque: a ``None``
    return means "interactive parity / nondeterministic" and downstream
    callers (``Generator.generate(*, seed=None)``) must skip MLX seeding,
    not silently fall back to ``0``.
    """
    run_block = cfg.get("run", None)
    if run_block is None:
        # No ``run`` section at all: defaults apply, deterministic=True.
        return _DEFAULT_SEED

    if hasattr(run_block, "to_dict"):
        run_dict = run_block.to_dict()
    elif isinstance(run_block, dict):
        run_dict = run_block
    else:
        raise TypeError(
            "cfg.run must be a mapping (YAML block / dict / Config); "
            f"got {type(run_block).__name__}={run_block!r}."
        )

    # Validate ``deterministic`` first so that ``deterministic=False`` users
    # are not forced to also keep ``seed`` well-formed (CB-003): a stale or
    # null seed must not block nondeterministic mode.
    deterministic = _validate_deterministic(run_dict)
    if not deterministic:
        return None
    return _validate_seed(run_dict)
