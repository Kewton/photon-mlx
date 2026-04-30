from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# CB-004 / CB-005: single allowlist for safe ``repo_id`` path segments.
# Matches ``scripts/build_symbol_graph.py``'s historical regex so the
# script, the pipeline factory, and demo entry points all reject the
# same traversal / shell-metacharacter / unicode shapes.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# CB-R2-001: cap ``repo_id`` length so downstream path concatenations
# (``<data_root>/indexes/<repo_id>/...``) stay well under the 255-byte
# POSIX ``NAME_MAX`` limit even after suffixes like ``/symbol_graph.json``.
# 64 leaves ample headroom for these suffixes on every supported FS.
_REPO_ID_MAX_LENGTH = 64


def validate_repo_id(repo_id: str) -> str:
    """Return ``repo_id`` unchanged iff it is a safe path segment.

    Enforces ``[A-Za-z0-9_-]+`` so that values like ``../outside``,
    ``/tmp/x``, ``a/b``, empty strings, or unicode/shell-metacharacters
    cannot escape ``<data_root>/indexes/`` via ``Path`` concatenation.
    Callers construct filesystem paths from this value, so fail-fast at
    the entry point (factory / CLI / demo) is preferable to defensive
    checks scattered across each index loader.

    CB-R2-001: values longer than ``_REPO_ID_MAX_LENGTH`` are rejected so
    overly long ids surface as a clear ``ValueError`` here rather than as
    a late ``OSError`` from ``Path`` operations on the resulting index
    directory.
    """
    if not isinstance(repo_id, str):
        raise TypeError(f"repo_id must be str, got {type(repo_id).__name__}")
    if not _REPO_ID_RE.match(repo_id):
        raise ValueError(
            f"repo_id must match [A-Za-z0-9_-]+ (got {repo_id!r}); "
            "unsafe characters or path traversal segments are rejected."
        )
    if len(repo_id) > _REPO_ID_MAX_LENGTH:
        raise ValueError(
            f"repo_id length {len(repo_id)} exceeds {_REPO_ID_MAX_LENGTH} "
            "characters; shorten it so derived filesystem paths stay "
            "within OS limits."
        )
    return repo_id


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    merged = {}
    for key in set(base) | set(override):
        if key in override and key in base:
            if isinstance(base[key], dict) and isinstance(override[key], dict):
                merged[key] = deep_merge(base[key], override[key])
            else:
                merged[key] = override[key]
        elif key in override:
            merged[key] = override[key]
        else:
            bv = base[key]
            merged[key] = deep_merge(bv, {}) if isinstance(bv, dict) else bv
    return merged


class Config:
    """Recursive dot-access wrapper for YAML config dicts."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                object.__setattr__(self, key, Config(value))
            elif isinstance(value, list):
                object.__setattr__(
                    self,
                    key,
                    [
                        Config(item) if isinstance(item, dict) else item
                        for item in value
                    ],
                )
            else:
                object.__setattr__(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a plain dict (recursive)."""
        out: dict[str, Any] = {}
        for key, value in vars(self).items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [
                    item.to_dict() if isinstance(item, Config) else item
                    for item in value
                ]
            else:
                out[key] = value
        return out

    def merge_override(self, override: dict[str, Any]) -> Config:
        """Return a new Config with *override* deep-merged on top."""
        return Config(deep_merge(self.to_dict(), override))

    def __repr__(self) -> str:
        return f"Config({vars(self)})"


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = Config(data)
    object.__setattr__(cfg, "_config_path", str(path))
    return cfg


def _get_graph_block_enabled(
    cfg: Config | dict[str, Any], sub_block: str, *, default: bool
) -> bool:
    """Return ``indexing.<sub_block>.enabled`` with a configurable default.

    Uses only the shared ``.get(key, default)`` protocol so both
    :class:`Config` instances and plain dicts work without branching.
    ``or {}`` guards against ``None`` values that would break ``.get()``.
    NOTE: ``isinstance(dict)`` check is intentionally absent (DR2-001) —
    Config objects are not dicts, so that check would silently fall back
    to the default for every Config input.

    CB-003: Non-bool values raise :class:`TypeError` (fail-fast).
    """
    indexing = cfg.get("indexing", {}) or {}
    block = indexing.get(sub_block, {}) or {}
    val = block.get("enabled", default)
    if not isinstance(val, bool):
        raise TypeError(
            f"indexing.{sub_block}.enabled must be a bool (true/false), "
            f"got {type(val).__name__}={val!r}"
        )
    return val


def is_symbol_graph_enabled(cfg: Config | dict[str, Any]) -> bool:
    """Return whether ``indexing.symbol_graph.enabled`` is on (default True).

    Accepts a :class:`Config` instance (which exposes ``.get()`` and
    recursively wraps nested dicts) or a plain ``dict``. Missing blocks
    default to ``True`` so configurations predating Issue #109 are unaffected.
    """
    return _get_graph_block_enabled(cfg, "symbol_graph", default=True)


def is_heading_graph_enabled(cfg: Config | dict[str, Any]) -> bool:
    """Return whether ``indexing.heading_graph.enabled`` is on (default False).

    Ships OFF by default (DR2-002 / release strategy: flag-off merge first).
    Enable explicitly in config to activate heading-based graph expansion.
    """
    return _get_graph_block_enabled(cfg, "heading_graph", default=False)
