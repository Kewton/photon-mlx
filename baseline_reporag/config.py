from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


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
