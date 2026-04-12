from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Recursive dot-access wrapper for YAML config dicts."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                object.__setattr__(self, key, Config(value))
            elif isinstance(value, list):
                object.__setattr__(self, key, [
                    Config(item) if isinstance(item, dict) else item
                    for item in value
                ])
            else:
                object.__setattr__(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({vars(self)})"


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data)
