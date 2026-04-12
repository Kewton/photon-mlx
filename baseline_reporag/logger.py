from __future__ import annotations

import json
import time
from pathlib import Path


class RunLogger:
    """Append-only JSONL logger for benchmark runs."""

    def __init__(self, log_dir: str | Path, run_id: str) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id
        self._path = self._log_dir / f"{run_id}.jsonl"

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_turn(self, record: dict) -> None:
        record = {"run_id": self._run_id, "logged_at": time.time(), **record}
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
