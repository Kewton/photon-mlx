"""LLM-free helpers for the Issue #135 training-corpus generator.

The companion script ``scripts/generate_institutional_training_corpus.py``
splits its work into three layers (DR1-001):

1. ``build_sessions`` — needs an LLM client to invent question/turn text.
2. ``verify_corpus`` — pure metrics + eval-leak gate; no LLM needed.
3. ``main`` — composes the two and writes JSONL output.

Everything in this module is LLM-free so unit tests can hit the
verification logic directly. Putting the helpers here also keeps
``generate_institutional_eval_set.py`` and the new training script
sharing the same JSONL plumbing rather than re-implementing it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, TypeVar

T = TypeVar("T")


def validate_eval_overlap(session_ids: Iterable[str], eval_path: Path) -> int:
    """Return the number of session IDs that appear in both inputs (DR4-005).

    The eval set lives at
    ``data/eval_sets/institutional_multi_turn_eval.jsonl`` (one JSON object
    per line, each with ``session_id``). Any positive return value means
    the training corpus has leaked into eval territory and the run must be
    aborted before tokenisation — never trim silently, since downstream
    ratio metrics depend on full corpus integrity.
    """
    eval_path = Path(eval_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"eval set not found: {eval_path}")
    eval_ids: set[str] = set()
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        sid = obj.get("session_id")
        if sid:
            eval_ids.add(sid)
    return len(set(session_ids) & eval_ids)


def split_train_val(
    items: list[T], *, val_ratio: float, seed: int
) -> tuple[list[T], list[T]]:
    """Deterministically split ``items`` into (train, val) lists.

    ``val_ratio`` must be in (0, 0.5) — the open interval excludes both
    "no validation" (use the caller's existing single-list path) and
    "more val than train" (a smell that should be re-thought, not
    silently allowed).
    """
    if not (0.0 < val_ratio < 0.5):
        raise ValueError(f"val_ratio must be in (0.0, 0.5), got {val_ratio} (DR1-005)")
    if not items:
        return [], []
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[:-n_val], shuffled[-n_val:]
