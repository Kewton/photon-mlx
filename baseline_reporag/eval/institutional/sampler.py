"""Document sampling utilities for institutional eval-set generation (Issue #110)."""

from __future__ import annotations

import random

from .corpus import DocIndex

_ARTICLE_CATEGORIES: frozenset[str] = frozenset(
    {"article_lookup", "penalty", "exception"}
)


def pick_category_docs(
    index: list[DocIndex],
    category: str,
    n: int,
    rng: random.Random,
) -> list[DocIndex]:
    """Return up to ``n`` docs suitable for the given category.

    ``article_lookup`` / ``penalty`` / ``exception`` are gated on the
    article-bearing subset. Other categories use the full index.
    """
    pool = list(index)
    if category in _ARTICLE_CATEGORIES:
        pool = [d for d in pool if d.has_articles]
    if category == "penalty":
        pool = [d for d in pool if d.has_penalty] or pool
    if category == "exception":
        pool = [d for d in pool if d.has_exception] or pool
    if not pool:
        return []
    rng.shuffle(pool)
    return pool[:n]


def sample_for_verification(
    rows: list[dict], *, n_per_category: int = 4, seed: int = 42
) -> list[dict]:
    """Pick ``n_per_category`` rows per category for human verification."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row.get("category", ""), []).append(row)
    out: list[dict] = []
    for cat in sorted(by_cat.keys()):
        candidates = list(by_cat[cat])
        rng.shuffle(candidates)
        out.extend(candidates[:n_per_category])
    return out
