from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import NamedTuple

from rank_bm25 import BM25Okapi

from ..ingestion.store import ChunkStore


class LexicalResult(NamedTuple):
    chunk_id: str
    score: float


# CJK ranges that should be tokenized as character bigrams for BM25:
# - U+3040–U+309F: Hiragana
# - U+30A0–U+30FF: Katakana (incl. ・ and Katakana-Hiragana Prolonged Sound Mark)
# - U+4E00–U+9FFF: CJK Unified Ideographs (Kanji / Hanzi / Hanja)
# - U+3005, U+3007: Iteration mark (々) and zero ideograph (〇)
# Whitespace, punctuation, and full-width forms are excluded so they act as
# bigram boundaries — a query like ``認定基準`` produces ``[認定, 定基, 基準]``
# while ``【認定の条件】`` (full-width brackets ignored as boundaries)
# produces ``[認定, 定の, の条, 条件]``.
_CJK_RUN = re.compile(r"[々〇぀-ゟ゠-ヿ一-鿿]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize for BM25.

    Combines two independent token streams:

    - **ASCII alphanumeric tokens** (legacy code/English path): camelCase
      split, lower-cased, split on non-``[a-z0-9_]``, filter ``len >= 2``.
    - **CJK character bigrams** (Issue #174): Japanese/Chinese/Korean text
      is split into runs of CJK characters, each run yields all overlapping
      2-char windows. A 1-char run yields the single character to keep
      retrievability for short kanji terms (e.g. ``区``).

    Without the CJK path, BM25 contributes 0 signal for Japanese corpora
    because the legacy regex strips every kana/kanji as a delimiter. This
    let semantically-similar-but-wrong chunks dominate ranking via the
    embedding signal alone.
    """
    # ASCII path (camelCase split + lowercase + alnum tokenization)
    ascii_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    ascii_text = ascii_text.lower()
    ascii_tokens = [t for t in re.split(r"[^a-z0-9_]+", ascii_text) if len(t) >= 2]

    # CJK path (character bigrams per CJK run)
    cjk_tokens: list[str] = []
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            cjk_tokens.append(run)
        else:
            cjk_tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    return ascii_tokens + cjk_tokens


class LexicalIndex:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []

    def build(self, store: ChunkStore, repo_id: str, repo_commit: str) -> None:
        corpus: list[list[str]] = []
        self._chunk_ids = []
        for chunk in store.iter_repo(repo_id, repo_commit):
            text = f"{chunk.rel_path} {chunk.section_header} {chunk.content}"
            corpus.append(_tokenize(text))
            self._chunk_ids.append(chunk.chunk_id)
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[LexicalResult]:
        if self._bm25 is None:
            raise RuntimeError("Index not built; call build() or load() first")
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [LexicalResult(cid, float(s)) for cid, s in ranked[:top_k]]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self._chunk_ids}, f)

    @classmethod
    def load(cls, path: str | Path) -> LexicalIndex:
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx._bm25 = data["bm25"]
        idx._chunk_ids = data["chunk_ids"]
        return idx
