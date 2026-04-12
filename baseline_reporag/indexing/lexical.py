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


def _tokenize(text: str) -> list[str]:
    # Split camelCase: fooBar -> foo bar
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    tokens = re.split(r"[^a-z0-9_]+", text)
    return [t for t in tokens if len(t) >= 2]


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
        ranked = sorted(zip(self._chunk_ids, scores),
                        key=lambda x: x[1], reverse=True)
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
