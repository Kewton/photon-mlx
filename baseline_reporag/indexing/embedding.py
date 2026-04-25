from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ..ingestion.store import ChunkStore

_E5_MODEL_PREFIX = "intfloat/multilingual-e5"


def _is_e5_standard(model_id: str) -> bool:
    if not model_id:
        return False
    return model_id.startswith(_E5_MODEL_PREFIX) and "instruct" not in model_id


def _assert_not_e5_instruct(model_id: str) -> None:
    if model_id and model_id.startswith(_E5_MODEL_PREFIX) and "instruct" in model_id:
        raise ValueError(
            f"E5 instruct variants ({model_id}) require task-specific prefix; "
            "not supported by this helper. Use config-based prefix injection instead."
        )


def _e5_passage_prefix(model_id: str, texts: list[str]) -> list[str]:
    _assert_not_e5_instruct(model_id)
    if _is_e5_standard(model_id):
        return [f"passage: {t}" for t in texts]
    return texts


def _e5_query_prefix(model_id: str, query: str) -> str:
    _assert_not_e5_instruct(model_id)
    if _is_e5_standard(model_id):
        return f"query: {query}"
    return query


class EmbeddingResult(NamedTuple):
    chunk_id: str
    score: float


class EmbeddingIndex:
    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._model_id = model_id
        self._model: SentenceTransformer | None = None
        self._embeddings: np.ndarray | None = None  # shape (N, D), float32
        self._chunk_ids: list[str] = []

    def _model_(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_id)
        return self._model

    def build(
        self,
        store: ChunkStore,
        repo_id: str,
        repo_commit: str,
        batch_size: int = 64,
    ) -> None:
        texts: list[str] = []
        self._chunk_ids = []
        for chunk in store.iter_repo(repo_id, repo_commit):
            text = f"{chunk.rel_path}\n{chunk.section_header}\n{chunk.content}"
            texts.append(text[:2048])  # truncate to avoid OOM
            self._chunk_ids.append(chunk.chunk_id)
        texts = _e5_passage_prefix(self._model_id, texts)
        self._embeddings = self._model_().encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def search(self, query: str, top_k: int = 20) -> list[EmbeddingResult]:
        if self._embeddings is None:
            raise RuntimeError("Index not built; call build() or load() first")
        encoded_query = _e5_query_prefix(self._model_id, query)
        q_emb = self._model_().encode(
            [encoded_query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        scores: np.ndarray = self._embeddings @ q_emb
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            EmbeddingResult(self._chunk_ids[i], float(scores[i])) for i in top_indices
        ]

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / "embeddings.npy", self._embeddings)
        (dir_path / "chunk_ids.json").write_text(
            json.dumps(self._chunk_ids), encoding="utf-8"
        )
        (dir_path / "model_id.txt").write_text(self._model_id, encoding="utf-8")

    @classmethod
    def load(cls, dir_path: str | Path) -> EmbeddingIndex:
        dir_path = Path(dir_path)
        model_id = (dir_path / "model_id.txt").read_text(encoding="utf-8").strip()
        idx = cls(model_id=model_id)
        idx._embeddings = np.load(dir_path / "embeddings.npy")
        idx._chunk_ids = json.loads(
            (dir_path / "chunk_ids.json").read_text(encoding="utf-8")
        )
        return idx
