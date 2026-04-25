from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ..ingestion.store import ChunkStore

logger = logging.getLogger(__name__)

_E5_MODEL_PREFIX = "intfloat/multilingual-e5"


def _is_e5_standard(model_id: str) -> bool:
    """E5 標準系 (multilingual-e5-{small,base,large}) を判定する。

    instruction-tuned 派生モデル (multilingual-e5-large-instruct 等) は
    異なる prefix 仕様 (task-specific instruction) のため除外する。
    空文字列 (model_id.txt が空のケース) には False を返す。
    """
    if not model_id:
        return False
    return model_id.startswith(_E5_MODEL_PREFIX) and "instruct" not in model_id


def _assert_not_e5_instruct(model_id: str) -> None:
    """E5 instruction-tuned 派生モデルへの silent 誤適用を防ぐ fail-fast。

    DR4-006: ValueError 本文には model_id 文字列を含めない (機密情報漏洩防止)。
    private HF org 名 / R&D project codename 等が log / Streamlit `st.error` /
    FastAPI traceback 経由で leak しないよう、識別情報は logger.debug() のみへ
    落とし、user 到達経路 (CLI stderr / UI / API response) には汎化したメッセージ
    のみを返す。
    """
    if model_id and model_id.startswith(_E5_MODEL_PREFIX) and "instruct" in model_id:
        logger.debug("E5 instruct variant rejected: model_id=%s", model_id)
        raise ValueError(
            "E5 instruct variants are not supported by this helper; "
            "use config-based prefix injection instead."
        )


def _e5_passage_prefix(model_id: str, texts: list[str]) -> list[str]:
    """E5 標準系モデル使用時に passage: prefix を前置する。

    model_id が空文字列または非 E5 の場合は既存挙動維持 (prefix 無し)。
    E5 instruct 派生モデルが渡された場合は ValueError で fail-fast。
    """
    _assert_not_e5_instruct(model_id)
    if _is_e5_standard(model_id):
        return [f"passage: {t}" for t in texts]
    return texts


def _e5_query_prefix(model_id: str, query: str) -> str:
    """E5 標準系モデル使用時に query: prefix を前置する。

    model_id が空文字列または非 E5 の場合は既存挙動維持 (prefix 無し)。
    E5 instruct 派生モデルが渡された場合は ValueError で fail-fast。
    """
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
