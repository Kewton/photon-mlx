from __future__ import annotations

from dataclasses import dataclass

from ..indexing.lexical import LexicalIndex
from ..indexing.embedding import EmbeddingIndex


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    lexical_score: float
    embedding_score: float


def _normalize(pairs: list[tuple[str, float]]) -> dict[str, float]:
    if not pairs:
        return {}
    max_s = max(s for _, s in pairs) or 1.0
    return {cid: s / max_s for cid, s in pairs}


def hybrid_search(
    query: str,
    lexical_index: LexicalIndex,
    embedding_index: EmbeddingIndex,
    lexical_top_k: int = 20,
    embedding_top_k: int = 20,
    fused_top_k: int = 16,
    lexical_weight: float = 0.45,
    embedding_weight: float = 0.45,
) -> list[RetrievalResult]:
    lex_raw = lexical_index.search(query, top_k=lexical_top_k)
    emb_raw = embedding_index.search(query, top_k=embedding_top_k)

    lex_norm = _normalize([(r.chunk_id, r.score) for r in lex_raw])
    emb_norm = _normalize([(r.chunk_id, r.score) for r in emb_raw])

    all_ids = set(lex_norm) | set(emb_norm)
    results = [
        RetrievalResult(
            chunk_id=cid,
            score=lexical_weight * lex_norm.get(cid, 0.0)
                  + embedding_weight * emb_norm.get(cid, 0.0),
            lexical_score=lex_norm.get(cid, 0.0),
            embedding_score=emb_norm.get(cid, 0.0),
        )
        for cid in all_ids
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:fused_top_k]
