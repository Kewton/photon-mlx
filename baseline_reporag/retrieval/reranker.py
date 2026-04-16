"""Cross-encoder reranker for post-retrieval re-ranking.

Two-stage approach:
1. Noise filter: remove known irrelevant file patterns that BM25/embedding
   rank highly due to surface-term overlap (llm-prompt.md, sponsors.yml,
   DISCUSSION_TEMPLATE, etc.).
2. Cross-encoder rerank: score remaining candidates by query-passage
   relevance using ms-marco-MiniLM-L-6-v2.  Uses English expansion terms
   as the reranking query when available, since the model is trained on
   English MS MARCO data.
"""

from __future__ import annotations

from .hybrid import RetrievalResult
from ..ingestion.store import ChunkStore

# File path patterns that are never useful for code analysis questions.
# Verified against FastAPI repo: these are meta-documents (LLM translation
# prompts, sponsor lists, GitHub templates) that BM25 ranks high due to
# ubiquitous FastAPI terminology.
_NOISE_PATTERNS: tuple[str, ...] = (
    "llm-prompt",
    "sponsors.yml",
    "language_names",
    "DISCUSSION_TEMPLATE",
    "general-llm-prompt",
)


def _is_noise(chunk_id: str) -> bool:
    path = chunk_id.split("::", 1)[1] if "::" in chunk_id else chunk_id
    return any(p in path for p in _NOISE_PATTERNS)


class CrossEncoderReranker:
    """Wraps sentence-transformers CrossEncoder for passage reranking.

    The model is loaded once at construction time.  ``rerank()`` is
    the only hot-path method (~50 ms for 16 candidates after warmup).

    Args:
        model_id: HuggingFace model ID.  Defaults to the lightweight
            ms-marco-MiniLM-L-6-v2 (22 M params).
        max_length: Tokenizer truncation.  256 covers most code chunks.
    """

    def __init__(
        self,
        model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 256,
    ) -> None:
        from sentence_transformers import CrossEncoder  # lazy import

        self._model = CrossEncoder(model_id, max_length=max_length)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        store: ChunkStore,
        top_k: int,
        rerank_query: str | None = None,
    ) -> list[RetrievalResult]:
        """Filter noise and re-rank by cross-encoder relevance.

        Steps:
        1. Remove chunks matching ``_NOISE_PATTERNS``.
        2. Score remaining candidates using ``rerank_query`` (English
           expansion terms) if provided, else ``query``.
        3. Return top ``top_k`` by cross-encoder score.

        If noise filtering removes all candidates, scoring falls back to
        the full unfiltered list to avoid empty results.
        """
        if not results:
            return results

        # Stage 1: noise filter
        clean = [r for r in results if not _is_noise(r.chunk_id)]
        if not clean:
            clean = results  # safety fallback

        # Stage 2: cross-encoder scoring
        # Use English expansion terms when the original query is non-ASCII;
        # the cross-encoder was trained on English and degrades on Japanese.
        scoring_query = rerank_query if rerank_query else query

        chunks = store.get_many([r.chunk_id for r in clean])
        content_map = {c.chunk_id: c.content[:600] for c in chunks}

        pairs = [(scoring_query, content_map.get(r.chunk_id, "")) for r in clean]
        scores: list[float] = self._model.predict(pairs).tolist()

        reranked = sorted(
            zip(clean, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                score=float(s),
                lexical_score=r.lexical_score,
                embedding_score=r.embedding_score,
            )
            for r, s in reranked[:top_k]
        ]
