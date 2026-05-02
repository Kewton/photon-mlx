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

from collections.abc import Sequence

from .hybrid import RetrievalResult
from ..ingestion.store import ChunkStore

# Mirror of configs/baseline.yaml retrieval.reranker.noise_patterns
# (FastAPI default). Update both when modifying.
_NOISE_PATTERNS: tuple[str, ...] = (
    "llm-prompt",
    "sponsors.yml",
    "language_names",
    "DISCUSSION_TEMPLATE",
    "general-llm-prompt",
)


def _format_passage_for_rerank(chunk: object, max_chars: int = 600) -> str:
    """Include stable document metadata in the reranker passage.

    BM25/embedding indexes already include path and section metadata, but the
    cross-encoder reranker used only raw content.  That lets generic chunks
    such as "required documents" outrank the correct numbered form when the
    discriminating signal lives in the filename or heading.  The format stays
    domain-agnostic: path, section, then body.
    """
    rel_path = str(getattr(chunk, "rel_path", "") or "")
    section = str(getattr(chunk, "section_header", "") or "")
    content = str(getattr(chunk, "content", "") or "")
    parts: list[str] = []
    if rel_path:
        parts.append(f"Document path: {rel_path}")
    if section:
        parts.append(f"Section: {section}")
    parts.append("Content:")
    parts.append(content[:max_chars])
    return "\n".join(parts)


class CrossEncoderReranker:
    """Wraps sentence-transformers CrossEncoder for passage reranking.

    The model is loaded once at construction time.  ``rerank()`` is
    the only hot-path method (~50 ms for 16 candidates after warmup).

    Args:
        model_id: HuggingFace model ID.  Defaults to the lightweight
            ms-marco-MiniLM-L-6-v2 (22 M params).
        max_length: Tokenizer truncation.  256 covers most code chunks.
        noise_patterns: File path patterns to filter out pre-ranking.
            - None (default): use built-in `_NOISE_PATTERNS` (backward compat).
            - Sequence of str: use as-is (empty sequence → noise filter
              explicit disable).
    """

    def __init__(
        self,
        model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 256,
        noise_patterns: Sequence[str] | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder  # lazy import

        self._model = CrossEncoder(model_id, max_length=max_length)
        self._noise_patterns: tuple[str, ...] = (
            _NOISE_PATTERNS if noise_patterns is None else tuple(noise_patterns)
        )

    def _is_noise(self, chunk_id: str) -> bool:
        path = chunk_id.split("::", 1)[1] if "::" in chunk_id else chunk_id
        return any(p in path for p in self._noise_patterns)

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
        clean = [r for r in results if not self._is_noise(r.chunk_id)]
        if not clean:
            clean = results  # safety fallback

        # Stage 2: cross-encoder scoring
        # Use English expansion terms when the original query is non-ASCII;
        # the cross-encoder was trained on English and degrades on Japanese.
        scoring_query = rerank_query if rerank_query else query

        chunks = store.get_many([r.chunk_id for r in clean])
        content_map = {
            c.chunk_id: _format_passage_for_rerank(c)
            for c in chunks
        }

        pairs = [(scoring_query, content_map.get(r.chunk_id, "")) for r in clean]
        scores: list[float] = self._model.predict(pairs).tolist()

        reranked = sorted(
            zip(clean, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        # DR1-003: preserve fused ``score``; write cross-encoder score to
        # ``reranker_score`` only.  Existing callers that read ``.score``
        # after rerank() now get the fused score, not the cross-encoder score.
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                score=r.score,
                lexical_score=r.lexical_score,
                embedding_score=r.embedding_score,
                reranker_score=float(s),
            )
            for r, s in reranked[:top_k]
        ]

    def rerank_with_debug(
        self,
        query: str,
        results: list[RetrievalResult],
        store: ChunkStore,
        top_k: int,
        rerank_query: str | None = None,
        rejected_debug_top_n: int = 10,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        """Rerank and return (top_k_rows, rejected_rows) each with reranker_score.

        Runs cross-encoder on ``top_k + rejected_debug_top_n`` candidates so
        that rejected rows also carry their cross-encoder score in
        ``reranker_score`` (Issue #176 AC-4).

        Existing ``rerank()`` is left unchanged for non-debug callers.
        """
        if not results:
            return [], []

        clean = [r for r in results if not self._is_noise(r.chunk_id)]
        if not clean:
            clean = results

        scoring_query = rerank_query if rerank_query else query

        chunks = store.get_many([r.chunk_id for r in clean])
        content_map = {
            c.chunk_id: _format_passage_for_rerank(c)
            for c in chunks
        }

        pairs = [(scoring_query, content_map.get(r.chunk_id, "")) for r in clean]
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(
            zip(clean, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        def _with_reranker_score(r: RetrievalResult, s: float) -> RetrievalResult:
            return RetrievalResult(
                chunk_id=r.chunk_id,
                score=r.score,
                lexical_score=r.lexical_score,
                embedding_score=r.embedding_score,
                reranker_score=float(s),
            )

        top_k_rows = [_with_reranker_score(r, s) for r, s in ranked[:top_k]]
        rejected_rows = [
            _with_reranker_score(r, s)
            for r, s in ranked[top_k : top_k + rejected_debug_top_n]
        ]
        return top_k_rows, rejected_rows
