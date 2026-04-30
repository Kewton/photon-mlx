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
    reranker_score: float | None = None


def _normalize(pairs: list[tuple[str, float]]) -> dict[str, float]:
    if not pairs:
        return {}
    max_s = max(s for _, s in pairs) or 1.0
    return {cid: s / max_s for cid, s in pairs}


def _belongs_to_repo(chunk_id: str, repo_id: str) -> bool:
    """Return True iff ``chunk_id`` was indexed under ``repo_id``.

    chunk_id format is ``{repo_id}::{rel_path}::{lines}``; we use the
    ``::`` boundary so that ``repo_id="foo"`` does not match ``foobar::``.
    """
    return chunk_id.startswith(f"{repo_id}::")


def _search_one(
    query: str,
    lexical_index: LexicalIndex,
    embedding_index: EmbeddingIndex,
    lexical_top_k: int,
    embedding_top_k: int,
    lexical_weight: float,
    embedding_weight: float,
    repo_id: str = "",
) -> dict[str, RetrievalResult]:
    """Run hybrid search for a single query; returns a chunk_id → result map.

    When ``repo_id`` is non-empty, results whose ``chunk_id`` does not begin
    with ``{repo_id}::`` are dropped before fusion (Issue #154 Bug 1: prevents
    cross-repo chunk leakage when an in-memory index ever contains chunks
    from more than one repo).
    """
    lex_raw = lexical_index.search(query, top_k=lexical_top_k)
    emb_raw = embedding_index.search(query, top_k=embedding_top_k)

    if repo_id:
        lex_raw = [r for r in lex_raw if _belongs_to_repo(r.chunk_id, repo_id)]
        emb_raw = [r for r in emb_raw if _belongs_to_repo(r.chunk_id, repo_id)]

    lex_norm = _normalize([(r.chunk_id, r.score) for r in lex_raw])
    emb_norm = _normalize([(r.chunk_id, r.score) for r in emb_raw])

    all_ids = set(lex_norm) | set(emb_norm)
    return {
        cid: RetrievalResult(
            chunk_id=cid,
            score=lexical_weight * lex_norm.get(cid, 0.0)
            + embedding_weight * emb_norm.get(cid, 0.0),
            lexical_score=lex_norm.get(cid, 0.0),
            embedding_score=emb_norm.get(cid, 0.0),
        )
        for cid in all_ids
    }


def hybrid_search(
    query: str,
    lexical_index: LexicalIndex,
    embedding_index: EmbeddingIndex,
    lexical_top_k: int = 20,
    embedding_top_k: int = 20,
    fused_top_k: int = 16,
    lexical_weight: float = 0.45,
    embedding_weight: float = 0.45,
    expanded_queries: list[str] | None = None,
    repo_id: str = "",
) -> list[RetrievalResult]:
    """Hybrid BM25 + embedding search with optional multi-query expansion.

    When *expanded_queries* is provided (non-empty list of additional query
    strings), each query is searched independently and the results are merged
    by taking the **max score** across queries for each chunk.  This broadens
    recall without penalising chunks that only match one query variant.

    When *repo_id* is provided, results whose ``chunk_id`` is not prefixed
    with ``{repo_id}::`` are dropped — see ``_search_one`` (Issue #154).
    """
    # Primary query
    merged: dict[str, RetrievalResult] = _search_one(
        query,
        lexical_index,
        embedding_index,
        lexical_top_k,
        embedding_top_k,
        lexical_weight,
        embedding_weight,
        repo_id=repo_id,
    )

    # Expanded queries — merge by max score
    for eq in expanded_queries or []:
        if not eq or eq == query:
            continue
        extra = _search_one(
            eq,
            lexical_index,
            embedding_index,
            lexical_top_k,
            embedding_top_k,
            lexical_weight,
            embedding_weight,
            repo_id=repo_id,
        )
        for cid, res in extra.items():
            if cid not in merged or res.score > merged[cid].score:
                merged[cid] = res

    results = sorted(merged.values(), key=lambda r: r.score, reverse=True)
    return results[:fused_top_k]


def _extract_extension(chunk_id: str) -> str:
    """Extract file extension from a chunk_id of the form ``repo::path::span``.

    >>> _extract_extension("repo::src/app.py::0-120")
    '.py'
    >>> _extract_extension("repo::README.md::0-50")
    '.md'
    >>> _extract_extension("repo::Makefile::0-10")
    ''
    """
    parts = chunk_id.split("::")
    if len(parts) >= 2:
        path = parts[1]
        dot = path.rfind(".")
        if dot != -1:
            return path[dot:]
    return ""


def apply_file_type_boost(
    results: list[RetrievalResult],
    boost: float = 0.0,
) -> list[RetrievalResult]:
    """Add a score bonus to ``.py`` chunks and re-sort.

    Designed to run **after** cross-encoder reranking (which overwrites
    the hybrid score) so that implementation files rank higher than
    docs/config files.  When *boost* is ``0.0`` the list is returned
    unchanged (no copy, no re-sort) to preserve backward-compatibility.
    """
    if boost == 0.0:
        return results
    boosted = []
    for r in results:
        ext = _extract_extension(r.chunk_id)
        new_score = r.score + boost if ext == ".py" else r.score
        boosted.append(
            RetrievalResult(
                chunk_id=r.chunk_id,
                score=new_score,
                lexical_score=r.lexical_score,
                embedding_score=r.embedding_score,
                reranker_score=r.reranker_score,
            )
        )
    return sorted(boosted, key=lambda x: x.score, reverse=True)
