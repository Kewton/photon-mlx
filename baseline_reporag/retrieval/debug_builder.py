"""Shared retrieval debug row builder for Issue #176.

Used by both baseline pipeline.py and photon_pipeline.py to avoid DRY
violations in retrieval_debug construction logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts import RetrievalDebugRow

if TYPE_CHECKING:
    from ..ingestion.store import ChunkStore
    from .graph_expansion import ExpandedChunkRef
    from .hybrid import RetrievalResult


def build_retrieval_debug_rows(
    raw_snapshot: list[RetrievalResult],
    reranked_top: list[RetrievalResult],
    rejected: list[RetrievalResult],
    expanded_refs: list[ExpandedChunkRef],
    store: ChunkStore,
    photon_scores: dict[str, float] | None = None,
    photon_current_scores: dict[str, float] | None = None,
    photon_session_scores: dict[str, float] | None = None,
) -> list[RetrievalDebugRow]:
    """Build skeleton RetrievalDebugRow list from retrieval pipeline outputs.

    used / citation_index are provisional (False / None) and must be
    finalised by calling finalise_retrieval_debug() after build_evidence_pack
    and resolve_citations complete.

    Args:
        raw_snapshot: RetrievalResult list captured before reranking (has
            lexical_score, embedding_score, fused score in .score).
        reranked_top: top-k rows returned by rerank_with_debug.
        rejected: rejected rows returned by rerank_with_debug (score = None
            if cross-encoder was not run on them).
        expanded_refs: ExpandedChunkRef list from expand_with_graph.
        store: ChunkStore for fetching rel_path / section metadata.
        photon_scores: Optional PHOTON pruning score by chunk_id.
        photon_current_scores: Optional score against the current question only.
        photon_session_scores: Optional score against the existing session state.
    """
    # Build score lookup from pre-rerank snapshot
    snapshot_map: dict[str, RetrievalResult] = {r.chunk_id: r for r in raw_snapshot}
    # reranker_score from reranked rows
    reranker_score_map: dict[str, float] = {
        r.chunk_id: r.reranker_score
        for r in reranked_top + rejected
        if r.reranker_score is not None
    }
    # source from expanded_refs
    source_map: dict[str, str] = {ref.chunk_id: ref.source for ref in expanded_refs}
    # All chunk IDs in order
    all_ids = [ref.chunk_id for ref in expanded_refs]

    # Fetch metadata from store
    chunks = store.get_many(all_ids)
    meta_map = {c.chunk_id: c for c in chunks}

    rows: list[RetrievalDebugRow] = []
    for cid in all_ids:
        chunk = meta_map.get(cid)
        raw = snapshot_map.get(cid)
        source = source_map.get(cid, "retrieval")

        if chunk is not None:
            rel_path = chunk.rel_path
            section = getattr(chunk, "section_header", None)
        else:
            rel_path = cid.split("::", 2)[1] if cid.count("::") >= 2 else cid
            section = None

        if raw is not None and source == "retrieval":
            bm25 = raw.lexical_score
            emb = raw.embedding_score
            fused = raw.score
        else:
            bm25 = None
            emb = None
            fused = None

        rows.append(
            RetrievalDebugRow(
                chunk_id=cid,
                rel_path=rel_path,
                section=section,
                bm25_score=bm25,
                embedding_score=emb,
                fused_score=fused,
                reranker_score=reranker_score_map.get(cid),
                used=False,
                citation_index=None,
                source=source,
                photon_score=(photon_scores or {}).get(cid),
                photon_current_score=(photon_current_scores or {}).get(cid),
                photon_session_score=(photon_session_scores or {}).get(cid),
            )
        )

    # Append rejected rows that are not already in expanded_refs
    expanded_ids = set(all_ids)
    for r in rejected:
        if r.chunk_id in expanded_ids:
            continue
        raw = snapshot_map.get(r.chunk_id)
        chunk = store.get_many([r.chunk_id])
        chunk = chunk[0] if chunk else None
        rel_path = (
            chunk.rel_path
            if chunk
            else r.chunk_id.split("::", 2)[1]
            if r.chunk_id.count("::") >= 2
            else r.chunk_id
        )
        section = getattr(chunk, "section_header", None) if chunk else None
        rows.append(
            RetrievalDebugRow(
                chunk_id=r.chunk_id,
                rel_path=rel_path,
                section=section,
                bm25_score=raw.lexical_score if raw else None,
                embedding_score=raw.embedding_score if raw else None,
                fused_score=raw.score if raw else None,
                reranker_score=r.reranker_score,
                used=False,
                citation_index=None,
                source="retrieval",
                photon_score=(photon_scores or {}).get(r.chunk_id),
                photon_current_score=(photon_current_scores or {}).get(r.chunk_id),
                photon_session_score=(photon_session_scores or {}).get(r.chunk_id),
            )
        )

    return rows


def finalise_retrieval_debug(
    rows: list[RetrievalDebugRow],
    pack_chunk_ids: list[str],
    cited_chunk_ids: list[str],
) -> list[RetrievalDebugRow]:
    """Finalise used / citation_index after build_evidence_pack and resolve_citations.

    Args:
        rows: skeleton rows from build_retrieval_debug_rows (or extended with
              PHOTON-specific rows).
        pack_chunk_ids: ordered chunk_ids in the evidence pack (determines
              citation_index = 1-based position).
        cited_chunk_ids: chunk_ids that appear as [C:n] in the answer.
    """
    pack_index: dict[str, int] = {cid: i + 1 for i, cid in enumerate(pack_chunk_ids)}
    cited_set: set[str] = set(cited_chunk_ids)

    finalised: list[RetrievalDebugRow] = []
    for row in rows:
        idx = pack_index.get(row.chunk_id)
        used = idx is not None
        citation_index = idx if row.chunk_id in cited_set else None
        finalised.append(
            RetrievalDebugRow(
                chunk_id=row.chunk_id,
                rel_path=row.rel_path,
                section=row.section,
                bm25_score=row.bm25_score,
                embedding_score=row.embedding_score,
                fused_score=row.fused_score,
                reranker_score=row.reranker_score,
                used=used,
                citation_index=citation_index,
                source=row.source,
                photon_score=row.photon_score,
                photon_current_score=row.photon_current_score,
                photon_session_score=row.photon_session_score,
            )
        )
    return finalised
