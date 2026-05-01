"""Tests for debug_builder.py (Issue #176 Phase B-1).

RED state: fails because
  - RetrievalResult has no reranker_score field (added in Phase C-1)
  - ExpandedChunkRef does not exist in graph_expansion (added in Phase C-3)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from baseline_reporag.contracts import RetrievalDebugRow
from baseline_reporag.retrieval.debug_builder import (
    build_retrieval_debug_rows,
    finalise_retrieval_debug,
)
from baseline_reporag.retrieval.graph_expansion import ExpandedChunkRef  # noqa: F401
from baseline_reporag.retrieval.hybrid import RetrievalResult


def _make_result(
    chunk_id: str,
    bm25: float = 0.5,
    emb: float = 0.6,
    fused: float = 0.55,
    reranker: float | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=fused,
        lexical_score=bm25,
        embedding_score=emb,
        reranker_score=reranker,
    )


def _make_ref(chunk_id: str, source: str) -> ExpandedChunkRef:
    return ExpandedChunkRef(chunk_id=chunk_id, source=source)


def _make_store(*chunk_ids: str) -> MagicMock:
    store = MagicMock()
    chunks = []
    for cid in chunk_ids:
        chunk = MagicMock()
        chunk.chunk_id = cid
        chunk.rel_path = f"src/{cid}.py"
        chunk.section_header = f"section_{cid}"
        chunks.append(chunk)

    def _get_many(ids: list[str]) -> list:
        return [c for c in chunks if c.chunk_id in ids]

    store.get_many.side_effect = _get_many
    return store


class TestBuildRetrievalDebugRows:
    def test_retrieval_source_scores_mapped(self) -> None:
        """retrieval-source chunk gets bm25/embedding/fused scores from raw_snapshot."""
        raw = [_make_result("c0", bm25=0.8, emb=0.7, fused=0.75)]
        reranked = [_make_result("c0", bm25=0.8, emb=0.7, fused=0.75, reranker=0.9)]
        rejected: list[RetrievalResult] = []
        refs = [_make_ref("c0", "retrieval")]
        store = _make_store("c0")

        rows = build_retrieval_debug_rows(raw, reranked, rejected, refs, store)

        assert len(rows) == 1
        row = rows[0]
        assert row.chunk_id == "c0"
        assert row.source == "retrieval"
        assert row.bm25_score == pytest.approx(0.8)
        assert row.embedding_score == pytest.approx(0.7)
        assert row.fused_score == pytest.approx(0.75)
        assert row.reranker_score == pytest.approx(0.9)
        assert row.used is False
        assert row.citation_index is None

    def test_graph_source_scores_are_none(self) -> None:
        """graph-source chunk has None bm25/embedding/fused scores."""
        raw: list[RetrievalResult] = []
        reranked: list[RetrievalResult] = []
        rejected: list[RetrievalResult] = []
        refs = [_make_ref("c_graph", "graph")]
        store = _make_store("c_graph")

        rows = build_retrieval_debug_rows(raw, reranked, rejected, refs, store)

        assert len(rows) == 1
        row = rows[0]
        assert row.chunk_id == "c_graph"
        assert row.source == "graph"
        assert row.bm25_score is None
        assert row.embedding_score is None
        assert row.fused_score is None
        assert row.reranker_score is None

    def test_photon_scores_are_attached_by_chunk_id(self) -> None:
        raw = [_make_result("c0", bm25=0.8, emb=0.7, fused=0.75)]
        reranked: list[RetrievalResult] = []
        rejected: list[RetrievalResult] = []
        refs = [_make_ref("c0", "retrieval"), _make_ref("c1", "photon_pruned")]
        store = _make_store("c0", "c1")

        rows = build_retrieval_debug_rows(
            raw,
            reranked,
            rejected,
            refs,
            store,
            photon_scores={"c0": 0.42, "c1": -0.13},
        )
        finalised = finalise_retrieval_debug(
            rows,
            pack_chunk_ids=["c0"],
            cited_chunk_ids=["c0"],
        )

        assert finalised[0].photon_score == pytest.approx(0.42)
        assert finalised[1].photon_score == pytest.approx(-0.13)

    def test_neighbor_source_scores_are_none(self) -> None:
        """neighbor-source chunk has None bm25/embedding/fused scores."""
        raw: list[RetrievalResult] = []
        reranked: list[RetrievalResult] = []
        rejected: list[RetrievalResult] = []
        refs = [_make_ref("c_nb", "neighbor")]
        store = _make_store("c_nb")

        rows = build_retrieval_debug_rows(raw, reranked, rejected, refs, store)

        assert len(rows) == 1
        row = rows[0]
        assert row.source == "neighbor"
        assert row.bm25_score is None
        assert row.embedding_score is None
        assert row.fused_score is None

    def test_rejected_rows_appended(self) -> None:
        """Rejected chunks not in expanded_refs appear at end with reranker_score."""
        raw = [_make_result("r0", bm25=0.3, emb=0.4, fused=0.35)]
        reranked: list[RetrievalResult] = []
        rejected = [_make_result("r0", reranker=0.2)]
        refs: list[ExpandedChunkRef] = []  # r0 not in expanded
        store = _make_store("r0")

        rows = build_retrieval_debug_rows(raw, reranked, rejected, refs, store)

        assert any(row.chunk_id == "r0" for row in rows)
        r0_row = next(r for r in rows if r.chunk_id == "r0")
        assert r0_row.reranker_score == pytest.approx(0.2)
        assert r0_row.source == "retrieval"

    def test_rel_path_from_store(self) -> None:
        """rel_path is taken from the store chunk metadata."""
        raw = [_make_result("c0")]
        reranked = [_make_result("c0")]
        rejected: list[RetrievalResult] = []
        refs = [_make_ref("c0", "retrieval")]
        store = _make_store("c0")

        rows = build_retrieval_debug_rows(raw, reranked, rejected, refs, store)

        assert rows[0].rel_path == "src/c0.py"

    def test_section_from_store(self) -> None:
        """section is taken from chunk.section_header."""
        raw = [_make_result("c0")]
        reranked = [_make_result("c0")]
        rejected: list[RetrievalResult] = []
        refs = [_make_ref("c0", "retrieval")]
        store = _make_store("c0")

        rows = build_retrieval_debug_rows(raw, reranked, rejected, refs, store)

        assert rows[0].section == "section_c0"

    def test_missing_from_store_uses_chunk_id_fallback(self) -> None:
        """When store has no metadata, rel_path falls back to chunk_id parsing."""
        raw = [_make_result("repo::src/foo.py::1-10")]
        reranked = [_make_result("repo::src/foo.py::1-10")]
        rejected: list[RetrievalResult] = []
        refs = [_make_ref("repo::src/foo.py::1-10", "retrieval")]
        store = MagicMock()
        store.get_many.return_value = []  # nothing in store

        rows = build_retrieval_debug_rows(raw, reranked, rejected, refs, store)

        assert rows[0].rel_path == "src/foo.py"
        assert rows[0].section is None


class TestFinaliseRetrievalDebug:
    def _make_skeleton_rows(self) -> list[RetrievalDebugRow]:
        return [
            RetrievalDebugRow(
                chunk_id="c0",
                rel_path="a.py",
                section=None,
                bm25_score=0.8,
                embedding_score=0.7,
                fused_score=0.75,
                reranker_score=0.9,
                used=False,
                citation_index=None,
                source="retrieval",
            ),
            RetrievalDebugRow(
                chunk_id="c1",
                rel_path="b.py",
                section=None,
                bm25_score=0.5,
                embedding_score=0.4,
                fused_score=0.45,
                reranker_score=0.6,
                used=False,
                citation_index=None,
                source="retrieval",
            ),
            RetrievalDebugRow(
                chunk_id="c_rejected",
                rel_path="c.py",
                section=None,
                bm25_score=0.1,
                embedding_score=0.1,
                fused_score=0.1,
                reranker_score=0.15,
                used=False,
                citation_index=None,
                source="retrieval",
            ),
        ]

    def test_used_flag_set_for_pack_chunks(self) -> None:
        """used=True for chunks in pack_chunk_ids, False for others."""
        rows = self._make_skeleton_rows()
        finalised = finalise_retrieval_debug(
            rows=rows,
            pack_chunk_ids=["c0", "c1"],
            cited_chunk_ids=[],
        )

        used_ids = {r.chunk_id for r in finalised if r.used}
        assert used_ids == {"c0", "c1"}
        assert not any(r.used for r in finalised if r.chunk_id == "c_rejected")

    def test_citation_index_set_for_cited_chunks(self) -> None:
        """citation_index matches 1-based position in pack_chunk_ids for cited chunks."""
        rows = self._make_skeleton_rows()
        finalised = finalise_retrieval_debug(
            rows=rows,
            pack_chunk_ids=["c0", "c1"],
            cited_chunk_ids=["c1"],
        )

        c0 = next(r for r in finalised if r.chunk_id == "c0")
        c1 = next(r for r in finalised if r.chunk_id == "c1")

        assert c0.citation_index is None  # used but not cited
        assert c1.citation_index == 2  # 1-based position in pack

    def test_citation_index_none_for_non_pack_chunks(self) -> None:
        """Chunks not in pack have citation_index=None even if in cited_chunk_ids."""
        rows = self._make_skeleton_rows()
        finalised = finalise_retrieval_debug(
            rows=rows,
            pack_chunk_ids=["c0"],
            cited_chunk_ids=["c_rejected"],  # rejected chunk, not in pack
        )

        rejected_row = next(r for r in finalised if r.chunk_id == "c_rejected")
        assert rejected_row.citation_index is None

    def test_cited_derived_from_citation_index(self) -> None:
        """cited = (citation_index is not None) per design decision DR1-004."""
        rows = self._make_skeleton_rows()
        finalised = finalise_retrieval_debug(
            rows=rows,
            pack_chunk_ids=["c0", "c1"],
            cited_chunk_ids=["c0"],
        )

        for row in finalised:
            cited = row.citation_index is not None
            if row.chunk_id == "c0":
                assert cited is True
            else:
                assert cited is False

    def test_score_fields_preserved(self) -> None:
        """Score fields are not modified by finalise_retrieval_debug."""
        rows = self._make_skeleton_rows()
        finalised = finalise_retrieval_debug(
            rows=rows,
            pack_chunk_ids=["c0"],
            cited_chunk_ids=["c0"],
        )

        c0 = next(r for r in finalised if r.chunk_id == "c0")
        assert c0.bm25_score == pytest.approx(0.8)
        assert c0.embedding_score == pytest.approx(0.7)
        assert c0.fused_score == pytest.approx(0.75)
        assert c0.reranker_score == pytest.approx(0.9)

    def test_empty_pack_all_unused(self) -> None:
        """Empty pack_chunk_ids → all rows have used=False."""
        rows = self._make_skeleton_rows()
        finalised = finalise_retrieval_debug(
            rows=rows,
            pack_chunk_ids=[],
            cited_chunk_ids=[],
        )
        assert all(not r.used for r in finalised)
        assert all(r.citation_index is None for r in finalised)
