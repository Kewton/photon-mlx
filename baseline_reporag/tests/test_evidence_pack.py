from __future__ import annotations

from unittest.mock import MagicMock

from baseline_reporag.generation.evidence_pack import (
    EvidencePack,
    build_evidence_pack,
)
from baseline_reporag.ingestion.chunker import Chunk
from baseline_reporag.memory.session import SessionState


def _make_chunk(
    chunk_id: str,
    rel_path: str = "app/main.py",
    content: str = "def hello(): pass",
    start_line: int = 1,
    end_line: int = 10,
    section_header: str = "",
) -> Chunk:
    """テスト用の Chunk を作成する。"""
    return Chunk(
        chunk_id=chunk_id,
        repo_id="test_repo",
        repo_commit="abc123",
        rel_path=rel_path,
        language="python",
        start_line=start_line,
        end_line=end_line,
        content=content,
        symbols=["hello"],
        section_header=section_header,
        file_header="# app/main.py",
    )


class TestFormatForPrompt:
    """EvidencePack.format_for_prompt() のフォーマット検証。"""

    def test_format_for_prompt_basic(self) -> None:
        """チャンクが正しくフォーマットされること。"""
        chunk = _make_chunk("c1", content="def main(): pass")
        pack = EvidencePack(chunks=[chunk], chunk_indices={"c1": 1})
        result = pack.format_for_prompt()
        assert "[C:1]" in result
        assert "app/main.py" in result
        assert "def main(): pass" in result

    def test_format_for_prompt_indices(self) -> None:
        """インデックスが 1-based で正確であること。"""
        chunks = [
            _make_chunk("c1", content="first"),
            _make_chunk("c2", content="second"),
            _make_chunk("c3", content="third"),
        ]
        indices = {"c1": 1, "c2": 2, "c3": 3}
        pack = EvidencePack(chunks=chunks, chunk_indices=indices)
        result = pack.format_for_prompt()
        assert "[C:1]" in result
        assert "[C:2]" in result
        assert "[C:3]" in result
        # Verify order: C:1 appears before C:2 before C:3
        assert result.index("[C:1]") < result.index("[C:2]")
        assert result.index("[C:2]") < result.index("[C:3]")

    def test_format_for_prompt_with_section_header(self) -> None:
        """section_header 付きチャンクが正しくフォーマットされること。"""
        chunk = _make_chunk(
            "c1",
            content="class MyRouter: pass",
            section_header="MyRouter",
        )
        pack = EvidencePack(chunks=[chunk], chunk_indices={"c1": 1})
        result = pack.format_for_prompt()
        assert "[MyRouter]" in result

    def test_format_for_prompt_empty_chunks(self) -> None:
        """空チャンクリストでエラーにならないこと。"""
        pack = EvidencePack(chunks=[], chunk_indices={})
        result = pack.format_for_prompt()
        assert result == ""


class TestBuildEvidencePackAdditionalPinnedIds:
    """Issue #103: ``additional_pinned_ids`` kw-only arg on
    :func:`build_evidence_pack`.

    Contract (design §4-2, DR2-003 / DR2-009):

    * Default ``None`` keeps existing call sites byte-identical (priority
      ordering / max_chunks / dedup all unchanged).
    * When provided, IDs are merged with ``session.pinned_chunk_ids`` and
      promoted to ``priority=0`` (above ``recent_cited`` priority=1).
    * Dedup is delegated to ``set(chunk_ids)`` and ``_merge_pinned_sets``;
      callers passing duplicates do not see double-counting.
    """

    def _make_store(self, chunks: list[Chunk]) -> object:
        store = MagicMock()
        by_id = {c.chunk_id: c for c in chunks}

        def _get_many(ids):
            return [by_id[cid] for cid in ids if cid in by_id]

        store.get_many.side_effect = _get_many
        return store

    def test_evidence_pack_additional_pinned_ids_default_none_preserves_legacy(
        self,
    ) -> None:
        """Omitting ``additional_pinned_ids`` must produce the identical
        ordering, indices, and chunk selection as the pre-#103 contract."""
        chunks = [_make_chunk(f"c{i}", content=f"chunk_{i}_body") for i in range(4)]
        store = self._make_store(chunks)
        session = SessionState(session_id="s1", repo_id="r", repo_commit="abc")
        pack_legacy = build_evidence_pack(
            chunk_ids=["c0", "c1", "c2", "c3"],
            store=store,
            session=session,
            max_chunks=4,
            max_tokens=16000,
            recent_citation_turns=2,
        )
        pack_new = build_evidence_pack(
            chunk_ids=["c0", "c1", "c2", "c3"],
            store=store,
            session=session,
            max_chunks=4,
            max_tokens=16000,
            recent_citation_turns=2,
            additional_pinned_ids=None,
        )
        assert [c.chunk_id for c in pack_legacy.chunks] == [
            c.chunk_id for c in pack_new.chunks
        ]
        assert pack_legacy.chunk_indices == pack_new.chunk_indices

    def test_evidence_pack_additional_pinned_ids_promoted_to_priority_zero(
        self,
    ) -> None:
        """``additional_pinned_ids`` must surface BEFORE ``recent_cited``
        chunks (priority 0 < priority 1)."""
        # 5 candidate chunks; max_chunks=2 so only the top-2 priority chunks
        # survive. Without pinning, c1 (recent_cited) would lead. With
        # pinning, c4 (additional pin) must outrank c1.
        chunks = [_make_chunk(f"c{i}", content=f"chunk_{i}_body") for i in range(5)]
        store = self._make_store(chunks)
        session = SessionState(session_id="s1", repo_id="r", repo_commit="abc")
        # Make c1 recently cited (priority=1).
        session.add_turn(
            question="prior question",
            answer="prior answer",
            cited_chunk_ids=["c1"],
        )
        pack = build_evidence_pack(
            chunk_ids=["c0", "c1", "c2", "c3", "c4"],
            store=store,
            session=session,
            max_chunks=2,
            max_tokens=16000,
            recent_citation_turns=2,
            additional_pinned_ids=["c4"],
        )
        ids = [c.chunk_id for c in pack.chunks]
        assert "c4" in ids, f"pinned chunk c4 must appear in pack: {ids!r}"
        # c4 (priority 0) must precede c1 (priority 1).
        assert ids.index("c4") < ids.index("c1") if "c1" in ids else True
        # Defensive: c4 should be the FIRST chunk because it is the only
        # priority-0 entry.
        assert ids[0] == "c4", f"expected c4 first, got {ids!r}"

    def test_evidence_pack_additional_pinned_ids_added_when_not_in_chunk_ids(
        self,
    ) -> None:
        """Issue #103 / CB-001 regression: ``additional_pinned_ids`` whose
        IDs are NOT already in ``chunk_ids`` MUST be added to the pack
        (rather than silently dropped). This is the entire purpose of the
        pinning channel — to rescue past-turn chunks that the current
        retrieval missed.

        Pre-fix bug: ``sorted(set(chunk_ids), ...)`` filters by
        ``set(chunk_ids)`` so any pinned ID outside that set is excluded
        even though ``priority()`` reports priority 0 for it. The fix
        unions ``additional_pinned_ids`` into the candidate set BEFORE
        the ``sorted(...)`` call.
        """
        # ``chunk_ids=['c0','c1']`` (current retrieval) does NOT contain
        # ``c4``. ``additional_pinned_ids=['c4']`` must still surface c4
        # in the resulting pack.
        chunks = [_make_chunk(f"c{i}", content=f"chunk_{i}_body") for i in range(5)]
        store = self._make_store(chunks)
        session = SessionState(session_id="s1", repo_id="r", repo_commit="abc")
        pack = build_evidence_pack(
            chunk_ids=["c0", "c1"],
            store=store,
            session=session,
            max_chunks=4,
            max_tokens=16000,
            recent_citation_turns=2,
            additional_pinned_ids=["c4"],
        )
        ids = [c.chunk_id for c in pack.chunks]
        assert "c4" in ids, (
            f"pinned chunk c4 (NOT in chunk_ids) must be added to pack: {ids!r}"
        )
        # c4 (priority 0) must precede c0/c1 (priority 2).
        assert ids[0] == "c4", f"expected c4 first, got {ids!r}"
        # Confirm store.get_many was called with c4 in the requested IDs.
        call_args = store.get_many.call_args_list
        requested_ids: list[str] = []
        for call in call_args:
            args, kwargs = call
            if args:
                requested_ids.extend(args[0])
        assert "c4" in requested_ids, (
            f"store.get_many must be invoked with c4 in the ID list: {requested_ids!r}"
        )
