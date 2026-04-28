"""Regression test: ``ChunkStore`` MUST be usable from a thread other than
the one that created it.

Bug: Streamlit caches the built pipeline in ``st.session_state`` and reruns
the script on a fresh thread on every UI interaction.  The default
``sqlite3.connect`` enforces same-thread use and raises
``ProgrammingError`` on cross-thread reuse, breaking multi-turn chat
mid-conversation. We open with ``check_same_thread=False``.

Repro before the fix:
    sqlite3.ProgrammingError: SQLite objects created in a thread can only
    be used in that same thread. The object was created in thread id ...
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from baseline_reporag.ingestion.chunker import Chunk
from baseline_reporag.ingestion.store import ChunkStore


def _make_chunk(chunk_id: str = "test::doc.md::1-10") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        repo_id="testrepo",
        repo_commit="abc",
        rel_path="doc.md",
        language="markdown",
        start_line=1,
        end_line=10,
        section_header="",
        file_header="",
        content="hello",
        symbols=[],
    )


class TestChunkStoreCrossThread:
    def test_can_query_from_different_thread(self, tmp_path: Path) -> None:
        """ChunkStore created in main thread + queried from worker thread
        must NOT raise sqlite3.ProgrammingError. Reproduces the Streamlit
        rerun-on-different-thread case observed in chat UI."""
        store = ChunkStore(tmp_path / "x.db")
        store.upsert(_make_chunk())
        store.commit()

        results: list[tuple[str, object]] = []

        def worker():
            try:
                count = store.count("testrepo", "abc")
                results.append(("ok", count))
            except sqlite3.ProgrammingError as exc:
                results.append(("err", str(exc)))

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)

        assert results, "worker thread did not finish"
        assert results[0][0] == "ok", (
            f"cross-thread query raised: {results[0][1]} — "
            "ChunkStore must be opened with check_same_thread=False so "
            "Streamlit's cached pipeline survives reruns on new threads"
        )
        assert results[0][1] == 1

    def test_can_insert_from_different_thread(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path / "y.db")
        results: list[tuple[str, object]] = []

        def worker():
            try:
                store.upsert(_make_chunk("from_other::a::1-1"))
                store.commit()
                results.append(("ok", store.count("testrepo", "abc")))
            except sqlite3.ProgrammingError as exc:
                results.append(("err", str(exc)))

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)

        assert results and results[0][0] == "ok"
        # Read-back must also work from main thread (round-trip).
        assert store.count("testrepo", "abc") == 1
