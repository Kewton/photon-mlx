from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator

from .chunker import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT PRIMARY KEY,
    repo_id        TEXT NOT NULL,
    repo_commit    TEXT NOT NULL,
    rel_path       TEXT NOT NULL,
    language       TEXT NOT NULL,
    start_line     INTEGER NOT NULL,
    end_line       INTEGER NOT NULL,
    content        TEXT NOT NULL,
    symbols        TEXT NOT NULL,
    section_header TEXT NOT NULL,
    file_header    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_repo    ON chunks(repo_id, repo_commit);
CREATE INDEX IF NOT EXISTS idx_path    ON chunks(repo_id, repo_commit, rel_path);
"""


def _row_to_chunk(row: tuple) -> Chunk:
    (chunk_id, repo_id, repo_commit, rel_path, language,
     start_line, end_line, content, symbols_json,
     section_header, file_header) = row
    return Chunk(
        chunk_id=chunk_id, repo_id=repo_id, repo_commit=repo_commit,
        rel_path=rel_path, language=language,
        start_line=start_line, end_line=end_line,
        content=content, symbols=json.loads(symbols_json),
        section_header=section_header, file_header=file_header,
    )


class ChunkStore:
    """SQLite-backed persistent chunk store."""

    def __init__(self, db_path: str | Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, chunk: Chunk) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO chunks
              (chunk_id, repo_id, repo_commit, rel_path, language,
               start_line, end_line, content, symbols, section_header, file_header)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (chunk.chunk_id, chunk.repo_id, chunk.repo_commit,
             chunk.rel_path, chunk.language,
             chunk.start_line, chunk.end_line,
             chunk.content, json.dumps(chunk.symbols),
             chunk.section_header, chunk.file_header),
        )

    def commit(self) -> None:
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        # Preserve input order
        by_id = {r[0]: _row_to_chunk(r) for r in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def iter_repo(self, repo_id: str, repo_commit: str) -> Iterator[Chunk]:
        rows = self._conn.execute(
            """SELECT * FROM chunks
               WHERE repo_id = ? AND repo_commit = ?
               ORDER BY rel_path, start_line""",
            (repo_id, repo_commit),
        )
        for row in rows:
            yield _row_to_chunk(row)

    def count(self, repo_id: str, repo_commit: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE repo_id = ? AND repo_commit = ?",
            (repo_id, repo_commit),
        ).fetchone()[0]

    def get_neighbors(self, chunk: Chunk, before: int, after: int) -> list[Chunk]:
        """Return the chunks immediately before and after `chunk` in the same file."""
        rows = self._conn.execute(
            """SELECT * FROM chunks
               WHERE repo_id = ? AND repo_commit = ? AND rel_path = ?
               ORDER BY start_line""",
            (chunk.repo_id, chunk.repo_commit, chunk.rel_path),
        ).fetchall()
        chunks = [_row_to_chunk(r) for r in rows]
        try:
            idx = next(i for i, c in enumerate(chunks) if c.chunk_id == chunk.chunk_id)
        except StopIteration:
            return []
        lo = max(0, idx - before)
        hi = min(len(chunks), idx + after + 1)
        return [c for c in chunks[lo:hi] if c.chunk_id != chunk.chunk_id]

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
