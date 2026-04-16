from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Turn:
    turn_id: int
    question: str
    answer: str
    cited_chunk_ids: list[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionState:
    session_id: str
    repo_id: str
    repo_commit: str
    turns: list[Turn] = field(default_factory=list)
    pinned_chunk_ids: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)  # cumulative

    def add_turn(
        self,
        question: str,
        answer: str,
        cited_chunk_ids: list[str],
    ) -> Turn:
        turn = Turn(
            turn_id=len(self.turns) + 1,
            question=question,
            answer=answer,
            cited_chunk_ids=cited_chunk_ids,
        )
        self.turns.append(turn)
        for cid in cited_chunk_ids:
            if cid not in self.cited_chunk_ids:
                self.cited_chunk_ids.append(cid)
        return turn

    def recent_history(self, max_turns: int = 4) -> list[Turn]:
        return self.turns[-max_turns:]

    def history_text(self, max_turns: int = 4) -> str:
        lines: list[str] = []
        for t in self.recent_history(max_turns):
            lines.append(f"Q{t.turn_id}: {t.question}")
            # Strip [C:N] markers from history: citation indices are local to each
            # turn's evidence pack and must not bleed into subsequent turns where
            # the pack differs, which would cause wrong_citation_indices errors.
            answer_stripped = re.sub(r"\[C:\d+\]", "", t.answer).strip()
            answer_preview = (
                answer_stripped[:400] + "..."
                if len(answer_stripped) > 400
                else answer_stripped
            )
            lines.append(f"A{t.turn_id}: {answer_preview}")
        return "\n".join(lines)


class SessionManager:
    def __init__(self, log_dir: str | Path | None = None) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._log_dir = Path(log_dir) if log_dir else None
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create(
        self,
        session_id: str,
        repo_id: str,
        repo_commit: str,
    ) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(
                session_id=session_id,
                repo_id=repo_id,
                repo_commit=repo_commit,
            )
        return self._sessions[session_id]

    def save(self, session: SessionState) -> None:
        if not self._log_dir:
            return
        path = self._log_dir / f"session_{session.session_id}.json"
        data = {
            "session_id": session.session_id,
            "repo_id": session.repo_id,
            "repo_commit": session.repo_commit,
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "question": t.question,
                    "answer": t.answer,
                    "cited_chunk_ids": t.cited_chunk_ids,
                    "timestamp": t.timestamp,
                }
                for t in session.turns
            ],
            "pinned_chunk_ids": session.pinned_chunk_ids,
            "cited_chunk_ids": session.cited_chunk_ids,
        }
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
