"""Turn-history panel helpers for the Streamlit chat UI (Issue #82 Wave 3).

This module is intentionally streamlit-free. The rendering side lives in
``app/photon_app.page_chat`` and consumes the dict returned by
:func:`format_turn_history_panel`.

Design reference: `workspace/design/issue-82-app-photon-features-design-policy.md`
§6.3 turn_history_panel.

The panel joins two state sources by ``turn_id`` (int):

* ``photon_turn_history`` — entries of ``PhotonSessionState.turn_history``
  (``photon_mlx/session.py``). Each entry has ``turn_id``, ``question_text``
  and ``timestamp`` attributes.
* ``session_manager_turns`` — ``SessionState.turns`` from
  ``baseline_reporag/memory/session.py``. Each entry has ``turn_id`` and
  ``cited_chunk_ids``.

Missing matches on the SessionManager side resolve to an empty cited list;
this keeps the UI resilient when the two stores briefly drift across
fail-closed paths (see design §3-1, DR2-004 in photon_pipeline.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TurnRow:
    """One rendered row: everything needed for a single table entry."""

    turn_id: int
    question_text: str
    timestamp: str
    cited_chunk_ids: list[str]


def format_turn_history_panel(
    photon_turn_history: list[Any] | None,
    session_manager_turns: list[Any] | None,
    working_memory_enabled: bool,
    max_turns: int,
) -> dict[str, Any]:
    """Build a render-ready payload for the turn-history panel.

    Args:
        photon_turn_history: Entries from
            :attr:`PhotonSessionState.turn_history` (ordered oldest→newest).
            ``None`` indicates a baseline_rag project (no PHOTON session).
        session_manager_turns: Entries from :attr:`SessionState.turns` used
            to look up ``cited_chunk_ids`` by ``turn_id``.
        working_memory_enabled: ``True`` iff
            ``cfg.session_memory.working_memory.enabled`` is set.
        max_turns: Maximum number of entries to retain in ``rows`` (keeps
            the last N after slicing).

    Returns:
        ``{"available": bool, "reason": str, "rows": list[TurnRow]}``.
        ``available=False`` with a ``reason`` string is returned for the
        three N/A paths (working_memory disabled, baseline_rag, or
        photon_turn_history is literally ``None``).
    """
    if not working_memory_enabled:
        return {
            "available": False,
            "reason": "N/A (working_memory disabled)",
            "rows": [],
        }
    if photon_turn_history is None:
        return {
            "available": False,
            "reason": "N/A (baseline_rag)",
            "rows": [],
        }
    if not photon_turn_history:
        return {"available": True, "reason": "", "rows": []}

    cited_by_tid: dict[int, list[str]] = {}
    if session_manager_turns:
        for t in session_manager_turns:
            try:
                tid = int(getattr(t, "turn_id", -1))
            except (TypeError, ValueError):
                continue
            cited = list(getattr(t, "cited_chunk_ids", []) or [])
            cited_by_tid[tid] = cited

    recent = list(photon_turn_history)[-max_turns:]
    rows: list[TurnRow] = []
    for ph in recent:
        try:
            tid = int(getattr(ph, "turn_id", -1))
        except (TypeError, ValueError):
            tid = -1
        rows.append(
            TurnRow(
                turn_id=tid,
                question_text=str(getattr(ph, "question_text", "")),
                timestamp=str(getattr(ph, "timestamp", "")),
                cited_chunk_ids=cited_by_tid.get(tid, []),
            )
        )
    return {"available": True, "reason": "", "rows": rows}
