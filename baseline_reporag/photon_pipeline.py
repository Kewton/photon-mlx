"""PHOTON-RAG pipeline integration (Issue #3).

Provides:
- build_pipeline(cfg) — factory that routes to RepoRAGPipeline or PhotonRAGPipeline
- PhotonRAGPipeline — PHOTON-enhanced RAG with drift tracking and fallback
- tokenize_evidence_pack() — encode evidence text for PHOTON prefill
- compute_confidence() — extract confidence from PHOTON logits
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .citation import compute_refusal_score, resolve_citations
from .checkpoints import maybe_download_checkpoint
from .config import Config
from .generation.evidence_pack import build_evidence_pack
from .generation.prompt import (
    _EVIDENCE_HEADER,
    build_messages,
    flatten_messages_for_plain_lm,
)
from .memory.session import SessionState
from .pipeline import QueryResult, RepoRAGPipeline, apply_citation_postprocess
from .profiler import TurnProfiler
from .retrieval.debug_builder import (
    build_retrieval_debug_rows,
    finalise_retrieval_debug,
)
from .retrieval.graph_expansion import ExpandedChunkRef, expand_with_graph
from .retrieval.hybrid import apply_file_type_boost, hybrid_search
from .retrieval.query_expansion import expand_query

if TYPE_CHECKING:
    # Issue #103 / DR2-008: ``TurnState`` is a PHOTON type; importing it at
    # runtime would force MLX/PHOTON load on baseline-only paths. The file
    # uses ``from __future__ import annotations`` so the cache type
    # ``dict[str, TurnState]`` resolves lazily.
    from photon_mlx.session import TurnState

_logger = logging.getLogger(__name__)
_RELATED_PAST_QUESTIONS_MAX = 3
_RELATED_PAST_EVIDENCE_TOP_K = 4
_PAST_CONTEXT_DECAY = 0.7
_PAST_CONTEXT_MIN_DECAY = 0.25
_ADMISSION_MIN_CURRENT_SCORE = 0.05
_REWRITE_HISTORY_MAX = 2
_PHOTON_CARRYOVER_MAX_TURNS = 2
_PHOTON_CARRYOVER_MIN_SIMILARITY = 0.75
_SEGMENT_MEMORY_FEATURE_MAX = 24
_SEGMENT_MEMORY_MIN_SIMILARITY = 0.06
_SUPPORT_GUARD_THRESHOLD = 0.48
_SPLIT_PART_RE = re.compile(r"^(?P<base>.+)#part(?P<idx>\d+)of(?P<total>\d+)$")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]+")
_STANDALONE_TOPIC_RE = re.compile(
    r"(?P<anchor>[A-Za-z0-9_\-\u3040-\u30ff\u3400-\u9fff・ー（）()]+?)"
    r"(?:の(?:概要|対象|必要書類|認定条件)|について|とは)"
)
_TOPIC_ANCHOR_NOISE = (
    "場合",
    "それ",
    "その",
    "どの",
    "どんな",
    "今回",
    "前回",
    "申請では",
    "文書",
)
_COMPARISON_MARKERS = (
    "比較",
    "違い",
    "違う",
    "比べ",
    "差分",
    "同じ",
    "異な",
    "versus",
    " vs ",
)
_FOLLOW_UP_MARKERS = (
    "その",
    "それ",
    "これ",
    "この",
    "あの",
    "上記",
    "前述",
    "さっき",
    "先ほど",
    "先程",
    "場合",
    "同じ",
    "違い",
    "比較",
    "続き",
    "申請では",
    "必要ですか",
    "書きますか",
    "含まれ",
    "ありますか",
    "どう",
    "では？",
    "it",
    "that",
    "those",
    "these",
    "they",
    "same",
    "difference",
    "compare",
    "above",
    "previous",
)


@dataclass(frozen=True)
class _CarryoverDecision:
    mode: str
    query: str
    similarity: float
    marker: bool
    reason: str


@dataclass
class _TopicSegmentState:
    current_segment_id: int = 1
    turn_segments: dict[int, int] | None = None

    def __post_init__(self) -> None:
        if self.turn_segments is None:
            self.turn_segments = {}


@dataclass(frozen=True)
class _PhotonCarryoverMatch:
    turn_id: int
    score: float


@dataclass(frozen=True)
class _SegmentMemoryDecision:
    applied: bool
    score: float
    reason: str
    query: str


@lru_cache(maxsize=1)
def _mlx_metal_available() -> bool:
    probe = "import mlx.core as mx; mx.array([1]); print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _require_mlx_metal() -> None:
    if not _mlx_metal_available():
        raise ImportError("PHOTON inference requires an accessible Metal device")


class _MxProxy:
    """Lazy ``mlx.core`` proxy so baseline-only imports do not require Metal."""

    _module: Any | None = None

    def _load(self) -> Any:
        if self._module is not None:
            return self._module
        if not _mlx_metal_available():
            raise ImportError("mlx.core requires an accessible Metal device")
        import mlx.core as loaded_mx

        self._module = loaded_mx
        return loaded_mx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


mx = _MxProxy()


# ---------------------------------------------------------------------------
# Two-pass search configuration (Issue #56)
# ---------------------------------------------------------------------------


def _resolve_two_pass_search_cfg(
    retrieval_cfg: Any,
    fused_top_k: int,
    evidence_max_chunks: int,
) -> tuple[bool, int, int]:
    """Resolve and validate ``retrieval.two_pass_search`` settings.

    Returns ``(enabled, pass1_top_k, pass2_top_k)``. ``enabled`` defaults to
    ``False`` when the section is missing so existing configs continue to work.

    Validation rules (design §4.5 / DR1-008):
    - ``pass1_top_k >= pass2_top_k >= 1`` — violation raises ``ValueError``
    - ``pass1_top_k < fused_top_k`` — warn and clamp up to ``fused_top_k``
      (avoids silently dropping candidates supplied by retrieval)

    Validation is performed even when ``enabled=False`` so mis-configurations
    surface early (Stage 3 S3-002).
    """
    section = (
        retrieval_cfg.get("two_pass_search", {}) if retrieval_cfg is not None else {}
    )
    if section is None:
        section = {}
    # Support both ``Config`` wrappers and plain dicts.
    getter = section.get

    enabled_raw = getter("enabled", False)
    enabled = bool(enabled_raw)
    pass1_top_k = getter("pass1_top_k", fused_top_k)
    pass2_top_k = getter("pass2_top_k", evidence_max_chunks)

    if not isinstance(pass1_top_k, int) or isinstance(pass1_top_k, bool):
        raise ValueError(
            "retrieval.two_pass_search.pass1_top_k must be an int, "
            f"got {type(pass1_top_k).__name__}"
        )
    if not isinstance(pass2_top_k, int) or isinstance(pass2_top_k, bool):
        raise ValueError(
            "retrieval.two_pass_search.pass2_top_k must be an int, "
            f"got {type(pass2_top_k).__name__}"
        )
    if pass2_top_k < 1:
        raise ValueError(
            f"retrieval.two_pass_search.pass2_top_k must be >= 1, got {pass2_top_k}"
        )
    if pass1_top_k < pass2_top_k:
        raise ValueError(
            "retrieval.two_pass_search.pass1_top_k must be >= pass2_top_k, "
            f"got pass1_top_k={pass1_top_k}, pass2_top_k={pass2_top_k}"
        )
    if pass1_top_k < fused_top_k:
        _logger.warning(
            "retrieval.two_pass_search.pass1_top_k (%d) < retrieval.fused_top_k "
            "(%d); clamping pass1_top_k up to fused_top_k to preserve retrieval "
            "candidates.",
            pass1_top_k,
            fused_top_k,
        )
        pass1_top_k = fused_top_k
    return enabled, pass1_top_k, pass2_top_k


def _nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _unique_candidate_indices_by_rank(
    ranked_chunk_ids: list[str],
    chunk_ids_for_scoring: list[str],
    limit: int,
) -> list[int]:
    """Return scoring indices for top-ranked retrieval/reranker chunks."""
    if limit <= 0:
        return []
    index_by_id = {cid: idx for idx, cid in enumerate(chunk_ids_for_scoring)}
    selected: list[int] = []
    seen: set[str] = set()
    for cid in ranked_chunk_ids:
        if cid in seen:
            continue
        seen.add(cid)
        idx = index_by_id.get(cid)
        if idx is None:
            continue
        selected.append(idx)
        if len(selected) >= limit:
            break
    return selected


def _merge_protected_and_photon_indices(
    *,
    ranked_chunk_ids: list[str],
    chunk_ids_for_scoring: list[str],
    photon_indices: list[int],
    protected_top_n: int,
) -> list[int]:
    protected_indices = _unique_candidate_indices_by_rank(
        ranked_chunk_ids,
        chunk_ids_for_scoring,
        protected_top_n,
    )
    valid_photon_indices = [
        idx for idx in photon_indices if 0 <= idx < len(chunk_ids_for_scoring)
    ]
    merged: list[int] = []
    seen: set[int] = set()
    for idx in protected_indices + valid_photon_indices:
        if idx in seen:
            continue
        seen.add(idx)
        merged.append(idx)
    return merged


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _text_features(text: str) -> set[str]:
    """Return language-agnostic-ish features for short query/document overlap."""
    normalised = _normalise_text(text)
    features = set(_ASCII_TOKEN_RE.findall(normalised))
    for segment in _CJK_RE.findall(normalised):
        if len(segment) <= 2:
            features.add(segment)
        else:
            features.update(segment[i : i + 2] for i in range(len(segment) - 1))
            features.update(segment[i : i + 3] for i in range(len(segment) - 2))
    if not features and normalised:
        compact = normalised.replace(" ", "")
        features.update(compact[i : i + 3] for i in range(max(1, len(compact) - 2)))
    return {f for f in features if f}


def _feature_overlap(a: str, b: str) -> float:
    left = _text_features(a)
    if not left:
        return 0.0
    right = _text_features(b)
    if not right:
        return 0.0
    return len(left & right) / len(left)


def _has_follow_up_marker(question: str) -> bool:
    q = _normalise_text(question)
    return any(marker.casefold() in q for marker in _FOLLOW_UP_MARKERS) or any(
        marker.casefold() in q for marker in _COMPARISON_MARKERS
    )


def _has_comparison_marker(question: str) -> bool:
    q = _normalise_text(question)
    return any(marker.casefold() in q for marker in _COMPARISON_MARKERS)


def _standalone_topic_anchors(question: str) -> list[str]:
    """Extract explicit subject anchors from standalone-looking questions."""
    anchors: list[str] = []
    seen: set[str] = set()
    for match in _STANDALONE_TOPIC_RE.finditer(_normalise_text(question)):
        anchor = match.group("anchor").strip()
        compact = re.sub(r"\s+", "", anchor)
        if len(compact) < 6:
            continue
        if any(noise in compact for noise in _TOPIC_ANCHOR_NOISE):
            continue
        if compact in seen:
            continue
        seen.add(compact)
        anchors.append(compact)
    return anchors


def _has_explicit_topic_switch_signal(
    question: str,
    prior_questions: list[str],
) -> bool:
    """Return True when the current question names a new standalone target.

    PHOTON can detect that two turns are semantically nearby, but that is not
    enough to prove the user wants carryover.  A question that explicitly names
    a new subject ("Xの概要", "Xについて" etc.) and has very low overlap with
    recent questions should start a new segment unless it is framed as a
    comparison.
    """
    if not prior_questions:
        return False
    if _has_follow_up_marker(question) or _has_comparison_marker(question):
        return False
    anchors = _standalone_topic_anchors(question)
    if not anchors:
        return False
    similarity = max((_feature_overlap(question, prev) for prev in prior_questions), default=0.0)
    if similarity >= 0.25:
        return False
    return all(
        max((_feature_overlap(anchor, prev) for prev in prior_questions), default=0.0)
        < 0.08
        for anchor in anchors
    )


def _recent_questions(session: SessionState | None, limit: int) -> list[str]:
    if session is None or limit <= 0:
        return []
    questions = [turn.question.strip() for turn in session.turns if turn.question]
    return questions[-limit:]


def _recent_questions_in_segment(
    session: SessionState | None,
    segment_state: _TopicSegmentState | None,
    segment_id: int,
    limit: int,
) -> list[str]:
    if session is None or segment_state is None or limit <= 0:
        return []
    turn_segments = segment_state.turn_segments or {}
    questions = [
        turn.question.strip()
        for turn in session.turns
        if turn.question and turn_segments.get(turn.turn_id) == segment_id
    ]
    return questions[-limit:]


def _recent_turns_in_segment(
    session: SessionState | None,
    segment_state: _TopicSegmentState | None,
    segment_id: int,
    limit: int,
) -> list[Any]:
    if session is None or segment_state is None or limit <= 0:
        return []
    turn_segments = segment_state.turn_segments or {}
    turns = [
        turn
        for turn in session.turns
        if turn_segments.get(turn.turn_id) == segment_id
    ]
    return turns[-limit:]


def _history_text_from_turns(turns: list[Any]) -> str:
    lines: list[str] = []
    for turn in turns:
        lines.append(f"Q{turn.turn_id}: {turn.question}")
        answer_stripped = re.sub(r"\[C:\d+\]", "", turn.answer).strip()
        answer_preview = (
            answer_stripped[:400] + "..."
            if len(answer_stripped) > 400
            else answer_stripped
        )
        lines.append(f"A{turn.turn_id}: {answer_preview}")
    return "\n".join(lines)


def _generation_history_text(
    session: SessionState | None,
    segment_state: _TopicSegmentState | None,
    *,
    segment_id: int,
    carryover_mode: str,
    max_turns: int = 4,
) -> str:
    """Return only the history that should be visible to generation."""
    if session is None or not session.turns or max_turns <= 0:
        return ""
    if carryover_mode in {"weak", "independent"}:
        return ""

    history_limit = max_turns
    if carryover_mode == "mixed":
        history_limit = min(max_turns, 2)

    turns = _recent_turns_in_segment(
        session,
        segment_state,
        segment_id,
        history_limit,
    )
    return _history_text_from_turns(turns)


def _resolve_context_carryover(
    question: str,
    session: SessionState | None,
    *,
    enabled: bool,
    rewrite_enabled: bool,
    rewrite_history_max: int,
    rewrite_questions: list[str] | None = None,
) -> _CarryoverDecision:
    """Classify whether this turn should carry prior context into retrieval.

    The heuristic intentionally avoids domain-specific entity rules. It uses
    anaphora/follow-up markers plus lexical/character-ngram similarity to
    decide whether history should be strong, mixed, or weak.
    """
    if not enabled or session is None or not session.turns:
        return _CarryoverDecision(
            mode="independent",
            query=question,
            similarity=0.0,
            marker=False,
            reason="no_history_or_disabled",
        )

    prior_questions = (
        list(rewrite_questions)
        if rewrite_questions is not None
        else _recent_questions(session, max(1, rewrite_history_max))
    )
    similarity = max((_feature_overlap(question, prev) for prev in prior_questions), default=0.0)
    marker = _has_follow_up_marker(question)

    if marker and similarity >= 0.08:
        mode = "strong"
        reason = "marker_and_similarity"
    elif marker:
        mode = "mixed"
        reason = "marker"
    elif similarity >= 0.18:
        mode = "mixed"
        reason = "similarity"
    else:
        mode = "weak"
        reason = "low_similarity_no_marker"

    rewritten = question
    if rewrite_enabled and mode in {"strong", "mixed"}:
        context = " ".join(q for q in prior_questions if q)
        if context:
            rewritten = f"{context} {question}"

    return _CarryoverDecision(
        mode=mode,
        query=rewritten,
        similarity=similarity,
        marker=marker,
        reason=reason,
    )


def _boost_carryover_with_photon_matches(
    decision: _CarryoverDecision,
    question: str,
    session: SessionState | None,
    *,
    matches: list[_PhotonCarryoverMatch],
    rewrite_enabled: bool,
    rewrite_history_max: int,
) -> _CarryoverDecision:
    """Promote weak lexical carryover when PHOTON finds related past turns."""
    if decision.mode not in {"weak", "independent"}:
        return decision
    if session is None or not session.turns or not matches:
        return decision

    turns_by_id = {turn.turn_id: turn for turn in session.turns}
    matched_questions: list[str] = []
    best_score = 0.0
    for match in sorted(matches, key=lambda item: item.turn_id):
        turn = turns_by_id.get(match.turn_id)
        if turn is None or not turn.question:
            continue
        matched_questions.append(turn.question.strip())
        best_score = max(best_score, float(match.score))

    if not matched_questions:
        return decision

    mode = "strong" if decision.marker else "mixed"
    rewritten = question
    if rewrite_enabled:
        history_limit = max(1, rewrite_history_max)
        context = " ".join(q for q in matched_questions[-history_limit:] if q)
        if context:
            rewritten = f"{context} {question}"

    return _CarryoverDecision(
        mode=mode,
        query=rewritten,
        similarity=max(decision.similarity, best_score),
        marker=decision.marker,
        reason=f"{decision.reason}+photon_related_turn",
    )


def _resolve_segment_memory(
    question: str,
    segment_questions: list[str],
    *,
    rewrite_enabled: bool,
    rewrite_history_max: int,
) -> _SegmentMemoryDecision:
    """Recover the active segment for terse follow-ups without entity rules.

    This intentionally uses only generic signals: current-query feature count,
    overlap with recent segment questions, and recency.  It avoids hard-coded
    business terms, document names, or field labels.
    """
    if not segment_questions:
        return _SegmentMemoryDecision(False, 0.0, "no_segment_questions", question)
    if _has_explicit_topic_switch_signal(question, segment_questions):
        return _SegmentMemoryDecision(False, 0.0, "explicit_topic_switch", question)

    feature_count = len(_text_features(question))
    similarity = max((_feature_overlap(question, prev) for prev in segment_questions), default=0.0)
    marker = _has_follow_up_marker(question)
    short_followup = feature_count <= _SEGMENT_MEMORY_FEATURE_MAX
    if not marker and not short_followup and similarity < _SEGMENT_MEMORY_MIN_SIMILARITY:
        return _SegmentMemoryDecision(False, similarity, "insufficient_signal", question)

    context = " ".join(segment_questions[-max(1, rewrite_history_max) :])
    rewritten = f"{context} {question}" if rewrite_enabled and context else question
    reason = "segment_memory_marker" if marker else "segment_memory_recent"
    return _SegmentMemoryDecision(True, similarity, reason, rewritten)


def _turn_decay(current_turn_id: int, past_turn_id: int, decay: float) -> float:
    distance = max(1, current_turn_id - past_turn_id)
    decay = min(1.0, max(0.0, float(decay)))
    return decay ** max(0, distance - 1)


def _normalised_score_map(results: list[Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for result in results:
        cid = str(getattr(result, "chunk_id", ""))
        if not cid:
            continue
        raw_score = getattr(result, "score", 0.0)
        if isinstance(raw_score, (int, float)):
            scores[cid] = max(scores.get(cid, 0.0), float(raw_score))
    max_score = max(scores.values(), default=0.0)
    if max_score <= 0.0:
        return scores
    return {cid: score / max_score for cid, score in scores.items()}


def _last_prune_score_map(photon_inference: Any, session_id: str) -> dict[str, float]:
    last_scores = getattr(photon_inference, "_last_prune_scores_by_session", {})
    if not isinstance(last_scores, dict):
        return {}
    maybe_scores = last_scores.get(session_id, {})
    if not isinstance(maybe_scores, dict):
        return {}
    return {
        str(cid): float(score)
        for cid, score in maybe_scores.items()
        if isinstance(score, (int, float))
    }


def _score_current_question_candidates(
    photon_inference: Any,
    *,
    chunk_texts: list[str],
    chunk_ids: list[str],
    question: str,
) -> dict[str, float]:
    """Score candidates against the current question without mutating session state."""
    if not chunk_texts or len(chunk_texts) <= 1 or not question.strip():
        return {}
    scratch_session_id = f"current-question-score-{uuid.uuid4().hex}"
    try:
        photon_inference.prune_evidence(
            chunk_texts=chunk_texts,
            chunk_ids=chunk_ids,
            session_id=scratch_session_id,
            max_chunks=max(1, len(chunk_texts) - 1),
            question=question,
        )
        return _last_prune_score_map(photon_inference, scratch_session_id)
    finally:
        last_scores = getattr(photon_inference, "_last_prune_scores_by_session", None)
        if isinstance(last_scores, dict):
            last_scores.pop(scratch_session_id, None)


def _admit_photon_indices(
    *,
    candidate_indices: list[int],
    protected_indices: list[int],
    chunk_ids_for_scoring: list[str],
    chunk_texts: list[str],
    retrieval_scores: dict[str, float],
    query: str,
    min_current_score: float,
) -> list[int]:
    """Filter PHOTON-selected candidates that are stale for the current query."""
    protected_set = set(protected_indices)
    admitted: list[int] = []
    min_current_score = max(0.0, float(min_current_score))
    for idx in candidate_indices:
        if idx in protected_set:
            admitted.append(idx)
            continue
        if not (0 <= idx < len(chunk_ids_for_scoring)):
            continue
        cid = chunk_ids_for_scoring[idx]
        retrieval_score = retrieval_scores.get(cid, 0.0)
        overlap = _feature_overlap(query, chunk_texts[idx])
        if max(retrieval_score, overlap) >= min_current_score:
            admitted.append(idx)
    return admitted


def _dual_score_candidate_indices(
    *,
    candidate_indices: list[int],
    protected_indices: list[int],
    chunk_ids_for_scoring: list[str],
    retrieval_scores: dict[str, float],
    current_scores: dict[str, float],
    session_scores: dict[str, float],
    carryover_mode: str,
    max_extra: int,
) -> list[int]:
    """Select non-protected candidates by retrieval + PHOTON current/session scores."""
    protected_set = {
        idx for idx in protected_indices if 0 <= idx < len(chunk_ids_for_scoring)
    }
    max_extra = max(0, int(max_extra))

    scored: list[tuple[float, int]] = []
    for idx in candidate_indices:
        if idx in protected_set or not (0 <= idx < len(chunk_ids_for_scoring)):
            continue
        cid = chunk_ids_for_scoring[idx]
        retrieval = retrieval_scores.get(cid, 0.0)
        current = current_scores.get(cid, 0.0)
        session = session_scores.get(cid, 0.0)
        stale_gap = max(0.0, session - current)
        if carryover_mode in {"strong", "mixed"}:
            score = (0.45 * retrieval) + (0.35 * current) + (0.20 * session)
            score -= 0.12 * stale_gap
        else:
            score = (0.60 * retrieval) + (0.40 * current)
            score -= 0.30 * stale_gap
        scored.append((score, idx))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return list(protected_indices) + [idx for _score, idx in scored[:max_extra]]


def _support_score_for_pack(
    *,
    question: str,
    pack_chunks: list[Any],
    retrieval_scores: dict[str, float],
    current_scores: dict[str, float],
    session_scores: dict[str, float],
) -> float:
    """Estimate how directly the selected evidence supports this turn."""
    scores: list[float] = []
    photon_scored = bool(current_scores or session_scores)
    for chunk in pack_chunks:
        cid = str(getattr(chunk, "chunk_id", ""))
        content = str(getattr(chunk, "content", ""))
        if not cid:
            continue
        scores.append(
            max(
                (0.35 if photon_scored else 1.0) * retrieval_scores.get(cid, 0.0),
                current_scores.get(cid, 0.0),
                0.5 * session_scores.get(cid, 0.0),
                _feature_overlap(question, content),
            )
        )
    return max(scores, default=0.0)


def _support_check_note(support_score: float, *, guard_active: bool) -> str:
    base = (
        "## Support Check\n"
        f"- Evidence support score: {support_score:.3f}\n"
        "- Before answering, verify that each claim is directly supported by the cited chunks.\n"
        "- If the evidence only partially supports a claim, state the limitation instead of filling gaps.\n"
        "- For questions about whether a named case, condition, item, or activity is included, "
        "do not treat broad examples, open-ended wording, or \"etc.\" as sufficient support unless "
        "the named item or an equivalent category is explicitly grounded in the evidence."
    )
    if guard_active:
        return (
            base
            + "\n- The support score is low. Prefer a cautious answer and say "
            "「根拠が不足しています」 for unsupported requested details."
        )
    return base


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _compose_evidence_frame_pins(
    *,
    current_query_ids: list[str],
    working_memory_ids: list[str] | None = None,
    related_past_ids: list[str] | None = None,
) -> list[str]:
    """Order evidence frame pins so current-query support cannot be crowded out."""
    return _dedupe_preserve_order(
        current_query_ids + (working_memory_ids or []) + (related_past_ids or [])
    )


def _split_sibling_chunk_ids(chunk_id: str) -> list[str]:
    """Return sibling chunk_ids for split chunks like ``...#part2of2``."""
    match = _SPLIT_PART_RE.match(chunk_id)
    if match is None:
        return []
    total = _nonnegative_int(match.group("total"))
    if total <= 1:
        return []
    base = match.group("base")
    return [f"{base}#part{i}of{total}" for i in range(1, total + 1)]


def _expand_related_past_refs(
    chunk_ids: list[str],
    *,
    store: Any,
    neighborhood_before: int,
    neighborhood_after: int,
) -> list[ExpandedChunkRef]:
    """Expand related-past evidence with split siblings and file neighbors."""
    ordered_ids = _dedupe_preserve_order([cid for cid in chunk_ids if cid])
    seen: set[str] = set()
    refs: list[ExpandedChunkRef] = []
    neighbor_seed_ids: list[str] = []

    def add(cid: str, source: str) -> None:
        if not cid or cid in seen:
            return
        seen.add(cid)
        refs.append(ExpandedChunkRef(chunk_id=cid, source=source))

    for cid in ordered_ids:
        add(cid, "related_past")
        neighbor_seed_ids.append(cid)
        for sibling_id in _split_sibling_chunk_ids(cid):
            if sibling_id != cid:
                add(sibling_id, "related_past_neighbor")
            neighbor_seed_ids.append(sibling_id)

    if neighborhood_before <= 0 and neighborhood_after <= 0:
        return refs

    chunks = store.get_many(_dedupe_preserve_order(neighbor_seed_ids))
    for chunk in chunks:
        try:
            neighbors = store.get_neighbors(
                chunk,
                before=neighborhood_before,
                after=neighborhood_after,
            )
        except Exception:
            neighbors = []
        if not isinstance(neighbors, list):
            neighbors = []
        for neighbor in neighbors:
            add(neighbor.chunk_id, "related_past_neighbor")
    return refs


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def tokenize_evidence_pack(
    text: str,
    tokenizer: Any,
    cfg: Any,
    max_tokens: int | None = None,
) -> mx.array:
    """Tokenize evidence text with chunk-aligned padding.

    Args:
        text: raw evidence text.
        tokenizer: tokenizer with ``encode()`` and ``pad_token_id``.
        cfg: a :class:`torch_ref.config.PhotonConfig` instance.  The baseline
            ``Config`` (from ``configs/baseline.yaml``) does **not** define
            ``model.max_position_embeddings`` and must not be passed here.
        max_tokens: hard cap on token count.  When ``None`` (default), the
            cap is taken from ``cfg.model.max_position_embeddings``.  Must be
            positive; a :class:`ValueError` is raised otherwise (DR1-001).

    Returns:
        mx.array of token ids, length is a multiple of prod(chunk_sizes).
    """
    if max_tokens is None:
        max_tokens = cfg.model.max_position_embeddings

    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    ids = tokenizer.encode(text)
    if not ids:
        return mx.array([], dtype=mx.int32)

    if len(ids) > max_tokens:
        ids = ids[:max_tokens]

    padding_multiple = prod(cfg.hierarchy.chunk_sizes)
    remainder = len(ids) % padding_multiple
    if remainder != 0:
        pad_count = padding_multiple - remainder
        ids = ids + [tokenizer.pad_token_id] * pad_count

    return mx.array(ids, dtype=mx.int32)


def compute_confidence(logits: mx.array) -> float:
    """Compute mean max-softmax confidence from logits.

    Args:
        logits: (B, seq_len, vocab_size) tensor.

    Returns:
        float in [0, 1].
    """
    probs = mx.softmax(logits, axis=-1)
    max_probs = mx.max(probs, axis=-1)
    return float(mx.mean(max_probs).item())


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def _build_baseline_deps(cfg: Config) -> dict[str, Any]:
    """Construct real baseline pipeline dependencies from config.

    The canonical implementation lives in
    :func:`baseline_reporag.pipeline_factory._build_baseline_deps_no_mlx`
    so the factory module can stay MLX-free at import time. This wrapper
    is preserved as a module attribute so existing tests that patch
    ``baseline_reporag.photon_pipeline._build_baseline_deps`` keep working
    (Issue #62 Phase 1 refactor R-1: single source of truth, no
    lockstep-drift risk).
    """
    from .pipeline_factory import _build_baseline_deps_no_mlx

    return _build_baseline_deps_no_mlx(cfg)


def _resolve_working_memory_cfg(raw: Any) -> Any:
    """Normalise ``session_memory.working_memory`` into a ``WorkingMemoryConfig``.

    Accepts ``None`` (feature disabled), a dict (YAML form), or an already
    constructed :class:`photon_mlx.session.WorkingMemoryConfig`. Anything
    else triggers a warning (type name only; raw values are never surfaced,
    design §7) and fails closed to ``None`` so the query path continues.

    Returns either a ``WorkingMemoryConfig`` instance or ``None``.
    """
    from photon_mlx.session import WorkingMemoryConfig

    if raw is None:
        return None
    if isinstance(raw, WorkingMemoryConfig):
        return raw
    # Support the baseline Config wrapper (has .to_dict()) and plain dicts.
    raw_dict: dict[str, Any]
    if isinstance(raw, Config):
        raw_dict = raw.to_dict()
    elif isinstance(raw, dict):
        raw_dict = dict(raw)
    else:
        _logger.warning(
            "session_memory.working_memory has unsupported type %s; "
            "disabling working memory for this session",
            type(raw).__name__,
        )
        return None
    try:
        return WorkingMemoryConfig(**raw_dict)
    except (TypeError, ValueError) as exc:
        # Intentionally omit the raw dict (may contain attacker-controlled
        # values from YAML). Only the exception class name is logged.
        _logger.warning(
            "WorkingMemoryConfig rejected session_memory.working_memory "
            "(%s); disabling working memory for this session",
            type(exc).__name__,
        )
        return None


def _extract_working_memory_cfg(cfg: Config) -> Any:
    """Pull ``session_memory.working_memory`` out of a baseline ``Config``.

    Uses ``getattr`` / ``get`` so missing sections surface as ``None``
    rather than raising (design §3-3 fail-closed rules).
    """
    session_memory = getattr(cfg, "session_memory", None)
    if session_memory is None:
        return None
    raw = None
    if hasattr(session_memory, "get"):
        raw = session_memory.get("working_memory", None)
    else:
        raw = getattr(session_memory, "working_memory", None)
    return _resolve_working_memory_cfg(raw)


def _build_photon_deps(cfg: Config) -> dict[str, Any]:
    """Construct PHOTON-specific dependencies from config."""
    _require_mlx_metal()

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from torch_ref.config import PhotonConfig

    from photon_mlx.inference import PhotonInference
    from photon_mlx.model import PhotonModel
    from photon_mlx.safe_recgen import SafeRecGenConfig, SafeRecGenController

    from torch_ref.config import (
        HierarchyConfig,
        ModelConfig,
        TokenizerConfig,
    )

    # Issue #55: wire long-context RoPE fields from baseline cfg so
    # `photon_long_context.yaml` reaches PhotonModel unchanged.  When the
    # baseline cfg lacks these keys (e.g. legacy 2048 profiles), we fall
    # back to ModelConfig defaults via ``rope_scaling_from``.
    scaling, factor = ModelConfig.rope_scaling_from(cfg.model)
    model_cfg = ModelConfig(
        architecture=cfg.model.get("architecture", "photon_decoder"),
        base_embed_dim=cfg.model.base_embed_dim,
        hidden_size=cfg.model.hidden_size,
        intermediate_size=cfg.model.intermediate_size,
        num_attention_heads=cfg.model.get("num_heads", 4),
        num_key_value_heads=cfg.model.get("num_heads", 4),
        head_dim=getattr(cfg.model, "head_dim", 64),
        max_position_embeddings=getattr(cfg.model, "max_position_embeddings", 2048),
        rope_theta=getattr(cfg.model, "rope_theta", 1_000_000.0),
        rope_scaling=scaling,
        rope_scale_factor=factor,
    )
    hierarchy_cfg = HierarchyConfig(
        levels=cfg.hierarchy.levels,
        chunk_sizes=cfg.hierarchy.chunk_sizes,
        encoder_layers_per_level=cfg.hierarchy.encoder_layers_per_level,
        decoder_layers_per_level=cfg.hierarchy.decoder_layers_per_level,
    )
    # Issue #138: ``tokenizer.vocab_size`` is the canonical source for
    # production photon configs (institutional_docs_photon.yaml,
    # photon_small.yaml, photon_long_context.yaml all set it under the
    # ``tokenizer:`` block). The legacy ``model.vocab_size`` lookup is kept
    # as a fallback so the ~17 unit tests that pre-date the
    # ``tokenizer:`` section keep working without modification.
    tokenizer_section = cfg.get("tokenizer")
    tokenizer_id: str | None = (
        getattr(tokenizer_section, "tokenizer_id", None)
        if tokenizer_section is not None
        else None
    )
    cfg_vocab_size: int
    if (
        tokenizer_section is not None
        and getattr(tokenizer_section, "vocab_size", None) is not None
    ):
        cfg_vocab_size = tokenizer_section.vocab_size
    else:
        cfg_vocab_size = cfg.model.get("vocab_size", 1000)
    tok_cfg = TokenizerConfig(vocab_size=cfg_vocab_size)
    photon_cfg = PhotonConfig(
        model=model_cfg,
        hierarchy=hierarchy_cfg,
        tokenizer=tok_cfg,
    )

    # Build the tokenizer before PhotonInference so both paths (question+evidence
    # prefill in PhotonRAGPipeline and chunk scoring in prune_evidence) share
    # the same instance (Issue #58).
    #
    # Issue #139: tokenizer_id is now required for provider=='photon'. The
    # legacy byte-mod stub-tokenizer fallback was deleted to remove a
    # structural path where production code could silently fall back onto a
    # test fixture (the same class of bug as S7-001 random-init weights).
    # Missing or unsafe tokenizer_id now raises ``ValueError`` at this
    # boundary.
    # Issue #148 Phase A0 / DR4-002: validate model_id against the HF repo-id
    # allowlist before any heavy construction.  model_id is untrusted yaml
    # input; ``_validate_repo_id`` rejects URL / local-path / traversal forms.
    # Skip validation when model_id is absent or empty (backwards compat for
    # configs that omit the field).
    raw_model_id = cfg.model.get("model_id", None)
    if raw_model_id:
        _validate_repo_id(raw_model_id, "model_id")

    if not tokenizer_id:
        raise ValueError(
            "cfg.tokenizer.tokenizer_id is required for provider=='photon'. "
            "Set the `tokenizer:` block with a valid tokenizer_id "
            "(e.g. 'mlx-community/Qwen2.5-Coder-14B-Instruct-4bit') in the yaml config."
        )
    tokenizer_id = _validate_tokenizer_id(tokenizer_id)
    tokenizer = _load_hf_tokenizer(tokenizer_id, photon_cfg.tokenizer.vocab_size)
    model = PhotonModel(photon_cfg)

    # Issue #148 Phase A0 / DR-1 (#135 S7-001 のフル実装): load checkpoint
    # weights when ``cfg.model.checkpoint_path`` is set.  The allowed root is
    # ``PHOTON_CHECKPOINT_ROOT`` (env var) or ``checkpoints/`` (default).
    # Security invariants (§6):
    # - root containment validated by ``_resolve_checkpoint_path``
    # - symlink escape rejected (``resolve(strict=True)`` follows symlinks)
    # - directory shape checked (weights.npz + state.json must exist)
    # - on failure: RuntimeError by default (fail-fast); WARNING + continue
    #   when ``PHOTON_ALLOW_RANDOM_INIT=1`` (unit/CI negative-path tests only —
    #   never set in production or Phase A eval)
    # - log messages use relative-to-root path only (never absolute path)
    raw_ckpt_path = getattr(cfg.model, "checkpoint_path", None)
    if raw_ckpt_path is None:
        raw_ckpt_path = (
            cfg.model.get("checkpoint_path", None)
            if hasattr(cfg.model, "get")
            else None
        )
    if raw_ckpt_path:
        checkpoint_repo_id = getattr(cfg.model, "checkpoint_repo_id", None)
        if checkpoint_repo_id is None and hasattr(cfg.model, "get"):
            checkpoint_repo_id = cfg.model.get("checkpoint_repo_id", None)
        checkpoint_revision = getattr(cfg.model, "checkpoint_revision", None)
        if checkpoint_revision is None and hasattr(cfg.model, "get"):
            checkpoint_revision = cfg.model.get("checkpoint_revision", None)
        maybe_download_checkpoint(
            raw_ckpt_path,
            repo_id=checkpoint_repo_id,
            revision=checkpoint_revision,
        )
        ckpt_path = _resolve_checkpoint_path(raw_ckpt_path)
        # Directory shape validation (weights.npz + state.json required).
        # CB-002 fix: use is_file() instead of exists() so that a directory
        # named "weights.npz/" or a broken symlink does not pass the check.
        for required_file in ("weights.npz", "state.json"):
            if not (ckpt_path / required_file).is_file():
                raise RuntimeError(
                    f"checkpoint directory is missing {required_file!r}. "
                    f"Expected a photon_mlx checkpoint directory containing "
                    f"weights.npz and state.json."
                )
        # Compute root-relative path for safe logging (never log absolute path).
        ckpt_root = Path(
            os.environ.get("PHOTON_CHECKPOINT_ROOT", "checkpoints")
        ).resolve()
        try:
            rel_ckpt = ckpt_path.relative_to(ckpt_root)
        except ValueError:
            rel_ckpt = ckpt_path.name
        try:
            _load_photon_checkpoint(model, ckpt_path)
            _logger.info("Loaded PHOTON checkpoint from %s", rel_ckpt)
        except Exception as exc:  # noqa: BLE001 — boundary normalization
            exc_class = type(exc).__name__
            allow_random_init = (
                os.environ.get("PHOTON_ALLOW_RANDOM_INIT", "0").strip() == "1"
            )
            if allow_random_init:
                _logger.warning(
                    "checkpoint load failed (%s) — continuing with random-init weights "
                    "because PHOTON_ALLOW_RANDOM_INIT=1. "
                    "Do NOT use random-init for production inference.",
                    exc_class,
                )
            else:
                raise RuntimeError(
                    f"checkpoint load failed ({exc_class}). "
                    f"PHOTON_ALLOW_RANDOM_INIT=1 may bypass this check, but is "
                    f"reserved for unit/CI negative-path tests — do not set it "
                    f"for production inference or Phase A eval."
                ) from None
    else:
        _logger.warning(
            "cfg.model.checkpoint_path is not set; PhotonModel will use "
            "random-init weights. Set checkpoint_path or PHOTON_CHECKPOINT_ROOT "
            "for production inference."
        )

    # Issue #64 / Codex CB-001: extract working memory policy once, pass it
    # into PhotonInference alongside the Issue #63 drift_level_weights below.
    working_memory_cfg = _extract_working_memory_cfg(cfg)

    safe_recgen_enabled = getattr(cfg.get("inference"), "safe_recgen_enabled", True)
    if safe_recgen_enabled:
        sr_cfg_data = cfg.get("safe_recgen")
        if sr_cfg_data is not None:
            triggers = sr_cfg_data.get("triggers")
            thresholds = sr_cfg_data.get("thresholds")
            # Issue #63 / DR1-010: alias resolution happens here, not inside
            # SafeRecGenConfig. The legacy YAML key
            # ``thresholds.latent_cosine_drift`` maps onto the new
            # ``latent_cosine_drift_top_threshold``; when both are present,
            # the new explicit ``latent_cosine_drift_top`` wins.
            legacy_top_threshold = (
                getattr(thresholds, "latent_cosine_drift", 0.18) if thresholds else 0.18
            )
            top_threshold = (
                getattr(thresholds, "latent_cosine_drift_top", legacy_top_threshold)
                if thresholds
                else legacy_top_threshold
            )
            # DR2-005: fall back to defaults for missing new keys.
            mid_threshold = (
                getattr(thresholds, "latent_cosine_drift_mid", 0.40)
                if thresholds
                else 0.40
            )
            token_threshold = (
                getattr(thresholds, "latent_cosine_drift_token", 0.30)
                if thresholds
                else 0.30
            )
            drift_level_weights = sr_cfg_data.get("drift_level_weights")
            if drift_level_weights is None:
                drift_level_weights = (0.2, 0.3, 0.5)
            sr_config = SafeRecGenConfig(
                enabled=True,
                trigger_exact_quote=getattr(triggers, "exact_quote", True)
                if triggers
                else True,
                trigger_diff_or_patch=getattr(triggers, "diff_or_patch", True)
                if triggers
                else True,
                trigger_high_risk_query=getattr(triggers, "high_risk_query", True)
                if triggers
                else True,
                trigger_topic_shift=getattr(triggers, "topic_shift", True)
                if triggers
                else True,
                trigger_latent_drift=getattr(triggers, "latent_drift", True)
                if triggers
                else True,
                trigger_low_confidence=getattr(triggers, "low_confidence", True)
                if triggers
                else True,
                # Legacy top-only threshold (kept in sync with the new field
                # for backward-compat log/schema consumers).
                latent_cosine_drift_threshold=top_threshold,
                topic_shift_score_threshold=getattr(
                    thresholds, "topic_shift_score", 0.65
                )
                if thresholds
                else 0.65,
                confidence_floor=getattr(thresholds, "confidence_floor", 0.40)
                if thresholds
                else 0.40,
                logit_kl_threshold=getattr(thresholds, "logit_kl", 0.75)
                if thresholds
                else 0.75,
                # Issue #63 new fields.
                latent_cosine_drift_top_threshold=top_threshold,
                latent_cosine_drift_mid_threshold=mid_threshold,
                latent_cosine_drift_token_threshold=token_threshold,
                drift_level_weights=drift_level_weights,
            )
        else:
            sr_config = SafeRecGenConfig(enabled=True)
        safe_recgen = SafeRecGenController(sr_config)
    else:
        sr_config = None
        safe_recgen = None

    # Issue #63 / DR1-005: pass drift_level_weights (not the whole
    # SafeRecGenConfig) into PhotonInference so the inference layer only
    # depends on what it actually needs (ISP).
    drift_weights_for_inference = (
        sr_config.drift_level_weights if sr_config is not None else None
    )
    photon_inference = PhotonInference(
        model,
        photon_cfg,
        tokenizer,
        drift_level_weights=drift_weights_for_inference,
        working_memory_cfg=working_memory_cfg,
    )

    return {
        "photon_inference": photon_inference,
        "safe_recgen": safe_recgen,
        "photon_cfg": photon_cfg,
        "tokenizer": tokenizer,
    }


# ---------------------------------------------------------------------------
# Issue #148 Phase A0 — HF repo-id allowlist (model_id) + checkpoint helpers
# ---------------------------------------------------------------------------

# Shared pattern: HF repo-id must be ``<org>/<name>`` with ASCII
# ``[A-Za-z0-9._-]`` only and exactly one slash.  Applies to both
# ``tokenizer.tokenizer_id`` (existing) and ``model.model_id`` (new).
_HF_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+$")


def _validate_repo_id(value: str, key: str) -> None:
    """Validate ``value`` against the HF repo-id allowlist for ``key``.

    Accepts ``<org>/<name>`` with ASCII ``[A-Za-z0-9._-]`` and exactly one
    slash.  Raises ``ValueError`` on unsafe input.

    CB-004 fix: raw input value is never embedded in the exception message
    to avoid leaking private slugs, token-like strings, or multi-line
    payloads into logs / UI / CI artifacts.  Error messages follow the same
    sanitization pattern as ``_validate_tokenizer_id``.

    Rejects:
    - URL forms (``://`` present)
    - Absolute / tilde paths (leading ``/`` or ``~``)
    - Path traversal (``..`` anywhere, or starts with ``../``)
    - Multiple slashes (``org/name/extra``)
    - No slash (``justname``)
    - Non-ASCII / shell-metacharacter forms
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    if "/" not in value:
        raise ValueError(
            f"{key} must be a HuggingFace repo-id in '<org>/<name>' form "
            f"(expected exactly one slash)"
        )
    if value.count("/") != 1:
        raise ValueError(
            f"{key} must contain exactly one slash (expected '<org>/<name>' form)"
        )
    if any(c in value for c in ("://", "\\", "\x00")):
        raise ValueError(
            f"{key} must not contain URL scheme or control characters "
            f"(expected '<org>/<name>' with [A-Za-z0-9._-] only)"
        )
    if value.startswith(("/", "~", ".")):
        raise ValueError(
            f"{key} must not start with '/', '~', or '.' "
            f"(path-like prefix not allowed; expected '<org>/<name>' form)"
        )
    if ".." in value:
        raise ValueError(
            f"{key} must not contain '..' "
            f"(path traversal not allowed; expected '<org>/<name>' form)"
        )
    if not _HF_REPO_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{key} has unsafe form (expected '<org>/<name>' with [A-Za-z0-9._-] only)"
        )


def _resolve_checkpoint_path(raw: str) -> Path:
    """Validate root containment and symlink-escape, return resolved path.

    The allowed root is ``PHOTON_CHECKPOINT_ROOT`` when set, otherwise the
    ``checkpoints/`` directory relative to the repository root (cwd).

    CB-001 fix: when ``raw`` is a relative path it is resolved against
    ``root`` (not against cwd) so that the documented idiom
    ``checkpoint_path: "mulmoclaude_step600"`` (relative to
    ``PHOTON_CHECKPOINT_ROOT``) works as intended.  Absolute paths (and
    ``~``-expanded paths) are resolved as-is so existing absolute-path
    configs continue to work unchanged.

    Raises ``RuntimeError`` if the resolved path escapes the root.
    """
    root = Path(os.environ.get("PHOTON_CHECKPOINT_ROOT", "checkpoints")).resolve()
    raw_path = Path(raw).expanduser()
    # Resolve relative paths against root (CB-001), absolute paths as-is.
    candidate_unresolved = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        candidate = candidate_unresolved.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            f"checkpoint_path does not exist or is inaccessible: {type(exc).__name__}"
        ) from None
    try:
        candidate.relative_to(root)
    except ValueError:
        raise RuntimeError(
            "checkpoint_path is outside the approved checkpoint roots. "
            "Set PHOTON_CHECKPOINT_ROOT or place the checkpoint under "
            "the repo 'checkpoints/' directory."
        ) from None
    return candidate


def _load_photon_checkpoint(model: Any, path: Path) -> Any:
    """Lazy wrapper around ``photon_mlx.trainer.load_checkpoint``.

    The separate function makes the call site patchable in tests without
    importing photon_mlx at module level (MLX-free import boundary).
    """
    from photon_mlx.trainer import load_checkpoint

    return load_checkpoint(model, path)


# Issue #139 / DR4-001 / Codex CB-001: ``tokenizer.tokenizer_id`` originates
# from yaml input and is treated as untrusted. ``transformers.AutoTokenizer
# .from_pretrained`` accepts both Hugging Face repo ids AND local filesystem
# paths, so naive regex allowlists let through values like ``../model``,
# ``org/..``, ``.cache/model`` that the HF loader could resolve as paths.
# We validate against the HF repo-id form (``<org>/<name>``) and additionally
# reject components that look path-like.
_TOKENIZER_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_TOKENIZER_ID_MAX_TOTAL_LEN = 200
_TOKENIZER_ID_MAX_COMPONENT_LEN = 96


def _validate_tokenizer_id(tokenizer_id: str) -> str:
    """Validate ``tokenizer_id`` against the HF repo-id allowlist.

    Returns the input unchanged on success. Raises ``ValueError`` with a
    sanitized message on failure (the raw input is never embedded directly
    in log/error output — see ``_display_tokenizer_id``).

    Hardening (Codex CB-001):

    - regex allowlist ``^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$``
    - total length cap (200) and per-component length cap (96)
    - rejects any component equal to ``.`` / ``..`` (would be path-like)
    - rejects components beginning with ``.`` (hides as hidden-file path)
    - rejects components containing ``..`` substring (path traversal)
    - rejects leading ``/``, ``.``, ``~`` overall (path-like prefix)
    """
    if not isinstance(tokenizer_id, str) or not tokenizer_id:
        raise ValueError("cfg.tokenizer.tokenizer_id must be a non-empty string")
    if len(tokenizer_id) > _TOKENIZER_ID_MAX_TOTAL_LEN:
        raise ValueError(
            "cfg.tokenizer.tokenizer_id exceeds maximum length "
            f"({_TOKENIZER_ID_MAX_TOTAL_LEN} chars)"
        )
    if tokenizer_id[0] in {"/", ".", "~"}:
        raise ValueError(
            "cfg.tokenizer.tokenizer_id must not start with '/', '.', or '~' "
            "(path-like prefix)"
        )
    if not _TOKENIZER_ID_PATTERN.fullmatch(tokenizer_id):
        raise ValueError(
            "cfg.tokenizer.tokenizer_id has unsafe form "
            "(expected '<org>/<name>' with [A-Za-z0-9._-] only)"
        )
    # _PATTERN guarantees exactly one '/' (no leading/trailing) so split is safe.
    org, name = tokenizer_id.split("/", 1)
    for component in (org, name):
        if len(component) > _TOKENIZER_ID_MAX_COMPONENT_LEN:
            raise ValueError(
                "cfg.tokenizer.tokenizer_id component exceeds maximum length "
                f"({_TOKENIZER_ID_MAX_COMPONENT_LEN} chars)"
            )
        if component in {".", ".."}:
            raise ValueError(
                "cfg.tokenizer.tokenizer_id components must not be '.' or '..' "
                "(path traversal)"
            )
        if component.startswith("."):
            raise ValueError(
                "cfg.tokenizer.tokenizer_id components must not start with '.' "
                "(hidden-file path-like form)"
            )
        if ".." in component:
            raise ValueError(
                "cfg.tokenizer.tokenizer_id components must not contain '..' "
                "(path traversal)"
            )
    return tokenizer_id


def _display_tokenizer_id(tokenizer_id: str) -> str:
    """Sanitized representation for log / error messages.

    Uses ``repr()`` so control characters (newline, etc.) are escape-printed,
    preventing log injection if an unsafe value somehow reaches an error path.
    """
    return repr(tokenizer_id)


def _load_hf_tokenizer(tokenizer_id: str, expected_vocab_size: int) -> Any:
    """Load the HuggingFace ``AutoTokenizer`` matched to ``tokenizer_id``.

    Issue #138: training (``scripts/generate_training_corpus.py``) loads a
    real Qwen subword tokenizer; inference must use the same tokenizer or
    PHOTON checkpoints return garbage at inference time. This helper is the
    single production code path that builds the inference-side tokenizer.

    Issue #139 / DR4-001 / DR4-002 hardening:

    - ``tokenizer_id`` must already be validated (callers in
      ``_build_photon_deps`` invoke ``_validate_tokenizer_id`` first).
    - ``trust_remote_code=False`` is fixed; do not relax it.
    - ``transformers.AutoTokenizer.from_pretrained`` failures (network /
      gated model / unknown id / cache miss) are normalized to
      ``ValueError`` so callers and ``docs/troubleshooting.md`` see a
      single failure mode. ``ImportError`` (transformers not installed)
      and ``ValueError`` from vocab-size mismatch (Issue #138 invariant)
      are preserved unchanged.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency-time error
        raise ImportError(
            "transformers is required for PHOTON inference (Issue #138). "
            "Install it via `pip install transformers`."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001 — Issue #139 boundary normalization
        # Hugging Face surfaces a wide family of exceptions here (OSError,
        # huggingface_hub.errors.HfHubHTTPError, RepositoryNotFoundError,
        # GatedRepoError, etc.) whose import paths drift across releases.
        # Normalize to ValueError at this boundary. We log the original
        # exception class + sanitized id at warning level so operators have
        # diagnostic breadcrumbs in private logs, then re-raise with
        # ``from None`` so the public traceback / __cause__ chain does NOT
        # carry the raw HF exception text (Codex CB-002): HF messages may
        # contain private paths, private model ids, or environment-specific
        # details that should not leak via Streamlit error banners /
        # operator log paste / Slack notifications.
        exc_class_name = type(exc).__name__
        _logger.warning(
            "PHOTON tokenizer load failed: id=%s exc_class=%s",
            _display_tokenizer_id(tokenizer_id),
            exc_class_name,
        )
        raise ValueError(
            f"failed to load tokenizer {_display_tokenizer_id(tokenizer_id)}: "
            f"{exc_class_name}"
        ) from None

    actual_vocab_size = getattr(tokenizer, "vocab_size", None)
    if actual_vocab_size is not None:
        # Issue #138 / #148 Phase A: allow vocab padding (e.g. Qwen2.5-Coder
        # tokenizer has 151643 tokens, trained model embeddings are padded to
        # 152064 = next multiple of 64 for tensor-core efficiency). Reject only
        # when the cfg vocab is *smaller* than the tokenizer (would index OOB).
        if actual_vocab_size > expected_vocab_size:
            raise ValueError(
                "tokenizer vocab_size mismatch (Issue #138): "
                f"tokenizer={actual_vocab_size} cfg={expected_vocab_size}. "
                "cfg.tokenizer.vocab_size must be >= tokenizer.vocab_size."
            )
        if actual_vocab_size < expected_vocab_size:
            _logger.info(
                "tokenizer vocab_size (%d) < cfg.tokenizer.vocab_size (%d) — "
                "treating as padded vocab; rows %d..%d are unreachable",
                actual_vocab_size,
                expected_vocab_size,
                actual_vocab_size,
                expected_vocab_size - 1,
            )
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            tokenizer.pad_token_id = eos_id
        else:
            tokenizer.pad_token_id = 0
    return tokenizer


def _clear_photon_session_state(photon_inference: Any, session_id: str) -> None:
    """Drop PHOTON coarse/prev state and cached logits for ``session_id``.

    Centralised fail-closed reset used in three places (design §8):
    - ``tokenize_evidence_pack`` failure in the pipeline (CB-001).
    - ``reprefill_hierarchy`` Safe RecGen action.
    - ``fallback_to_baseline_path`` Safe RecGen action.

    ``prev_logits`` must be cleared alongside ``current_state`` /
    ``prev_state`` because ``PhotonSessionState.update()`` derives
    ``token_agreement`` / ``logit_kl`` from ``prev_logits`` independently
    of the hierarchy; leaving it set would leak stale drift into the next
    turn (Codex CB-004).
    """
    photon_session = photon_inference._sessions.get(session_id)
    if photon_session is None:
        return
    # Issue #64: delegate to PhotonSessionState.reset_working_memory() so
    # ``turn_history`` is cleared atomically alongside the stale latents
    # while ``drift_history`` / ``turn_count`` are preserved for telemetry.
    photon_session.reset_working_memory()


def _photon_session_has_pruning_state(photon_inference: Any, session_id: str) -> bool:
    """Return whether PHOTON has a usable coarse state for pruning."""
    sessions = getattr(photon_inference, "_sessions", None)
    if not isinstance(sessions, dict):
        return False
    photon_session = sessions.get(session_id)
    current_state = getattr(photon_session, "current_state", None)
    level_states = getattr(current_state, "level_states", None)
    return bool(level_states)


def build_pipeline(cfg: Config) -> RepoRAGPipeline | PhotonRAGPipeline:
    """Factory: create the right pipeline based on cfg.model.provider.

    CB-004 (codex-fix): the canonical factory lives in
    ``baseline_reporag.pipeline_factory`` so baseline-only entry points can
    route via a module that does not import MLX at load time.  This
    function is a thin backward-compat re-export; prefer importing from
    ``baseline_reporag.pipeline_factory`` directly.
    """
    from .pipeline_factory import build_pipeline as _factory_build_pipeline

    return _factory_build_pipeline(cfg)


# ---------------------------------------------------------------------------
# PhotonRAGPipeline
# ---------------------------------------------------------------------------


class PhotonRAGPipeline:
    """PHOTON-enhanced RepoRAG pipeline with drift tracking and fallback."""

    def __init__(
        self,
        cfg: Config,
        baseline_deps: dict[str, Any],
        photon_deps: dict[str, Any],
    ) -> None:
        self.cfg = cfg
        self.baseline = RepoRAGPipeline(
            config=cfg,
            store=baseline_deps["store"],
            lexical=baseline_deps["lexical"],
            embedding=baseline_deps["embedding"],
            graph=baseline_deps["graph"],
            sessions=baseline_deps["sessions"],
            generator=baseline_deps["generator"],
            logger=baseline_deps["logger"],
            reranker=baseline_deps["reranker"],
        )
        self.photon_inference = photon_deps["photon_inference"]
        self.safe_recgen = photon_deps["safe_recgen"]
        self.photon_cfg = photon_deps["photon_cfg"]
        self.tokenizer = photon_deps["tokenizer"]
        # Issue #103: 1-session-1-entry sidecar cache for past-turn pinning.
        # write/read/pop must always go through ``query()`` or
        # :meth:`_clear_photon_session_artifacts` so the lifecycle invariant
        # documented in design §3 (write at end of Turn N, pop at start of
        # Turn N+1) is preserved.
        self._relevant_past_turn_cache: dict[str, TurnState] = {}
        self._topic_segment_cache: dict[str, _TopicSegmentState] = {}

    def _clear_photon_session_artifacts(self, session_id: str) -> None:
        """Centralised reset for PHOTON state + Issue #103 sidecar cache.

        Replaces direct ``_clear_photon_session_state`` call sites
        (``tokenize_evidence_pack`` fail-closed, Safe RecGen
        ``reprefill_hierarchy`` / ``fallback_to_baseline_path``) so cache
        cleanup is *always* paired with PHOTON state reset.

        DR1-003 / DR1-007: ``artifacts ⊃ state + cache``. Any future
        session-delete / session-reset API (SessionManager, FastAPI,
        CLI) MUST funnel through this single entry point before mutating
        ``PhotonInference._sessions``. The pop-then-clear order matters
        only insofar as both happen — the cache pop is idempotent
        (``dict.pop(..., None)``) so missing entries are not an error.
        """
        self._relevant_past_turn_cache.pop(session_id, None)
        _clear_photon_session_state(self.photon_inference, session_id)

    def _segment_state_for_session(
        self,
        session_id: str,
        session: SessionState,
    ) -> _TopicSegmentState:
        state = self._topic_segment_cache.get(session_id)
        if state is None:
            state = _TopicSegmentState()
            self._topic_segment_cache[session_id] = state
        turn_segments = state.turn_segments or {}
        for turn in session.turns:
            turn_segments.setdefault(turn.turn_id, state.current_segment_id)
        state.turn_segments = turn_segments
        return state

    def _photon_carryover_matches_for_question(
        self,
        *,
        session_id: str,
        question: str,
        max_turns: int,
        min_similarity: float,
    ) -> list[_PhotonCarryoverMatch]:
        """Compare a transient question state with PHOTON turn history.

        This intentionally uses ``hierarchical_prefill`` rather than
        ``session_forward`` so carryover classification can consult PHOTON
        semantics without mutating the session or appending a turn.
        """
        max_turns = _nonnegative_int(max_turns)
        if max_turns <= 0:
            return []
        min_similarity = max(0.0, min(1.0, float(min_similarity)))
        sessions = getattr(self.photon_inference, "_sessions", None)
        if not isinstance(sessions, dict):
            return []
        photon_session = sessions.get(session_id)
        turn_history = getattr(photon_session, "turn_history", None)
        if not turn_history:
            return []

        try:
            evidence_tokens = tokenize_evidence_pack(
                question,
                self.tokenizer,
                self.photon_cfg,
            )
            if getattr(evidence_tokens, "size", 0) <= 0:
                return []
            _logits, current_state = self.photon_inference.hierarchical_prefill(
                evidence_tokens.reshape(1, -1)
            )
            current_levels = getattr(current_state, "level_states", None)
            if not current_levels:
                return []

            from photon_mlx.session import cosine_distance

            current_top = current_levels[-1]
            scored: list[_PhotonCarryoverMatch] = []
            for past_turn in list(turn_history)[-max_turns:]:
                past_state = getattr(past_turn, "hierarchical_state", None)
                past_levels = getattr(past_state, "level_states", None)
                if not past_levels:
                    continue
                try:
                    score = 1.0 - cosine_distance(current_top, past_levels[-1])
                except (RuntimeError, TypeError, ValueError):
                    continue
                if not math.isfinite(score) or score < min_similarity:
                    continue
                try:
                    turn_id = int(getattr(past_turn, "turn_id"))
                except (TypeError, ValueError):
                    continue
                scored.append(_PhotonCarryoverMatch(turn_id=turn_id, score=score))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _logger.warning(
                "PHOTON carryover matching failed; using lexical carryover "
                "decision (fail-closed, reason=%s)",
                type(exc).__name__,
            )
            return []

        scored.sort(key=lambda item: (item.score, item.turn_id), reverse=True)
        selected = scored[:max_turns]
        return sorted(selected, key=lambda item: item.turn_id)

    @staticmethod
    def _extract_pinned_chunk_ids(
        session: SessionState | None,
        matched: TurnState,
        max_pinned: int,
    ) -> list[str] | None:
        """Look up cited chunks for the matched PHOTON ``turn_id``.

        DR2-004: PHOTON ``turn_count`` and Baseline ``len(turns)`` can drift
        across fail-closed paths (tokenize failure / Safe RecGen reset
        clears ``turn_history`` while preserving ``turn_count``; baseline
        always appends). We therefore guard with
        ``session.turns[idx].turn_id == matched.turn_id`` before trusting
        the index. On drift we fall back to a linear scan; if that still
        fails to locate the matched turn we fail closed (return ``None``)
        rather than risk pinning the wrong chunks.

        DR2-009: dedup is delegated to
        :func:`baseline_reporag.generation.evidence_pack._merge_pinned_sets`
        (set union). This helper performs only the slice; double-counting
        is impossible by construction.

        DR3-001 / DR4-001: the linear-search fallback is O(N) over
        ``session.turns`` but only fires after fail-closed drift; the
        whole helper is wrapped in ``prof.phase("past_turn_pinning")`` by
        the caller. Production telemetry intentionally does not surface
        ``matched_turn_id`` / ``scanned_turns`` — long-session diagnosis
        is restricted to self-hosted benchmarks and unit tests.
        """
        if session is None or not session.turns:
            return None
        idx = matched.turn_id - 1
        if 0 <= idx < len(session.turns):
            candidate = session.turns[idx]
        else:
            candidate = None
        if candidate is None or candidate.turn_id != matched.turn_id:
            # turn_id alignment is broken (PHOTON / Baseline drift after
            # fail-closed). Linear search; failing that, fail closed.
            for t in session.turns:
                if t.turn_id == matched.turn_id:
                    candidate = t
                    break
            else:
                return None
        cited = candidate.cited_chunk_ids
        return list(cited[:max_pinned]) if cited else None

    @staticmethod
    def _extract_related_questions(
        session: SessionState | None,
        matched_turns: list[TurnState],
    ) -> list[str]:
        """Map PHOTON-matched turn ids to prior user questions in turn order."""
        return [
            question
            for _turn_id, question in PhotonRAGPipeline._extract_related_question_pairs(
                session,
                matched_turns,
            )
        ]

    @staticmethod
    def _extract_related_question_pairs(
        session: SessionState | None,
        matched_turns: list[TurnState],
    ) -> list[tuple[int, str]]:
        """Map PHOTON-matched turn ids to ``(turn_id, question)`` in order."""
        if session is None or not session.turns or not matched_turns:
            return []

        questions: list[tuple[int, str]] = []
        seen_turn_ids: set[int] = set()
        for matched in sorted(matched_turns, key=lambda turn: turn.turn_id):
            if matched.turn_id in seen_turn_ids:
                continue
            seen_turn_ids.add(matched.turn_id)
            question = ""
            idx = matched.turn_id - 1
            if 0 <= idx < len(session.turns):
                candidate = session.turns[idx]
                if candidate.turn_id == matched.turn_id:
                    question = candidate.question
            if not question:
                for turn in session.turns:
                    if turn.turn_id == matched.turn_id:
                        question = turn.question
                        break
            if not question:
                question = getattr(matched, "question_text", "")
            if question.strip():
                questions.append((int(matched.turn_id), question.strip()))
        return questions

    # ---------------------------------------------------------------
    # Issue #62 Phase 1: opt-in PHOTON single-path generation
    # ---------------------------------------------------------------

    @staticmethod
    def _resolve_photon_max_new_tokens(
        followup_tokens: int | None,
        inference_cfg: Any,
        cfg: Config,
    ) -> int:
        """Resolve the Phase 1 ``max_new_tokens`` contract (DR-62-005 / DR1-004).

        Precedence:
        1. ``followup_tokens`` when non-None (multi-turn cap).
        2. ``inference.answer_max_new_tokens`` when set.
        3. ``generation.max_new_tokens`` when a top-level generation section
           exists (non-photon configs).
        4. Hard default ``512`` (matches Qwen first-turn behaviour).

        Strict type enforcement (DR4-003): rejects ``bool`` and non-``int``,
        rejects values < 1.
        """
        if followup_tokens is not None:
            raw_value: Any = followup_tokens
        else:
            raw_value = getattr(inference_cfg, "answer_max_new_tokens", None)
            if raw_value is None:
                generation_cfg = cfg.get("generation")
                if generation_cfg is not None:
                    raw_value = getattr(generation_cfg, "max_new_tokens", 512)
                else:
                    raw_value = 512

        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValueError(
                "PHOTON max_new_tokens must be a positive int, "
                f"got {type(raw_value).__name__}"
            )
        if raw_value < 1:
            raise ValueError(f"PHOTON max_new_tokens must be >= 1, got {raw_value}")
        return raw_value

    @staticmethod
    def _run_qwen_generation(
        generator: Any,
        messages: list[dict],
        *,
        max_new_tokens: int | None,
        seed: int | None,
    ) -> str:
        """Run Qwen with the baseline call shape when no cap is specified."""
        if max_new_tokens is None:
            if seed is not None:
                return generator.generate(messages, seed=seed)
            return generator.generate(messages)
        if seed is not None:
            return generator.generate(
                messages, max_new_tokens=max_new_tokens, seed=seed
            )
        return generator.generate(messages, max_new_tokens=max_new_tokens)

    def _run_photon_generation(
        self,
        *,
        messages: list[dict],
        bl: RepoRAGPipeline,
        cfg: Config,
        inference_cfg: Any,
        followup_tokens: int | None,
        fallback_policy: str,
        seed: int | None = None,
    ) -> tuple[str, str, str | None]:
        """Execute the PHOTON generation branch with fail-closed semantics.

        Returns ``(answer, generator_used, generator_fallback_reason)``.

        Contract (design §8.2 + §9):

        - ``_TokenizerEncodeFailure`` / ``ValueError`` / ``RuntimeError`` →
          fall back to Qwen unless ``fallback_policy == "abort"`` in which
          case a ``RuntimeError`` is raised with a sanitized message.
        - Empty PHOTON output → fall back with ``generator_fallback_reason
          == "empty_output"``.
        - Security logging: warning message uses ``type(exc).__name__``
          only; the raw exception body is never logged (Stage 4 DR4-002).

        Issue #143: when ``seed`` is provided, both Qwen fallback paths
        propagate it into ``Generator.generate`` so eval scripts get a
        reproducible Qwen output even when PHOTON falls back. Default
        ``seed=None`` preserves the legacy single-positional call shape
        used by every existing MagicMock test in this module.
        """
        from photon_mlx.inference import _TokenizerEncodeFailure

        prompt_text = flatten_messages_for_plain_lm(messages)  # DR-62-003
        photon_max_new = self._resolve_photon_max_new_tokens(
            followup_tokens, inference_cfg, cfg
        )

        try:
            photon_answer = self.photon_inference.generate_answer(
                prompt_text,
                max_new_tokens=photon_max_new,
            )
        except (_TokenizerEncodeFailure, ValueError, RuntimeError) as exc:
            reason = type(exc).__name__
            if fallback_policy == "abort":
                # Sanitized error — do NOT include exc body in the message.
                raise RuntimeError(
                    "PHOTON generation failed and fallback policy=abort"
                ) from None
            # Stage 4 DR4-002: log the closed-enum reason only; the raw
            # exception body must not appear in the warning.
            _logger.warning(
                "PHOTON generation failed; falling back to Qwen (reason=%s)",
                reason,
            )
            # Issue #143 / DR3-002: ``if seed is not None`` (NOT
            # ``if seed:``) — seed=0 is a valid deterministic seed.
            qwen_answer = self._run_qwen_generation(
                bl.generator,
                messages,
                max_new_tokens=followup_tokens,
                seed=seed,
            )
            return qwen_answer, "qwen", reason

        # DR1-001: empty / whitespace-only output is fail-closed.
        if not photon_answer or not photon_answer.strip():
            _logger.warning(
                "PHOTON returned empty answer; falling back to Qwen (reason=%s)",
                "empty_output",
            )
            if fallback_policy == "abort":
                raise RuntimeError("PHOTON generation failed and fallback policy=abort")
            qwen_answer = self._run_qwen_generation(
                bl.generator,
                messages,
                max_new_tokens=followup_tokens,
                seed=seed,
            )
            return qwen_answer, "qwen", "empty_output"

        return photon_answer, "photon", None

    def query(
        self,
        question: str,
        session_id: str = "",
        repo_id: str = "",
        *,
        seed: int | None = None,
    ) -> QueryResult:
        """Run PHOTON-enhanced query with drift tracking and evidence pruning.

        On follow-up turns (turn 2+), PHOTON coarse state is used to prune
        the evidence pack from max_chunks down to pruned_max_chunks, halving
        LLM prefill time while retaining the most session-relevant chunks.

        Issue #143: ``seed`` (keyword-only, default ``None``) is forwarded
        into every Qwen call site (Qwen-only path + 2 fallback paths) so
        eval scripts can reproduce Qwen sampling. ``seed=None`` keeps
        interactive callers unaffected (CLI / server / Streamlit) and
        preserves the legacy single-positional Qwen call shape so the
        17+ existing MagicMock tests in ``test_photon_pipeline.py`` keep
        passing without TypeError.
        """
        cfg = self.cfg
        bl = self.baseline  # access baseline components without calling query()
        prof = TurnProfiler()
        prof.start()

        session_id = session_id or str(uuid.uuid4())
        repo_id = repo_id or cfg.repo.repo_id
        session = bl.sessions.get_or_create(
            session_id,
            repo_id,
            cfg.repo.repo_commit,
        )

        photon_session_id = session_id or "default"

        is_follow_up = len(session.turns) > 0
        segment_state = self._segment_state_for_session(photon_session_id, session)
        inference_cfg = cfg.get("inference")
        context_carryover_enabled = (
            bool(getattr(inference_cfg, "context_carryover_enabled", True))
            if inference_cfg is not None
            else True
        )
        context_rewrite_enabled = (
            bool(getattr(inference_cfg, "context_rewrite_enabled", True))
            if inference_cfg is not None
            else True
        )
        rewrite_history_max = (
            _nonnegative_int(
                getattr(
                    inference_cfg,
                    "context_rewrite_history_max",
                    _REWRITE_HISTORY_MAX,
                ),
                default=_REWRITE_HISTORY_MAX,
            )
            if inference_cfg is not None
            else _REWRITE_HISTORY_MAX
        )
        photon_carryover_enabled = (
            bool(getattr(inference_cfg, "photon_carryover_enabled", True))
            if inference_cfg is not None
            else True
        )
        photon_carryover_max_turns = (
            _nonnegative_int(
                getattr(
                    inference_cfg,
                    "photon_carryover_max_turns",
                    _PHOTON_CARRYOVER_MAX_TURNS,
                ),
                default=_PHOTON_CARRYOVER_MAX_TURNS,
            )
            if inference_cfg is not None
            else _PHOTON_CARRYOVER_MAX_TURNS
        )
        photon_carryover_min_similarity = (
            float(
                getattr(
                    inference_cfg,
                    "photon_carryover_min_similarity",
                    _PHOTON_CARRYOVER_MIN_SIMILARITY,
                )
            )
            if inference_cfg is not None
            else _PHOTON_CARRYOVER_MIN_SIMILARITY
        )
        initial_carryover = _resolve_context_carryover(
            question,
            session,
            enabled=context_carryover_enabled,
            rewrite_enabled=context_rewrite_enabled,
            rewrite_history_max=rewrite_history_max,
        )
        explicit_topic_switch = _has_explicit_topic_switch_signal(
            question,
            _recent_questions(session, max(1, rewrite_history_max)),
        )
        if initial_carryover.mode == "weak" and explicit_topic_switch:
            initial_carryover = _CarryoverDecision(
                mode="weak",
                query=question,
                similarity=initial_carryover.similarity,
                marker=initial_carryover.marker,
                reason=f"{initial_carryover.reason}+explicit_topic_switch",
            )
        photon_carryover_matches: list[_PhotonCarryoverMatch] = []
        if (
            is_follow_up
            and photon_carryover_enabled
            and initial_carryover.mode == "weak"
            and not explicit_topic_switch
        ):
            effective_min_similarity = photon_carryover_min_similarity
            if not initial_carryover.marker:
                effective_min_similarity = min(1.0, effective_min_similarity + 0.1)
            photon_carryover_matches = self._photon_carryover_matches_for_question(
                session_id=photon_session_id,
                question=question,
                max_turns=photon_carryover_max_turns,
                min_similarity=effective_min_similarity,
            )
            initial_carryover = _boost_carryover_with_photon_matches(
                initial_carryover,
                question,
                session,
                matches=photon_carryover_matches,
                rewrite_enabled=context_rewrite_enabled,
                rewrite_history_max=rewrite_history_max,
            )
        current_segment_id = segment_state.current_segment_id
        segment_memory = _SegmentMemoryDecision(False, 0.0, "not_considered", question)
        if is_follow_up and initial_carryover.mode == "weak" and not explicit_topic_switch:
            previous_segment_questions = _recent_questions_in_segment(
                session,
                segment_state,
                current_segment_id,
                max(1, rewrite_history_max),
            )
            if not previous_segment_questions:
                previous_segment_questions = _recent_questions(
                    session,
                    max(1, rewrite_history_max),
                )
            segment_memory = _resolve_segment_memory(
                question,
                previous_segment_questions,
                rewrite_enabled=context_rewrite_enabled,
                rewrite_history_max=rewrite_history_max,
            )
            if segment_memory.applied:
                initial_carryover = _CarryoverDecision(
                    mode="mixed",
                    query=segment_memory.query,
                    similarity=max(initial_carryover.similarity, segment_memory.score),
                    marker=initial_carryover.marker or _has_follow_up_marker(question),
                    reason=f"{initial_carryover.reason}+{segment_memory.reason}",
                )
        if is_follow_up and initial_carryover.mode == "weak":
            current_segment_id += 1
            segment_state.current_segment_id = current_segment_id
        segment_questions = _recent_questions_in_segment(
            session,
            segment_state,
            current_segment_id,
            max(1, rewrite_history_max),
        )
        carryover = _resolve_context_carryover(
            question,
            session,
            enabled=context_carryover_enabled,
            rewrite_enabled=context_rewrite_enabled,
            rewrite_history_max=rewrite_history_max,
            rewrite_questions=segment_questions,
        )
        carryover = _boost_carryover_with_photon_matches(
            carryover,
            question,
            session,
            matches=photon_carryover_matches,
            rewrite_enabled=context_rewrite_enabled,
            rewrite_history_max=rewrite_history_max,
        )
        if (
            is_follow_up
            and carryover.mode == "weak"
            and not explicit_topic_switch
            and not segment_memory.applied
        ):
            memory_questions = segment_questions or _recent_questions(
                session,
                max(1, rewrite_history_max),
            )
            segment_memory = _resolve_segment_memory(
                question,
                memory_questions,
                rewrite_enabled=context_rewrite_enabled,
                rewrite_history_max=rewrite_history_max,
            )
        if segment_memory.applied and carryover.mode == "weak":
            carryover = _CarryoverDecision(
                mode="mixed",
                query=segment_memory.query,
                similarity=max(carryover.similarity, segment_memory.score),
                marker=carryover.marker or _has_follow_up_marker(question),
                reason=f"{carryover.reason}+{segment_memory.reason}",
            )
        if is_follow_up and initial_carryover.mode == "weak":
            carryover = _CarryoverDecision(
                mode="weak",
                query=question,
                similarity=initial_carryover.similarity,
                marker=initial_carryover.marker,
                reason=initial_carryover.reason,
            )
        retrieval_query = carryover.query
        if is_follow_up and carryover.mode == "weak":
            # A fresh independent question should not reuse a stale PHOTON
            # hierarchy from an earlier topic. The current turn will prefill a
            # new state after evidence is built.
            self._clear_photon_session_artifacts(photon_session_id)

        # --- Query expansion ---
        qe_cfg = cfg.retrieval.query_expansion
        if qe_cfg.get("enabled", False):
            _queries = expand_query(retrieval_query, mapping=qe_cfg.get("domain_map"))
            expansion_terms: str | None = _queries[1] if len(_queries) > 1 else None
        else:
            expansion_terms = None

        # --- Two-pass search configuration (Issue #56) ---
        two_pass_enabled, pass1_top_k, pass2_top_k = _resolve_two_pass_search_cfg(
            cfg.retrieval,
            fused_top_k=cfg.retrieval.fused_top_k,
            evidence_max_chunks=cfg.evidence_pack.max_chunks,
        )
        effective_fused_top_k = (
            max(cfg.retrieval.fused_top_k, pass1_top_k)
            if two_pass_enabled and not is_follow_up
            else cfg.retrieval.fused_top_k
        )

        # --- Retrieval ---
        with prof.phase("retrieval"):
            raw = hybrid_search(
                query=retrieval_query,
                lexical_index=bl.lexical,
                embedding_index=bl.embedding,
                lexical_top_k=cfg.retrieval.lexical_top_k,
                embedding_top_k=cfg.retrieval.embedding_top_k,
                fused_top_k=effective_fused_top_k,
                lexical_weight=cfg.retrieval.weights.lexical,
                embedding_weight=cfg.retrieval.weights.embedding,
                expanded_queries=[expansion_terms] if expansion_terms else [],
                repo_id=repo_id,
            )

        # Snapshot pre-rerank scores for debug rows (Issue #176)
        raw_snapshot = list(raw)

        # --- Reranking ---
        # On follow-up turns, PHOTON pruning handles chunk selection.  On turn
        # 1 (or when no reranker is configured), reranking runs as usual.  The
        # current-turn Safe RecGen fallback decision is now computed *after*
        # the evidence pack is built (see design §5.3 / Issue #58), so it no
        # longer gates reranking or pruning within this turn.
        reranked_top: list = []
        rejected_debug: list = []
        with prof.phase("reranking"):
            if bl.reranker is not None and not is_follow_up:
                reranked_top, rejected_debug = bl.reranker.rerank_with_debug(
                    query=question,
                    results=raw,
                    store=bl.store,
                    top_k=cfg.retrieval.rerank_top_k,
                    rerank_query=expansion_terms,
                    rejected_debug_top_n=10,
                )
                raw = reranked_top

        # --- File-type boost ---
        file_type_boost = cfg.retrieval.get("file_type_boost", 0.0)
        if file_type_boost:
            raw = apply_file_type_boost(raw, boost=file_type_boost)

        # --- Graph expansion ---
        with prof.phase("graph_expansion"):
            expanded_refs: list[ExpandedChunkRef] = expand_with_graph(
                results=raw,
                store=bl.store,
                graph=bl.graph,
                repo_id=repo_id,
                repo_commit=cfg.repo.repo_commit,
                max_hops=cfg.retrieval.graph_expansion.max_hops,
                max_nodes=cfg.retrieval.graph_expansion.max_nodes,
                neighborhood_before=cfg.retrieval.neighborhood_expansion.before,
                neighborhood_after=cfg.retrieval.neighborhood_expansion.after,
            )
        expanded_ids = [ref.chunk_id for ref in expanded_refs]

        # --- Evidence pruning (PHOTON-guided) and Pass 1 scoring (Issue #56) ---
        # Uses the *previous* turn's coarse state on Turn 2+ (1-pass constraint,
        # design §4); Turn 1 optionally scores with a question-derived transient
        # coarse_vec when two_pass_search.enabled=true (Issue #56, DR1-003).
        inference_cfg = cfg.get("inference")
        pruning_enabled = (
            getattr(inference_cfg, "evidence_pruning_enabled", False)
            if inference_cfg is not None
            else False
        )
        pruned_max_chunks = (
            getattr(inference_cfg, "pruned_max_chunks", 8)
            if inference_cfg is not None
            else 8
        )
        protected_top_n = (
            _nonnegative_int(getattr(inference_cfg, "pruning_protected_top_n", 0))
            if inference_cfg is not None
            else 0
        )
        photon_top_m = (
            _nonnegative_int(getattr(inference_cfg, "pruning_photon_top_m", 0))
            if inference_cfg is not None
            else 0
        )
        related_questions_max = (
            _nonnegative_int(
                getattr(
                    inference_cfg,
                    "related_past_questions_max",
                    _RELATED_PAST_QUESTIONS_MAX,
                ),
                default=_RELATED_PAST_QUESTIONS_MAX,
            )
            if inference_cfg is not None
            else _RELATED_PAST_QUESTIONS_MAX
        )
        related_evidence_top_k = (
            _nonnegative_int(
                getattr(
                    inference_cfg,
                    "related_past_evidence_top_k",
                    _RELATED_PAST_EVIDENCE_TOP_K,
                ),
                default=_RELATED_PAST_EVIDENCE_TOP_K,
            )
            if inference_cfg is not None
            else _RELATED_PAST_EVIDENCE_TOP_K
        )
        past_context_decay = (
            float(getattr(inference_cfg, "past_context_decay", _PAST_CONTEXT_DECAY))
            if inference_cfg is not None
            else _PAST_CONTEXT_DECAY
        )
        past_context_min_decay = (
            float(
                getattr(
                    inference_cfg,
                    "past_context_min_decay",
                    _PAST_CONTEXT_MIN_DECAY,
                )
            )
            if inference_cfg is not None
            else _PAST_CONTEXT_MIN_DECAY
        )
        admission_min_current_score = (
            float(
                getattr(
                    inference_cfg,
                    "admission_min_current_score",
                    _ADMISSION_MIN_CURRENT_SCORE,
                )
            )
            if inference_cfg is not None
            else _ADMISSION_MIN_CURRENT_SCORE
        )
        effective_max_chunks = cfg.evidence_pack.max_chunks
        do_pass1 = two_pass_enabled and not is_follow_up
        do_pass2plus = pruning_enabled and is_follow_up
        split_pruning_enabled = (
            inference_cfg is not None
            and (
                hasattr(inference_cfg, "pruning_protected_top_n")
                or hasattr(inference_cfg, "pruning_photon_top_m")
            )
            and do_pass2plus
        )
        photon_pruned_ids: list[str] = []
        photon_score_map: dict[str, float] = {}
        photon_current_score_map: dict[str, float] = {}
        photon_session_score_map: dict[str, float] = {}
        photon_pruning_applied = bool(do_pass1 or do_pass2plus)
        photon_scoring_mode: str | None = None
        dual_score_pruning_applied = False
        current_query_frame_ids: list[str] = []
        if do_pass1 or do_pass2plus:
            chunks_for_scoring = bl.store.get_many(expanded_ids)
            chunk_texts = [c.content for c in chunks_for_scoring]
            chunk_ids_for_scoring = [c.chunk_id for c in chunks_for_scoring]
            if do_pass1:
                pruning_question: str | None = question
                photon_scoring_mode = "turn1_question"
            elif carryover.mode == "weak":
                pruning_question = retrieval_query
                photon_scoring_mode = "topic_switch_current_query"
            elif _photon_session_has_pruning_state(
                self.photon_inference, photon_session_id
            ):
                pruning_question = None
                photon_scoring_mode = "session_state"
            else:
                pruning_question = retrieval_query
                photon_scoring_mode = "question_fallback_no_state"

            if split_pruning_enabled:
                scoring_max_chunks = photon_top_m
                selected_indices = []
                if photon_top_m > 0:
                    with prof.phase("evidence_pruning"):
                        selected_indices = self.photon_inference.prune_evidence(
                            chunk_texts=chunk_texts,
                            chunk_ids=chunk_ids_for_scoring,
                            session_id=photon_session_id,
                            max_chunks=photon_top_m,
                            question=pruning_question,
                        )
                    selection_scores = _last_prune_score_map(
                        self.photon_inference,
                        photon_session_id,
                    )
                    if pruning_question is None:
                        photon_session_score_map = dict(selection_scores)
                        photon_current_score_map = _score_current_question_candidates(
                            self.photon_inference,
                            chunk_texts=chunk_texts,
                            chunk_ids=chunk_ids_for_scoring,
                            question=retrieval_query,
                        )
                    else:
                        photon_current_score_map = dict(selection_scores)
                ranked_chunk_ids = [
                    str(getattr(result, "chunk_id", "")) for result in raw
                ]
                protected_indices = _unique_candidate_indices_by_rank(
                    ranked_chunk_ids,
                    chunk_ids_for_scoring,
                    protected_top_n,
                )
                current_query_frame_ids = [
                    chunk_ids_for_scoring[idx]
                    for idx in protected_indices
                    if 0 <= idx < len(chunk_ids_for_scoring)
                ]
                retrieval_scores = _normalised_score_map(list(raw_snapshot) + list(raw))
                if photon_top_m > 0:
                    selected_indices = _dual_score_candidate_indices(
                        candidate_indices=list(range(len(chunk_ids_for_scoring))),
                        protected_indices=protected_indices,
                        chunk_ids_for_scoring=chunk_ids_for_scoring,
                        retrieval_scores=retrieval_scores,
                        current_scores=photon_current_score_map,
                        session_scores=photon_session_score_map,
                        carryover_mode=carryover.mode,
                        max_extra=photon_top_m,
                    )
                    dual_score_pruning_applied = True
                else:
                    selected_indices = _merge_protected_and_photon_indices(
                        ranked_chunk_ids=ranked_chunk_ids,
                        chunk_ids_for_scoring=chunk_ids_for_scoring,
                        photon_indices=[],
                        protected_top_n=protected_top_n,
                    )
            else:
                scoring_max_chunks = pass2_top_k if do_pass1 else pruned_max_chunks
                with prof.phase("pass1_scoring" if do_pass1 else "evidence_pruning"):
                    selected_indices = self.photon_inference.prune_evidence(
                        chunk_texts=chunk_texts,
                        chunk_ids=chunk_ids_for_scoring,
                        session_id=photon_session_id,
                        max_chunks=scoring_max_chunks,
                        question=pruning_question,
                    )
                selection_scores = _last_prune_score_map(
                    self.photon_inference,
                    photon_session_id,
                )
                if pruning_question is None:
                    photon_session_score_map = dict(selection_scores)
                    photon_current_score_map = _score_current_question_candidates(
                        self.photon_inference,
                        chunk_texts=chunk_texts,
                        chunk_ids=chunk_ids_for_scoring,
                        question=retrieval_query,
                    )
                else:
                    photon_current_score_map = dict(selection_scores)
            selected_set = {chunk_ids_for_scoring[i] for i in selected_indices}
            photon_pruned_ids = [
                cid for cid in chunk_ids_for_scoring if cid not in selected_set
            ]
            photon_score_map = _last_prune_score_map(
                self.photon_inference,
                photon_session_id,
            )
            selected_ids = [chunk_ids_for_scoring[i] for i in selected_indices]
            if do_pass2plus and not current_query_frame_ids:
                fallback_current_limit = protected_top_n or min(4, len(selected_ids))
                ranked_chunk_ids = [
                    str(getattr(result, "chunk_id", "")) for result in raw
                ]
                current_query_frame_ids = _dedupe_preserve_order(
                    [cid for cid in ranked_chunk_ids if cid in set(selected_ids)]
                )[:fallback_current_limit]
            expanded_ids = selected_ids
            effective_max_chunks = (
                len(expanded_ids) if split_pruning_enabled else scoring_max_chunks
            )

        # --- Issue #103: read cached past-turn pin (before evidence pack) ---
        # DR2-001: use the module-level helper (PhotonRAGPipeline has no
        # ``_resolve_working_memory_cfg`` method). DR2-002: the helper
        # returns ``None`` when the YAML lacks ``working_memory:`` or the
        # block is malformed — guard before accessing fields.
        working_memory_cfg = _extract_working_memory_cfg(cfg)
        pinning_enabled = (
            working_memory_cfg is not None
            and working_memory_cfg.past_turn_pinning_enabled
        )
        additional_pinned_ids: list[str] | None = None
        if pinning_enabled and is_follow_up and carryover.mode != "weak":
            cached_turn = self._relevant_past_turn_cache.pop(photon_session_id, None)
            cached_segment = (
                (segment_state.turn_segments or {}).get(int(cached_turn.turn_id))
                if cached_turn is not None
                else None
            )
            if cached_turn is not None and cached_segment == current_segment_id:
                additional_pinned_ids = self._extract_pinned_chunk_ids(
                    session,
                    cached_turn,
                    working_memory_cfg.max_pinned_chunks,
                )
        elif pinning_enabled and is_follow_up:
            self._relevant_past_turn_cache.pop(photon_session_id, None)

        # Build skeleton debug rows (Issue #176): expanded_refs holds source
        # tags; photon_pruned and working_memory rows are appended after.
        all_refs_for_debug: list[ExpandedChunkRef] = list(expanded_refs)
        for cid in photon_pruned_ids:
            all_refs_for_debug.append(
                ExpandedChunkRef(chunk_id=cid, source="photon_pruned")
            )
        if additional_pinned_ids:
            for cid in additional_pinned_ids:
                all_refs_for_debug.append(
                    ExpandedChunkRef(chunk_id=cid, source="working_memory")
                )

        frame_pinned_ids = _compose_evidence_frame_pins(
            current_query_ids=current_query_frame_ids,
            working_memory_ids=additional_pinned_ids,
        )

        # --- Evidence pack ---
        with prof.phase("evidence_pack"):
            pack = build_evidence_pack(
                chunk_ids=expanded_ids,
                store=bl.store,
                session=session,
                max_chunks=effective_max_chunks,
                max_tokens=cfg.evidence_pack.max_tokens,
                additional_pinned_ids=frame_pinned_ids or None,
            )

        # --- PHOTON prefill on question + evidence (new coarse state) ---
        # Issue #58: the coarse state is now built from the concatenation of
        # the question and the evidence text so drift, Safe RecGen, and the
        # next turn's prune_evidence operate in a richer semantic space.
        # Fail-closed: if tokenization fails we clear the PHOTON session
        # state and fall through to the baseline generation path rather than
        # silently reusing a stale coarse state on the next turn (design §8
        # + CB-001).
        evidence_text_for_photon = pack.format_for_prompt()
        photon_input_text = question + "\n\n" + evidence_text_for_photon
        drift = None
        drift_dict = None
        confidence = 1.0
        tokenization_failed = False
        related_past_questions: list[str] = []
        related_past_question_pairs: list[tuple[int, str]] = []
        try:
            evidence_tokens = tokenize_evidence_pack(
                photon_input_text,
                self.tokenizer,
                self.photon_cfg,
            )
        except Exception as exc:
            # Security logging (Issue #58 CB-001 + Issue #64 Codex CB-002):
            # log only the closed-enum exception class name. The tokenizer
            # was handed ``question + evidence_pack``; a pathological or
            # mis-configured tokenizer could echo that payload back in its
            # exception message, so surfacing ``str(exc)`` / ``%s % exc``
            # would leak question/evidence fragments to log sinks (design §7
            # bars raw question_text and attacker-controlled values from
            # fail-closed telemetry).
            _logger.warning(
                "tokenize_evidence_pack failed; clearing PHOTON session "
                "state and falling back to baseline path for this turn "
                "(fail-closed, CB-001, Codex CB-002, reason=%s)",
                type(exc).__name__,
            )
            tokenization_failed = True
            evidence_tokens = mx.array([], dtype=mx.int32)
            # Explicit fail-closed: drop any prior coarse/prev state so the
            # next turn cannot reuse a stale hierarchy.  No raw input text,
            # token ids, or latents are retained (design §8). Issue #103
            # routes through ``_clear_photon_session_artifacts`` so the
            # past-turn pin sidecar cache is also dropped.
            self._clear_photon_session_artifacts(photon_session_id)

        if evidence_tokens.size > 0:
            input_ids = evidence_tokens.reshape(1, -1)
            logits, drift = self.photon_inference.session_forward(
                input_ids,
                session_id=photon_session_id,
                repo_id=repo_id or "unknown",
                repo_commit="HEAD",
                question=question,
            )
            confidence = compute_confidence(logits)
            drift_dict = drift.as_dict() if drift else None
            if (
                is_follow_up
                and working_memory_cfg is not None
                and carryover.mode != "weak"
            ):
                photon_session = self.photon_inference._sessions.get(photon_session_id)
                finder = (
                    getattr(photon_session, "find_relevant_past_turns", None)
                    if photon_session is not None
                    else None
                )
                if callable(finder):
                    try:
                        matched_turns = finder(
                            photon_session.current_state,
                            max_turns=related_questions_max,
                        )
                    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                        _logger.warning(
                            "find_relevant_past_turns failed; skipping related "
                            "questions (fail-closed, reason=%s)",
                            type(exc).__name__,
                        )
                        matched_turns = []
                    if isinstance(matched_turns, list):
                        related_past_question_pairs = self._extract_related_question_pairs(
                            session,
                            matched_turns,
                        )
                        current_turn_id = len(session.turns) + 1
                        related_past_question_pairs = [
                            (turn_id, related_question)
                            for turn_id, related_question in related_past_question_pairs
                            if (segment_state.turn_segments or {}).get(turn_id)
                            == current_segment_id
                            and _turn_decay(
                                current_turn_id,
                                turn_id,
                                past_context_decay,
                            )
                            >= past_context_min_decay
                        ]
                        related_past_questions = [
                            question for _turn_id, question in related_past_question_pairs
                        ]

        related_past_evidence_ids: list[str] = []
        if is_follow_up and related_past_question_pairs and related_evidence_top_k > 0:
            with prof.phase("retrieval"):
                current_turn_id = len(session.turns) + 1
                for turn_id, related_question in related_past_question_pairs:
                    decay_weight = _turn_decay(
                        current_turn_id,
                        turn_id,
                        past_context_decay,
                    )
                    effective_related_top_k = max(
                        1,
                        min(
                            related_evidence_top_k,
                            round(related_evidence_top_k * decay_weight),
                        ),
                    )
                    related_raw = hybrid_search(
                        query=related_question,
                        lexical_index=bl.lexical,
                        embedding_index=bl.embedding,
                        lexical_top_k=cfg.retrieval.lexical_top_k,
                        embedding_top_k=cfg.retrieval.embedding_top_k,
                        fused_top_k=effective_related_top_k,
                        lexical_weight=cfg.retrieval.weights.lexical,
                        embedding_weight=cfg.retrieval.weights.embedding,
                        expanded_queries=[retrieval_query],
                        repo_id=repo_id,
                    )
                    related_past_evidence_ids.extend(
                        str(getattr(result, "chunk_id", "")) for result in related_raw
                    )
            related_past_evidence_ids = [
                cid for cid in _dedupe_preserve_order(related_past_evidence_ids) if cid
            ]

        related_past_refs = _expand_related_past_refs(
            related_past_evidence_ids,
            store=bl.store,
            neighborhood_before=cfg.retrieval.neighborhood_expansion.before,
            neighborhood_after=cfg.retrieval.neighborhood_expansion.after,
        )
        related_past_pack_ids = [ref.chunk_id for ref in related_past_refs]

        if related_past_pack_ids:
            related_existing = {ref.chunk_id for ref in all_refs_for_debug}
            for ref in related_past_refs:
                if ref.chunk_id not in related_existing:
                    all_refs_for_debug.append(ref)
                    related_existing.add(ref.chunk_id)
            combined_pinned_ids = _compose_evidence_frame_pins(
                current_query_ids=current_query_frame_ids,
                working_memory_ids=additional_pinned_ids,
                related_past_ids=related_past_pack_ids,
            )
            with prof.phase("evidence_pack"):
                pack = build_evidence_pack(
                    chunk_ids=expanded_ids,
                    store=bl.store,
                    session=session,
                    max_chunks=min(
                        cfg.evidence_pack.max_chunks,
                        effective_max_chunks + len(related_past_pack_ids),
                    ),
                    max_tokens=cfg.evidence_pack.max_tokens,
                    additional_pinned_ids=combined_pinned_ids,
                )

        # --- Safe RecGen evaluation (uses new coarse state) ---
        fallback_dict = None
        if self.safe_recgen is not None and drift is not None:
            decision = self.safe_recgen.evaluate(
                question, drift=drift, confidence=confidence
            )
            fallback_dict = decision.as_dict()
        fallback_actions = (
            set(fallback_dict.get("actions", [])) if fallback_dict else set()
        )

        # A fallback that invalidates the PHOTON hierarchy must clear the
        # session state (including prev_logits) so subsequent turns do not
        # reuse a coarse state or drift reference from a stale topic
        # (design §8 fail-closed; Codex CB-004). Issue #103 routes through
        # ``_clear_photon_session_artifacts`` so the past-turn pin sidecar
        # cache is dropped in lockstep with PHOTON state.
        if fallback_actions & {"reprefill_hierarchy", "fallback_to_baseline_path"}:
            self._clear_photon_session_artifacts(photon_session_id)

        # --- Issue #103: write past-turn pin cache for next turn ---
        # 3 branches (design §4-3):
        #   OFF                       → skip entirely (no profiler phase).
        #   drift is None             → pop only (DR2-011 stale-cache safety).
        #   drift is not None         → try find_relevant_past_turn.
        # DR4-001: production observability is limited to the
        # ``past_turn_pinning`` phase duration and the failure exception
        # class name — turn_id, similarity, and scanned_turns are NOT
        # logged so attacker-controlled YAML or pathological session state
        # cannot leak into log sinks.
        if pinning_enabled:
            if drift is None:
                # tokenize fail-closed / Safe RecGen reset / session_forward
                # not run: drop any stale cache entry from the prior turn so
                # the next turn cannot consume a misaligned pin.
                self._relevant_past_turn_cache.pop(photon_session_id, None)
            else:
                with prof.phase("past_turn_pinning"):
                    photon_session = self.photon_inference._sessions.get(
                        photon_session_id
                    )
                    match: TurnState | None
                    if photon_session is not None:
                        try:
                            match = photon_session.find_relevant_past_turn(
                                photon_session.current_state
                            )
                        except (AttributeError, RuntimeError, ValueError) as exc:
                            # DR1-002 + Codex CB-001/CB-002: closed exception
                            # set (no Pokémon catch). Only the type name is
                            # surfaced, never raw message content.
                            _logger.warning(
                                "find_relevant_past_turn failed; skipping "
                                "pin cache (fail-closed, reason=%s)",
                                type(exc).__name__,
                            )
                            match = None
                    else:
                        # PHOTON session was never initialised for this
                        # session_id — keep the cache empty.
                        match = None

                    if match is not None:
                        self._relevant_past_turn_cache[photon_session_id] = match
                    else:
                        self._relevant_past_turn_cache.pop(photon_session_id, None)

        retrieval_support_scores = _normalised_score_map(list(raw_snapshot) + list(raw))
        support_score = _support_score_for_pack(
            question=question,
            pack_chunks=list(pack.chunks),
            retrieval_scores=retrieval_support_scores,
            current_scores=photon_current_score_map,
            session_scores=photon_session_score_map,
        )
        support_guard_active = support_score < _SUPPORT_GUARD_THRESHOLD

        # --- Generation (Issue #62 Phase 1: opt-in PHOTON single-path) ---
        # DR-62-001 / DR4-003: strict bool validation for the opt-in flag.
        raw_photon_gen_enabled = (
            getattr(inference_cfg, "photon_generation_enabled", False)
            if inference_cfg is not None
            else False
        )
        if not isinstance(raw_photon_gen_enabled, bool):
            raise ValueError(
                "inference.photon_generation_enabled must be bool, "
                f"got {type(raw_photon_gen_enabled).__name__}"
            )
        photon_gen_enabled = raw_photon_gen_enabled

        # DR4-004: closed-enum validation for the deployment policy knob.
        fallback_policy = (
            getattr(inference_cfg, "generation_fallback_policy", "qwen")
            if inference_cfg is not None
            else "qwen"
        )
        if fallback_policy not in {"qwen", "abort"}:
            raise ValueError(
                "inference.generation_fallback_policy must be 'qwen' or 'abort', "
                f"got {fallback_policy!r}"
            )

        generator_used = "qwen"
        generator_fallback_reason: str | None = None

        with prof.phase("generation"):
            evidence_text = pack.format_for_prompt()
            evidence_text = (
                _support_check_note(
                    support_score,
                    guard_active=support_guard_active,
                )
                + "\n\n"
                + evidence_text
            )
            is_first_turn = len(session.turns) == 0
            if is_first_turn:
                evidence_text = f"{_EVIDENCE_HEADER}\n\n{evidence_text}"
            generation_history = _generation_history_text(
                session,
                segment_state,
                segment_id=current_segment_id,
                carryover_mode=carryover.mode,
                max_turns=4,
            )
            messages = build_messages(
                question=question,
                evidence_text=evidence_text,
                history_text=generation_history,
                related_questions=related_past_questions,
                include_few_shot=is_first_turn,
            )
            # Keep PHOTON follow-up generation aligned with the baseline path:
            # do not impose a shorter per-turn cap here. The generator will use
            # cfg.generation.max_new_tokens unless an explicit PHOTON generation
            # limit is configured.
            followup_tokens = None

            if photon_gen_enabled:
                (
                    answer,
                    generator_used,
                    generator_fallback_reason,
                ) = self._run_photon_generation(
                    messages=messages,
                    bl=bl,
                    cfg=cfg,
                    inference_cfg=inference_cfg,
                    followup_tokens=followup_tokens,
                    fallback_policy=fallback_policy,
                    seed=seed,
                )
            else:
                # Issue #143 / DR3-002: ``if seed is not None`` (NOT
                # ``if seed:``); seed=0 is a valid deterministic seed.
                # Default ``seed=None`` keeps the legacy single-positional
                # call shape for interactive callers + 17+ MagicMock
                # tests.
                answer = self._run_qwen_generation(
                    bl.generator,
                    messages,
                    max_new_tokens=followup_tokens,
                    seed=seed,
                )

        # --- Citation ---
        with prof.phase("citation"):
            citation = resolve_citations(answer, pack)
            answering_cfg = getattr(cfg, "answering", None)
            if answering_cfg is not None:
                postprocess_enabled = answering_cfg.get(
                    "citation_postprocess_enabled", True
                )
            else:
                postprocess_enabled = True
            if not isinstance(postprocess_enabled, bool):
                raise RuntimeError(
                    "answering.citation_postprocess_enabled must be bool, "
                    f"got {type(postprocess_enabled)}"
                )
            answer, citation, citation_postprocessed = apply_citation_postprocess(
                answer, pack, citation, enabled=postprocess_enabled
            )

        # Finalise debug rows: used/citation_index now determinable (Issue #176)
        pack_chunk_ids = [c.chunk_id for c in pack.chunks]
        debug_rows = build_retrieval_debug_rows(
            raw_snapshot=raw_snapshot,
            reranked_top=reranked_top
            if bl.reranker is not None and not is_follow_up
            else list(raw),
            rejected=rejected_debug,
            expanded_refs=all_refs_for_debug,
            store=bl.store,
            photon_scores=photon_score_map,
            photon_current_scores=photon_current_score_map,
            photon_session_scores=photon_session_score_map,
        )
        debug_rows = finalise_retrieval_debug(
            rows=debug_rows,
            pack_chunk_ids=pack_chunk_ids,
            cited_chunk_ids=citation.cited_chunk_ids,
        )

        latency, memory = prof.finish()

        # --- Session update ---
        session_cited_ids = [] if citation_postprocessed else citation.cited_chunk_ids
        turn = session.add_turn(question, answer, session_cited_ids)
        if segment_state.turn_segments is None:
            segment_state.turn_segments = {}
        segment_state.turn_segments[turn.turn_id] = current_segment_id
        segment_state.current_segment_id = current_segment_id
        bl.sessions.save(session)

        # --- Log ---
        bl.logger.log_turn(
            {
                "session_id": session_id,
                "turn_id": turn.turn_id,
                "repo_id": repo_id,
                "repo_commit": cfg.repo.repo_commit,
                "model_id": cfg.model.model_id,
                "question": question,
                "answer": answer,
                "retrieval_chunk_ids": [r.chunk_id for r in raw],
                "evidence_pack_ids": [c.chunk_id for c in pack.chunks],
                "cited_chunk_ids": citation.cited_chunk_ids,
                "wrong_citation_indices": citation.wrong_citation_indices,
                "no_citation": citation.no_citation,
                "citation_postprocessed": citation_postprocessed,
                "latency": latency.as_dict(),
                "memory": memory.as_dict(),
                "fallback_flag": bool(
                    fallback_dict and fallback_dict.get("should_fallback")
                ),
                "fallback_reason": (
                    fallback_dict.get("reasons") if fallback_dict else None
                ),
                "evidence_pruning_applied": (pruning_enabled and is_follow_up),
                "photon_tokenization_failed": tokenization_failed,
                "photon_pruning_applied": photon_pruning_applied,
                "photon_scoring_applied": bool(photon_score_map),
                "photon_scored_count": len(photon_score_map),
                "photon_scoring_mode": photon_scoring_mode,
                "context_carryover_mode": carryover.mode,
                "context_carryover_reason": carryover.reason,
                "context_carryover_similarity": carryover.similarity,
                "rewritten_query": retrieval_query if retrieval_query != question else None,
                "topic_segment_id": current_segment_id,
                "segment_memory_applied": segment_memory.applied,
                "segment_memory_score": segment_memory.score,
                "dual_score_pruning_applied": dual_score_pruning_applied,
                "support_score": support_score,
                "support_guard_active": support_guard_active,
                "photon_carryover_applied": bool(photon_carryover_matches),
                "photon_carryover_turn_ids": [
                    match.turn_id for match in photon_carryover_matches
                ],
                "photon_carryover_scores": [
                    match.score for match in photon_carryover_matches
                ],
                "admission_min_current_score": admission_min_current_score,
                # Issue #62 Phase 1: generation-level observability.
                # ``generator_used`` ∈ {"photon", "qwen"} and
                # ``generator_fallback_reason`` is a closed enum (§7.2):
                # None | "_TokenizerEncodeFailure" | "ValueError"
                #      | "RuntimeError" | "empty_output".
                "generator_used": generator_used,
                "generator_fallback_reason": generator_fallback_reason,
            }
        )

        r_score, r_matches = compute_refusal_score(answer)
        result = QueryResult(
            answer=answer,
            session_id=session_id,
            turn_id=turn.turn_id,
            cited_chunk_ids=citation.cited_chunk_ids,
            wrong_citation_indices=citation.wrong_citation_indices,
            no_citation=citation.no_citation,
            latency=latency,
            memory=memory,
            citation_postprocessed=citation_postprocessed,
            # Issue #62 Phase 1 (CB-003 codex-fix): expose the generator
            # that produced ``answer`` on the structured result so
            # comparison tools can distinguish a real PHOTON answer from
            # a Qwen fallback without having to parse the log stream.
            generator_used=generator_used,
            generator_fallback_reason=generator_fallback_reason,
            retrieval_debug=debug_rows,
            photon_pruning_applied=photon_pruning_applied,
            photon_scoring_applied=bool(photon_score_map),
            photon_scored_count=len(photon_score_map),
            photon_scoring_mode=photon_scoring_mode,
            context_carryover_mode=carryover.mode,
            context_carryover_reason=carryover.reason,
            context_carryover_similarity=carryover.similarity,
            rewritten_query=retrieval_query if retrieval_query != question else None,
            topic_segment_id=current_segment_id,
            segment_memory_applied=segment_memory.applied,
            segment_memory_score=segment_memory.score,
            dual_score_pruning_applied=dual_score_pruning_applied,
            support_score=support_score,
            support_guard_active=support_guard_active,
            refusal_score=r_score,
            refusal_matches=r_matches,
        )

        # Attach PHOTON metadata
        result.drift_metrics = drift_dict
        result.confidence = confidence
        result.fallback_decision = fallback_dict

        return result
