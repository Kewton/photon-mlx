from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .citation import CitationResult, resolve_citations
from .generation.evidence_pack import EvidencePack

_CITATION_RE = re.compile(r"\[C:(\d+)\]")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_][a-z0-9_.-]*")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]+")


@dataclass(frozen=True)
class CitationEligibilityScore:
    citation_index: int
    chunk_id: str
    score: float
    context_relevance: float
    question_relevance: float
    answer_relevance: float
    retrieval_score: float
    current_score: float
    session_score: float
    eligible: bool


@dataclass(frozen=True)
class CitationBudgetResult:
    answer: str
    citation: CitationResult
    changed: bool
    removed_indices: list[int]
    replaced_indices: dict[int, int]
    scores: list[CitationEligibilityScore]


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _text_features(text: str) -> set[str]:
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
    return {feature for feature in features if feature}


def _feature_overlap(a: str, b: str) -> float:
    left = _text_features(a)
    if not left:
        return 0.0
    right = _text_features(b)
    if not right:
        return 0.0
    return len(left & right) / len(left)


def _chunk_text(chunk: Any) -> str:
    return "\n".join(
        [
            str(getattr(chunk, "rel_path", "") or ""),
            str(getattr(chunk, "section_header", "") or ""),
            str(getattr(chunk, "content", "") or ""),
        ]
    )


def _normalise_score_map(score_map: dict[str, float] | None) -> dict[str, float]:
    if not score_map:
        return {}
    clean = {
        str(key): float(value)
        for key, value in score_map.items()
        if isinstance(value, (int, float))
    }
    max_score = max(clean.values(), default=0.0)
    if max_score <= 0.0:
        return clean
    return {key: value / max_score for key, value in clean.items()}


def _citation_contexts(answer: str) -> dict[int, str]:
    contexts: dict[int, list[str]] = {}
    for segment in re.split(r"(?<=[。.!?！？])\s+|\n+", answer):
        if not segment.strip():
            continue
        for match in _CITATION_RE.finditer(segment):
            index = int(match.group(1))
            contexts.setdefault(index, []).append(segment)
    return {index: "\n".join(parts) for index, parts in contexts.items()}


def compute_citation_eligibility_scores(
    *,
    question: str,
    answer: str,
    pack: EvidencePack,
    context_text: str = "",
    retrieval_scores: dict[str, float] | None = None,
    current_scores: dict[str, float] | None = None,
    session_scores: dict[str, float] | None = None,
    min_score: float = 0.08,
    relative_floor: float = 0.35,
) -> list[CitationEligibilityScore]:
    retrieval = _normalise_score_map(retrieval_scores)
    current = _normalise_score_map(current_scores)
    session = _normalise_score_map(session_scores)
    full_context = "\n".join(part for part in (question, context_text) if part.strip())
    citation_contexts = _citation_contexts(answer)

    raw_scores: list[tuple[int, str, float, float, float, float, float, float, float]] = []
    for chunk in pack.chunks:
        chunk_id = str(getattr(chunk, "chunk_id", ""))
        citation_index = pack.chunk_indices.get(chunk_id)
        if citation_index is None:
            continue
        text = _chunk_text(chunk)
        local_answer = citation_contexts.get(citation_index, answer)
        context_relevance = _feature_overlap(full_context, text)
        question_relevance = _feature_overlap(question, text)
        answer_relevance = _feature_overlap(local_answer, text)
        retrieval_score = retrieval.get(chunk_id, 0.0)
        current_score = current.get(chunk_id, 0.0)
        session_score = session.get(chunk_id, 0.0)
        stale_gap = max(0.0, session_score - max(current_score, context_relevance))
        score = (
            (0.38 * context_relevance)
            + (0.22 * question_relevance)
            + (0.16 * answer_relevance)
            + (0.14 * current_score)
            + (0.07 * retrieval_score)
            + (0.03 * session_score)
            - (0.12 * stale_gap)
        )
        raw_scores.append(
            (
                citation_index,
                chunk_id,
                max(0.0, min(1.0, score)),
                context_relevance,
                question_relevance,
                answer_relevance,
                retrieval_score,
                current_score,
                session_score,
            )
        )

    best = max((item[2] for item in raw_scores), default=0.0)
    threshold = max(min_score, best * relative_floor)
    return [
        CitationEligibilityScore(
            citation_index=index,
            chunk_id=chunk_id,
            score=score,
            context_relevance=context_relevance,
            question_relevance=question_relevance,
            answer_relevance=answer_relevance,
            retrieval_score=retrieval_score,
            current_score=current_score,
            session_score=session_score,
            eligible=score >= threshold,
        )
        for (
            index,
            chunk_id,
            score,
            context_relevance,
            question_relevance,
            answer_relevance,
            retrieval_score,
            current_score,
            session_score,
        ) in raw_scores
    ]


def apply_citation_budget_rerank(
    *,
    question: str,
    answer: str,
    pack: EvidencePack,
    citation: CitationResult,
    context_text: str = "",
    retrieval_scores: dict[str, float] | None = None,
    current_scores: dict[str, float] | None = None,
    session_scores: dict[str, float] | None = None,
    max_citations: int = 8,
) -> CitationBudgetResult:
    if not answer.strip() or not pack.chunks:
        return CitationBudgetResult(answer, citation, False, [], {}, [])

    scores = compute_citation_eligibility_scores(
        question=question,
        answer=answer,
        pack=pack,
        context_text=context_text,
        retrieval_scores=retrieval_scores,
        current_scores=current_scores,
        session_scores=session_scores,
    )
    by_index = {score.citation_index: score for score in scores}
    ranked = sorted(scores, key=lambda score: (-score.score, score.citation_index))
    eligible_indices = [score.citation_index for score in ranked if score.eligible]
    if not eligible_indices and ranked:
        eligible_indices = [ranked[0].citation_index]
    allowed_ordered = eligible_indices[: max(1, max_citations)]
    allowed = set(allowed_ordered)

    used_allowed: set[int] = set()
    removed: list[int] = []
    replaced: dict[int, int] = {}

    def _replace(match: re.Match[str]) -> str:
        original = int(match.group(1))
        score = by_index.get(original)
        if original in allowed and score is not None:
            used_allowed.add(original)
            return match.group(0)

        replacement = next(
            (
                candidate
                for candidate in allowed_ordered
                if candidate not in used_allowed and candidate != original
            ),
            None,
        )
        if replacement is not None:
            used_allowed.add(replacement)
            replaced[original] = replacement
            return f"[C:{replacement}]"
        removed.append(original)
        return ""

    new_answer = _CITATION_RE.sub(_replace, answer)
    new_answer = re.sub(r"\s+([。,.、])", r"\1", new_answer)
    changed = new_answer != answer
    if not changed:
        return CitationBudgetResult(answer, citation, False, [], {}, scores)
    return CitationBudgetResult(
        new_answer,
        resolve_citations(new_answer, pack),
        True,
        removed,
        replaced,
        scores,
    )
