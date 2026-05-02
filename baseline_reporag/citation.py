from __future__ import annotations

import re
from dataclasses import dataclass

from .generation.evidence_pack import EvidencePack
from .ingestion.store import Chunk

# Issue #154 Bug 2: phrases that mark an answer as a legitimate refusal
# ("we don't have evidence to answer"). When present, the answer must be
# graded as no-citation regardless of any formal ``[C:N]`` markers — baseline
# was producing refusals like ``根拠が不足しています ... [C:1]`` and the
# regex-only check was treating them as cited, unfairly penalising PHOTON
# which honestly omits the marker on refusals.
REFUSAL_PATTERNS: tuple[str, ...] = (
    "根拠が不足しています",
    "根拠不足",
    "情報がありません",
    "情報はありません",
    "わかりません",
    "判断できません",
    "特定できません",
    "確認できません",
    "確認できない",
    # Issue #177: phrases observed in actual refusal responses
    "該当する情報は含まれていません",
    "該当する情報が含まれていません",
    "見当たりません",
)

_CITATION_MARKER_RE = re.compile(r"[\[【［]\s*C\s*[:：]\s*(\d+)\s*[\]】］]")

_UNCERTAIN_ANSWER_MARKERS: tuple[str, ...] = (
    "確認できません",
    "確認できない",
    "判断できません",
    "判断できない",
    "特定できません",
    "特定できない",
    "根拠が不足しています",
    "根拠不足",
    "明記されていません",
    "明記がありません",
)

_AFFIRMATIVE_INCLUSION_MARKERS: tuple[str, ...] = (
    "はい",
    "対象です",
    "対象になります",
    "対象に含まれます",
    "含まれます",
    "可能です",
    "できます",
    "該当します",
    "yes",
    "is eligible",
    "are eligible",
    "is included",
    "are included",
    "is covered",
    "are covered",
)

_INCLUSION_QUESTION_MARKERS: tuple[str, ...] = (
    "対象になりますか",
    "対象ですか",
    "対象に含まれますか",
    "含まれますか",
    "該当しますか",
    "だけでも対象",
    "だけでも可能",
    "でも対象",
    "eligible",
    "included",
    "covered",
)

_TERM_BOUNDARY_MARKERS: tuple[str, ...] = (
    "だけでも",
    "でも",
    "は対象",
    "が対象",
    "は含ま",
    "が含ま",
    "は該当",
    "が該当",
)

_TERM_STRIP_CHARS = " 　、。・:：;；?？!！「」『』（）()[]【】"

_GENERIC_QUERY_TERMS: tuple[str, ...] = (
    "それ",
    "その",
    "これ",
    "この",
    "場合",
    "対象",
    "該当",
    "可能",
    "申請",
    "文書",
    "明記",
    "条件",
    "必要",
)


def is_refusal_answer(answer: str) -> bool:
    """True if *answer* contains any known refusal / abstain phrase."""
    if not answer:
        return False
    return any(p in answer for p in REFUSAL_PATTERNS)


def compute_refusal_score(answer: str) -> tuple[float, list[str]]:
    """Return (score, matched_phrases) for *answer*.

    score is 1.0 when any REFUSAL_PATTERNS phrase is found, 0.0 otherwise.
    matched_phrases lists every pattern that was detected.
    """
    if not answer:
        return 0.0, []
    matches = [p for p in REFUSAL_PATTERNS if p in answer]
    return (1.0, matches) if matches else (0.0, [])


@dataclass
class CitationResult:
    cited_chunk_ids: list[str]
    cited_chunks: list[Chunk]
    wrong_citation_indices: list[int]  # [C:N] indices absent from the pack
    no_citation: bool
    is_refusal: bool = False


@dataclass
class ClaimSupportGuard:
    applied: bool
    reason: str | None = None
    unsupported_terms: list[str] | None = None


def normalise_citation_markers(answer: str) -> str:
    """Canonicalise supported citation-marker variants to ``[C:N]``.

    LLMs often emit harmless spacing/bracket variants such as ``[ C:1 ]`` or
    ``【C：2】``.  Normalising before citation resolution keeps the displayed
    answer and structured ``cited_chunk_ids`` aligned.
    """
    return _CITATION_MARKER_RE.sub(lambda match: f"[C:{int(match.group(1))}]", answer)


def resolve_citations(answer: str, pack: EvidencePack) -> CitationResult:
    """Parse [C:N] references in answer and map them to chunk IDs."""
    indices = [int(m.group(1)) for m in _CITATION_MARKER_RE.finditer(answer)]

    # Build reverse map: 1-based index -> Chunk
    index_to_chunk: dict[int, Chunk] = {}
    for chunk, idx in zip(pack.chunks, pack.chunk_indices.values()):
        index_to_chunk[idx] = chunk

    cited_chunk_ids: list[str] = []
    wrong_indices: list[int] = []
    seen: set[str] = set()

    for idx in set(indices):
        chunk = index_to_chunk.get(idx)
        if chunk:
            if chunk.chunk_id not in seen:
                cited_chunk_ids.append(chunk.chunk_id)
                seen.add(chunk.chunk_id)
        else:
            wrong_indices.append(idx)

    cited_chunks = [c for c in pack.chunks if c.chunk_id in seen]

    return CitationResult(
        cited_chunk_ids=cited_chunk_ids,
        cited_chunks=cited_chunks,
        wrong_citation_indices=wrong_indices,
        no_citation=len(indices) == 0,
        is_refusal=is_refusal_answer(answer),
    )


def _normalise_for_support(text: str) -> str:
    return re.sub(r"\s+", "", text.casefold())


def _is_inclusion_question(question: str) -> bool:
    q = _normalise_for_support(question)
    return any(marker.casefold() in q for marker in _INCLUSION_QUESTION_MARKERS)


def _is_affirmative_inclusion_answer(answer: str) -> bool:
    if not answer:
        return False
    if any(marker in answer for marker in _UNCERTAIN_ANSWER_MARKERS):
        return False
    normalised = _normalise_for_support(answer)
    return any(marker.casefold() in normalised for marker in _AFFIRMATIVE_INCLUSION_MARKERS)


def _extract_condition_terms(question: str) -> list[str]:
    """Extract concrete condition/item terms from yes/no inclusion questions.

    This intentionally avoids domain-specific dictionaries.  It looks for the
    user-named item immediately before generic inclusion markers such as
    "だけでも対象" and keeps only terms that are specific enough to require
    explicit support.
    """
    candidates: list[str] = []
    for marker in _TERM_BOUNDARY_MARKERS:
        index = question.find(marker)
        if index <= 0:
            continue
        prefix = question[:index].strip(_TERM_STRIP_CHARS)
        prefix = re.split(r"[。．.!！?？\n]", prefix)[-1].strip(_TERM_STRIP_CHARS)
        if marker == "でも" and prefix.endswith("だけ"):
            prefix = prefix[: -len("だけ")].strip(_TERM_STRIP_CHARS)
        if prefix:
            candidates.append(prefix)

    # Quoted terms are also concrete user-named conditions.
    candidates.extend(re.findall(r"[「『\"]([^」』\"]{2,40})[」』\"]", question))

    terms: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        term = candidate.strip(_TERM_STRIP_CHARS)
        for generic in _GENERIC_QUERY_TERMS:
            term = term.replace(generic, "")
        term = term.strip(_TERM_STRIP_CHARS)
        if len(term) < 2:
            continue
        key = _normalise_for_support(term)
        if key and key not in seen:
            seen.add(key)
            terms.append(term)
    return terms[:3]


def _evidence_text_for_guard(pack: EvidencePack) -> str:
    parts: list[str] = []
    for chunk in pack.chunks:
        parts.extend(
            [
                chunk.rel_path,
                chunk.section_header or "",
                chunk.content,
            ]
        )
    return "\n".join(parts)


def _term_is_explicitly_supported(term: str, evidence_text: str) -> bool:
    normalised_term = _normalise_for_support(term)
    if not normalised_term:
        return True
    normalised_evidence = _normalise_for_support(evidence_text)
    if normalised_term in normalised_evidence:
        return True
    # Conservative fallback for scripts/languages where spacing or punctuation
    # splits the phrase but the important pieces still appear together.
    pieces = [
        p
        for p in re.split(r"[\s　、。・/／()（）【】「」『』:：;；]+", term)
        if len(p) >= 2
    ]
    return bool(pieces) and all(_normalise_for_support(p) in normalised_evidence for p in pieces)


def apply_claim_support_guard(
    *,
    question: str,
    answer: str,
    pack: EvidencePack,
    citation: CitationResult,
) -> tuple[str, CitationResult, ClaimSupportGuard]:
    """Guard unsupported affirmative inclusion answers.

    Retrieval relevance is not the same as claim support.  For yes/no questions
    asking whether a concrete user-named item/condition is included or eligible,
    an affirmative answer is only allowed when that concrete term appears in the
    evidence.  Otherwise we replace the answer with a cautious, cited statement.
    """
    if not pack.chunks:
        return answer, citation, ClaimSupportGuard(applied=False)
    if not _is_inclusion_question(question):
        return answer, citation, ClaimSupportGuard(applied=False)
    if not _is_affirmative_inclusion_answer(answer):
        return answer, citation, ClaimSupportGuard(applied=False)

    terms = _extract_condition_terms(question)
    if not terms:
        return answer, citation, ClaimSupportGuard(applied=False)

    evidence_text = _evidence_text_for_guard(pack)
    unsupported = [
        term for term in terms if not _term_is_explicitly_supported(term, evidence_text)
    ]
    if not unsupported:
        return answer, citation, ClaimSupportGuard(applied=False)

    cited_index = 1
    if citation.cited_chunk_ids:
        cited_index = pack.chunk_indices.get(citation.cited_chunk_ids[0], 1)
    term_text = "、".join(f"「{term}」" for term in unsupported)
    guarded_answer = (
        f"提供された文書では、{term_text}が対象に含まれるかどうかは確認できません。"
        f"関連する記載はありますが、当該条件を明示的に対象とする根拠が不足しています [C:{cited_index}]。"
    )
    guarded_citation = resolve_citations(guarded_answer, pack)
    return (
        guarded_answer,
        guarded_citation,
        ClaimSupportGuard(
            applied=True,
            reason="unsupported_affirmative_inclusion_claim",
            unsupported_terms=unsupported,
        ),
    )
