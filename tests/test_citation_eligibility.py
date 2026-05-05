from __future__ import annotations

from baseline_reporag.citation import resolve_citations
from baseline_reporag.citation_eligibility import (
    apply_citation_budget_rerank,
    compute_citation_eligibility_scores,
)
from baseline_reporag.generation.evidence_pack import EvidencePack
from baseline_reporag.ingestion.chunker import Chunk


def _chunk(chunk_id: str, rel_path: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        repo_id="repo",
        repo_commit="commit",
        rel_path=rel_path,
        language="markdown",
        start_line=1,
        end_line=5,
        content=content,
        symbols=[],
        section_header="",
        file_header="",
    )


def _pack() -> EvidencePack:
    chunks = [
        _chunk(
            "digital",
            "葛飾区デジタル化支援事業費補助金のご案内/document.md",
            "デジタル化支援事業費補助金の必要書類は事業計画書、企業概要、見積書です。",
        ),
        _chunk(
            "safety1",
            "セーフティネット保証1号認定のご案内/document.md",
            "セーフティネット保証1号の必要書類は認定書、登記簿謄本、決算書類です。",
        ),
    ]
    return EvidencePack(chunks=chunks, chunk_indices={"digital": 1, "safety1": 2})


def test_citation_eligibility_prefers_active_topic_over_stale_context() -> None:
    pack = _pack()
    scores = compute_citation_eligibility_scores(
        question="必要書類は何ですか？",
        context_text="葛飾区デジタル化支援事業費補助金の対象を教えて",
        answer="葛飾区デジタル化支援事業費補助金の必要書類です [C:1][C:2]。",
        pack=pack,
        session_scores={"safety1": 1.0},
        current_scores={"digital": 1.0, "safety1": 0.05},
    )

    by_id = {score.chunk_id: score for score in scores}
    assert by_id["digital"].score > by_id["safety1"].score
    assert by_id["digital"].eligible is True
    assert by_id["safety1"].eligible is False


def test_citation_budget_rerank_replaces_unrelated_citation() -> None:
    pack = _pack()
    answer = "葛飾区デジタル化支援事業費補助金の必要書類です [C:2]。"
    citation = resolve_citations(answer, pack)

    result = apply_citation_budget_rerank(
        question="必要書類は何ですか？",
        context_text="葛飾区デジタル化支援事業費補助金の対象を教えて",
        answer=answer,
        pack=pack,
        citation=citation,
        current_scores={"digital": 1.0, "safety1": 0.05},
        session_scores={"safety1": 1.0},
    )

    assert result.changed is True
    assert result.replaced_indices == {2: 1}
    assert "[C:1]" in result.answer
    assert result.citation.cited_chunk_ids == ["digital"]


def test_citation_budget_rerank_keeps_multiple_relevant_citations() -> None:
    chunks = [
        _chunk(
            "six",
            "セーフティネット保証6号認定のご案内/document.md",
            "6号の必要書類は破綻金融機関との取引確認資料です。",
        ),
        _chunk(
            "eight",
            "セーフティネット保証8号認定のご案内/document.md",
            "8号の必要書類は債権譲渡通知書と事業計画書です。",
        ),
    ]
    pack = EvidencePack(chunks=chunks, chunk_indices={"six": 1, "eight": 2})
    answer = "6号は取引確認資料 [C:1]、8号は債権譲渡通知書 [C:2] が必要です。"
    citation = resolve_citations(answer, pack)

    result = apply_citation_budget_rerank(
        question="必要書類は6号や8号と同じですか？",
        context_text="セーフティネット保証7号だけ詳しく教えて",
        answer=answer,
        pack=pack,
        citation=citation,
        current_scores={"six": 0.8, "eight": 0.8},
    )

    assert result.changed is False
    assert result.citation.cited_chunk_ids == ["six", "eight"]


def test_citation_budget_rerank_does_not_self_replace_budget_excess() -> None:
    chunks = [
        _chunk("a", "active-a.md", "active topic alpha"),
        _chunk("b", "active-b.md", "active topic beta"),
    ]
    pack = EvidencePack(chunks=chunks, chunk_indices={"a": 1, "b": 2})
    answer = "alpha [C:1] beta [C:2]"
    citation = resolve_citations(answer, pack)

    result = apply_citation_budget_rerank(
        question="active topic",
        context_text="active topic",
        answer=answer,
        pack=pack,
        citation=citation,
        retrieval_scores={"a": 1.0, "b": 0.9},
        max_citations=1,
    )

    assert result.changed is True
    assert result.replaced_indices == {}
    assert result.removed_indices == [2]
    assert result.citation.cited_chunk_ids == ["a"]
