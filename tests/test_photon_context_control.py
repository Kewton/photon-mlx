from __future__ import annotations

from baseline_reporag.memory.session import SessionState
from baseline_reporag.photon_pipeline import (
    _PhotonCarryoverMatch,
    _TopicSegmentState,
    _admit_photon_indices,
    _boost_carryover_with_photon_matches,
    _compose_evidence_frame_pins,
    _dual_score_candidate_indices,
    _generation_history_text,
    _has_explicit_topic_switch_signal,
    _chunk_id_audit_rows,
    _recent_questions_in_segment,
    _retrieval_result_audit_rows,
    _resolve_context_carryover,
    _resolve_segment_memory,
    _support_check_note,
    _support_score_for_pack,
    _turn_decay,
)


def _session_with_questions(*questions: str) -> SessionState:
    session = SessionState("s1", "repo", "commit")
    for i, question in enumerate(questions, start=1):
        session.add_turn(question, f"answer {i}", [])
    return session


def test_context_carryover_weak_for_independent_topic_switch() -> None:
    session = _session_with_questions(
        "セーフティネット保証4号の認定条件を教えて",
        "創業して間もない場合はどの様式になりますか？",
    )

    decision = _resolve_context_carryover(
        "起業家・創業支援融資の概要を教えて",
        session,
        enabled=True,
        rewrite_enabled=True,
        rewrite_history_max=2,
    )

    assert decision.mode == "weak"
    assert decision.query == "起業家・創業支援融資の概要を教えて"


def test_photon_related_turn_boosts_weak_carryover() -> None:
    session = _session_with_questions("セーフティネット保証4号の認定条件を教えて")
    question = "創業直後の様式は？"
    decision = _resolve_context_carryover(
        question,
        session,
        enabled=True,
        rewrite_enabled=True,
        rewrite_history_max=1,
    )

    boosted = _boost_carryover_with_photon_matches(
        decision,
        question,
        session,
        matches=[_PhotonCarryoverMatch(turn_id=1, score=0.91)],
        rewrite_enabled=True,
        rewrite_history_max=1,
    )

    assert decision.mode == "weak"
    assert boosted.mode == "mixed"
    assert "photon_related_turn" in boosted.reason
    assert "セーフティネット保証4号" in boosted.query
    assert question in boosted.query


def test_explicit_topic_switch_blocks_photon_carryover_boost() -> None:
    session = _session_with_questions(
        "セーフティネット保証4号の認定条件を教えて",
        "創業して間もない場合はどの様式になりますか？",
    )
    question = "起業家・創業支援融資の概要を教えて"
    decision = _resolve_context_carryover(
        question,
        session,
        enabled=True,
        rewrite_enabled=True,
        rewrite_history_max=2,
    )

    assert _has_explicit_topic_switch_signal(
        question,
        [turn.question for turn in session.turns],
    )

    boosted = _boost_carryover_with_photon_matches(
        decision,
        question,
        session,
        matches=[] if _has_explicit_topic_switch_signal(question, [turn.question for turn in session.turns]) else [_PhotonCarryoverMatch(turn_id=2, score=0.95)],
        rewrite_enabled=True,
        rewrite_history_max=2,
    )

    assert boosted.mode == "weak"
    assert boosted.query == question


def test_explicit_comparison_target_does_not_signal_topic_switch() -> None:
    assert not _has_explicit_topic_switch_signal(
        "生産性向上・事業拡大融資と比べると？",
        ["起業家・創業支援融資の必要書類を教えて"],
    )


def test_comparison_question_is_treated_as_followup() -> None:
    session = _session_with_questions("起業家・創業支援融資の概要を教えて")

    decision = _resolve_context_carryover(
        "個人用と違う点だけ整理して",
        session,
        enabled=True,
        rewrite_enabled=True,
        rewrite_history_max=1,
    )

    assert decision.mode == "mixed"
    assert "起業家・創業支援融資" in decision.query


def test_conditional_question_is_treated_as_followup() -> None:
    session = _session_with_questions("中小企業融資申込書には法人用と個人用がありますか？")

    decision = _resolve_context_carryover(
        "法人の場合に必要な記入項目を教えて",
        session,
        enabled=True,
        rewrite_enabled=True,
        rewrite_history_max=1,
    )

    assert decision.mode in {"mixed", "strong"}
    assert "中小企業融資申込書" in decision.query


def test_photon_related_turn_boost_keeps_stronger_lexical_decision() -> None:
    session = _session_with_questions("起業家・創業支援融資の概要を教えて")
    decision = _resolve_context_carryover(
        "その計画書には資金計画も書きますか？",
        session,
        enabled=True,
        rewrite_enabled=True,
        rewrite_history_max=1,
    )

    boosted = _boost_carryover_with_photon_matches(
        decision,
        "その計画書には資金計画も書きますか？",
        session,
        matches=[_PhotonCarryoverMatch(turn_id=1, score=0.95)],
        rewrite_enabled=True,
        rewrite_history_max=1,
    )

    assert decision.mode in {"mixed", "strong"}
    assert boosted == decision


def test_context_carryover_rewrites_ambiguous_followup() -> None:
    session = _session_with_questions("起業家・創業支援融資の概要を教えて")

    decision = _resolve_context_carryover(
        "その計画書には資金計画も書きますか？",
        session,
        enabled=True,
        rewrite_enabled=True,
        rewrite_history_max=1,
    )

    assert decision.mode in {"strong", "mixed"}
    assert "起業家・創業支援融資" in decision.query
    assert "その計画書" in decision.query


def test_rewrite_history_can_be_limited_to_current_topic_segment() -> None:
    session = _session_with_questions(
        "first topic question",
        "new topic overview",
    )
    segment_state = _TopicSegmentState(
        current_segment_id=2,
        turn_segments={1: 1, 2: 2},
    )

    segment_questions = _recent_questions_in_segment(
        session,
        segment_state,
        segment_id=2,
        limit=2,
    )
    decision = _resolve_context_carryover(
        "その詳細は？",
        session,
        enabled=True,
        rewrite_enabled=True,
        rewrite_history_max=2,
        rewrite_questions=segment_questions,
    )

    assert "new topic overview" in decision.query
    assert "first topic question" not in decision.query


def test_generation_history_uses_only_current_topic_segment() -> None:
    session = _session_with_questions(
        "セーフティネット保証4号の概要を教えて",
        "起業家・創業支援融資の概要を教えて",
        "申請ではどんな計画書が必要ですか？",
    )
    segment_state = _TopicSegmentState(
        current_segment_id=2,
        turn_segments={1: 1, 2: 2, 3: 2},
    )

    history = _generation_history_text(
        session,
        segment_state,
        segment_id=2,
        carryover_mode="strong",
        max_turns=4,
    )

    assert "起業家・創業支援融資" in history
    assert "申請ではどんな計画書" in history
    assert "セーフティネット保証4号" not in history


def test_generation_history_is_empty_for_weak_carryover() -> None:
    session = _session_with_questions("前の話題")
    segment_state = _TopicSegmentState(current_segment_id=2, turn_segments={1: 1})

    history = _generation_history_text(
        session,
        segment_state,
        segment_id=2,
        carryover_mode="weak",
        max_turns=4,
    )

    assert history == ""


def test_mixed_generation_history_keeps_recent_segment_turns_only() -> None:
    session = _session_with_questions(
        "old same segment",
        "recent same segment 1",
        "recent same segment 2",
    )
    segment_state = _TopicSegmentState(
        current_segment_id=1,
        turn_segments={1: 1, 2: 1, 3: 1},
    )

    history = _generation_history_text(
        session,
        segment_state,
        segment_id=1,
        carryover_mode="mixed",
        max_turns=4,
    )

    assert "old same segment" not in history
    assert "recent same segment 1" in history
    assert "recent same segment 2" in history


def test_segment_memory_recovers_terse_followup_without_topic_terms() -> None:
    decision = _resolve_segment_memory(
        "必要書類は何ですか？",
        ["葛飾区デジタル化支援事業費補助金の対象を教えて"],
        rewrite_enabled=True,
        rewrite_history_max=1,
    )

    assert decision.applied
    assert "葛飾区デジタル化支援事業費補助金" in decision.query
    assert "必要書類" in decision.query


def test_turn_decay_reduces_older_context() -> None:
    assert _turn_decay(5, 4, 0.7) == 1.0
    assert _turn_decay(5, 3, 0.7) == 0.7
    assert _turn_decay(5, 2, 0.7) == 0.7 * 0.7


def test_admission_keeps_protected_and_filters_stale_photon_candidate() -> None:
    admitted = _admit_photon_indices(
        candidate_indices=[0, 1, 2],
        protected_indices=[0],
        chunk_ids_for_scoring=["current", "stale", "semantic"],
        chunk_texts=[
            "current protected text",
            "unrelated old context",
            "資金計画と調達方法について記載します",
        ],
        retrieval_scores={"current": 1.0, "stale": 0.0, "semantic": 0.0},
        query="資金計画は書きますか？",
        min_current_score=0.05,
    )

    assert admitted == [0, 2]


def test_dual_score_pruning_prefers_current_support_over_stale_session_only() -> None:
    selected = _dual_score_candidate_indices(
        candidate_indices=[0, 1, 2, 3],
        protected_indices=[0],
        chunk_ids_for_scoring=["protected", "current", "stale", "balanced"],
        retrieval_scores={
            "protected": 1.0,
            "current": 0.8,
            "stale": 0.1,
            "balanced": 0.5,
        },
        current_scores={"current": 0.9, "stale": 0.1, "balanced": 0.5},
        session_scores={"current": 0.1, "stale": 0.95, "balanced": 0.5},
        carryover_mode="weak",
        max_extra=2,
    )

    assert selected == [0, 1, 3]


class _Chunk:
    def __init__(
        self,
        chunk_id: str,
        content: str = "",
        rel_path: str = "",
        section_header: str = "",
    ) -> None:
        self.chunk_id = chunk_id
        self.content = content
        self.rel_path = rel_path
        self.section_header = section_header


def test_support_score_uses_photon_current_score_when_available() -> None:
    score = _support_score_for_pack(
        question="資金計画は必要ですか？",
        pack_chunks=[_Chunk("c1", "unrelated"), _Chunk("c2", "資金計画を記載します")],
        retrieval_scores={"c1": 0.1, "c2": 0.2},
        current_scores={"c2": 0.88},
        session_scores={"c1": 0.9},
    )

    assert score == 0.88


def test_support_check_note_requires_explicit_support_for_inclusion_claims() -> None:
    note = _support_check_note(0.82, guard_active=False)

    assert "named case, condition, item, or activity" in note
    assert "etc." in note


class _Store:
    def __init__(self, chunks: list[_Chunk]) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}

    def get_many(self, chunk_ids: list[str]) -> list[_Chunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]


class _Result:
    def __init__(self, chunk_id: str, score: float) -> None:
        self.chunk_id = chunk_id
        self.score = score
        self.lexical_score = score / 2
        self.embedding_score = score / 3
        self.reranker_score = None


def test_retrieval_audit_rows_include_stage_rank_scores_and_metadata() -> None:
    rows = _retrieval_result_audit_rows(
        [_Result("c1", 0.9)],
        store=_Store([_Chunk("c1", rel_path="docs/a.md", section_header="A")]),
    )

    assert rows == [
        {
            "rank": 1,
            "chunk_id": "c1",
            "rel_path": "docs/a.md",
            "section": "A",
            "score": 0.9,
            "lexical_score": 0.45,
            "embedding_score": 0.3,
            "reranker_score": None,
        }
    ]


def test_chunk_id_audit_rows_mark_source_and_citation() -> None:
    rows = _chunk_id_audit_rows(
        ["c1", "c2"],
        store=_Store([
            _Chunk("c1", rel_path="docs/a.md", section_header="A"),
            _Chunk("c2", rel_path="docs/b.md", section_header="B"),
        ]),
        source_map={"c2": "neighbor"},
        cited_chunk_ids=["c2"],
    )

    assert rows[0]["source"] is None
    assert rows[0]["cited"] is False
    assert rows[1]["source"] == "neighbor"
    assert rows[1]["cited"] is True


def test_evidence_frame_pins_prioritise_current_query_support() -> None:
    pins = _compose_evidence_frame_pins(
        current_query_ids=["current-a", "current-b"],
        working_memory_ids=["memory-a", "current-a"],
        related_past_ids=["related-a", "memory-a"],
    )

    assert pins == ["current-a", "current-b", "memory-a", "related-a"]
