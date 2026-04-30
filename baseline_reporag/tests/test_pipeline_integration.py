"""Pipeline integration tests for evidence header and few-shot injection.

Mock strategy (from design policy Section 7):
- Generator: MagicMock, capture .generate() call args
- ChunkStore: MagicMock with get_many() returning test chunks
- LexicalIndex / EmbeddingIndex / SymbolGraph: MagicMock
- RunLogger: MagicMock
- SessionManager: real instance (in-memory)
- hybrid_search / expand_with_graph: patched to return test chunk IDs
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

# Guard: pipeline.py → generator.py triggers MLX import at collection time.
# Provide stub modules so tests are collected even without MLX installed.
# Only inject stubs when MLX is genuinely absent (not already loaded).
if importlib.util.find_spec("mlx") is None:
    for _mod in ("mlx", "mlx.core", "mlx_lm", "mlx_lm.sample_utils"):
        if _mod not in sys.modules:
            _stub = ModuleType(_mod)
            _stub.make_sampler = lambda **kw: None  # type: ignore[attr-defined]
            sys.modules[_mod] = _stub

from baseline_reporag.config import Config  # noqa: E402
from baseline_reporag.ingestion.chunker import Chunk  # noqa: E402
from baseline_reporag.memory.session import SessionManager  # noqa: E402
from baseline_reporag.pipeline import RepoRAGPipeline  # noqa: E402
from baseline_reporag.retrieval.graph_expansion import ExpandedChunkRef  # noqa: E402


def _make_test_chunks() -> list[Chunk]:
    """テスト用の固定チャンクを作成する。"""
    return [
        Chunk(
            chunk_id="chunk_0",
            repo_id="test_repo",
            repo_commit="abc123",
            rel_path="app/main.py",
            language="python",
            start_line=1,
            end_line=10,
            content="def main():\n    app = FastAPI()\n    return app",
            symbols=["main"],
            section_header="main",
            file_header="# app/main.py",
        ),
        Chunk(
            chunk_id="chunk_1",
            repo_id="test_repo",
            repo_commit="abc123",
            rel_path="app/router.py",
            language="python",
            start_line=1,
            end_line=15,
            content="router = APIRouter()\n@router.get('/')\ndef index(): pass",
            symbols=["index"],
            section_header="index",
            file_header="# app/router.py",
        ),
    ]


def _build_test_pipeline(
    generator: MagicMock | None = None,
) -> RepoRAGPipeline:
    """テスト用の RepoRAGPipeline を構築する。"""
    cfg_data = {
        "repo": {
            "repo_id": "test_repo",
            "repo_commit": "abc123",
        },
        "retrieval": {
            "lexical_top_k": 10,
            "embedding_top_k": 10,
            "fused_top_k": 8,
            "rerank_top_k": 8,
            "weights": {"lexical": 0.5, "embedding": 0.5},
            "graph_expansion": {"max_hops": 1, "max_nodes": 16},
            "neighborhood_expansion": {"before": 1, "after": 1},
            "query_expansion": {"enabled": False},
            "reranker": {"enabled": False},
            "file_type_boost": 0.0,
        },
        "evidence_pack": {
            "max_chunks": 16,
            "max_tokens": 16000,
        },
        "model": {
            "model_id": "test-model",
        },
    }
    config = Config(cfg_data)

    mock_store = MagicMock()
    mock_store.get_many.return_value = _make_test_chunks()

    mock_gen = generator or MagicMock()
    if generator is None:
        mock_gen.generate.return_value = "Answer with [C:1]."

    sessions = SessionManager()  # real in-memory instance

    return RepoRAGPipeline(
        config=config,
        store=mock_store,
        lexical=MagicMock(),
        embedding=MagicMock(),
        graph=MagicMock(),
        sessions=sessions,
        generator=mock_gen,
        logger=MagicMock(),
    )


# Mock return values for hybrid_search and expand_with_graph
_MOCK_CHUNK_IDS = ["chunk_0", "chunk_1"]


def _mock_hybrid_search(**kwargs):  # noqa: ANN003
    """hybrid_search のモック: RetrievalResult のリストを返す。"""
    from baseline_reporag.retrieval.hybrid import RetrievalResult

    return [
        RetrievalResult(chunk_id=cid, score=1.0, lexical_score=0.5, embedding_score=0.5)
        for cid in _MOCK_CHUNK_IDS
    ]


def _mock_expand_with_graph(**kwargs):  # noqa: ANN003
    """expand_with_graph のモック: ExpandedChunkRef リストを返す。"""
    return [
        ExpandedChunkRef(chunk_id=cid, source="retrieval") for cid in _MOCK_CHUNK_IDS
    ]


@patch(
    "baseline_reporag.pipeline.expand_with_graph",
    side_effect=_mock_expand_with_graph,
)
@patch(
    "baseline_reporag.pipeline.hybrid_search",
    side_effect=_mock_hybrid_search,
)
class TestPipelineEvidenceHeader:
    """pipeline.py の evidence header 挿入ロジックを検証する。"""

    def test_first_turn_includes_evidence_header(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """First turn の evidence_text に _EVIDENCE_HEADER が含まれること。"""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "Answer with [C:1]."
        pipeline = _build_test_pipeline(generator=mock_gen)
        pipeline.query("test question", session_id="s1")
        messages = mock_gen.generate.call_args[0][0]
        user_content = messages[1]["content"]
        assert "IMPORTANT: You MUST cite" in user_content

    def test_second_turn_excludes_evidence_header(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """2nd turn の evidence_text に _EVIDENCE_HEADER が含まれないこと。"""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "Answer with [C:1]."
        pipeline = _build_test_pipeline(generator=mock_gen)
        pipeline.query("first question", session_id="s1")
        pipeline.query("second question", session_id="s1")
        # 2nd call args
        messages = mock_gen.generate.call_args[0][0]
        user_content = messages[1]["content"]
        assert "IMPORTANT: You MUST cite" not in user_content

    def test_first_turn_includes_few_shot(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """First turn で few-shot example が含まれること。"""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "Answer with [C:1]."
        pipeline = _build_test_pipeline(generator=mock_gen)
        pipeline.query("test question", session_id="s1")
        messages = mock_gen.generate.call_args[0][0]
        user_content = messages[1]["content"]
        assert "Example 1" in user_content
        assert "Q: Where is the main router defined?" in user_content

    def test_follow_up_turn_includes_few_shot(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """Follow-up でも few-shot は含まれること（_FORMAT_HINT は全ターン）。"""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "Answer with [C:1]."
        pipeline = _build_test_pipeline(generator=mock_gen)
        pipeline.query("first question", session_id="s1")
        pipeline.query("second question", session_id="s1")
        # 2nd call args
        messages = mock_gen.generate.call_args[0][0]
        user_content = messages[1]["content"]
        assert "Example 1" in user_content
        assert "Q: Where is the main router defined?" in user_content


def _build_postprocess_pipeline(
    enabled: bool,
    answer: str = "This answer has no citation.",
) -> RepoRAGPipeline:
    """Build a pipeline with citation_postprocess_enabled controlled."""
    cfg_data = {
        "repo": {
            "repo_id": "test_repo",
            "repo_commit": "abc123",
        },
        "retrieval": {
            "lexical_top_k": 10,
            "embedding_top_k": 10,
            "fused_top_k": 8,
            "rerank_top_k": 8,
            "weights": {"lexical": 0.5, "embedding": 0.5},
            "graph_expansion": {"max_hops": 1, "max_nodes": 16},
            "neighborhood_expansion": {"before": 1, "after": 1},
            "query_expansion": {"enabled": False},
            "reranker": {"enabled": False},
            "file_type_boost": 0.0,
        },
        "evidence_pack": {
            "max_chunks": 16,
            "max_tokens": 16000,
        },
        "model": {
            "model_id": "test-model",
        },
        "answering": {
            "citation_postprocess_enabled": enabled,
        },
    }
    config = Config(cfg_data)

    mock_store = MagicMock()
    mock_store.get_many.return_value = _make_test_chunks()

    mock_gen = MagicMock()
    mock_gen.generate.return_value = answer

    sessions = SessionManager()

    return RepoRAGPipeline(
        config=config,
        store=mock_store,
        lexical=MagicMock(),
        embedding=MagicMock(),
        graph=MagicMock(),
        sessions=sessions,
        generator=mock_gen,
        logger=MagicMock(),
    )


@patch(
    "baseline_reporag.pipeline.expand_with_graph",
    side_effect=_mock_expand_with_graph,
)
@patch(
    "baseline_reporag.pipeline.hybrid_search",
    side_effect=_mock_hybrid_search,
)
class TestCitationPostprocess:
    """post-processing ON/OFF の統合テスト。"""

    def test_postprocess_enabled_adds_citation(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """enabled=True のとき no_citation 回答に [C:1] が付与される。"""
        pipeline = _build_postprocess_pipeline(
            enabled=True, answer="No citation in this answer."
        )
        result = pipeline.query("test question", session_id="s1")
        assert result.citation_postprocessed is True
        assert "[C:1]" in result.answer
        assert result.no_citation is False

    def test_postprocess_disabled_preserves_no_citation(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """enabled=False のとき no_citation 回答が変更されない。"""
        original_answer = "No citation in this answer."
        pipeline = _build_postprocess_pipeline(enabled=False, answer=original_answer)
        result = pipeline.query("test question", session_id="s1")
        assert result.citation_postprocessed is False
        assert result.answer == original_answer
        assert result.no_citation is True

    def test_postprocess_skips_when_already_cited(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """回答に既に [C:1] がある場合は post-process フラグが立たない。"""
        pipeline = _build_postprocess_pipeline(
            enabled=True, answer="Answer with [C:1]."
        )
        result = pipeline.query("test question", session_id="s1")
        assert result.citation_postprocessed is False
        assert result.no_citation is False

    def test_postprocess_skips_abstain_marker(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """ABSTAIN_MARKER を含む回答には [C:1] が付与されない。"""
        from baseline_reporag.generation.prompt import ABSTAIN_MARKER

        abstain_answer = f"{ABSTAIN_MARKER}。詳細は不明です。"
        pipeline = _build_postprocess_pipeline(enabled=True, answer=abstain_answer)
        result = pipeline.query("test question", session_id="s1")
        assert result.citation_postprocessed is False
        assert result.answer == abstain_answer

    def test_postprocess_does_not_pollute_session(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """機械付与した cited_chunk_ids は session に積まれない。"""
        pipeline = _build_postprocess_pipeline(enabled=True, answer="No citation here.")
        result = pipeline.query("test question", session_id="s1")
        assert result.citation_postprocessed is True
        # Session should not contain the auto-attached citation
        session = pipeline.sessions.get_or_create("s1", "test_repo", "abc123")
        assert session.cited_chunk_ids == []

    def test_postprocess_logs_field(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """logger.log_turn に citation_postprocessed フィールドが入る。"""
        pipeline = _build_postprocess_pipeline(enabled=True, answer="No citation here.")
        pipeline.query("test question", session_id="s1")
        log_payload = pipeline.logger.log_turn.call_args[0][0]
        assert "citation_postprocessed" in log_payload
        assert log_payload["citation_postprocessed"] is True


# ---------------------------------------------------------------------------
# Issue #109: graph=None type compatibility at pipeline layer
# ---------------------------------------------------------------------------


@patch(
    "baseline_reporag.pipeline.expand_with_graph",
    side_effect=_mock_expand_with_graph,
)
@patch(
    "baseline_reporag.pipeline.hybrid_search",
    side_effect=_mock_hybrid_search,
)
class TestGraphNoneTypeCompatibility:
    """Issue #109: ``graph=None`` must not cause assembly / type errors.

    Actual ``graph is None`` branching is covered in
    ``test_graph_expansion.py``. This test only verifies that the
    pipeline can be built and a turn completes when ``graph=None`` is
    plumbed through (``expand_with_graph`` is still patched).
    """

    def test_pipeline_accepts_graph_none(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        cfg_data = {
            "repo": {"repo_id": "test_repo", "repo_commit": "abc123"},
            "retrieval": {
                "lexical_top_k": 10,
                "embedding_top_k": 10,
                "fused_top_k": 8,
                "rerank_top_k": 8,
                "weights": {"lexical": 0.5, "embedding": 0.5},
                "graph_expansion": {"max_hops": 1, "max_nodes": 16},
                "neighborhood_expansion": {"before": 1, "after": 1},
                "query_expansion": {"enabled": False},
                "reranker": {"enabled": False},
                "file_type_boost": 0.0,
            },
            "evidence_pack": {"max_chunks": 16, "max_tokens": 16000},
            "model": {"model_id": "test-model"},
        }
        config = Config(cfg_data)

        mock_store = MagicMock()
        mock_store.get_many.return_value = _make_test_chunks()

        mock_gen = MagicMock()
        mock_gen.generate.return_value = "Answer with [C:1]."

        sessions = SessionManager()

        pipeline = RepoRAGPipeline(
            config=config,
            store=mock_store,
            lexical=MagicMock(),
            embedding=MagicMock(),
            graph=None,  # Issue #109: symbol-graph disabled path.
            sessions=sessions,
            generator=mock_gen,
            logger=MagicMock(),
        )
        result = pipeline.query("test question", session_id="s1")
        assert result.answer.endswith("[C:1].")


# ---------------------------------------------------------------------------
# Issue #143: seed propagation from pipeline.query into Generator.generate.
#
# Contract (work-plan §Step 2.2 / DR3-002):
# - ``RepoRAGPipeline.query(seed=None)`` (default) MUST call the generator
#   with the same single-positional-arg shape as before so the existing
#   17+ MagicMock tests in this file (``mock_gen.generate.return_value =
#   ...``) keep passing without TypeError on extra kwargs.
# - ``RepoRAGPipeline.query(seed=42)`` MUST call ``generator.generate``
#   with ``seed=42`` keyword propagated.
# ---------------------------------------------------------------------------


@patch(
    "baseline_reporag.pipeline.expand_with_graph",
    side_effect=_mock_expand_with_graph,
)
@patch(
    "baseline_reporag.pipeline.hybrid_search",
    side_effect=_mock_hybrid_search,
)
class TestSeedPropagation:
    """RepoRAGPipeline.query forwards the seed kwarg to Generator.generate."""

    def test_query_without_seed_uses_legacy_call_shape(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """``seed=None`` (default) -> generator.generate called WITHOUT seed kwarg.

        Backwards-compat invariant for the 17+ existing MagicMock tests.
        """
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "Answer with [C:1]."
        pipeline = _build_test_pipeline(generator=mock_gen)
        pipeline.query("test question", session_id="s1")
        call = mock_gen.generate.call_args
        assert "seed" not in call.kwargs, (
            f"seed kwarg leaked into legacy path: {call.kwargs}"
        )

    def test_query_with_seed_propagates_to_generator(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """``seed=42`` -> generator.generate(messages, seed=42)."""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "Answer with [C:1]."
        pipeline = _build_test_pipeline(generator=mock_gen)
        pipeline.query("test question", session_id="s1", seed=42)
        call = mock_gen.generate.call_args
        assert call.kwargs.get("seed") == 42, (
            f"seed=42 did not propagate; kwargs={call.kwargs}"
        )

    def test_query_with_seed_zero_propagates(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """``seed=0`` MUST propagate (DR3-002: ``if seed:`` is a silent bug).

        Falsy-but-valid seed: ``if seed:`` would silently drop the call,
        leaving the eval nondeterministic. The implementation must use
        ``if seed is not None:``.
        """
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "Answer with [C:1]."
        pipeline = _build_test_pipeline(generator=mock_gen)
        pipeline.query("test question", session_id="s1", seed=0)
        call = mock_gen.generate.call_args
        assert call.kwargs.get("seed") == 0, (
            f"seed=0 silently dropped; kwargs={call.kwargs}"
        )


# ---------------------------------------------------------------------------
# Issue #176: retrieval_debug populated in QueryResult
# ---------------------------------------------------------------------------


@patch(
    "baseline_reporag.pipeline.expand_with_graph",
    side_effect=_mock_expand_with_graph,
)
@patch(
    "baseline_reporag.pipeline.hybrid_search",
    side_effect=_mock_hybrid_search,
)
class TestRetrievalDebug:
    """Issue #176: QueryResult.retrieval_debug must be populated after query."""

    def test_retrieval_debug_is_not_none(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """retrieval_debug must be a non-None list after a query turn."""
        pipeline = _build_test_pipeline()
        result = pipeline.query("test question", session_id="s1")
        assert result.retrieval_debug is not None, (
            "QueryResult.retrieval_debug must be populated (Issue #176)"
        )

    def test_retrieval_debug_sources_valid(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """All retrieval_debug rows must have a recognized source value."""
        pipeline = _build_test_pipeline()
        result = pipeline.query("test question", session_id="s1")
        assert result.retrieval_debug is not None
        valid_sources = {
            "retrieval",
            "graph",
            "neighbor",
            "photon_pruned",
            "working_memory",
        }
        for row in result.retrieval_debug:
            assert row.source in valid_sources, (
                f"Unexpected source={row.source!r} for chunk {row.chunk_id}"
            )

    def test_retrieval_debug_has_used_rows(
        self,
        mock_search: MagicMock,
        mock_expand: MagicMock,
    ) -> None:
        """At least one row must have used=True (chunks were added to evidence pack)."""
        pipeline = _build_test_pipeline()
        result = pipeline.query("test question", session_id="s1")
        assert result.retrieval_debug is not None
        assert any(r.used for r in result.retrieval_debug), (
            "At least one row must be used=True after evidence pack construction"
        )
