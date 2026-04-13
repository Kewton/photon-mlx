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
            "weights": {"lexical": 0.5, "embedding": 0.5},
            "graph_expansion": {"max_hops": 1, "max_nodes": 16},
            "neighborhood_expansion": {"before": 1, "after": 1},
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
    """expand_with_graph のモック: chunk ID リストを返す。"""
    return _MOCK_CHUNK_IDS


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
        assert "Example:" in user_content
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
        assert "Example:" in user_content
        assert "Q: Where is the main router defined?" in user_content
