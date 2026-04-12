"""
Baseline RepoRAG – FastAPI server.

Start with:
    python -m baseline_reporag.server --config configs/local.baseline.yaml
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from .citation import resolve_citations
from .config import Config, load_config
from .generation.evidence_pack import build_evidence_pack
from .generation.generator import Generator
from .generation.prompt import build_messages
from .indexing.embedding import EmbeddingIndex
from .indexing.lexical import LexicalIndex
from .indexing.symbol_graph import SymbolGraph
from .ingestion.store import ChunkStore
from .logger import RunLogger
from .memory.session import SessionManager
from .retrieval.graph_expansion import expand_with_graph
from .retrieval.hybrid import hybrid_search

app = FastAPI(title="baseline-reporag")
_state: dict = {}


def _index_dir(config: Config) -> Path:
    return Path(config.paths.data_root) / "indexes" / config.repo.repo_id


def init_app(config: Config) -> None:
    idx_dir = _index_dir(config)
    _state["config"] = config
    _state["store"] = ChunkStore(idx_dir / "chunks.db")
    _state["lexical"] = LexicalIndex.load(idx_dir / "lexical.pkl")
    _state["embedding"] = EmbeddingIndex.load(idx_dir / "embedding")
    _state["graph"] = SymbolGraph.load(idx_dir / "symbol_graph.json")
    _state["sessions"] = SessionManager(
        log_dir=Path(config.paths.log_root) / "sessions"
    )
    _state["generator"] = Generator(
        model_id=config.model.model_id,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
    )
    run_id = (
        f"baseline_{config.repo.repo_id}"
        f"_{time.strftime('%Y%m%d')}"
        f"_{config.repo.repo_commit[:7]}"
    )
    _state["logger"] = RunLogger(config.paths.log_root, run_id)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    session_id: str = ""
    repo_id: str = ""


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    turn_id: int
    cited_chunk_ids: list[str]
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    cfg: Config = _state["config"]
    t0 = time.perf_counter()

    session_id = req.session_id or str(uuid.uuid4())
    repo_id = req.repo_id or cfg.repo.repo_id
    session = _state["sessions"].get_or_create(
        session_id, repo_id, cfg.repo.repo_commit
    )

    raw = hybrid_search(
        query=req.question,
        lexical_index=_state["lexical"],
        embedding_index=_state["embedding"],
        lexical_top_k=cfg.retrieval.lexical_top_k,
        embedding_top_k=cfg.retrieval.embedding_top_k,
        fused_top_k=cfg.retrieval.fused_top_k,
        lexical_weight=cfg.retrieval.weights.lexical,
        embedding_weight=cfg.retrieval.weights.embedding,
    )

    expanded_ids = expand_with_graph(
        results=raw,
        store=_state["store"],
        graph=_state["graph"],
        repo_id=repo_id,
        repo_commit=cfg.repo.repo_commit,
        max_hops=cfg.retrieval.graph_expansion.max_hops,
        max_nodes=cfg.retrieval.graph_expansion.max_nodes,
        neighborhood_before=cfg.retrieval.neighborhood_expansion.before,
        neighborhood_after=cfg.retrieval.neighborhood_expansion.after,
    )

    pack = build_evidence_pack(
        chunk_ids=expanded_ids,
        store=_state["store"],
        session=session,
        max_chunks=cfg.evidence_pack.max_chunks,
        max_tokens=cfg.evidence_pack.max_tokens,
    )

    messages = build_messages(
        question=req.question,
        evidence_text=pack.format_for_prompt(),
        history_text=session.history_text(max_turns=4),
    )
    answer = _state["generator"].generate(messages)
    citation = resolve_citations(answer, pack)

    turn = session.add_turn(req.question, answer, citation.cited_chunk_ids)
    _state["sessions"].save(session)

    latency_ms = (time.perf_counter() - t0) * 1000

    _state["logger"].log_turn({
        "session_id": session_id,
        "turn_id": turn.turn_id,
        "repo_id": repo_id,
        "repo_commit": cfg.repo.repo_commit,
        "model_id": cfg.model.model_id,
        "question": req.question,
        "answer": answer,
        "retrieval_chunk_ids": [r.chunk_id for r in raw],
        "evidence_pack_ids": [c.chunk_id for c in pack.chunks],
        "cited_chunk_ids": citation.cited_chunk_ids,
        "wrong_citation_indices": citation.wrong_citation_indices,
        "no_citation": citation.no_citation,
        "latency_ms": latency_ms,
        "fallback_flag": False,
        "fallback_reason": None,
    })

    return QueryResponse(
        answer=answer,
        session_id=session_id,
        turn_id=turn.turn_id,
        cited_chunk_ids=citation.cited_chunk_ids,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Baseline RepoRAG server")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg = load_config(args.config)
    init_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
