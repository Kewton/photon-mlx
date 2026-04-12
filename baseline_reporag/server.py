"""
Baseline RepoRAG – FastAPI server.

Start with:
    python -m baseline_reporag.server --config configs/baseline.yaml
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from .config import Config, load_config
from .generation.generator import Generator
from .indexing.embedding import EmbeddingIndex
from .indexing.lexical import LexicalIndex
from .indexing.symbol_graph import SymbolGraph
from .ingestion.store import ChunkStore
from .logger import RunLogger
from .memory.session import SessionManager
from .pipeline import RepoRAGPipeline

app = FastAPI(title="baseline-reporag")
_pipeline: RepoRAGPipeline | None = None


def _build_pipeline(config: Config) -> RepoRAGPipeline:
    idx_dir = Path(config.paths.data_root) / "indexes" / config.repo.repo_id
    run_id = (
        f"baseline_{config.repo.repo_id}"
        f"_{time.strftime('%Y%m%d')}"
        f"_{config.repo.repo_commit[:7]}"
    )
    return RepoRAGPipeline(
        config=config,
        store=ChunkStore(idx_dir / "chunks.db"),
        lexical=LexicalIndex.load(idx_dir / "lexical.pkl"),
        embedding=EmbeddingIndex.load(idx_dir / "embedding"),
        graph=SymbolGraph.load(idx_dir / "symbol_graph.json"),
        sessions=SessionManager(log_dir=Path(config.paths.log_root) / "sessions"),
        generator=Generator(
            model_id=config.model.model_id,
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
        ),
        logger=RunLogger(config.paths.log_root, run_id),
    )


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
    memory_peak_mb: float


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = _pipeline.query(
        question=req.question,
        session_id=req.session_id,
        repo_id=req.repo_id,
    )
    return QueryResponse(
        answer=result.answer,
        session_id=result.session_id,
        turn_id=result.turn_id,
        cited_chunk_ids=result.cited_chunk_ids,
        latency_ms=result.latency.total_ms,
        memory_peak_mb=result.memory.peak_mb,
    )


def main() -> None:
    import argparse
    import uvicorn

    global _pipeline

    parser = argparse.ArgumentParser(description="Baseline RepoRAG server")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    _pipeline = _build_pipeline(load_config(args.config))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
