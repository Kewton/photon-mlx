"""
Baseline RepoRAG – FastAPI server.

Start with:
    python -m baseline_reporag.server --config configs/baseline.yaml
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from pydantic import BaseModel

from .config import Config, load_config

# CB-004 (codex-fix): import from the lightweight factory so baseline-only
# deployments can boot the FastAPI server without MLX installed.  The
# PHOTON pipeline type is only needed for the type annotation below.
from .pipeline import RepoRAGPipeline
from .pipeline_factory import build_pipeline

if TYPE_CHECKING:  # pragma: no cover - type hint only
    from .photon_pipeline import PhotonRAGPipeline

app = FastAPI(title="baseline-reporag")
_pipeline: "RepoRAGPipeline | PhotonRAGPipeline | None" = None


def _build_pipeline(config: Config) -> "RepoRAGPipeline | PhotonRAGPipeline":
    """Delegate to the provider-aware factory so ``model.provider`` wires
    to the right pipeline (Issue #62 Phase 1 Stage 3 DR3-001).
    """
    return build_pipeline(config)


def _resolve_config_path(config_path: str) -> str:
    path = Path(config_path)
    if path.exists() or path.is_absolute() or path.parent.name != "configs":
        return config_path
    try:
        packaged = resources.files("configs").joinpath(path.name)
    except ModuleNotFoundError:
        return config_path
    return str(packaged) if packaged.is_file() else config_path


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
    refusal_score: float | None = None


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
        refusal_score=result.refusal_score,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    import uvicorn

    global _pipeline

    parser = argparse.ArgumentParser(description="Baseline RepoRAG server")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    _pipeline = _build_pipeline(load_config(_resolve_config_path(args.config)))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
