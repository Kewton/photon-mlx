"""Issue #114 — YAML invariant tests for the global pipeline factory.

Locks down the global-default ``reranker.model_id`` in
``configs/baseline.yaml`` so a future Issue cannot silently swap it.
#96 was reverted because an institutional-only evaluation result was
promoted to the global default; this test forces an intentional update
of both the YAML and the assertion in the same commit, surfacing intent
and preventing #96-style regressions.

The invariant is checked via ``baseline_reporag.config.load_config`` so
the assertion goes through the same defaulting / merge path that
``pipeline_factory`` consumes — a raw ``yaml.safe_load`` would skip
``Config.merge_override`` and risk drifting from runtime behaviour.
"""

from __future__ import annotations

from pathlib import Path

from baseline_reporag.config import load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

GLOBAL_DEFAULT_RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_baseline_yaml_reranker_model_id_unchanged() -> None:
    """``configs/baseline.yaml`` reranker.model_id must remain the global
    English default.

    Changing this requires updating both the YAML and this assertion in
    the same commit. That intentional friction prevents another
    institutional-domain evaluation result from leaking into the global
    default — the failure mode that motivated the #96 revert. Issue
    #114 evaluates a multilingual reranker for the institutional config
    only; the global default stays unchanged.
    """
    cfg = load_config(CONFIGS_DIR / "baseline.yaml")
    assert cfg.retrieval.reranker.model_id == GLOBAL_DEFAULT_RERANKER_MODEL_ID
