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

Issue #133 — institutional 用 invariant の placeholder を追加 (skip)。
A/B 実機評価で採用 variant が決定したら、skip を外して採用後の
embedding/reranker model_id を assertion 化する。これにより
institutional プロファイルから global default への cascade を防止する
(設計書 §3.2 / 判断 #4)。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from baseline_reporag.config import load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

GLOBAL_DEFAULT_RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Issue #133 で採用 variant が決定したら以下を実値で埋め、@pytest.mark.skip を外す。
# 採用判断は reports/institutional_retrieval_ab.md で行う。
INSTITUTIONAL_RERANKER_MODEL_ID: str | None = None
INSTITUTIONAL_EMBEDDING_MODEL_ID: str | None = None


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


@pytest.mark.skipif(
    INSTITUTIONAL_RERANKER_MODEL_ID is None,
    reason="Issue #133: 採用 variant 決定後に有効化 (現在 A/B 評価中)",
)
def test_institutional_yaml_reranker_model_id_pinned() -> None:
    """``configs/institutional_docs.yaml`` reranker.model_id を採用後値に pin する。

    Issue #133 の A/B 評価で採用 variant が決まったら、
    ``INSTITUTIONAL_RERANKER_MODEL_ID`` を埋めて skip を外すこと。
    これにより institutional プロファイルから global default
    (``configs/baseline.yaml``) への cascade を防ぐ。
    """
    cfg = load_config(CONFIGS_DIR / "institutional_docs.yaml")
    assert cfg.retrieval.reranker.model_id == INSTITUTIONAL_RERANKER_MODEL_ID


@pytest.mark.skipif(
    INSTITUTIONAL_EMBEDDING_MODEL_ID is None,
    reason="Issue #133: 採用 variant 決定後に有効化 (現在 A/B 評価中)",
)
def test_institutional_yaml_embedding_model_id_pinned() -> None:
    """``configs/institutional_docs.yaml`` embedding.model_id を採用後値に pin する。"""
    cfg = load_config(CONFIGS_DIR / "institutional_docs.yaml")
    assert cfg.indexing.embedding.model_id == INSTITUTIONAL_EMBEDDING_MODEL_ID
