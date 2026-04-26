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


# ---------------------------------------------------------------------------
# Issue #139 — Phase A: PHOTON profile yaml must declare ``tokenizer.vocab_size``
# and ``tokenizer.tokenizer_id`` so ``_build_photon_deps`` (which now raises
# ValueError on missing tokenizer_id) cannot silently break a yaml. Phase B
# (other ``getattr default`` patterns) is tracked separately.
#
# Hardening (Issue #139 / DR4-004): this test is **never** skipped or xfailed.
# Adding ``@pytest.mark.skip`` / ``skipif`` / ``xfail`` here would re-open the
# CI gate that the design closes — review must reject any such patch.
# ---------------------------------------------------------------------------


def _is_photon_profile_yaml(path: Path, cfg) -> bool:
    """Return True if ``path`` is a PHOTON-profile yaml.

    Filename judgment is the main signal — every PHOTON yaml in this repo is
    named ``photon_*.yaml`` or ``institutional_docs_photon.yaml``. A
    ``model.provider == 'photon'`` check is kept as a fallback so a future
    yaml that follows a different naming scheme but still declares
    ``provider=='photon'`` is also covered (DR Stage 1 / DR1-006).
    """
    name = path.name
    if name.startswith("photon_") or name == "institutional_docs_photon.yaml":
        return True
    model_section = getattr(cfg, "model", None)
    if model_section is None:
        return False
    return getattr(model_section, "provider", None) == "photon"


def test_photon_yaml_has_required_tokenizer_fields() -> None:
    """Every PHOTON-profile yaml in ``configs/`` must declare
    ``tokenizer.vocab_size`` and ``tokenizer.tokenizer_id``.

    Issue #139: ``_build_photon_deps`` raises ``ValueError`` if
    ``tokenizer.tokenizer_id`` is missing or unsafe (the previous
    ``_StubTokenizer`` fallback was deleted). This invariant test catches
    yaml-side regressions at CI time so a future PR adding a new PHOTON
    profile config without a ``tokenizer:`` block fails the merge gate
    rather than the production server start-up.

    ``tokenizer.vocab_size`` is the canonical embedding size (Issue #138)
    and is also pinned here.

    Non-PHOTON yaml (e.g. ``configs/baseline.yaml`` with provider=mlx_lm,
    ``configs/eval.yaml`` benchmark runner config) are filtered out by
    ``_is_photon_profile_yaml``.
    """
    failures: list[tuple[str, str]] = []
    for yaml_path in sorted(CONFIGS_DIR.glob("*.yaml")):
        cfg = load_config(yaml_path)
        if not _is_photon_profile_yaml(yaml_path, cfg):
            continue
        tok = getattr(cfg, "tokenizer", None)
        for key in ("vocab_size", "tokenizer_id"):
            value = getattr(tok, key, None) if tok is not None else None
            if value in (None, ""):
                failures.append((str(yaml_path), f"tokenizer.{key}"))
    assert not failures, (
        f"PHOTON profile yaml missing required tokenizer fields: {failures}"
    )
