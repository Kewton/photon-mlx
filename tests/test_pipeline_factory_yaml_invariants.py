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

# Issue #148 Phase A0: pin the LLM model_id used for generation in the global
# default profile.  Phase C (adoption LLM evaluation) updates this constant in
# the same commit that swaps the yaml field so the change is intentional and
# visible in code review.
#
# 採用切替 (2026-04-28, Phase B/C): Qwen 2.5 → Qwen 3.5 no-think。
# 評価エビデンス: reports/qwen_model_matrix_20260428_400cmp_report.md (400-sample)
#   - Baseline+Qwen3.5: static p50 -38.6%, multi-turn p50 -46.2%
#   - PHOTON+Qwen3.5: static p50 -44.3%, multi-turn p50 -43.6%, NC 0.00%
GLOBAL_DEFAULT_GENERATION_MODEL_ID = "mlx-community/Qwen3.5-9B-MLX-4bit"

# Issue #137 Phase B: 5-variant A/B で V4 (bge-m3 + bge-reranker-v2-m3, 8192 chars)
# 採用 (NC -6.90pt vs V0 12.93%、reports/institutional_retrieval_ab.md 参照)。
# 定数を実値に置換することで skipif 条件 (is None) が False になり test が自動活性化する。
INSTITUTIONAL_RERANKER_MODEL_ID: str | None = "BAAI/bge-reranker-v2-m3"
INSTITUTIONAL_EMBEDDING_MODEL_ID: str | None = "BAAI/bge-m3"
# Issue #137 V4 採用に伴い max_input_chars=8192 を pin (fallback 2048 への drift を防止)。
INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS: int | None = 8192


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


@pytest.mark.skipif(
    INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS is None,
    reason="Issue #137: V4 (bge-m3 8192 chars) 採用時のみ有効化",
)
def test_institutional_yaml_embedding_max_input_chars_pinned() -> None:
    """``configs/institutional_docs.yaml`` embedding.max_input_chars を採用値に pin する。

    fallback 2048 への drift (config を編集して max_input_chars を消す等) を防ぐ。
    cfg.indexing.embedding は Config オブジェクト (baseline_reporag/config.py) で
    attribute access と .get() の両 API を提供する。max_input_chars は optional
    フィールドのため、未宣言時に AttributeError を起こさない .get() 経路を採用
    (build_indexes.py の getattr fallback パターンと整合)。
    """
    cfg = load_config(CONFIGS_DIR / "institutional_docs.yaml")
    assert (
        cfg.indexing.embedding.get("max_input_chars")
        == INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS
    )


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


def test_baseline_yaml_generation_model_id_pinned() -> None:
    """``configs/baseline.yaml`` model.model_id must remain the global default LLM.

    Issue #148 Phase A0: pin the generation LLM to prevent silent swap.
    Phase C (adoption LLM evaluation) updates both the YAML and this assertion
    in the same commit so the change is intentional and traceable.

    ``cfg.model.model_id`` is the field that ``pipeline_factory`` passes to
    the generation back-end (mlx_lm / transformers). ``configs/baseline.yaml``
    does not have a separate ``generation.model_id`` field.
    """
    cfg = load_config(CONFIGS_DIR / "baseline.yaml")
    assert cfg.model.model_id == GLOBAL_DEFAULT_GENERATION_MODEL_ID


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


# ---------------------------------------------------------------------------
# Issue #135 Phase 8 — institutional_docs_photon.yaml の checkpoint_path を
# institutional retrain step_003000 に昇格 (Phase 7 refusal-aware Turn 5-6
# NC = 0.00% で採用判定)。再学習を経て mulmoclaude step_000600 へ戻すような
# silent rollback を CI で検出する。
# ---------------------------------------------------------------------------


# Phase 8 採用 checkpoint。値変更は本 Issue #135 の前提を覆すため、必ず
# 設計方針書 + reports/ + bug check report をセットで更新すること。
INSTITUTIONAL_PHOTON_ADOPTED_CHECKPOINT = (
    "photon_institutional_retrain_20260428/step_003000"
)


def test_institutional_docs_photon_checkpoint_path_is_phase8_adopted() -> None:
    """``configs/institutional_docs_photon.yaml`` の ``model.checkpoint_path``
    は Phase 8 採用 ckpt (step_003000) に固定される。

    Issue #135 Phase 8: refusal-aware Turn 5-6 NC = 0.00% (Issue #154 Bug 2
    観点) を達成した institutional retrain 成果物を production photon path
    として採用。エビデンスは
    ``reports/institutional_photon_mt_eval_v2_3k.md`` および
    ``reports/institutional_photon_mt_eval_v2_3k_bug_check.md``。

    Silent な rollback (例: 旧 ``step_000600`` への戻し) を CI で防ぐため、
    値を厳格 pin する。値変更には本 Issue の前提見直し + 採用判定報告書の
    更新がセットで必要。
    """
    cfg = load_config(CONFIGS_DIR / "institutional_docs_photon.yaml")
    assert cfg.model.checkpoint_path == INSTITUTIONAL_PHOTON_ADOPTED_CHECKPOINT, (
        f"institutional_docs_photon.yaml の checkpoint_path が "
        f"Phase 8 採用 ckpt から逸脱: got {cfg.model.checkpoint_path!r}, "
        f"expected {INSTITUTIONAL_PHOTON_ADOPTED_CHECKPOINT!r}"
    )
