"""Unit tests for baseline_reporag.retrieval.query_expansion.

Covers Issue #111: domain-agnostic generalization of `expand_query`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from baseline_reporag.config import Config, load_config
from baseline_reporag.retrieval.query_expansion import (
    _JP_TO_CODE,
    _normalize_mapping,
    expand_query,
)


# --------------------------------------------------------------------------- #
# _normalize_mapping unit tests
# --------------------------------------------------------------------------- #


def test_normalize_mapping_none_returns_builtin():
    result = _normalize_mapping(None)
    assert result is _JP_TO_CODE


def test_normalize_mapping_config_converts_to_dict():
    cfg = Config({"ミドルウェア": ["middleware", "Middleware"]})
    result = _normalize_mapping(cfg)
    assert isinstance(result, dict)
    assert result == {"ミドルウェア": ["middleware", "Middleware"]}


def test_normalize_mapping_dict_identity():
    mapping = {"ミドルウェア": ["middleware"]}
    result = _normalize_mapping(mapping)
    assert result is mapping


def test_normalize_mapping_empty_dict_preserved():
    result = _normalize_mapping({})
    assert result == {}


# --------------------------------------------------------------------------- #
# expand_query behavior tests (Cases 1-7, 11)
# --------------------------------------------------------------------------- #


def test_case_1_mapping_none_matches_builtin():
    """Case 1: mapping=None → backward compat with built-in _JP_TO_CODE."""
    query = "ミドルウェアの使い方"
    result_none = expand_query(query, mapping=None)
    result_default = expand_query(query)
    assert result_none == result_default
    # Verify actual expansion happened from _JP_TO_CODE
    assert len(result_none) == 2
    assert "middleware" in result_none[1]


def test_case_2_mapping_plain_dict():
    """Case 2: mapping=<plain dict> → expansion from given dict."""
    mapping = {"ミドルウェア": ["middleware", "MW"]}
    result = expand_query("ミドルウェアの設定", mapping=mapping)
    assert result[0] == "ミドルウェアの設定"
    assert len(result) == 2
    assert "middleware" in result[1]
    assert "MW" in result[1]


def test_case_3_mapping_config_wrapper():
    """Case 3: mapping=<Config wrapper> → same result as plain dict."""
    plain = {"ミドルウェア": ["middleware", "MW"]}
    wrapped = Config(plain)
    query = "ミドルウェアの設定"
    result_plain = expand_query(query, mapping=plain)
    result_wrapped = expand_query(query, mapping=wrapped)
    assert result_plain == result_wrapped


def test_case_4a_empty_mapping_no_identifier():
    """Case 4a: mapping={} + query without identifier → [query] only."""
    result = expand_query("テスト", mapping={})
    assert result == ["テスト"]


def test_case_4b_empty_mapping_with_identifier():
    """Case 4b: mapping={} + query with identifier → identifier extraction runs."""
    # Identifier is embedded in a no-space token so query.split() doesn't
    # include it verbatim, triggering the identifier-extraction path.
    result = expand_query("ApiRouterを使う方法", mapping={})
    # Japanese expansion disabled, but identifier extraction still runs
    assert result[0] == "ApiRouterを使う方法"
    assert len(result) == 2
    assert "ApiRouter" in result[1]


def test_case_5_mapping_with_undefined_key():
    """Case 5: mapping contains undefined key for this query → no expansion."""
    mapping = {"存在しないキー": ["nothing"]}
    result = expand_query("ミドルウェアの設定", mapping=mapping)
    # No Japanese expansion, no identifiers → [query] only
    assert result == ["ミドルウェアの設定"]


def test_case_6_identifier_and_japanese_combined():
    """Case 6: identifier extraction + Japanese expansion together."""
    mapping = {"ミドルウェア": ["middleware"]}
    # Identifier embedded without whitespace so identifier extraction fires.
    query = "ApiRouterのミドルウェア設定"
    result = expand_query(query, mapping=mapping)
    assert result[0] == query
    assert len(result) == 2
    assert "middleware" in result[1]
    assert "ApiRouter" in result[1]


def test_case_7_snapshot_regression_full_builtin_pass_through():
    """Case 7: passing full _JP_TO_CODE externally → equals mapping=None result."""
    queries = [
        "ミドルウェアの使い方",
        "依存性注入の例",
        "認証と認可の違い",
        "非同期処理のデッドロック調査",
        "APIRouter と include_router",
        "OpenAPI スキーマ生成",
    ]
    for q in queries:
        assert expand_query(q, mapping=_JP_TO_CODE) == expand_query(q, mapping=None)


# --------------------------------------------------------------------------- #
# Case 11: SoT equivalence (baseline.yaml ↔ module constant)
# --------------------------------------------------------------------------- #


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


REPRESENTATIVE_QUERIES = [
    "ミドルウェアの使い方",
    "依存性注入の例",
    "認証と認可の違い",
    "非同期処理のデッドロック調査",
    "APIRouter と include_router",
    "OpenAPI スキーマ生成",
    "エラーハンドリングのベストプラクティス",
    "クエリパラメータのバリデータ",
]


def test_case_11_sot_equivalence_expand_query():
    """Case 11: baseline.yaml domain_map == module constant for representative queries."""
    cfg = load_config(REPO_ROOT / "configs" / "baseline.yaml")
    domain_map = cfg.retrieval.query_expansion.get("domain_map")
    assert domain_map is not None, (
        "baseline.yaml must declare retrieval.query_expansion.domain_map"
    )
    for q in REPRESENTATIVE_QUERIES:
        assert expand_query(q, mapping=domain_map) == expand_query(q, mapping=None), (
            f"SoT divergence on query: {q}"
        )


def test_case_11_sot_equivalence_noise_patterns():
    """Case 11 (paired): baseline.yaml noise_patterns == module _NOISE_PATTERNS."""
    from baseline_reporag.retrieval.reranker import _NOISE_PATTERNS

    cfg = load_config(REPO_ROOT / "configs" / "baseline.yaml")
    yaml_patterns = cfg.retrieval.reranker.get("noise_patterns")
    assert yaml_patterns is not None, (
        "baseline.yaml must declare retrieval.reranker.noise_patterns"
    )
    assert tuple(yaml_patterns) == _NOISE_PATTERNS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
