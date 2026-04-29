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


# --------------------------------------------------------------------------- #
# Issue #175 (G2): institutional 同義語辞書テスト
# --------------------------------------------------------------------------- #


_INSTITUTIONAL_CONFIGS = (
    "configs/institutional_docs.yaml",
    "configs/institutional_docs_photon.yaml",
    "configs/institutional_docs_photon_retrain.yaml",
)


def _load_institutional_domain_map(rel_path: str) -> dict[str, list[str]]:
    cfg = load_config(REPO_ROOT / rel_path)
    domain_map = cfg.retrieval.query_expansion.get("domain_map")
    assert domain_map is not None, f"{rel_path} must declare domain_map"
    if hasattr(domain_map, "to_dict"):
        return domain_map.to_dict()
    return dict(domain_map)


class TestInstitutionalDomainMap:
    """Issue #175: institutional configs MUST share the same domain_map content
    so retrieval behaviour is identical across baseline / PHOTON / retrain
    profiles. Empty domain_map (the pre-#175 state) MUST not regress."""

    @pytest.mark.parametrize("config_path", _INSTITUTIONAL_CONFIGS)
    def test_domain_map_is_non_empty(self, config_path: str) -> None:
        domain_map = _load_institutional_domain_map(config_path)
        assert len(domain_map) >= 20, (
            f"{config_path}: institutional domain_map must contain ≥20 entries "
            f"(AC1 from Issue #175). Got {len(domain_map)}."
        )

    def test_domain_map_content_identical_across_configs(self) -> None:
        """SoT equivalence: 3 institutional configs share identical domain_map."""
        first = _load_institutional_domain_map(_INSTITUTIONAL_CONFIGS[0])
        for cfg_path in _INSTITUTIONAL_CONFIGS[1:]:
            other = _load_institutional_domain_map(cfg_path)
            assert other == first, (
                f"{cfg_path} domain_map differs from {_INSTITUTIONAL_CONFIGS[0]}"
            )


class TestInstitutionalSynonymExpansion:
    """Issue #175 AC3 / AC4: query が synonym key を含む場合に展開語が
    expand_query() の出力に登場することを E2E で確認。"""

    @pytest.fixture(scope="class")
    def domain_map(self) -> dict[str, list[str]]:
        return _load_institutional_domain_map("configs/institutional_docs.yaml")

    def test_ac3_認定基準_expansion(self, domain_map) -> None:
        """AC3: ``認定基準`` query → ``認定の基準`` / ``認定要件`` / ``対象事業者``
        が展開語に含まれる。"""
        queries = expand_query(
            "セーフティネット保証1号の認定基準を簡潔に教えて", mapping=domain_map
        )
        assert len(queries) == 2, "expansion must produce 1 combined extra query"
        expanded = queries[1]
        assert "認定の基準" in expanded
        assert "認定要件" in expanded
        assert "対象事業者" in expanded

    def test_ac4_申請期限_expansion(self, domain_map) -> None:
        """AC4: ``申請期限`` query → ``提出期限`` / ``締切`` / ``期日`` が
        展開語に含まれる (cap=8 内に複数 entries が co-fire するケース)."""
        queries = expand_query("補助金の申請期限はいつですか?", mapping=domain_map)
        expanded = queries[1]
        # 「申請」「申請期限」「期限」「補助金」が co-fire するため cap=8 で絞られる。
        # 締切 (申請期限+期限 双方の値) は確実に登場するはず。
        assert "締切" in expanded, f"'締切' missing — expansion: {expanded!r}"

    def test_必要書類_expansion(self, domain_map) -> None:
        """『必要書類』query → 添付書類 / 提出書類 / 提出資料 が展開語に登場."""
        queries = expand_query("必要書類を教えてください", mapping=domain_map)
        expanded = queries[1]
        assert "添付書類" in expanded or "提出書類" in expanded

    def test_no_match_query_returns_only_original(self, domain_map) -> None:
        """domain_map のどの key にも match しない query は expand されない."""
        # ASCII identifier も含まないシンプル日本語
        queries = expand_query("こんにちは", mapping=domain_map)
        assert queries == ["こんにちは"]

    def test_reverse_direction_補助_helps_query_about_助成(self, domain_map) -> None:
        """双方向性: 「補助」「助成」のどちらの表記でも query → corpus 展開が効く."""
        # corpus 表現が「補助」、利用者 query が「助成」
        queries_helping_search = expand_query("助成の上限額", mapping=domain_map)
        # 助成 → 補助 が含まれる
        assert "補助" in queries_helping_search[1]
        # 逆方向
        queries_back = expand_query("補助率について", mapping=domain_map)
        # 補助率 → 補助割合 / 補助 → 助成 が含まれる
        expanded = queries_back[1]
        assert "補助割合" in expanded or "助成" in expanded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
