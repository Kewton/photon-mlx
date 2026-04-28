# Issue #140 マルチステージ設計レビュー完了報告

## 概要

Issue #140 の設計方針書 (`workspace/design/issue-140-review-process-design-policy.md`) に対し、`/multi-stage-design-review 140` を 4 ステージ完走しました。Stage 1-2 (opus) で設計原則 / 整合性、Stage 3-4 (Codex) で影響範囲 / セキュリティを段階的にレビューし、合計 **26 finding** (Must Fix 8 / Should Fix 14 / Nice to Have 4) を抽出・全件設計方針書に反映済みです。

## ステージ別結果

| Stage | レビュー種別 | レビュアー | 指摘数 (MF/SF/NTH) | 対応数 | ステータス |
|-------|------------|----------|-----------------|-------|----------|
| 1 | 通常レビュー (設計原則) | **opus** | 10 (3/5/2) | 10 | 完了 (reviewer 検証: opus) |
| 2 | 整合性レビュー | **opus** | 10 (2/6/2) | 10 | 完了 (reviewer 検証: opus) |
| 3 | 影響分析レビュー | **codex** | 3 (1/2/0) | 3 | 完了 (reviewer="codex" ✓) |
| 4 | セキュリティレビュー | **codex** | 3 (2/1/0) | 3 | 完了 (reviewer="codex" ✓) |

**合計**: 26 finding (Must Fix 8 / Should Fix 14 / Nice to Have 4)、全件反映。

## reviewer フィールド検証

- `stage3-review-result.json`: `reviewer = "codex"` ✓
- `stage4-review-result.json`: `reviewer = "codex"` ✓
- Codex stage の結果ファイルは Claude による不正な上書きなし

## 主要な反映内容

### Stage 1 通常レビュー (opus / 設計原則)

**Must Fix (3 件)**:
- DR1-001: 設計判断 #3 の WARNING 抑制実装が `conftest.py:_make` の実体と矛盾 → `_tiny_cfg` ベース + `_photon_cfg` ヘルパー方式に書き換え
- DR1-002: (a) ModelConfig 採用確定後の S7-003 (WorkingMemoryConfig 案) 受入条件の取り扱いが未宣言 → 対象外宣言を追記
- DR1-003: Issue 受入条件 Task 4 の yaml 互換 test が §7 に欠落 → §7.4 を新設

**Should Fix (5 件)**: SRP/KISS トレードオフの根拠補強、`_photon_cfg` ヘルパー定義、Markdown snippet 動作 smoke test、行番号参照の脆弱性、実装順序の依存逆転

**Nice to Have (2 件)**: grep 対象の Single Source of Truth、「必須」の段階的厳格化注記

### Stage 2 整合性レビュー (opus / 設計書 vs 実コード vs Issue)

**Must Fix (2 件)**:
- DR2-001: `_tiny_cfg` ヘルパーが `photon_mlx/tests/` 配下 4 ファイルで重複定義されている事実が未カバー → test_inference.py のみ更新する方針を明示
- DR2-002: §7.4 yaml load 互換テストが `load_photon_config` の `__post_init__` 再実行 / `_validate_cross_config` 経路を考慮していない → 注意点を追記

**Should Fix (6 件)**: 既存 test 件数表現の経路別分割、`_build_real_model` / `_stub_tokenizer` の未定義の解消、summary 表の列数変更明示、reviewer snippet を issue-review/design-review で 2 種類に分離、`REVIEWER_SNIPPET_RE` の頑健性、yaml 互換 test の早期化

**Nice to Have (2 件)**: test の意味的強度、「7 個 → 5 個」表記の単純化

### Stage 3 影響分析レビュー (codex)

**Must Fix (1 件)**:
- DR3-001: WARNING 抑制対象を `test_inference.py` のみに限定すると `test_generate.py` / `test_session.py` の実 `PhotonInference` 経路で WARNING が漏れる → 3 ファイルすべての `_tiny_cfg` を `float('inf')` 化する方針に拡張

**Should Fix (2 件)**:
- DR3-002: yaml 互換 test の配置先を `photon_mlx/tests/test_config.py` と明記
- DR3-003: §7.5 smoke test の `...` プレースホルダを実装可能な具体コードに置換

### Stage 4 セキュリティレビュー (codex)

**Must Fix (2 件)**:
- DR4-001: reviewer 検証 snippet が未検証の `ISSUE` を path / Python code に展開 → path traversal / code injection リスク → `ISSUE` を数値のみ許可、Python は `sys.argv[1]` 経由で path を受け渡す方式に変更
- DR4-002: `embedding_random_init_threshold` の型・範囲検証が欠落、`float('inf')` を test default にすると production validation と衝突 → `__post_init__` で型・範囲検証を入れ、test suppress 値を有限の `TEST_EMBEDDING_RANDOM_INIT_THRESHOLD = 1e9` に変更

**Should Fix (1 件)**:
- DR4-003: reviewer 値の raw log 出力でログ汚染 → 比較は raw 値、ログ出力前に制御文字を置換

## 設計方針書の現状

- 場所: `workspace/design/issue-140-review-process-design-policy.md`
- セクション構成: §1 ゴール / §2 技術スコープ / §3 アーキテクチャ判断 (#1〜#6) / §4 レイヤー別変更マップ / §5 セキュリティ設計 / §6 後方互換性ロールバック / §7 テスト戦略 (7.1〜7.5) / §8 実装順序 (14 ステップ) / §9 品質基準 / §10 参考資料
- ソースコード変更は **未着手**。本コマンドのスコープは設計方針書のレビューと改善のみ。

## 次のアクション

- [x] Stage 1: 通常レビュー (設計原則) 完了
- [x] Stage 2: 整合性レビュー 完了
- [x] Stage 3: 影響分析レビュー 完了
- [x] Stage 4: セキュリティレビュー 完了
- [ ] Phase 4: `/work-plan 140` で作業計画立案
- [ ] Phase 5: `/pm-auto-dev 140` で TDD 実装
- [ ] Phase 6: 完了報告と最終検証
