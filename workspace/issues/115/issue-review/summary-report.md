# Issue #115 マルチステージレビュー完了報告

実行日: 2026-04-26 〜 2026-04-27
対象: feat(app): Streamlit wizard に制度文書 domain template と日本語 prompt 調整

## 仮説検証結果（Phase 0.5）

| # | 仮説/主張 | 判定 |
|---|----------|------|
| 1 | `app/components/wizard.py` の `_WIZARD_BASE_PROFILES` に既存 profile が定義されている | Confirmed (ただし定数名は実コードと乖離) |
| 2 | `app/photon_app.py` の wizard UI で profile 選択肢が表示されている | Confirmed |
| 3 | `baseline_reporag/generation/prompt.py` に日本語判定ロジックが既にある | Partially Confirmed (LLM 任せで実装側に判定機能なし) |
| 4 | `configs/institutional_docs.yaml` が #112 で作成済 | Confirmed |
| 5 | bge-reranker-v2-m3 が #114 の結果として現在の reranker config に反映 | **Rejected** (#114 PR は guard test 追加のみ、reranker は据え置き) |
| 6 | wizard.py / prompt.py 関連テストが存在 | Confirmed |

## ステージ別結果

| Stage | レビュー種別 | レビュアー | 指摘数 (Must/Should/Nice) | 対応数 | ステータス |
|-------|------------|----------|---|-------|----------|
| 1 | 通常レビュー（1回目） | opus | 8 (3/4/1) | - | 完了 |
| 2 | 指摘事項反映（1回目） | sonnet | - | 8 | 完了 |
| 3 | 影響範囲レビュー（1回目） | opus | 8 (4/3/1) | - | 完了 |
| 4 | 指摘事項反映（1回目） | sonnet | - | 8 | 完了 |
| 5 | 通常レビュー（2回目） | **codex (gpt-5.5 high)** | 4 (1/3/0) | - | 完了 |
| 6 | 指摘事項反映（2回目） | codex | - | 4 | 完了 |
| 7 | 影響範囲レビュー（2回目） | **codex (gpt-5.5 high)** | 3 (1/2/0) | - | 完了 |
| 8 | 指摘事項反映（2回目） | codex | - | 3 | 完了 |

**合計**: 23 件指摘 (9 Must Fix / 12 Should Fix / 2 Nice to Have) → 23 件全反映

## Issue 本文の変遷

- 元 Issue: 1789 chars
- Stage 2 反映後: 6562 chars
- Stage 4 反映後: 12094 chars
- Stage 6 反映後: ~13000 chars
- 最終 (Stage 8 反映後): **17196 chars**

## 重要な改善点ハイライト

1. **#114 結論誤認の修正** (S1-001): bge-reranker-v2-m3 の固有名詞を削除し、reranker/embedding 採用判定は #133/#137 の institutional A/B 結果に委ねる方針に変更
2. **`_WIZARD_BASE_PROFILES` 不在問題の解決** (S1-002): 実コード位置 (`app/photon_app.py:1234-1238` ハードコードリスト) を明記
3. **detect_language 仕様の完全明文化** (S1-003 + S5-003 + S7-002): codepoint 範囲、ja 30%/en 50% 閾値、edge case (0文字 / 非空白0 / 空白のみ / 絵文字) を含む完全仕様
4. **dead config key の除去** (S3-001): `ingestion.chunker_type: markdown` は実コードに存在しないため merge から削除
5. **embedding index 不整合リスクの排除** (S3-002): `_DOMAIN_TEMPLATES` 適用は base=`institutional_docs` 限定
6. **build_messages() シグネチャ維持方針** (S3-004): callsite 不変で内部 detect、photon_pipeline.py 含む既存 callsite/test に影響なし
7. **#137 merge-order guard 追加** (S7-001): #137 が先 merge された場合は bge-m3 / bge-reranker-v2-m3 / batch_size=32 / max_input_chars=8192 を反映する条件付き仕様
8. **PHOTON follow-up 経路のテスト追加** (S7-003): `test_photon_pipeline.py` を影響ファイルに追加

## 次のアクション

- [x] Issue 本文の最終確認
- [ ] `/design-policy 115` で設計方針策定
- [ ] `/multi-stage-design-review 115` で設計レビュー
- [ ] `/work-plan 115` で作業計画立案
- [ ] `/pm-auto-dev 115` で TDD 実装

## 出力ファイル一覧

- `original-issue.json` (元 Issue snapshot)
- `hypothesis-verification.md` (Phase 0.5)
- `stage1-review-context.json` / `stage1-review-result.json`
- `stage2-apply-result.json`
- `stage3-issue-snapshot.json` / `stage3-review-result.json`
- `stage4-apply-result.json`
- `stage5-issue-snapshot.md` / `stage5-review-result.json`
- `stage6-apply-result.json` / `stage6-issue-body.md`
- `stage7-issue-snapshot.md` / `stage7-review-result.json`
- `stage8-apply-result.json` / `stage8-issue-body.md`
- `summary-report.md` (本ファイル)

## レビュー手法

- 1回目イテレーション (Stage 1-4): Claude Opus 4.7 をサブエージェントとして利用
- 2回目イテレーション (Stage 5-8): **Codex (gpt-5.5 high)** に commandmatedev send 経由で委譲し、異モデルクロスレビュー実施
- 全 Codex 結果ファイルの `reviewer` フィールドが `"codex"` であることを確認済
