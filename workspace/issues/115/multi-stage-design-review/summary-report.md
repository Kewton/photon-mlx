# Issue #115 マルチステージ設計レビュー完了報告

実行日: 2026-04-27
対象: 設計方針書 `workspace/design/issue-115-wizard-jp-prompt-design-policy.md`

## ステージ別結果

| Stage | レビュー種別 | レビュアー | 指摘数 (Must/Should/Nice) | 対応数 | ステータス |
|-------|------------|----------|---|-------|----------|
| 1 | 通常レビュー (設計原則 SOLID/KISS/YAGNI/DRY) | opus | 6 (0/3/3) | 6 | 完了 |
| 2 | 整合性レビュー | opus | 9 (4/3/2) | 9 | 完了 |
| 3 | 影響分析レビュー | **codex (gpt-5.5 high)** | 3 (1/2/0) | 3 | 完了 |
| 4 | セキュリティレビュー | **codex (gpt-5.5 high)** | 0 (0/0/0) | 0 | 完了 (問題なし) |

**合計**: 18 件指摘 (5 Must Fix / 8 Should Fix / 5 Nice to Have) → 18 件全反映

## 重要な改善点ハイライト

1. **DR1-001 (DRY)**: `_DOMAIN_TEMPLATES` と `configs/institutional_docs.yaml` の整合 CI guard test を完了条件 §10 必須項目 (#13) に格上げ
2. **DR1-002 (KISS)**: `detect_language()` を 1 ループ集計化、ja/en 分母非対称の理由をコメント化
3. **DR1-003 (YAGNI)**: pre/post #137 確定運用ルール (PR 着手時 git diff で確定→不採用側削除) を §3-1 に明記
4. **DR1-005 (SRP)**: `_resolve_system_prompt(question)` helper 切り出しを §3-2 に追記、test pyramid 明確化
5. **DR2-001/DR2-007**: §3-1 literal を pre-#137 確定形に整理、Issue 本文「並列開発」節と整合
6. **DR2-002**: CI guard test 配置先を `tests/test_photon_app_components.py` に確定、yaml 直接 parse 方式を明記
7. **DR2-003**: Issue 受入条件を 13→14 項目に拡張 (CI guard 追加)、設計書 §10 完了条件本文も 14 項目に修正
8. **DR2-004**: selectbox label を `'Base profile'`→`'Config template'` に修正 (実コード一致)
9. **DR2-005**: §3-3 test 一覧に `TestResolveSystemPrompt` (3 tests) 追加
10. **DR2-006**: Issue body L11 の `_DOMAIN_TEMPLATES` 型注釈を tuple-key 形式に修正
11. **DR3-001**: Streamlit Project `repo_id` と generated YAML `repo.repo_id` 不一致を保存前 validation で拒否、テスト・リスク・完了条件追加
12. **DR3-002**: `_resolve_system_prompt()` 実装例を §3-2 に明示、build_messages サンプルを helper 呼び出しに修正
13. **DR3-003**: #135/#139 が `photon_pipeline.py` / `test_photon_pipeline.py` を触る rebase 影響をリスク表に追記

## セキュリティ判定 (Stage 4)

Codex セキュリティレビューで **findings 0 件**。以下が全て許容:
- `eval()` / `exec()` 不使用
- subprocess は argv list + `shell=False` 維持
- `yaml.safe_load` + `_assert_safe_yaml` allowlist 前提
- `_DOMAIN_TEMPLATES` の path key はコード定数の `tuple[str, ...]`
- `_JP_INSTITUTIONAL_HINT` 追加で `flatten_messages_for_plain_lm()` の DR-62-003 role-boundary 防御に影響なし
- `[C:N]` citation 形式と `ABSTAIN_MARKER` 不変、`apply_citation_postprocess()` 互換維持
- `detect_language()` は question のみ単一 pass、DoS 面の追加コスト限定的

## 設計書の変遷

- 初版: 約 11000 chars
- Stage 1+2 反映後: 約 16000 chars
- Stage 3+4 反映後: 約 22000 chars (DR3-001 の repo_id validation 設計追加で増加)

## 次のアクション

- [x] 設計方針書の最終確認
- [ ] `/work-plan 115` で作業計画立案
- [ ] `/pm-auto-dev 115` で TDD 実装

## レビュー手法

- Stage 1-2 (設計原則・整合性): Claude Opus 4.7 サブエージェント
- Stage 3-4 (影響範囲・セキュリティ): **Codex (gpt-5.5 high)** に commandmatedev 経由で委譲、異モデルクロスレビュー実施
- 全 Codex 結果ファイルの `reviewer` フィールドが `"codex"` であることを確認済
