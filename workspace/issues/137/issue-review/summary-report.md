# Issue #137 マルチステージレビュー完了報告

**Issue**: #137 feat(retrieval): institutional 多言語 embedding/reranker 5-variant 実機 A/B (#133 Phase B)
**完了日**: 2026-04-26
**worktree**: photon-mlx-feature-issue-137-institutional-ab

## 仮説検証結果（Phase 0.5）

13 件の事実関係主張をコードベース照合で検証。

| # | 仮説/主張 | 判定 |
|---|----------|------|
| 1 | PR #136 (`96e7b45`) で EmbeddingIndex.max_input_chars 設定可能化 | Confirmed |
| 2 | `configs/_experiments/institutional_V[0-4].yaml` 既存 | **Rejected** (gitignore のみ、未作成) |
| 3 | global default embedding = `intfloat/multilingual-e5-small` | **Rejected** (実体は `sentence-transformers/all-MiniLM-L6-v2`) |
| 4 | global default reranker = `cross-encoder/ms-marco-MiniLM-L-6-v2` | Confirmed |
| 5 | institutional_docs.yaml に max_input_chars フィールドあり | **Partially Confirmed** (fallback 2048 で動作) |
| 6 | invariant test 2 件 (skipif) 存在 | Confirmed |
| 7 | guard test (#132) で global 不変保護 | Confirmed (実体は #114) |
| 8 | `aggregate_institutional_baseline.py` 再利用可 | Confirmed |
| 9 | 3 scripts (ingest/build/eval) `--config`/`--repo-id` 対応 | Confirmed |
| 10 | workspace artifacts (work-plan / design) 存在 | **Unverifiable** (本 worktree 未取り込み) |
| 11 | V0 institutional NC ~11.21% | Confirmed (`reports/institutional_baseline_static.md`) |
| 12 | EmbeddingIndex は model 非依存で variant サポート | Confirmed |
| 13 | bge-m3 で 8192 char truncate 可能 | Confirmed |

## ステージ別結果

| Stage | レビュー種別 | レビュアー | 指摘数 | 反映数 | ステータス |
|-------|------------|----------|--------|--------|----------|
| 0.5 | 仮説検証 | claude-opus | 13 件中 4 件問題発見 | - | 完了 |
| 1 | 通常レビュー（1回目） | **claude-opus** | 11 (MF:3, SF:5, NH:3) | - | 完了 |
| 2 | 指摘事項反映（1回目・通常） | claude-sonnet | - | 11/11 | 完了 |
| 3 | 影響範囲レビュー（1回目） | **claude-opus** | 11 (MF:0, SF:6, NH:5) | - | 完了 |
| 4 | 指摘事項反映（1回目・影響範囲） | claude-sonnet | - | 10/11 (S3-011 確認のみ) | 完了 |
| 5 | 通常レビュー（2回目） | **codex** | 2 (MF:2, SF:0, NH:0) | - | 完了 |
| 6 | 指摘事項反映（2回目・通常） | codex | - | 2/2 | 完了 |
| 7 | 影響範囲レビュー（2回目） | **codex** | 2 (MF:1, SF:1, NH:0) | - | 完了 |
| 8 | 指摘事項反映（2回目・影響範囲） | codex | - | 2/2 | 完了 |

**合計**: 26 findings (6 Must Fix, 12 Should Fix, 8 Nice to Have) → 25 applied + 1 record-only

## 主要な改善ポイント

### Iteration 1 (claude-opus)

**Must Fix 3 件**:
1. S1-001: V0「現 default」が global default 事実と矛盾 → institutional/global 区別を明記
2. S1-002: variant configs「gitignore 済」表現が誤誘導 → 「Phase B で新規作成」に修正
3. S1-003: V4 max_input_chars=8192 の明示宣言が必要 → variant config + institutional_docs.yaml 両方に明記

**Should Fix 11 件**: 採用判定基準のタイブレーカー定義、受入条件の論理矛盾解消、想定 compute repo_id 列追加、invariant test 活性化のテスト名/定数名明記、リソース見積もり (ディスク/HF cache/RAM)、強制再 build 手順、SQLite lock 競合回避、device plumbing 不在、deployment.md 更新、PR 戦略 3 パターン

### Iteration 2 (codex クロスレビュー)

**Must Fix 3 件** (opus が見落とした重要バグ):
1. S5-001: `run_baseline_eval.py --repo-id` は index load 先を変えない → variant config の `repo.repo_id` 明示が必須
2. S5-002: `aggregate_institutional_baseline.py` は複数 predictions を合算する → variant ごとに個別集計が必要
3. S7-001: eval 実行コマンドに `--eval-set data/eval_sets/institutional_static_eval.jsonl` 明記なし → デフォルトで FastAPI 120Q が走る危険

**Should Fix 1 件**:
- S7-002: `load_config()` に `extends`/include 機能なし → variant config は full copy 必須

## 結論

Codex クロスレビューにより、claude-opus が見落とした実装上の致命的なバグ 3 件 (index load の不整合、aggregator の合算問題、eval set の取り違え) を補足できた。Issue 本文は Phase B 実機実行に必要な手順・前提・受入条件を網羅した状態に到達。

## 次のアクション

- [x] Phase 1: マルチステージIssueレビュー
- [ ] Phase 2: 設計方針書確認・作成 (`/design-policy 137`)
- [ ] Phase 3: マルチステージ設計レビュー
- [ ] Phase 4: 作業計画立案
- [ ] Phase 5: TDD自動開発
