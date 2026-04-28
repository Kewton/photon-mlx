# Issue #145 マルチステージレビュー完了報告

実施日: 2026-04-28
対象: Issue #145 「test(photon): real-weight integration test (split from #139, depends on #135)」
最終本文長: 17,777 文字 (起票時 5,193 → +12,584)

---

## 仮説検証結果（Phase 0.5）

| # | 仮説/主張 | 判定 |
|---|----------|------|
| H1 | `model.checkpoint_path` 機構は #135 で導入された | Confirmed |
| H2 | Issue 起票時 #135 は main 未マージ | Rejected (2026-04-28 時点で main マージ済 / commit 8c13517) |
| H3 | 現行 main に `checkpoint_path` / `load_checkpoint` 不在 | Rejected (line 256/365-417/626-633 で実装済) |
| H4 | `_build_photon_deps` 経由で load する候補方針 B が実現可能 | Confirmed |
| H5 | `pipeline.last_pruning_score_distribution.std()` で random-init 検出 | Rejected (該当 public API なし) |
| H6 | random-init silently 動作型事故を CI 検出可能 | Partially Confirmed |

---

## ステージ別結果

| Stage | レビュー種別 | レビュアー | Must Fix | Should Fix | Nice to Have | 反映 | ステータス |
|-------|------------|----------|---------|-----------|-------------|-----|----------|
| 0.5 | 仮説検証 | claude (主) | - | - | - | - | 完了 (6 仮説検証) |
| 1 | 通常レビュー (1回目) | claude-opus | 2 | 5 | 2 | - | 完了 |
| 2 | 指摘事項反映 (1回目通常) | sonnet | - | - | - | 9 件全反映 | 完了 |
| 3 | 影響範囲レビュー (1回目) | claude-opus | 2 | 6 | 2 | - | 完了 |
| 4 | 指摘事項反映 (1回目影響範囲) | sonnet | - | - | - | 9 件反映 / 1 件据置 | 完了 |
| 5 | 通常レビュー (2回目) | **codex** | 2 | 3 | 0 | - | 完了 (reviewer="codex" 検証済) |
| 6 | 指摘事項反映 (2回目通常) | codex | - | - | - | issue_updated=True | 完了 |
| 7 | 影響範囲レビュー (2回目) | **codex** | 1 | 4 | 0 | - | 完了 (reviewer="codex" 検証済) |
| 8 | 指摘事項反映 (2回目影響範囲) | codex | - | - | - | applied_findings 計上済 | 完了 |

**合計指摘**: Must Fix 7 / Should Fix 18 / Nice to Have 4

---

## reviewer フィールド検証 (Issue #140 / S7-001 follow-up)

| ファイル | reviewer | 期待値 | 結果 |
|---------|----------|-------|------|
| stage5-review-result.json | codex | codex | OK |
| stage7-review-result.json | codex | codex | OK |

WARNING なし。Codex 担当 Stage 5/7 が正しく Codex により実行されている。

---

## 主要な Issue 本文の変更

1. **依存ステータス更新** (S1-001): "#135/#139 共に main マージ済 (2026-04-28 時点)" を明記、線 256/365-417/626-633 の既存実装を参照
2. **架空 API 削除** (S1-002): `last_pruning_score_distribution.std()` を削除し、実在 API 4 案 (a) integrity hash / (b) weight identity / (c) PHOTON_ALLOW_RANDOM_INIT bypass / (d) `_score_prune_candidates` private 呼出 に書き換え
3. **API 名修正** (S3-001): `_load_checkpoint_into_model` → `_load_photon_checkpoint` (実体名)
4. **test 配置詳細化** (S3-002 + S7-002): `tests/__init__.py` と `tests/integration/conftest.py` の追加要否を作業項目化
5. **CI/CD 戦略確定** (S7-001): `weekly_eval.yml` への integration step 追加を scope 内必須化、`continue-on-error: true` 禁止で fail-fast 検出を保証
6. **MLX nondeterminism 対策** (S3-005): seed 固定、deterministic flag を test 側で明示
7. **PHOTON_CHECKPOINT_ROOT test 設定** (S3-007): tmp_path 経由 + env var override の両ルート定義

---

## 次のアクション

- [ ] Phase 2: 設計方針書作成 (`/design-policy 145`)
- [ ] Phase 3: マルチステージ設計レビュー (`/multi-stage-design-review 145`)
- [ ] Phase 4: 作業計画立案 (`/work-plan 145`)
- [ ] Phase 5: TDD自動開発 (`/pm-auto-dev 145`)
