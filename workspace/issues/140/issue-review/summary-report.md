# Issue #140 マルチステージレビュー完了報告

## 概要

Issue #140 ("process(review): Codex multi-stage design review 必須化 + scaffolding 命名禁止 checklist (S7-001 follow-up)") に対し、`/multi-stage-issue-review 140` を 8 ステージ完走 (Phase 0.5 仮説検証 + Stage 1-8) しました。1回目イテレーション (opus) と 2回目イテレーション (Codex) でクロスレビューを行い、合計 **29 finding** (Must Fix 9 / Should Fix 8 / Nice to Have 5 + Codex 7) を抽出・全件 Issue 本文に反映済みです。

## 仮説検証結果（Phase 0.5）

| # | 仮説/主張 | 判定 |
|---|----------|------|
| H1 | S7-001 は「設計レビュー Stage 7 (Codex クロスレビュー)」で発見 | **Rejected** (design review は 4 stages、issue review の Stage 7 が正しい) |
| H2 | `/multi-stage-design-review` は Stage 7-8 が Codex | **Rejected** (Stage 3-4 が Codex) |
| H3 | Codex stage が optional のため bug 混入 | **Partially Rejected** (構造課題は Confirmed、Stage 番号は誤り) |
| H4 | scaffolding 命名 (`_StubTokenizer`) が production に残存 | **Confirmed** (`baseline_reporag/photon_pipeline.py:451`) |
| H5 | code review checklist で禁止項目化されていない | **Confirmed** |
| H6 | embedding access path = `model.embed_tokens.weight` | **Rejected** (`model.token_embed.weight`) |
| H7 | `docs/code_review_checklist.md` は未作成 | **Confirmed** |
| H8 | #138 は先行解消すべき | **Confirmed** (CLOSED 済み) |
| H9 | #139 と並列可能 | **Confirmed** (OPEN) |

## ステージ別結果

| Stage | レビュー種別 | レビュアー | 指摘数 | 内訳 (MF/SF/NTH) | 対応数 | ステータス |
|-------|------------|----------|-------|-----------------|-------|----------|
| 0.5 | 仮説検証 | opus | 9 件 | - | - | 完了 |
| 1 | 通常レビュー（1回目） | **opus** | 11 | 5/4/2 | 11 | 完了 |
| 2 | 指摘事項反映（1回目） | sonnet | - | - | 11 | 完了 |
| 3 | 影響範囲レビュー（1回目） | **opus** | 11 | 4/4/3 | 11 | 完了 |
| 4 | 指摘事項反映（1回目） | sonnet | - | - | 11 | 完了 |
| 5 | 通常レビュー（2回目） | **codex** | 4 | 2/2/0 | 4 | 完了 |
| 6 | 指摘事項反映（2回目） | codex | - | - | 4 | 完了 |
| 7 | 影響範囲レビュー（2回目） | **codex** | 3 | 1/2/0 | 3 | 完了 |
| 8 | 指摘事項反映（2回目） | codex | - | - | 3 | 完了 |

**合計**: 29 finding (Must Fix 12 / Should Fix 12 / Nice to Have 5)、全件反映。

## reviewer フィールド検証

- `stage5-review-result.json`: `reviewer = "codex"` ✓
- `stage7-review-result.json`: `reviewer = "codex"` ✓
- いずれも Claude による不正な上書きは無し

## 主要な反映内容 (Issue 本文への影響)

### イテレーション 1 (opus)

**Must Fix (9 件)**:
- S1-001: Stage 番号誤認 (multi-stage-design-review は 4 stages、Codex は Stage 3-4 / multi-stage-issue-review は 8 stages、Codex は Stage 5-8) を全面修正
- S1-002: `model.embed_tokens.weight` → `model.token_embed.weight` に修正
- S1-003: `.claude/agents/` → `.claude/commands/` に修正
- S1-004: norm 閾値の校正方針を 4 項目で明確化、config 化要件を追加
- S1-005: Phase 完了判定基準を `>= 0` から「結果ファイル存在 + reviewer=codex + Stage 別 finding 数表」に再定義
- S3-001: `_check_weight_initialization` に try/except + isinstance ガード追加 (MagicMock model 対策)
- S3-002: Task 3 のスコープを docs 整備のみに縮小、test 実装は #139 に委譲
- S3-003: 新規 Task 5 (CLAUDE.md スラッシュコマンド表更新) を追加
- S3-004: Stage 6/8 が apply-only である旨を明示、completion report の集計対象を review stage に限定

**Should Fix (8 件)**: scaffolding 例外定義、test 方針、ロールバック方針、PhotonInference 呼び出し点の WARNING 副作用、CI grep 自動化レイヤ、test 配置先 `tests/test_skill_descriptions.py`、閾値 field 設置先 3 候補、Phase 1 適用範囲の明示

**Nice to Have (5 件)**: finding ID 命名体系の脚注、受入条件の細分化、ロールバック手順の field 設置先依存、セキュリティ方針 (ログに weight tensor 値を出さない)

### イテレーション 2 (codex)

**Must Fix (3 件)**:
- S5-001: `multi-stage-issue-review` の Stage 5-8 自動スキップ判定が Codex 必須化と矛盾 → 廃止方針を追記
- S5-002: "strict 化" と "WARNING のみ" の failure policy を統一 (構造化検証は必ず行うが初期実装は WARNING のみ)
- S7-001: Task 2 の reviewer 検証 test fixture の実装形態 (Markdown snippet vs helper script) を明確化

**Should Fix (4 件)**:
- S5-003: `configs/photon_*.yaml` 個数を 7 → 5 (現存ファイル名を明記)
- S5-004: `baseline_reporag/tests/conftest.py` を「新規、必要時」と明記
- S7-002: auto-skip 廃止の波及範囲 (summary report テンプレート、完了条件) を全箇所更新
- S7-003: WorkingMemoryConfig 案採用時の `_UNSET`/`None`/明示 config の閾値解決規則と baseline YAML roundtrip test 要件

## 最終 Issue 本文

- 場所: GitHub Issue #140 (https://github.com/Kewton/photon-mlx/issues/140)
- 最終更新: 2026-04-26 13:43:30 UTC
- 長さ: 15051 chars / 21561 bytes
- 構成: 背景 / ゴール 5 項目 / Task 1-5 / 受入条件 / 後方互換性ロールバック / 並列性依存関係 / 影響ファイル / 関連
- ローカルバックアップ: `workspace/issues/140/issue-review/stage8-issue-body.md`

## 次のアクション

- [x] Phase 1: マルチステージIssueレビュー完了
- [ ] Phase 2: `/design-policy 140` で設計方針書作成
- [ ] Phase 3: `/multi-stage-design-review 140` で設計レビュー
- [ ] Phase 4: `/work-plan 140` で作業計画立案
- [ ] Phase 5: `/pm-auto-dev 140` で TDD 実装
- [ ] Phase 6: 完了報告と最終検証
