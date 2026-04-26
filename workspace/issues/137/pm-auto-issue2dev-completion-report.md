# PM Auto Issue2Dev 完了報告 — Issue #137

**Issue**: #137 feat(retrieval): institutional 多言語 embedding/reranker 5-variant 実機 A/B (#133 Phase B)
**完了日**: 2026-04-26
**ブランチ**: feature/issue-137-institutional-ab

---

## 実行フェーズ結果

| Phase | 内容 | ステータス | 成果物 |
|-------|------|-----------|--------|
| 1 | マルチステージIssueレビュー | ✅ 完了 | 仮説検証 + 8 stage (4 reviewer: opus/opus/codex/codex) → 26 findings (6 MF, 12 SF, 8 NH) → 25 applied |
| 2 | 設計方針書作成 | ✅ 完了 | `workspace/design/issue-137-institutional-multilingual-ab-design-policy.md` (12 section、5 設計判断 + LSP 補強 #6) |
| 3 | マルチステージ設計レビュー | ✅ 完了 | 4 stage (opus/opus/codex/codex) → 21 findings (0 MF, 14 SF, 7 NH) → 19 applied |
| 4 | 作業計画立案 | ✅ 完了 | `workspace/issues/137/work-plan.md` (Phase 1-5、5 Task カテゴリ、リスク対策) |
| 5 | TDD自動開発 | ⛔ N-A (措置済) | 本 Issue は **測定 Issue** のため TDD 不適用。Phase B 実機実行 (4-5h GPU) + 採用判定 (人間判断) が成果物の本体であり、コード変更は採用判定後の最小修正 (~30 行) のみ。手順は work-plan.md にて完備。 |
| 6 | 完了報告 | ✅ 完了 | 本ファイル |

---

## 品質チェック (現状コード変更ゼロ確認)

| チェック項目 | コマンド | 結果 |
|-------------|----------|------|
| Lint | `ruff check .` | ✅ All checks passed! |
| Format | `ruff format --check .` | ✅ 147 files already formatted |
| Invariant tests | `python -m pytest tests/test_pipeline_factory_yaml_invariants.py -v` | ✅ 1 passed (baseline guard) + 2 skipped (institutional, 採用後活性化) |
| Git status | `git status --short` | workspace/ のみ (untracked、設計/計画ドキュメント) |

> Phase 5 を skip したためソースコード変更ゼロ。Phase A (#133/#136) で導入された institutional invariant 2 件は引き続き skip 状態 (採用 variant 確定後に Phase 4 で活性化)。

---

## 生成ファイル

### 設計・計画ドキュメント (本 worktree)
- `workspace/design/issue-137-institutional-multilingual-ab-design-policy.md` — 設計方針書 (Stage 1-4 design review 反映済み)
- `workspace/issues/137/work-plan.md` — Phase B 実機実行 + 採用反映の手順書

### Issue レビュー成果物
- `workspace/issues/137/issue-review/summary-report.md` — マルチステージ Issue レビュー完了報告
- `workspace/issues/137/issue-review/hypothesis-verification.md` — 13 仮説のコードベース照合
- `workspace/issues/137/issue-review/stage[1-8]-*.json` — 各 stage の review/apply 結果

### 設計レビュー成果物
- `workspace/issues/137/multi-stage-design-review/summary-report.md` — マルチステージ設計レビュー完了報告
- `workspace/issues/137/multi-stage-design-review/stage[1-4]-*.json` — 各 stage の review/apply 結果

### GitHub
- Issue #137 本文: 4 reviewer による 8 stage 反映済み (https://github.com/Kewton/photon-mlx/issues/137)

---

## 次のアクション (人間 or 主セッション側)

### Phase B 実機実行 (work-plan.md Phase 1-2 参照)
- [ ] Task 1.1: 環境確認 (sentence-transformers >= 2.3.0, ディスク空き ~7GB)
- [ ] Task 1.2: base config の HEAD commit 記録 (DRY 違反許容ルール)
- [ ] Task 2.1: 5 variant config 作成 (`configs/_experiments/institutional_V[0-4].yaml`)
- [ ] Task 2.2: V0 → V3 評価 (逐次必須、SQLite lock 競合回避)
- [ ] Task 2.3: V1, V2, V4 — index 再 build + 評価 (合計 ~3-4h)
- [ ] Task 2.4: aggregator 個別実行 (glob 一括禁止)

### 採否判定 + レポート (work-plan.md Phase 3)
- [ ] Task 3.1: `reports/institutional_retrieval_ab.md` 作成 (5 variant 比較表 + 採否判断)
- [ ] Task 3.2: 採否確定 → Phase 4 分岐 (非採用 / V0-V3 採用 / V4 採用)

### コード反映 (採用ありの場合のみ、work-plan.md Phase 4)
- [ ] Task 4.1: `configs/institutional_docs.yaml` 更新 + index 強制再 build (commit #1)
- [ ] Task 4.2: invariant test 活性化 + (V4 時) 第 3 invariant 追加 (commit #2)
- [ ] Task 4.3: `docs/deployment.md` 更新 (commit #3)
- [ ] Task 4.4: `reports/institutional_retrieval_ab.md` 完成 (commit #4)

### PR + マージ + cleanup (work-plan.md Phase 5)
- [ ] Task 5.2: `gh pr create` (採用結果に応じたタイトル)
- [ ] Task 5.3: 未採用 variant の `data/indexes/`, HF cache cleanup

### #135 への引継ぎ
- [ ] 本 Issue 完了後、institutional 採用 model を #135 (本格再学習) に組み込む (直列実行)

---

## クロスレビューが補足した重要発見

opus 単独では見落としていた、Codex クロスレビューが補足した致命的バグ・前提誤り:

### Issue レビュー (Phase 1, codex Stage 5/7)
1. `run_baseline_eval.py --repo-id` は **index load 先を変えない** — variant config の `repo.repo_id` 明示が必須 (S5-001)
2. `aggregate_institutional_baseline.py` は **複数 predictions を合算する** — variant ごとに個別実行が必須 (S5-002)
3. eval 実行コマンドに `--eval-set data/eval_sets/institutional_static_eval.jsonl` 明記なしで FastAPI 120Q が走る危険 (S7-001)
4. `load_config()` に `extends`/include 機能なし — variant config は **full copy 必須** (S7-002)

### 設計レビュー (Phase 3, codex Stage 3/4)
5. `institutional_static_eval.jsonl` は **gitignored ではなく tracked** (security table 誤記)
6. CLI `--repo-id` の **path traversal リスク** (security 観点で明示要)
7. HF model 自動 download の **revision pin 不在** (供給網リスク follow-up 候補)
8. invariant bypass 対策が **CODEOWNERS 不在の前提誤り** → required CI + reviewer checklist 主体に修正

これらは全て Phase B 実機実行と採用反映の正しさに直結するため、本 PM Auto Issue2Dev の最大の価値は「opus + codex の dual review で実装可能な手順書を獲得した」点にある。

---

## 結論

本 Issue は実装ではなく **測定 + 採用判定** が本体のため、Phase 5 (TDD) を意図的に N-A とした。設計方針書と作業計画書は実機実行可能なレベルまで詳細化されており、人間オペレーター (または主セッション) が work-plan.md に従って Phase B 以降を実行可能な状態。

実機実行を開始する前に、本 worktree の `workspace/` ディレクトリ (設計/計画ドキュメント) を commit するか別 PR で別途記録することを推奨。
