# Issue #144 マルチステージ設計レビュー完了報告

**対象**: workspace/design/issue-144-ruri-v2-build-design-policy.md
**完了日時**: 2026-04-28
**実行者**: Claude (Stage 1-2) + Codex (Stage 3-4) クロスレビュー

---

## ステージ別結果

| Stage | レビュー種別 | レビュアー | 指摘数 (M/S/N) | 反映数 | ステータス |
|-------|------------|----------|--------------|-------|----------|
| 1 | 通常レビュー (設計原則) | claude-opus | 2 / 5 / 3 | 7 (Nice to Have 3 件 deferred) | 完了 |
| 2 | 整合性レビュー | claude-opus | 0 / 3 / 2 | 3 (Nice to Have 2 件 deferred) | 完了 |
| 3 | 影響分析レビュー | **codex** | 0 / 3 / 0 | 3 | 完了 (reviewer=codex 検証済) |
| 4 | セキュリティレビュー | **codex** | 0 / 3 / 0 | 3 | 完了 (reviewer=codex 検証済) |

**集計**:
- Total findings: 18 件 (Must Fix 2 / Should Fix 14 / Nice to Have 5)
- Total applied: 16 件 (Stage 1/2 で deferred とした Nice to Have 5 件を除く全件)
- 設計方針書: 4 stage で計 4 回更新済み

---

## reviewer 検証結果 (Issue #140 / S7-001 follow-up)

```
OK: workspace/issues/144/multi-stage-design-review/stage3-review-result.json reviewer=codex
OK: workspace/issues/144/multi-stage-design-review/stage4-review-result.json reviewer=codex
```

WARNING なし。Codex 担当 Stage 3/4 ともに `reviewer="codex"` 検証 PASS。

---

## Codex クロスレビューで発見された主要 finding

### Stage 3 (影響分析): Should Fix 3 件

- **DR3-001**: V2 build/eval の生成 artifact が `embedding/` と predictions JSONL だけに過小記載 → `data/indexes/institutional_documents_V2/` ディレクトリ単位 (chunks.db、lexical.pkl、embedding 内訳) と log artifact (`logs/bench_variant_*.jsonl`、`logs/sessions/session_eval-*.json`) を追加。
- **DR3-002**: weekly_eval は workflow logic 不変だが `requirements.txt` 追加で install step に限定波及 → §5.3 の表現を「workflow file / eval 対象は不変。install step は huggingface_hub 追加後の依存解決を通る」に修正。
- **DR3-003**: Option B の `.gitignore` 条件付き変更が §5.1 から漏れ → §5.1 と §9 Task 0 に Option B 採用時の `.gitignore .venv-ruri-*/` 追加を追記。

### Stage 4 (セキュリティ): Should Fix 3 件

- **DR4-001**: gitignored variant config の typo / tampering / 過大値検証 gate 不足 → §3.4 に Step 2.5 variant config security validation を追加 (固定値 assert、`repo_id`、sentinel commit、max_input_chars 等)。
- **DR4-002**: HF cache full path と raw traceback を Issue コメント貼付する運用が log hygiene と衝突 → 成功 log は filename + file size のみ、失敗時は exception type + sanitized summary のみ記録する方針に変更 (HF token/PAT/full path/raw traceback を Issue コメントに貼らない)。
- **DR4-003**: `huggingface_hub` direct dep 追加時の resolver 監査不足 → Task 0 受入条件に `pip check` + `pip show huggingface_hub transformers sentence-transformers` の実行・resolved version 記録を追加。

---

## Stage 1-2 (Claude opus) の主要 finding 概要

### Stage 1 (設計原則 SOLID/KISS/YAGNI/DRY): Must Fix 2、Should Fix 5

- **DR1-001 (SRP)**: `fetch_ruri_files()` / `verify_load()` の責務記述追加。
- **DR1-003 (separation)**: §3.4 Step 2 「手で編集」を yq + python -c + チェックリストの 3 段階に置換。
- **DR1-002/004/005/006/007** (Should Fix): fail-fast 連動表記、invariant test 不破壊根拠、既存 venv 前提、相互参照、冪等性関数別分割を反映。

### Stage 2 (整合性): Should Fix 3

- **DR2-001/002/003**: Issue 本文との 1:1 対応 (normalize 行追加、完了条件 2 項目追加、`(推定)` 削除と出典 line 追記)。

---

## 主要な改善点 (設計方針書の品質向上)

1. **Function-level 設計の明示**: §3.1 の `fetch_ruri_files()` / `verify_load()` に責務・成功条件・失敗時動作・出力形式の 4 項目を追加。
2. **再現性のある config 生成手順**: §3.4 Step 2 を yq → python -c → 手編集チェックリストの 3 段階に格上げ。
3. **Variant config security validation gate**: §3.4 Step 2.5 で固定値 assert (DR4-001)。
4. **Log hygiene**: HF cache path / raw traceback / token を Issue コメント貼付しない方針を §3.1 / §6 / §8 / §9 に明文化 (DR4-002)。
5. **Resolver 監査**: `pip check` + `pip show` を Task 0 受入条件に追加 (DR4-003)。
6. **影響範囲の網羅化**: V2 build/eval が生成する artifact 一式 (chunks.db、lexical.pkl、embedding 内訳、bench_variant log、session log) を §5.1 / §7 / §9 に統一。
7. **CI 影響の正確化**: weekly_eval は workflow logic 不変、install step に限定波及 (DR3-002)。
8. **Option B 条件付き変更**: `.gitignore` 拡張を §5.1 / §9 Task 0 に明記 (DR3-003)。

---

## 次のアクション

- [x] Phase 3: マルチステージ設計レビュー — **完了**
- [ ] Phase 4: 作業計画立案 (`/work-plan 144`)
- [ ] Phase 5: TDD 自動開発 (`/pm-auto-dev 144`)
- [ ] Phase 6: 完了報告

---

## 出力ファイル

```
workspace/issues/144/multi-stage-design-review/
├── stage1-review-result.json (claude-opus)
├── stage1-apply-result.json (claude-opus)
├── stage2-review-result.json (claude-opus)
├── stage2-apply-result.json (claude-opus)
├── stage3-review-result.json (codex, reviewer=codex 検証 PASS)
├── stage3-apply-result.json (codex)
├── stage4-review-result.json (codex, reviewer=codex 検証 PASS)
├── stage4-apply-result.json (codex)
└── summary-report.md (本ファイル)

workspace/design/
└── issue-144-ruri-v2-build-design-policy.md (Stage 1-4 全件反映済み)
```
