# Issue #144 マルチステージレビュー完了報告

**対象 Issue**: test(retrieval): ruri-small-v2 (V2) build failure follow-up — manual spiece.model download workaround
**完了日時**: 2026-04-28
**実行者**: Claude (Stage 1-4) + Codex (Stage 5-8) クロスレビュー

---

## 仮説検証結果（Phase 0.5）

| # | 仮説/主張 | 判定 |
|---|----------|------|
| H1 | spiece.model cache 漏れ | Unverifiable |
| H2 | sentence-transformers 互換性問題 | Unverifiable |
| H3 | AutoTokenizer/AutoProcessor fallback 失敗 | Unverifiable |
| H4 | ruri-base も同じ問題 | Partially Confirmed |
| H5 | 環境のライブラリバージョン | Unverifiable |
| H6 | EmbeddingIndex は SentenceTransformer を直接呼ぶ | Confirmed |
| H7 | configs/institutional_docs.yaml は V4 採用 | Confirmed |
| H8 | 4 variant メトリクス確定 + V2 のみ補完 | Partially Confirmed |
| H9 | V4 採用判定は不変 | Confirmed (構造的推測) |

申し送り事項は Stage 1 / Stage 3 で全件カバー済み。

---

## ステージ別結果

| Stage | レビュー種別 | レビュアー | 指摘数 (M/S/N) | 反映数 | ステータス |
|-------|------------|----------|--------------|-------|----------|
| 1 | 通常レビュー（1回目） | claude-opus | 1 / 4 / 3 | - | 完了 |
| 2 | 指摘事項反映（1回目通常） | claude-sonnet | - | 8 (全件) | 完了 |
| 3 | 影響範囲レビュー（1回目） | claude-opus | 0 / 2 / 6 | - | 完了 |
| 4 | 指摘事項反映（1回目影響範囲） | claude-sonnet | - | 5 (3 件 deferred) | 完了 |
| 5 | 通常レビュー（2回目） | **codex** | 1 / 0 / 0 | - | 完了 (reviewer=codex 検証済) |
| 6 | 指摘事項反映（2回目通常） | codex | - | 1 | 完了 |
| 7 | 影響範囲レビュー（2回目） | **codex** | 0 / 3 / 0 | - | 完了 (reviewer=codex 検証済) |
| 8 | 指摘事項反映（2回目影響範囲） | codex | - | 3 | 完了 |

**集計**:
- Total findings: 16 件 (Must Fix 2 / Should Fix 9 / Nice to Have 9)
- Total applied: 17 件 (Stage 4 で deferred とした 3 件を除く全件)
- GitHub Issue 本文: Stage 2/4/6/8 で計 4 回更新済み

---

## reviewer 検証結果 (Issue #140 / S7-001 follow-up)

```
OK: workspace/issues/144/issue-review/stage5-review-result.json reviewer=codex
OK: workspace/issues/144/issue-review/stage7-review-result.json reviewer=codex
```

WARNING なし。Codex 担当 Stage 5/7 ともに `reviewer="codex"` 検証 PASS。

---

## Codex クロスレビューで発見された主要 finding

### Stage 5 (通常 2回目): Must Fix 1 件
- **S5-001**: Task 2 の ingest 手順が現行 CLI と不一致 (`--repo` 必須引数欠落、`--commit` sentinel 欠落、`configs/_experiments/` ディレクトリ未存在)。本ままでは V2 build 再実行手順が fail-fast で失敗する。

→ **Stage 6 で Issue 本文に反映済み**: `mkdir -p configs/_experiments` と `--repo /path/to/institutional_documents --commit 9e500539...` を Task 2 手順に追加。

### Stage 7 (影響範囲 2回目): Should Fix 3 件
- **S7-001**: Option B fallback の `pip install -e .` が現 repo で実行不能 (pyproject.toml/setup.py 不在)。
- **S7-002**: Option B venv `.venv-ruri-v2` が `.gitignore` 対象外で worktree 汚染リスク。
- **S7-003**: V2 採用低確率分岐に invariant test (`tests/test_pipeline_factory_yaml_invariants.py`) と `docs/deployment.md` の更新が含まれていない。

→ **Stage 8 で Issue 本文に反映済み**: Option B 手順を `pip install -r requirements.txt + version constraint` に修正、`.gitignore` 拡張方針を追加、低確率採用分岐に invariant test/deployment docs 更新を含めた。

---

## 主要な改善点 (Issue 本文の品質向上)

1. **V2 build 再実行手順の完全化**: configs/_experiments/institutional_V2.yaml 再生成、`scripts/ingest_repo.py --repo --commit --config --repo-id`、`scripts/build_indexes.py`、`scripts/run_baseline_eval.py` の 4 step を CLI 引数込みで明記 (S1-001 + S5-001)。
2. **V2 設定の固定パラメータ表**: chunker max_chars=800、reranker=ms-marco、max_input_chars=2048、batch_size=32 を表で固定 (S1-004)。
3. **実行順序とフォールバック分岐**: Option A → load 検証 → 成功/Option B/Option C の 5 分岐 (S1-003)。
4. **依存契約の明示**: requirements.txt に huggingface_hub 追加 (S3-001)、Option B は専用 venv で隔離 + `.gitignore` 拡張 (S7-002)。
5. **真因表現の慎重化**: 「## 真因」→「## 推定真因 (実測ベースの仮説)」(S1-008)。
6. **PR #142 / Issue #137 状態整合**: MERGED 済 / closed 済を本文に反映、本 Task 3 は新規 PR 起点と明記 (S3-002)。
7. **V2 採用低確率分岐の網羅化**: invariant test と deployment docs を更新対象に追加 (S7-003)。

---

## 次のアクション

- [x] Phase 1: マルチステージIssueレビュー — **完了**
- [ ] Phase 2: 設計方針書確認・作成 (`/design-policy 144`)
- [ ] Phase 3: マルチステージ設計レビュー (`/multi-stage-design-review 144`)
- [ ] Phase 4: 作業計画立案 (`/work-plan 144`)
- [ ] Phase 5: TDD 自動開発 (`/pm-auto-dev 144`)
- [ ] Phase 6: 完了報告

---

## 出力ファイル

```
workspace/issues/144/issue-review/
├── original-issue.json
├── hypothesis-verification.md
├── stage1-review-context.json / stage1-review-result.json (claude-opus)
├── stage2-updated-body.md / stage2-apply-result.json (claude-sonnet)
├── stage3-review-context.json / stage3-review-result.json (claude-opus)
├── stage4-updated-body.md / stage4-apply-result.json (claude-sonnet)
├── stage5-review-result.json (codex, reviewer=codex 検証 PASS)
├── stage6-prev-body.md / stage6-updated-body.md / stage6-apply-result.json (codex)
├── stage7-review-result.json (codex, reviewer=codex 検証 PASS)
├── stage8-prev-body.md / stage8-updated-body.md / stage8-apply-result.json (codex)
└── summary-report.md (本ファイル)
```
