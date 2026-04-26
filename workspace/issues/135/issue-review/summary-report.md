# Issue #135 マルチステージレビュー完了報告

**Issue**: feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合 (#117 conditional, #113 follow-up)
**完了日**: 2026-04-26
**実行コマンド**: `/multi-stage-issue-review 135`

## 仮説検証結果（Phase 0.5）

12 件の事実主張を検証し、10 件 Confirmed / 1 件 Unverifiable / 1 件 (mulmoclaude 600 step checkpoint の物理存在) は実装着手前に補足要。

| # | 主張 | 判定 |
|---|------|------|
| 1 | #113 PR #134 (e281660) merged | ✅ Confirmed |
| 2 | #113 実測値 (NC / latency) | ✅ Confirmed |
| 3 | 設計 §9 仮説 C (Turn 5-6 NC > 6%) | ✅ Confirmed |
| 4 | #117 Epic Phase 2 conditional 表 | ✅ Confirmed |
| 5 | mulmoclaude 600 step 完了 | 🔲 Unverifiable |
| 6 | institutional_documents 4228 md | ✅ Confirmed |
| 7 | institutional_multi_turn_eval.jsonl (30 sessions) | ✅ Confirmed |
| 8 | generate_institutional_eval_set.py | ✅ Confirmed |
| 9 | run_multi_turn_eval.py | ✅ Confirmed |
| 10 | configs/institutional_docs_photon.yaml | ✅ Confirmed |
| 11 | trainer.py 主要 API (resume_from / cosine schedule / gradient accumulation 等) | ✅ Confirmed |
| 12 | data.py で JP/EN mix 実現可能 | ✅ Confirmed |

## ステージ別結果

| Stage | レビュー種別 | レビュアー | Must / Should / Nice | 反映 | ステータス |
|-------|------------|-----------|---------------------|------|-----------|
| 1 | 通常レビュー（1回目） | Claude opus | 3 / 8 / 3 = 14 | - | 完了 |
| 2 | 指摘事項反映（1回目） | sonnet | - | 14/14 | 完了 |
| 3 | 影響範囲レビュー（1回目） | Claude opus | 3 / 7 / 2 = 12 | - | 完了 |
| 4 | 指摘事項反映（1回目） | sonnet | - | 12/12 | 完了 |
| 5 | 通常レビュー（2回目） | Codex (gpt-5.5 high) | 3 / 3 / 0 = 6 | - | 完了 |
| 6 | 指摘事項反映（2回目） | Codex | - | 6/6 | 完了 |
| 7 | 影響範囲レビュー（2回目） | Codex (gpt-5.5 high) | 2 / 4 / 0 = 6 | - | 完了 |
| 8 | 指摘事項反映（2回目） | Codex | - | 6/6 | 完了 |

**累計指摘**: Must Fix 11 / Should Fix 22 / Nice to Have 5 = **38 件**、すべて Issue 本文に反映済。

## 主要改善点

### 1 回目イテレーション (Claude opus)

- **C-3 受入条件**: 「FastAPI 系 retrieval 性能 regression 5pp 以内」を `MT NC 6.7% +5pp 以内 (= 11.7% 以下)` に測定可能化。
- **B-1 ハイパーパラメータ**: `lr=3e-5, micro_batch=2, grad_accum=16, warmup_ratio=0.0, min_lr=3e-6` を初期値として固定。
- **A-1 学習 corpus**: sessions 数 ≥ 2,000、JP token 比 ≥ 50%、eval set 重複禁止、`scripts/generate_institutional_training_corpus.py` 新規追加を明記。
- **C-2 採用基準**: 「最低条件 (Turn 5-6 NC < 6% AND latency ≤ 13.6s) を満たす checkpoint のうち NC 最小」に書き換え。
- **A-0 機密保護**: `.gitignore` に `data/training/` 追加を最初の commit で実施。
- **D ロールアウト**: `configs/institutional_docs_photon_retrain.yaml` 新規分離、CLAUDE.md / gate2_v5 / roadmap.md 更新を明記。

### 2 回目イテレーション (Codex 独自発見)

- **S5-001 LR 起点**: `warmup_ratio=0.0` で実装は `lr=3e-5 → min_lr=3e-6` 起点 (本文の「min_lr→max_lr」表現を修正)。
- **S5-002 累計 step**: `resume_from` 後の `max_steps` は累計 step として扱われるため、checkpoint 名・eval 対象の表現を明確化。
- **S5-003 generate script**: 入力 source を `institutional_documents` raw corpus に修正、CLI signature を明示。
- **S5-004 token 計算**: `micro_batch × grad_accum × context_length = 65,536 token/step` で 20K step ≈ 1.31B token。compute コスト再見積もり。
- **S5-006 turn separator**: `<|turn_sep|>` を special token から「既存 tokenizer の通常文字列 delimiter」に変更 (vocab/embedding resize 不要)。
- **S7-001 checkpoint load 経路**: PHOTON eval/runtime が採用 checkpoint を実際にロードする経路を明示 (現状未確認の最大破壊リスク)。
- **S7-002 leaf path 伝播**: 累計 step 化した checkpoint leaf path を config/report で具体化。
- **S7-005 #137 競合**: Open な #137 (institutional 多言語 embedding A/B) との GPU/config 競合を反映。
- **S7-006 GPU 占有再見積もり**: 65,536 tokens/step 反映後の GPU 占有時間を再計算。

## 残課題と次フェーズへの申し送り

1. **mulmoclaude 600 step checkpoint の物理確認** (Phase 0.5 Unverifiable): 設計方針書策定時に実 checkpoint パス・val_loss=0.4525 ログを取得し、紛失時の risk 対応 (scratch 学習) を明文化する。
2. **学習 corpus の元データ確認**: ユーザー提供情報により corpus は `/Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents/` (4228 .md) で確認済。設計方針書で `--corpus-dir` 引数として固定する。
3. **PHOTON eval/runtime の checkpoint load 経路確認** (S7-001): 設計方針書または実装段階で `baseline_reporag/photon_pipeline.py` の checkpoint 読込経路を実装レベルで確定する。

## 次のアクション

- [x] Issueの最終確認 (`gh issue view 135` で更新確認済)
- [ ] `/design-policy 135` で設計方針策定 (Phase 2)
- [ ] `/multi-stage-design-review 135` で設計レビュー (Phase 3)
- [ ] `/work-plan 135` で作業計画立案 (Phase 4)
- [ ] `/pm-auto-dev 135` または分割 Issue で TDD 実装 (Phase 5)

## 出力ファイル

```
workspace/issues/135/issue-review/
├── original-issue.json
├── hypothesis-verification.md
├── stage1-review-context.json / stage1-review-result.json
├── stage2-apply-context.json / stage2-apply-result.json / stage2-new-body.md
├── stage3-review-context.json / stage3-review-result.json
├── stage4-apply-context.json / stage4-apply-result.json / stage4-new-body.md
├── stage5-review-result.json
├── stage6-apply-result.json / stage6-new-body.md
├── stage7-review-result.json
├── stage8-apply-result.json / stage8-new-body.md
└── summary-report.md (本ファイル)
```

GitHub Issue: https://github.com/Kewton/photon-mlx/issues/135
