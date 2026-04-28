# Issue #135 仮説検証レポート

**検証日**: 2026-04-26
**対象 Issue**: #135 — feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合
**検証完了度**: 10/12 (83.3%)

## 抽出した仮説・前提条件

Issue 本文から事実主張として抽出したもの:

| # | 主張カテゴリ | 内容 |
|---|------------|------|
| 1 | 前提 (履歴) | #113 PR #134 がマージ済み (commit e281660) |
| 2 | 仮説 (実測根拠) | #113 の実測値 (NC overall, Turn 5-6, latency p50) |
| 3 | 仮説 (判定基準) | 設計 §9 仮説 C (Turn 5-6 NC > 6%) 該当 |
| 4 | 前提 (Epic 計画) | #117 Epic Phase 2 conditional 表 |
| 5 | 前提 (現状) | 既存 mulmoclaude (英語コード) 600 step 完了 |
| 6 | 前提 (corpus) | institutional_documents 4228 md |
| 7 | 前提 (eval set) | data/eval_sets/institutional_multi_turn_eval.jsonl 存在 |
| 8 | 前提 (script) | scripts/generate_institutional_eval_set.py 存在 |
| 9 | 前提 (script) | scripts/run_multi_turn_eval.py 存在 |
| 10 | 前提 (config) | configs/institutional_docs_photon.yaml 存在 |
| 11 | 前提 (実装) | photon_mlx/trainer.py が継続学習可能 |
| 12 | 前提 (実装) | photon_mlx/data.py で JP/EN mix 実装可能 |

## 検証結果

| # | 主張 | 判定 | 根拠 |
|---|------|------|------|
| 1 | PR #134 merged (e281660) | ✅ Confirmed | `git log` |
| 2 | #113 実測値 | ✅ Confirmed | `reports/institutional_photon_mt_eval.md` L1-180 すべて一致 |
| 3 | 設計 §9 仮説 C | ✅ Confirmed | `reports/institutional_photon_mt_eval.md` §8, `workspace/mvp/roadmap.md` L88-94 |
| 4 | #117 Epic Phase 2 conditional 表 | ✅ Confirmed | `workspace/mvp/roadmap.md` L86-94 (`< 3%` / `3-6%` / `> 6%` の 3 段階判定) |
| 5 | mulmoclaude 600 step 完了 | 🔲 Unverifiable | `workspace/memo.md` L140 / `workspace/mvp/roadmap.md` L17 で記述あり (val_loss 0.4525) だが、`checkpoints/` 配下の物理ファイルは未確認 (git LFS or 外部保管の可能性) |
| 6 | institutional_documents 4228 md | ✅ Confirmed | `reports/institutional_baseline_static.md` L3, L36 |
| 7 | institutional_multi_turn_eval.jsonl | ✅ Confirmed | 30 行 (= 30 sessions × 6 turns = 180 turns) |
| 8 | generate_institutional_eval_set.py | ✅ Confirmed | 存在、`baseline_reporag.eval.institutional` 配下を呼ぶ CLI ラッパー |
| 9 | run_multi_turn_eval.py | ✅ Confirmed | 存在、`build_pipeline(cfg)` で baseline/photon 両 provider 対応 |
| 10 | institutional_docs_photon.yaml | ✅ Confirmed | 存在、`paths.checkpoint_root: ./checkpoints`, `provider: photon`, photon_small 相当 |
| 11 | trainer.py 主要 API | ✅ Confirmed | L253-430 に train loop / checkpoint 保存 / cosine schedule / resume_from / gradient accumulation / early stopping / float16 mixed precision を実装済み |
| 12 | data.py で JP/EN mix 実装可能 | ✅ Confirmed | `load_jsonl` → `pack_sequences` → `create_batches` の構造で 2 corpus concat + shuffle により 50/50 mix を最小改修で実現可能 |

## Stage 1 レビューへの申し送り事項

### 確認事項 (レビュー時に再確認すべき)

- **#5 mulmoclaude 600 step checkpoint の物理存在**: Issue 本文では「既存 mulmoclaude (英語コード) 50% を保持」「(catastrophic forgetting 回避)」と前提されている。Stage 1 レビューでは「実 checkpoint パス・val_loss 実測ログ」を Issue に明記すべきか検討。
- **#11 resume_from メカニズム**: trainer.py に既存 (L281-283) ある。Issue ゴール「continual learning」は実装上自然だが、設計方針書で明示が必要。
- **#12 data.py の mix 実装方針**: 現状 `iterate_batches()` は単一 corpus_path 前提。Issue 受入条件で「`iterate_batches` を corpus 複数対応に拡張」と明記すべきか検討。

### Rejected 仮説

なし (Issue 本文の主張はすべて Confirmed または Unverifiable)。

### 補足提案

1. `reports/institutional_photon_mt_eval.md` の数値は Issue と完全一致しており、根拠の信頼性は高い。
2. `workspace/mvp/roadmap.md` の conditional 表 (3-6% で軽量 fine-tune 5K step、>6% で本格再学習 10-20K step) は Issue 本文の「10-20K step」「JP 50%+」と整合している。
3. eval pipeline (`scripts/run_multi_turn_eval.py`) は再学習後の比較に再利用可能。`reports/institutional_photon_mt_eval_v2.md` 出力先も既存パターン踏襲で問題ない。

## 結論

Issue #135 は**事実関係に問題なし**。ゴール・アプローチ・受入条件すべて根拠ある実測値・既存実装に基づいている。Stage 1 (通常レビュー) では以下にフォーカスすべき:

- **A-1 学習 corpus 構築の具体性** (sessions 数、turn 構成、generation script の拡張範囲)
- **B-1 ハイパーパラメータの根拠** (10K-20K step の妥当性、learning rate 数値)
- **C-2 採用基準の優先順位** (Turn 5-6 NC < 6% AND latency 維持の同時成立条件)
- **mulmoclaude checkpoint 物理確認の必要性** (Unverifiable 項目の補足)
