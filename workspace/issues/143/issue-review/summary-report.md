# Issue #143 マルチステージレビュー完了報告

実施日: 2026-04-28
対象: `fix(eval): institutional eval reproducibility — Qwen 14B nondeterminism causes ±1.7pt baseline drift between runs`
ブランチ: `feature/issue-143-eval-reproducibility`

---

## 仮説検証結果（Phase 0.5）

| # | 仮説/主張 | 判定 |
|---|----------|------|
| H1 | 既存スクリプト (`run_baseline_eval.py`, `aggregate_institutional_baseline.py`, `generation/generator.py`) に seed/--runs 機能なし | Confirmed |
| H2 | `temperature=0.2, do_sample=False` で「greedy decoding 想定」だが `make_sampler(temp=0.2)` 経路 | Partially Confirmed |
| H3 | nondeterminism: institutional `llm_client.py` には seed 固定あるが baseline には無い (経路不統一) | Confirmed |
| H4 | V0 baseline drift: 11.21% (4/25) → 12.93% (4/26) = +1.72pt | Confirmed |
| H5 | multi-turn eval も同 `build_pipeline` 経由で同 nondeterminism 影響下 | Confirmed |
| H6 | Issue #135 (seed perturb), #138 (tokenizer fix), #156 (refusal-aware bug) との関係 | Confirmed |

→ 仮説検証レポート: `hypothesis-verification.md`

---

## ステージ別結果

| Stage | レビュー種別 | Reviewer | Must Fix | Should Fix | Nice to Have | 対応数 | ステータス |
|-------|------------|----------|---------|----------|------|------|----------|
| 1 | 通常レビュー（1回目） | claude-opus | 3 | 5 | 3 | - | 完了 |
| 2 | 指摘事項反映（1回目） | sonnet | - | - | - | 11/11 | 完了 |
| 3 | 影響範囲レビュー（1回目） | claude-opus | 3 | 4 | 2 | - | 完了 |
| 4 | 指摘事項反映（1回目） | sonnet | - | - | - | 9/9 | 完了 |
| 5 | 通常レビュー（2回目） | **codex** | 3 | 3 | 0 | - | 完了 (reviewer=codex 検証済) |
| 6 | 指摘事項反映（2回目） | **codex** | - | - | - | 6/6 | 完了 |
| 7 | 影響範囲レビュー（2回目） | **codex** | 1 | 4 | 1 | - | 完了 (reviewer=codex 検証済) |
| 8 | 指摘事項反映（2回目） | **codex** | - | - | - | 6/6 | 完了 |
| **合計** | - | - | **10** | **16** | **6** | **32/32** | **全反映済** |

### Codex reviewer 検証結果 (Issue #140 / S7-001 follow-up)

- `stage5-review-result.json`: `reviewer="codex"` ✓
- `stage7-review-result.json`: `reviewer="codex"` ✓
- WARNING なし

---

## 主要な発見

### 1回目イテレーション (Claude opus)

**Stage 1 通常レビュー**:
- Must Fix: 影響ファイル `mlx_lm.py` 誤記、Task 1 サンプルが起動時 1 回 seed (per-query 不足)、`mlx_lm.generate()` に seed 引数なし API 前提
- Should Fix: greedy decoding 表現乖離、既存 `llm_client.py` 統一方針欠落、テスト粒度・Task 4 schema・Task 順序未定義
- Nice to Have: 事実と推定の分離、Task 2 スコープ、multi-turn eval スコープ

**Stage 3 影響範囲レビュー**:
- Must Fix: `Generator.generate(messages, ..., *, seed)` の真 API 反映、CLAUDE.md メトリクス再計測リスク、既存 MagicMock 17+ 件後方互換
- Should Fix: CLI/server interactive UX 退化、Issue #156 衝突、weekly_eval.yml + ci_eval_check.py 整合、文書更新タスク
- Nice to Have: PR #142 (MERGED) 状態反映、Issue #138 (CLOSED) 表示

### 2回目イテレーション (Codex 独立クロスレビュー)

**Stage 5 通常レビュー (Codex)**:
- Must Fix: API 設計の本文内矛盾 (`Generator(seed=None)` と default `seed=42` が併存)、`run_baseline_eval.py --repo-id` の silent bug (build_pipeline 前に cfg.repo.repo_id 反映なし)、Task 3 `--runs` schema 未定義 (run_index/seed の predictions 必須出力と aggregator per-run 計算)
- Should Fix: 既存 `cfg.run.seed`/`cfg.run.deterministic` の未使用 (config 駆動への切り替え)、in-process `PYTHONHASHSEED` の誤用、multi-turn predictions と aggregator schema 不整合

**Stage 7 影響範囲レビュー (Codex)**:
- Must Fix: `cfg.run.seed`/`deterministic` の型検証と `run` ブロック欠落 fallback 設計
- Should Fix: weekly_eval.yml の `--runs` 採用方針未確定、Streamlit eval runner と EvalJob schema 未対応、`scripts/compare_generators.py` の seed 伝播漏れ、`query(seed=...)` の伝播 unit test 不足
- Nice to Have: #156 関連 (REQUIRED_FIELDS が `is_refusal` + `run_index/run_seed/run_id` で増分される旨)

### Codex 独立観点の価値

- **silent bug 発見** (S5-004): `run_baseline_eval.py` の `--repo-id` が build_pipeline 前に cfg.repo.repo_id へ反映されず、override 時に空 retrieval になる既存 bug を発見
- **API 設計矛盾発見** (S5-001): opus の Stage 1+3 反映で生まれた "default seed=42 vs Generator(seed=None)" の矛盾を独立観点で検出
- **config 駆動への昇格** (S5-002): 既存 `cfg.run.seed`/`deterministic` の活用を促す設計修正
- **影響範囲の漏れ発見** (S7-003, S7-004): Streamlit eval runner / `compare_generators.py` の seed 伝播漏れを発見

---

## 最終 Issue 状態

- GitHub URL: https://github.com/Kewton/photon-mlx/issues/143
- 現在の state: OPEN
- 反映 findings: 32/32 (Must Fix 10、Should Fix 16、Nice to Have 6)
- Issue 本文サイズ: 約 19KB (Stage 8 反映後)
- Markdown 構造: 維持 (背景/原因/影響/ゴール/変更内容/受入条件/影響ファイル/内部依存関係/並列性/関連)

---

## 次のアクション

- [x] Issue の最終確認
- [ ] **/design-policy 143 で設計方針策定** ← 次のフェーズ
- [ ] /multi-stage-design-review 143 で設計レビュー
- [ ] /work-plan 143 で作業計画立案
- [ ] /pm-auto-dev 143 で TDD 自動開発
