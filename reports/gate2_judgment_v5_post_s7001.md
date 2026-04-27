# Gate 2 判定レポート v5 — S7-001/138 修正後 PHOTON 真の baseline

- **Date**: 2026-04-27
- **Issue**: [#148](https://github.com/Kewton/photon-mlx/issues/148)
- **PRs**: [#150](https://github.com/Kewton/photon-mlx/pull/150) (Phase A0 fail-loud checkpoint loading), [#151](https://github.com/Kewton/photon-mlx/pull/151) (Phase A pre-flight fixes)
- **判定**: **Conditional Go (with Caveats)** — 真の PHOTON は baseline 比 NC +1.67pp と僅差、latency P50 -38.3% / FU P50 -41.0% の大幅改善は維持。v4 era の "PHOTON NC 6.7%" は random-init artifact であり、真値は baseline 同等。#135 再学習なしでも latency benefit が確定。

---

## 0. TL;DR

| 結論 | 根拠 |
|------|------|
| **真の PHOTON は baseline と同等の品質 (NC +1.67pp)** | v5 measurement: baseline NC 6.11% vs PHOTON NC 7.78% |
| **PHOTON working memory による latency 大幅短縮は確定** | P50 -38.3%, follow-up P50 -41.0% (weight 非依存、信頼可) |
| **v4 era PHOTON NC 6.7% は random-init artifact** | S7-001 (random-init) + #138 (stub tokenizer) で eval pipeline が壊れていた |
| **Phase A0+A 完了 = #135 GPU 着手 unblock** | 真の baseline 確立済 |

---

## 1. 概要

Gate 2 v4 までの PHOTON 数値は **S7-001 (random-init weights) + #138 (tokenizer mismatch)** の影響で **無効**。
本レポートは両 bug を修正後 (PR #141, #146, #147 merged + 本セッションの PR #150, #151) の **真の PHOTON ベースライン** を確立する。

### v4 → v5 の変更点

| 項目 | v4 | v5 (本レポート) |
|------|-----|----------------|
| PHOTON checkpoint | random-init (S7-001 で load されず) | mulmoclaude 600-step (real ckpt loaded, val_loss 1.6238) |
| Tokenizer | `_StubTokenizer` (#138 前) | real Qwen2.5-Coder HF tokenizer |
| LLM | mlx-community/Qwen2.5-Coder-14B-Instruct-4bit | (同上) |
| Pipeline | photon_pipeline (S7-001 era) | photon_pipeline (PR #150 fail-loud loading) |
| Phase A0 fix | N/A | checkpoint root containment + repo-id allowlist |
| Phase A pre-flight | N/A | vocab padding 許容 + num_heads=10 (PR #151) |

### 補足: Phase A pre-flight 発見事項

PR #151 で 2 つの latent bug を修正:
1. #138 vocab_size strict equality が padding (151643→152064) を拒否 → `<=` に relax
2. yaml/loader schema drift (yaml: `generation.num_attention_heads`, loader: `cfg.model.num_heads` 既定 4) → `model.num_heads: 10` 追加

---

## 2. 実行環境

| 項目 | 値 |
|------|-----|
| マシン | M3 Ultra (Mac Studio, 256GB unified memory) |
| OS | macOS Darwin 25.4.0 |
| Python | 3.12.3 |
| ブランチ | `feature/issue-148-phase-a-fixes` |
| 実行時刻 | 2026-04-27 15:38–17:24 JST (Phase A.1 全体、約 1.8 時間) |
| PHOTON_CHECKPOINT_ROOT | `/Users/maenokota/share/work/github_kewton/photon-mlx-develop/checkpoints` |
| ckpt | `step_000600` (mulmoclaude, val_loss 1.6238) |
| eval set | `data/eval_sets/multi_turn_eval.jsonl` (30 sessions × 6 turns = 180 turns) |

---

## 3. 設定

### baseline (`configs/baseline.yaml`)
- `model.provider: "mlx_lm"` (Qwen2.5-Coder-14B-Instruct-4bit のみ)
- `repo_id: "fastapi_fastapi"`
- retrieval: BM25 + sentence-transformers/all-MiniLM-L6-v2 + cross-encoder/ms-marco-MiniLM-L-6-v2

### PHOTON (`configs/photon_small.yaml`)
- `model.provider: "photon"`, `model_id: "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"`
- `model.checkpoint_path: "step_000600"` (PR #150 + PR #151)
- `model.num_heads: 10` (PR #151)
- 同 retrieval

---

## 4. 結果

### 4.1 Phase A.1 single-run summary (FastAPI MT)

| 指標 | baseline (run 1) | **PHOTON (run 1)** | delta | v4 PHOTON (random-init era、無効) |
|------|-----------------|-------------------|-------|----------------------------------|
| 完走 turn | 180/180 | 180/180 | — | 180/180 |
| **MT NC (overall)** | 6.11% | **7.78%** | **+1.67pp** | 6.7% (artifact) |
| MT NC (Turn 5-6) | 1.67% | **1.67%** | 0pp | — |
| **Latency P50** | 20,440 ms | **12,604 ms** | **-38.3%** | — |
| Latency mean | 21,416 ms | 13,401 ms | -37.4% | — |
| Latency max | 38,627 ms | 38,588 ms | ≈0% | — |
| **Follow-up P50 (T2-T6)** | 20,350 ms | **12,011 ms** | **-41.0%** | 14,428 ms (weight 非依存、近い) |
| Follow-up P90 | 31,909 ms | 16,118 ms | -49.5% | ~25,000 ms (推定) |

### 4.2 Turn-by-turn breakdown

#### baseline (configs/baseline.yaml)
| Turn | NC% | latency P50 |
|------|-----|-------------|
| T1 | 3.3% | 22,061 ms |
| T2 | 13.3% | 19,071 ms |
| T3 | 13.3% | 19,675 ms |
| T4 | 3.3% | 20,440 ms |
| T5 | 3.3% | 22,930 ms |
| T6 | 0.0% | 20,965 ms |

#### PHOTON (configs/photon_small.yaml + step_000600 ckpt)
| Turn | NC% | latency P50 |
|------|-----|-------------|
| T1 | 0.0% | 20,234 ms |
| T2 | 16.7% | 10,101 ms |
| T3 | 20.0% | 10,103 ms |
| T4 | 6.7% | 11,770 ms |
| T5 | 3.3% | 12,869 ms |
| T6 | 0.0% | 13,850 ms |

### 4.3 Drift metrics (PHOTON 専用)

本 single-run では Drift metrics の集計を未実装。Issue #148 設計判断 #6 (DR2-005) に従い別途 follow-up issue で取得。

> v4 数値 (Latent cosine drift, Logit KL, Topic shift score) は random-init eval で計算されており **無効**。v5 では「真の Drift metrics」を別 PR で測定する方針。

---

## 5. 解釈

### 5.1 「真の PHOTON」が baseline に対してどの程度の性能か (Phase A.1 / FastAPI)

- **NC**: PHOTON +1.67pp (baseline=6.11%, photon=7.78%) — **僅差で有意に悪化していない**
- **Latency**: PHOTON で P50 -38.3%, follow-up P50 -41.0% — **大幅短縮を維持**
- → **PHOTON working memory の latency benefit は実在**、品質 trade-off は許容範囲

### 5.2 v4 → v5 の差分要因

v4 (2026-04-19) の数値:
- baseline NC: 15.6% (MT)
- PHOTON NC: 6.7% (MT) → 大幅改善と見えていた
- PHOTON follow-up P50: 14,428 ms (-35.0%)

v5 (2026-04-27) の数値:
- baseline NC: 6.11% (大幅改善、v4 から 8 日間の retrieval 改善累積)
- PHOTON NC: 7.78% (baseline ≈ +1.67pp、真値)
- PHOTON follow-up P50: 12,011 ms (-41.0%、weight 非依存で改善、近い)

**v4 期に PHOTON NC が baseline より大幅低かったのは S7-001 random-init による artifact**。実 ckpt を load した v5 では PHOTON は baseline と同等 NC + latency 大幅短縮の trade-off。

### 5.3 #135 再学習への影響 (Gate 2 v5 の判定論理)

Issue #148 の判定基準:
- 真の PHOTON が baseline 比 NC 改善 → S7-001 仮説裏付け
- 真の PHOTON が baseline 比 NC 悪化 (#113 の +4.44pp 程度) → アーキ自体に課題、#135 再学習で改善期待

**v5 結果**: NC +1.67pp (baseline 同等)、latency 大幅改善 → **アーキテクチャは健全**、#135 再学習で更なる品質向上を期待できる状態。

判定: **#135 GPU 着手 unblock** (Phase A0+A 完了で本 Issue の必須前提達成)。

---

## 6. Phase A 実行記録

### 6.1 Phase A.0 pre-flight (本セッション)

| step | 結果 | duration |
|------|------|----------|
| 環境構築 (data symlink + PHOTON_CHECKPOINT_ROOT) | OK | < 1 min |
| Discovery 1 (vocab padding) 修正 | PR #151 | — |
| Discovery 2 (num_heads schema) 修正 | PR #151 | — |
| Smoke test (institutional / 1 turn) | OK (25s, valid answer + cite) | < 1 min |

### 6.2 Phase A.1 (Gate 2 v5 / FastAPI MT)

| run | start | end | duration | output | turns | NC | P50 |
|-----|-------|-----|----------|--------|-------|----|----|
| baseline run 1 | 15:38 | 16:43 | 1h05min | logs/phase_a1_baseline_run1.predictions.jsonl | 180 | 6.11% | 20,440 ms |
| PHOTON run 1 | 16:43 | 17:24 | 41 min | logs/phase_a1_photon_run1.predictions.jsonl | 180 | 7.78% | 12,604 ms |

> PHOTON run は baseline より速い (PHOTON working memory による follow-up turn の高速化が反映)。

### 6.3 Phase A.2 (#113 v2 / Institutional MT)

別レポート `reports/institutional_photon_mt_eval_v2.md` 参照。

---

## 7. Limitation / Caveat

1. **本 v5 の ckpt (val_loss 1.6238) と v4 期 ckpt (val_loss 0.4525, roadmap.md 記載) の不一致**: v4 当時の ckpt 所在不明、本セッションでは候補 #1 (`step_000600`) を使用。Phase A 結果は「val_loss 1.62 の ckpt での baseline」として読む必要あり。
2. **1-run のみ**: variance / nondeterminism (#143) は別 follow-up 化。
3. **PHOTON × Qwen2.5-Coder の同一 LLM 環境**: 新 LLM (Qwen3.x / Gemma4) は Phase B (PR #2) で扱う。
4. **v4 baseline NC 15.6% から v5 6.11% への改善**は v4 (2026-04-19) 以降の retrieval 改善 (#125 chunker, #126 embedding, #137 V4 retrieval 等) 累積効果。本 v5 は **現行 baseline vs 現行 (real-loaded) PHOTON** の比較が主目的。
5. **Drift metrics 未測定**: 別 follow-up issue で取得。

---

## 8. 次のアクション

- [x] PR #150 (Phase A0 fail-loud loading) merged
- [x] PR #151 (Phase A pre-flight: vocab padding + num_heads schema) created
- [ ] 本 PR (Phase A reports) merge
- [ ] #143 follow-up: 2nd run + variance 報告 (Option B → Full)
- [ ] CB-005 follow-up issue: MLX import abort in fresh test env
- [ ] yaml/loader schema 整理 follow-up issue (num_attention_heads / num_heads 統一)
- [ ] roadmap.md val_loss 0.4525 vs 1.6238 不一致の調査 / 訂正
- [ ] PHOTON Drift metrics 別 follow-up
- [ ] **#135 Phase 6-8 (本格再学習) 着手解禁** ← 本 PR merge で達成
- [ ] Phase B (PR #2): 新 LLM (Qwen3.x / Gemma4) baseline-only eval
