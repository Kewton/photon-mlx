# 制度文書 PHOTON MT 測定レポート v2 (Issue #148 Phase A.2)

**実測対象 corpus**: `institutional_documents` (post-#125+#126 状態、#137 V4 retrieval 採用後)
**eval set**: `data/eval_sets/institutional_multi_turn_eval.jsonl` (30 sessions × 6 turns = 180 turns)
**実測日**: 2026-04-27
**Issue**: [#148](https://github.com/Kewton/photon-mlx/issues/148)
**Branch**: `feature/issue-148-phase-a-fixes`
**前バージョン**: `reports/institutional_photon_mt_eval.md` (S7-001 era、v1 = 無効)

---

## 0. TL;DR

| 結論 | 根拠 |
|------|------|
| **真の PHOTON は制度文書 corpus で baseline 比 NC +5.56pp 悪化** | v2 measurement: baseline NC 7.78% vs PHOTON NC 13.33% |
| **Turn 5-6 で更に悪化** | baseline=5.00% vs PHOTON=11.67% (+6.67pp) |
| **Latency 大幅短縮は維持** | P50 -34.0%, follow-up P50 -42.3% |
| **mulmoclaude ckpt は英語コード訓練 — 日本語制度文書ドメインで弱い** | Issue #148 仮説 C (本格再学習が必要) を裏付け |
| **#135 Phase 6-8 (日本語制度文書 corpus での再学習) が品質改善の鍵** | 本 v2 結果が #135 着手の justification |

---

## 1. 概要

#113 の v1 PHOTON 数値 (NC 11.39%, NC Turn 5-6 = 10.83%) は **S7-001 (random-init weights) + #138 (tokenizer mismatch)** の影響で **無効**。
本 v2 は両 bug 修正後 (PR #141, #146, #147 merged + PR #150 fail-loud + PR #151 pre-flight) の **真の PHOTON 制度文書 baseline** を確立する。

### v1 → v2 の変更点

| 項目 | v1 (#113) | v2 (本レポート) |
|------|-----------|----------------|
| PHOTON checkpoint | random-init (S7-001 で load されず) | mulmoclaude 600-step (real ckpt loaded, val_loss 1.6238) |
| Tokenizer | `_StubTokenizer` (#138 前) | real Qwen2.5-Coder HF tokenizer |
| LLM | mlx-community/Qwen2.5-Coder-14B-Instruct-4bit | (同上) |
| Phase A0 fix | N/A | PR #150 fail-loud loading + root containment + repo-id allowlist |
| Phase A pre-flight | N/A | PR #151 vocab padding 許容 + num_heads=10 |

---

## 2. 実行環境

| 項目 | 値 |
|------|-----|
| マシン | M3 Ultra (Mac Studio, 256GB unified memory) |
| OS | macOS Darwin 25.4.0 |
| Python | 3.12.3 |
| ブランチ | `feature/issue-148-phase-a-fixes` |
| 実行時刻 | 2026-04-27 17:25–19:05 JST (Phase A.2 全体、約 1.7 時間) |
| PHOTON_CHECKPOINT_ROOT | `/Users/maenokota/share/work/github_kewton/photon-mlx-develop/checkpoints` |
| ckpt | `step_000600` (mulmoclaude, val_loss 1.6238) |

---

## 3. 設定

### baseline (`configs/institutional_docs.yaml`)
- `model.provider: "mlx_lm"` (Qwen2.5-Coder-14B-Instruct-4bit のみ)
- `repo_id: "institutional_documents"`
- chunker: max_chars=800 (post-#126)
- E5 prefix: on (post-#125)
- retrieval: BM25 + bge-m3 (#137 V4) + bge-reranker-v2-m3

### PHOTON (`configs/institutional_docs_photon.yaml`)
- `model.provider: "photon"`
- `model.checkpoint_path: "step_000600"` (PR #150 + PR #151)
- `model.num_heads: 10` (PR #151)
- 同 retrieval (V4)

---

## 4. 結果

### 4.1 Phase A.2 single-run summary

| 指標 | baseline (run 1) | **PHOTON (run 1)** | delta | v1 PHOTON (random-init era、無効) |
|------|-----------------|-------------------|-------|----------------------------------|
| 完走 turn | 180/180 | 180/180 | — | 180/180 |
| **MT NC (overall)** | 7.78% | **13.33%** | **+5.56pp** | 11.39% (artifact) |
| **MT NC (Turn 5-6)** | 5.00% | **11.67%** | **+6.67pp** | 10.83% (artifact) |
| **Latency P50** | 19,383 ms | **12,800 ms** | **-34.0%** | — |
| Latency mean | 20,145 ms | 12,936 ms | -35.8% | — |
| Latency max | 37,109 ms | 24,729 ms | -33.4% | — |
| **Follow-up P50 (T2-T6)** | 20,149 ms | **11,628 ms** | **-42.3%** | (-44.9% by v1、weight 非依存で近い) |
| Follow-up P90 | 28,012 ms | 17,865 ms | -36.2% | — |

### 4.2 Turn-by-turn breakdown

#### baseline (configs/institutional_docs.yaml)
| Turn | NC% | latency P50 |
|------|-----|-------------|
| T1 | 13.3% | 16,837 ms |
| T2 | 0.0% | 18,398 ms |
| T3 | 13.3% | 21,099 ms |
| T4 | 10.0% | 17,342 ms |
| T5 | 10.0% | 20,388 ms |
| T6 | 0.0% | 23,383 ms |

#### PHOTON (configs/institutional_docs_photon.yaml + step_000600 ckpt)
| Turn | NC% | latency P50 |
|------|-----|-------------|
| T1 | 10.0% | 17,398 ms |
| T2 | 0.0% | 10,621 ms |
| T3 | 16.7% | 11,814 ms |
| T4 | 30.0% | 10,174 ms |
| T5 | 23.3% | 11,563 ms |
| T6 | 0.0% | 15,002 ms |

> **観察**: PHOTON の T4 NC=30.0% / T5 NC=23.3% が baseline (T4=10.0%, T5=10.0%) より顕著に悪化。日本語制度文書のドメイン外実行で 中盤 turn の retrieval が弱体化していると推察。

### 4.3 Drift metrics (PHOTON 専用)

本 single-run では Drift metrics の集計を未実装。Issue #148 設計判断 #6 / DR2-005 に従い別 follow-up issue で取得。

> v1 数値は random-init eval で **無効**。v2 では「真の Drift metrics」を別 PR で測定する方針。

---

## 5. 解釈

### 5.1 真の PHOTON が制度文書 corpus でどの程度動作するか

- **NC overall +5.56pp 悪化** (7.78% → 13.33%)
- **NC Turn 5-6 +6.67pp 悪化** (5.00% → 11.67%) — multi-turn 持続力に課題
- **Latency P50 -34.0%、follow-up -42.3%** — 速度面は明確に向上

→ **品質と速度の trade-off が明確、品質側が課題**。
→ Issue #148 仮説 C「アーキ自体に課題、#135 再学習で改善が期待される」を支持。

### 5.2 mulmoclaude ckpt のドメイン適合性

- **訓練データ**: 主に英語コード (mulmoclaude)
- **eval ドメイン**: 日本語制度文書
- **ドメイン外実行のため retrieval-aware decoding 弱体化**
- 特に T4-T5 (中盤 follow-up) で NC が大きく上昇 (working memory が日本語ドメインで安定しない)

### 5.3 v1 → v2 の差分要因

| 指標 | v1 PHOTON | v2 PHOTON | 解釈 |
|------|-----------|-----------|------|
| NC overall | 11.39% (random-init artifact) | 13.33% | 真値は v1 より僅かに悪化 (random-init の方が偶然「ぼやけて」NC が低めだった) |
| NC Turn 5-6 | 10.83% (random-init) | 11.67% | 同等の magnitude、v1 の数値は偶然近かった |
| Follow-up P50 | (latency 改善 -44.9%) | -42.3% | weight 非依存、近い |

**v1 の "PHOTON NC 11.39%" は random-init artifact**。真の PHOTON は v1 より僅かに悪化した数値だが、概ね同等の magnitude。

### 5.4 #135 Phase 6-8 (本格再学習) の justification

本 v2 結果は **#135 着手の必要性を強く支持**:
- 速度面では PHOTON working memory が機能 (-42% follow-up latency)
- 品質面では英語コード訓練の ckpt では日本語制度文書に対応できず +5.56pp 悪化
- → **日本語制度文書 corpus で再学習すれば品質改善が期待される** (=#135 Phase 6-8)

---

## 6. Phase A.2 実行記録

| run | start | end | duration | output | turns | NC | P50 |
|-----|-------|-----|----------|--------|-------|-----|-----|
| baseline run 1 | 17:25 | 18:25 | 60 min | logs/phase_a2_baseline_run1.predictions.jsonl | 180 | 7.78% | 19,383 ms |
| PHOTON run 1 | 18:25 | 19:05 | 40 min | logs/phase_a2_photon_run1.predictions.jsonl | 180 | 13.33% | 12,800 ms |

> PHOTON run は baseline より速い (PHOTON working memory による follow-up turn の高速化が反映)。

---

## 7. Limitation / Caveat

1. **本 v2 の ckpt (val_loss 1.6238) と v1 期 ckpt の不一致**: v1 当時の ckpt 所在不明、本セッションでは候補 #1 (`step_000600`) を使用。
2. **1-run のみ**: variance / nondeterminism (#143) は別 follow-up 化。
3. **mulmoclaude ckpt は英語コード訓練 — 日本語制度文書はドメイン外**: NC 高めは想定通り。#135 再学習で日本語ドメインに適応する見込み。
4. **Drift metrics 未測定**: 別 follow-up issue で取得。
5. **embedding model (bge-m3, multilingual)** は日本語に強いが、**LLM の generation step** が日本語制度文書 + PHOTON working memory の組合せで弱い (主因と推察)。

---

## 8. 次のアクション

- [x] PR #150 (Phase A0 fail-loud loading) merged
- [x] PR #151 (Phase A pre-flight: vocab padding + num_heads schema) created
- [ ] 本 PR (Phase A reports) merge
- [ ] **#135 Phase 6-8 着手解禁 (= 日本語制度文書 corpus で本格再学習)** — 本 v2 結果が必要性を裏付け
- [ ] #143 follow-up: 2nd run + variance 報告
- [ ] PHOTON Drift / Safe RecGen 指標が v2 で収集される場合、別 report に切り出し検討
- [ ] CB-005 / yaml schema / roadmap.md / data 共有 follow-up issue (PR #150/#151 で記録済)
