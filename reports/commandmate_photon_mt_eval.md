# Commandmate PHOTON MT 測定レポート (Issue #148 Phase A.3)

**実測対象 corpus**: `commandmate` (TypeScript + Next.js + Markdown docs, 5,737 chunks after cleanup)
**eval set**: `data/eval_sets/commandmate_multi_turn_eval.jsonl` (15 sessions × 6 turns = 90 turns、本 PR で新規生成)
**実測日**: 2026-04-27
**Issue**: [#148](https://github.com/Kewton/2photon-mlx/issues/148)
**Branch**: `feature/issue-148-phase-a3-commandmate`
**位置づけ**: Phase A.1 (FastAPI/Python) + A.2 (Institutional/JP) と並べて、**訓練ドメイン仮説** (mulmoclaude=TS で訓練済み ckpt なら TS ドメインで PHOTON 勝つはず) を検証する 3 番目の測定点。

---

## 0. TL;DR

| 結論 | 根拠 |
|------|------|
| **訓練ドメイン仮説は否定的**: TS+Next.js (訓練に最も近い) でも PHOTON が baseline より NC +20.00pp 悪化 | 3 dataset 比較表 (§4) |
| **ドメイン距離と PHOTON 不利の単調関係は成立せず** | commandmate(近)=+20pp / FastAPI(中)=+1.67pp / Institutional(遠)=+5.56pp |
| **#135 single-domain retrain では不十分かもしれない** — アーキテクチャ自体の課題が浮上 | Turn 5-6 で全 dataset 共通に NC 悪化、commandmate では +43.33pp |
| **Latency 改善は確定** | 全 dataset で -34〜48% の P50 短縮 (weight 非依存、信頼可) |
| **Caveat (要重視)**: commandmate eval は abstract な質問が多く baseline NC 42% も高い。retrieval 失敗連鎖が PHOTON を不利にしている可能性あり | §6 Limitation |

---

## 1. 概要と動機

PR #152 (Phase A.1 + A.2) の結果から:
- Phase A.1 (FastAPI MT, Python ドメイン): PHOTON NC +1.67pp (僅差)
- Phase A.2 (#113 v2, JP 制度文書ドメイン): PHOTON NC +5.56pp (悪化)

仮説: **訓練ドメインから遠いほど PHOTON が劣後** (institutional > FastAPI > commandmate の順で悪い見込み)。

検証手段: mulmoclaude ckpt は **TypeScript で訓練** されているため、**commandmate (Next.js + TS)** が **訓練ドメインに最も近い** corpus。ここで PHOTON が baseline 以上なら仮説成立。

**結果**: 仮説は否定される (commandmate で最も悪化)。アーキ自体の見直しが必要かもしれない。

---

## 2. 実行環境

| 項目 | 値 |
|------|-----|
| マシン | M3 Ultra (Mac Studio, 256GB unified memory) |
| OS | macOS Darwin 25.4.0 |
| Python | 3.12.3 |
| ブランチ | `feature/issue-148-phase-a3-commandmate` |
| 実行時刻 | 2026-04-27 20:00–21:30 JST (約 1.5 時間) |
| commandmate worktree | `/Users/maenokota/share/work/github_kewton/commandmate-issue-676` (commit `78be750e9c`) |
| PHOTON_CHECKPOINT_ROOT | `/Users/maenokota/share/work/github_kewton/photon-mlx-develop/checkpoints` |
| ckpt | `step_000600` (mulmoclaude, val_loss 1.6238) |

---

## 3. 設定

### 3.1 Ingest + index

```bash
python scripts/ingest_repo.py \
  --repo /Users/maenokota/share/work/github_kewton/commandmate-issue-676 \
  --repo-id commandmate \
  --config configs/baseline.yaml

# Cleanup excluded paths (node_modules / .next / dev-reports etc. が ingester で除外漏れ)
sqlite3 data/indexes/commandmate/chunks.db "DELETE FROM chunks WHERE rel_path LIKE 'node_modules/%' OR ..."

python scripts/build_indexes.py --repo-id commandmate \
  --commit 78be750e9cbb72935e40a28354f3e35d02da1eed \
  --config configs/baseline.yaml
```

注: `**/node_modules/**` 等の exclude パターンが fnmatch で完全には効かず、98k chunks (typescript.js 等) を後処理で削除。最終 5,737 chunks (src=2053, tests=2263, docs=1321 等)。

### 3.2 baseline (`configs/baseline.yaml`, --repo-id commandmate)
- `model.provider: "mlx_lm"` (Qwen2.5-Coder-14B-Instruct-4bit)
- retrieval: BM25 + sentence-transformers/all-MiniLM-L6-v2 + cross-encoder/ms-marco-MiniLM-L-6-v2

### 3.3 PHOTON (`configs/photon_small.yaml`, --repo-id commandmate)
- `model.provider: "photon"`
- `model.checkpoint_path: "step_000600"` (mulmoclaude)
- `model.num_heads: 10`
- 同 retrieval

### 3.4 Eval set 設計

新規 `scripts/generate_commandmate_eval_set.py` で 15 sessions × 6 turns を hand-craft:
- worktree 機能 / tmux 統合 / Claude CLI 連携 / SQLite 永続化 / Next.js UI / GitHub Issue / CLI tools / i18n / tests / multi-agent / config / deployment / セキュリティ / リアルタイム UI / agent 自動化
- 各 session は topic_narrowing 形式 (T1: 概要 → T6: 詳細)

---

## 4. 結果

### 4.1 Phase A.3 single-run summary

| 指標 | baseline (run 1) | **PHOTON (run 1)** | delta |
|------|-----------------|-------------------|-------|
| 完走 turn | 90/90 | 90/90 | — |
| **MT NC (overall)** | 42.22% | **62.22%** | **+20.00pp** |
| **MT NC (Turn 5-6)** | 23.33% | **66.67%** | **+43.33pp** |
| Latency P50 | 12,473 ms | **6,854 ms** | **-45.1%** |
| Latency mean | 13,484 ms | 8,222 ms | -39.0% |
| Latency max | 35,956 ms | 36,652 ms | ≈0% |
| **Follow-up P50 (T2-T6)** | 11,898 ms | **6,179 ms** | **-48.1%** |

### 4.2 Turn-by-turn breakdown

#### baseline
| Turn | NC% | latency P50 |
|------|-----|-------------|
| T1 | 66.7% | 13,316 ms |
| T2 | 80.0% | 9,563 ms |
| T3 | 26.7% | 15,122 ms |
| T4 | 33.3% | 10,689 ms |
| T5 | 26.7% | 12,500 ms |
| T6 | 20.0% | 13,501 ms |

> **観察**: T1 の baseline NC=66.7% は **eval question の vague さ** に起因 (例: 「全体構造とディレクトリ役割を教えてください」は specific chunk に紐付かない)。T2 で 80% に上昇するのは T1 で context 不足 + RAG 失敗の連鎖。T3 以降 retrieval が回復。

#### PHOTON
| Turn | NC% | latency P50 |
|------|-----|-------------|
| T1 | 73.3% | 13,005 ms |
| T2 | 60.0% | 5,270 ms |
| T3 | 53.3% | 6,799 ms |
| T4 | 53.3% | 7,081 ms |
| T5 | 66.7% | 6,775 ms |
| T6 | 66.7% | 5,411 ms |

> **観察**: PHOTON は T2 以降の latency が baseline の半分以下 (working memory 効果)、しかし NC が **常時 50% 以上** で baseline より大幅悪化。特に T5-T6 で 66.7% と PHOTON 単独で更に悪化、working memory が context 不足下で誤って drift している兆候。

---

## 5. 3-dataset 比較表 (Phase A.1 + A.2 + A.3 を並べて訓練ドメイン仮説検証)

| Dataset | 言語 / ドメイン | 訓練距離 | baseline NC | **PHOTON NC** | **Δ NC** | baseline P50 | PHOTON P50 | Δ P50 |
|---------|----------------|---------|-------------|---------------|----------|--------------|------------|-------|
| **commandmate** | **TS + Next.js** | **近 (訓練)** | 42.22% | **62.22%** | **+20.00pp** | 12,473 ms | 6,854 ms | -45.1% |
| FastAPI | Python | 中 | 6.11% | 7.78% | +1.67pp | 20,440 ms | 12,604 ms | -38.3% |
| Institutional | JP 法規 | 遠 | 7.78% | 13.33% | +5.56pp | 19,383 ms | 12,800 ms | -34.0% |

### 5.1 仮説検証

**仮説**: 訓練ドメイン (TS) 近い corpus で PHOTON が baseline 以上、または同等。

**結果**: **否定**。
- commandmate (TS、最近) で **最も悪化** (+20.00pp)
- ドメイン距離との単調関係なし (FastAPI < Institutional < commandmate の順で悪化)

### 5.2 共通の傾向 (3 dataset 共通)

1. **Latency**: 全 dataset で PHOTON が -34〜48% 短縮 (working memory 機能、weight 非依存で確定)
2. **Turn 5-6 で必ず NC が悪化**: PHOTON working memory が late turn で drift する系統的傾向
3. **NC 絶対値は eval set / corpus 質に強く依存**: commandmate 42% baseline は eval question vague に起因 (caveat §6)

---

## 6. Limitation / Caveat (重要)

### 6.1 commandmate eval set の質に関する自己評価

本 PR で hand-craft した 15 sessions × 6 turns の eval set には以下の問題がある:

1. **Q1 が abstract**: 「全体構造を教えてください」「Worktree 機能はどう実装されていますか?」など、specific chunk に直接紐付かない概念質問が多い。baseline NC=66.7% は eval set の質自体に起因する可能性が大。
2. **reference_chunk_ids 未設定**: 各 turn の正解 chunk ID が空、自動 grading が機能しない (NC ベースの判定のみ可)。
3. **15 sessions のみ**: Phase A.1/A.2 の 30 sessions と比べて N が小さい、ばらつきの影響が大きい。
4. **生成方法**: hand-craft で人間が書いたもので、現実のユーザー質問分布と乖離する可能性。

### 6.2 retrieval 失敗の連鎖

T1 で baseline が 66.7% NC = 過半の質問で retrieval 失敗 → T2 以降 working memory に context 不足の chunks が混入 → PHOTON はこの状態で更に drift。これは **PHOTON の弱点を強調する eval 条件** であり、real-world workload とは異なる可能性。

### 6.3 corpus の質

commandmate は active development 中の repo で、**実装と docs の乖離** が大きい。docs が古い箇所 (例: agent 切り替え機能の記述) があり、retrieval は新コードを取るが docs が question に答えやすく書かれていない。FastAPI / Institutional は安定 corpus で、この gap が小さい。

### 6.4 single-run のみ (variance 不明)

Phase A.1/A.2 と同様、本 PR も Option B (1 run/cell) で実行。#143 (Qwen nondeterminism) follow-up で 2nd run + variance 取得が望ましい。

### 6.5 ckpt の不一致 (継続)

mulmoclaude 600-step ckpt の val_loss=1.6238 は roadmap.md の 0.4525 と一致しない。実際の "mulmoclaude TS で訓練" の正確な内容も未確認 (state.json に dataset 情報は無い)。

---

## 7. 解釈と次のアクション

### 7.1 訓練ドメイン仮説否定の意味

仮説 (近=PHOTON 勝) は不成立。考えられる原因:

1. **アーキ自体の課題**: PHOTON working memory が **multi-turn で drift**、ドメインと無関係に late turn 品質低下。ckpt 改善 (#135 retrain) では本質改善にならない可能性。
2. **eval set 質の影響**: 特に commandmate で baseline NC=42% という high-NC ベースで PHOTON 効果が hidden。**より specific な eval set** で再測定が必要。
3. **ckpt の訓練データが TS でない**: mulmoclaude の訓練 corpus 構成が不明、TS で実は訓練されていない可能性 (state.json には dataset 情報なし)。

### 7.2 #135 への含意

- **Phase A.1 + A.2 の結論**: "#135 (Japanese institutional retrain) で品質改善が期待される" → 本 A.3 で **疑念**
- 訓練ドメイン適合だけでは PHOTON 改善しないなら、**#135 は domain adaptation だけでなくアーキ調整も検討すべき**
- 具体的には: PHOTON working memory の drift control (Safe RecGen 強化) を retrain と並行して検討

### 7.3 推奨フォローアップ

1. **commandmate eval set 改善**: より specific で reference_chunk_ids 設定済の eval set を生成、再測定
2. **mulmoclaude ckpt の訓練 corpus 確認**: state.json / training scripts から dataset 構成を特定
3. **PHOTON drift metrics 取得**: 本 PR で未測定の latent drift / logit KL / topic shift を 3 dataset で並べて比較、drift の dataset 依存性を特定
4. **#135 設計再検討**: retrain alone vs retrain + Safe RecGen 強化 vs アーキ変更、の選択肢を比較

---

## 8. Phase A.3 実行記録

| run | start | end | duration | output | turns | NC | P50 |
|-----|-------|-----|----------|--------|-------|-----|-----|
| ingest + index | 19:55 | 20:00 | 5 min | data/indexes/commandmate/ | 5,737 chunks | — | — |
| eval set 生成 | 20:00 | 20:01 | 1 min | data/eval_sets/commandmate_multi_turn_eval.jsonl | 90 | — | — |
| baseline run 1 | 20:01 | 20:51 | 50 min | logs/phase_a3_baseline_run1.predictions.jsonl | 90 | 42.22% | 12,473 ms |
| PHOTON run 1 | 20:51 | 21:30 | 39 min | logs/phase_a3_photon_run1.predictions.jsonl | 90 | 62.22% | 6,854 ms |

---

## 9. 次のアクション

- [x] Phase A.3 完了 (本 PR)
- [ ] commandmate eval set 改善 (specific question + reference_chunk_ids、別 follow-up)
- [ ] mulmoclaude ckpt 訓練 corpus の確認 (state.json / training script からの dataset 特定)
- [ ] PHOTON drift metrics 取得 (3 dataset で並べた比較、別 follow-up)
- [ ] #135 設計再検討 (retrain だけで品質改善するかの再検証、アーキ調整の必要性検討)
- [ ] Phase B (新 LLM Qwen3.x / Gemma4 baseline-only eval) 検討
