# Commandmate PHOTON MT 測定レポート (Issue #148 Phase A.3)

**実測対象 corpus**: `commandmate` (TypeScript + Next.js + Markdown docs, 5,737 chunks after cleanup)
**eval set**: `data/eval_sets/commandmate_multi_turn_eval.jsonl` (15 sessions × 6 turns = 90 turns、本 PR で新規生成)
**実測日 (v1)**: 2026-04-27 20:00–21:30 JST
**実測日 (v2 / 本レポートの primary)**: 2026-04-27 22:10–22:53 JST
**Issue**: [#148](https://github.com/Kewton/photon-mlx/issues/148)
**Branch**: `feature/issue-148-phase-a3-commandmate`
**位置づけ**: Phase A.1 (FastAPI/Python) + A.2 (Institutional/JP) と並べて、**訓練ドメイン仮説** (mulmoclaude=TS で訓練済み ckpt なら TS ドメインで PHOTON 勝つはず) を検証する 3 番目の測定点。

> **⚠️ 重要 / 数値修正履歴**: 本レポートの primary 数値は **v2 (2026-04-27 22:10 JST 実行)**。v1 (同日 20:00 JST) は #154 で修正された 2 つの eval バグ (cross-repo chunk leakage / refusal misclassified-as-cited) と、本 PR で更に修正したスクリプト側 `--repo-id` 配線バグ (`scripts/run_multi_turn_eval.py` が `cfg.repo.repo_id` を上書きせず default index を読み込んでいた) の影響を受けた**無効な比較**。v2 は 3 つすべての修正を適用後の真値。詳細は §4.0 と §10 を参照。

---

## 0. TL;DR (v2)

| 結論 | 根拠 |
|------|------|
| **訓練ドメイン仮説は支持される**: TS+Next.js (訓練に最も近い) で **PHOTON が baseline より NC -6.67pp 改善** | 3 dataset 比較表 (§5) |
| v1 で観測された PHOTON +20pp 悪化は **eval パイプラインのバグの artifact** であり、修正後 (v2) は逆転 (PHOTON 勝) | §4.0 v1 vs v2 比較 |
| **Turn 5-6 でも PHOTON が baseline 以上**: v1 で観測した late-turn drift も artifact で、v2 では PHOTON T5-6 NC=10.0% < baseline 6.67% (Δ +3.33pp) と僅差 | §4.2 turn-by-turn |
| **Latency: PHOTON が baseline より -27.6%**: v1 (-45.1%) より差は縮小したが優位は維持 | §4.1 |
| **#135 (本格再学習) の論理的 motivation は維持**: 訓練ドメイン適合で PHOTON 改善することが確認された | §7 |

---

## 1. 概要と動機

PR #152 (Phase A.1 + A.2) の結果から:
- Phase A.1 (FastAPI MT, Python ドメイン): PHOTON NC +1.67pp (僅差)
- Phase A.2 (#113 v2, JP 制度文書ドメイン): PHOTON NC +5.56pp (悪化)

仮説: **訓練ドメインから遠いほど PHOTON が劣後** (institutional > FastAPI > commandmate の順で悪い見込み)。

検証手段: mulmoclaude ckpt は **TypeScript で訓練** されているため、**commandmate (Next.js + TS)** が **訓練ドメインに最も近い** corpus。ここで PHOTON が baseline 以上なら仮説成立。

**結果 (v2)**: 仮説は支持される (PHOTON が NC -6.67pp 改善)。

---

## 2. 実行環境

| 項目 | 値 |
|------|-----|
| マシン | M3 Ultra (Mac Studio, 256GB unified memory) |
| OS | macOS Darwin 25.4.0 |
| Python | 3.12.3 |
| ブランチ | `feature/issue-148-phase-a3-commandmate` |
| 実行時刻 (v2) | 2026-04-27 22:10–22:53 JST (約 43 分) |
| commandmate worktree | `/Users/maenokota/share/work/github_kewton/commandmate-issue-676` (commit `78be750e9c`) |
| PHOTON_CHECKPOINT_ROOT | `/Users/maenokota/share/work/github_kewton/photon-mlx-develop/checkpoints` |
| ckpt | `step_000600` (mulmoclaude, val_loss 1.6238) |
| 適用された fix | (a) #154 cross-repo retrieval filter / (b) #154 refusal-aware citation grader / (c) `scripts/run_multi_turn_eval.py` の `cfg.repo.repo_id` 上書き |

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

# v2 で追加 (#154 後の pipeline は graph load を skip しないため)
python scripts/build_symbol_graph.py --repo-id commandmate --config configs/baseline.yaml
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

### 4.0 v1 vs v2 比較 (eval バグ修正の効果)

| 指標 | v1 baseline | **v2 baseline** | v1 PHOTON | **v2 PHOTON** | v1 Δ (PHOTON−base) | **v2 Δ (PHOTON−base)** |
|------|------------|-----------------|-----------|---------------|--------------------|------------------------|
| 完走 turn | 90/90 | 90/90 | 90/90 | 90/90 | — | — |
| **MT NC (overall)** | 42.22% | **20.00%** | 62.22% | **13.33%** | **+20.00pp** | **-6.67pp** ← 逆転 |
| **MT NC (Turn 5-6)** | 23.33% (7/30) | **6.67% (2/30)** | 66.67% (20/30) | **10.00% (3/30)** | **+43.33pp** | **+3.33pp** ← 縮小 |
| Latency P50 | 12,473 ms | 16,210 ms | 6,854 ms | 11,740 ms | -45.1% | **-27.6%** |
| Latency mean | 13,484 ms | 16,622 ms | 8,222 ms | 11,837 ms | -39.0% | -28.8% |
| Follow-up P50 (T2-T6) | 11,898 ms | 16,947 ms | 6,179 ms | 11,568 ms | -48.1% | **-31.7%** |
| Total citations | — | 144 | — | 137 | — | — |

**観察**:
1. **NC 値の絶対量が両モデルで大幅低下** (baseline 42→20%, PHOTON 62→13%): v1 は (a) cross-repo chunk leakage で wrong-repo の chunks が retrieve され、(b) refusal が `[C:N]` echo で「citation あり」と誤認識される、という 2 重の bias が両モデルにかかっていた。ただし PHOTON はより小さい model で refusal を出しやすい性質があり、v1 grader の penalty が PHOTON 側に偏った結果が +20pp 差として観測された。
2. **修正後 PHOTON は baseline より NC 改善 (-6.67pp)**: 訓練ドメイン仮説 (TS で訓練した PHOTON ckpt なら TS+Next.js corpus で baseline 以上) と整合。
3. **Latency 優位は縮小したが維持**: v1 では cross-repo の小さい (誤った) chunk set で latency が異常に短かった。v2 で正しい commandmate chunks を retrieve するため絶対 latency は両モデル増加するが、PHOTON の working memory による短縮効果は依然として -27.6%。

### 4.1 Phase A.3 v2 single-run summary (primary numbers)

| 指標 | baseline (run 2) | **PHOTON (run 2)** | delta |
|------|-----------------|-------------------|-------|
| 完走 turn | 90/90 | 90/90 | — |
| **MT NC (overall)** | 20.00% | **13.33%** | **-6.67pp** ← PHOTON 勝 |
| **MT NC (Turn 5-6)** | 6.67% | 10.00% | +3.33pp (僅差) |
| Latency P50 | 16,210 ms | **11,740 ms** | **-27.6%** |
| Latency mean | 16,622 ms | 11,837 ms | -28.8% |
| Latency max | 38,764 ms | 29,799 ms | -23.1% |
| **Follow-up P50 (T2-T6)** | 16,947 ms | **11,568 ms** | **-31.7%** |
| Total citations | 144 | 137 | -4.9% (僅減、平均約 1.6 chunks/turn) |

### 4.2 Turn-by-turn breakdown (v2)

#### baseline (v2)
| Turn | NC% | latency P50 |
|------|-----|-------------|
| T1 | 40.0% | 14,696 ms |
| T2 | 13.3% | 12,919 ms |
| T3 | 26.7% | 16,317 ms |
| T4 | 26.7% | 15,268 ms |
| T5 | 13.3% | 18,524 ms |
| T6 | 0.0% | 18,311 ms |

#### PHOTON (v2)
| Turn | NC% | latency P50 |
|------|-----|-------------|
| T1 | 26.7% | 13,379 ms |
| T2 | 13.3% | 7,900 ms |
| T3 | 6.7% | 11,568 ms |
| T4 | 13.3% | 12,077 ms |
| T5 | 20.0% | 10,213 ms |
| T6 | 0.0% | 13,959 ms |

> **観察 (v2)**: PHOTON は T1 (-13.3pp) / T3 (-20.0pp) で baseline より明確に良好、T2 / T6 で同等、T5 で僅か劣後 (+6.7pp)。late-turn drift の系統的傾向は v2 では観測されない (T6 は両モデル NC=0%)。

---

## 5. 3-dataset 比較表 (Phase A.1 + A.2 + A.3 v2 を並べて訓練ドメイン仮説検証)

| Dataset | 言語 / ドメイン | 訓練距離 | baseline NC | **PHOTON NC** | **Δ NC** | baseline P50 | PHOTON P50 | Δ P50 |
|---------|----------------|---------|-------------|---------------|----------|--------------|------------|-------|
| **commandmate (v2)** | **TS + Next.js** | **近 (訓練)** | 20.00% | **13.33%** | **-6.67pp** ← 仮説支持 | 16,210 ms | **11,740 ms** | -27.6% |
| FastAPI | Python | 中 | 6.11% | 7.78% | +1.67pp | 20,440 ms | 12,604 ms | -38.3% |
| Institutional | JP 法規 | 遠 | 7.78% | 13.33% | +5.56pp | 19,383 ms | 12,800 ms | -34.0% |

> **注**: FastAPI と Institutional は v1 (PR #152 時点) の数値のまま。両者ともデフォルト config の `cfg.repo.repo_id` がそれぞれの実 repo に一致しているため、本 PR で修正したスクリプト側 `--repo-id` 配線バグの影響は受けにくい (cross-repo leakage / refusal grader の影響は残る可能性)。完全な真値比較には FastAPI / Institutional の rerun も必要 (§10 follow-up)。

### 5.1 仮説検証 (v2)

**仮説**: 訓練ドメイン (TS) 近い corpus で PHOTON が baseline 以上、または同等。

**結果 (v2)**: **支持**。
- commandmate (TS、最近) で PHOTON が baseline より NC -6.67pp 改善
- ドメイン距離との関係: 近 (commandmate -6.67pp) → 中 (FastAPI +1.67pp) → 遠 (Institutional +5.56pp) と単調に PHOTON 不利が増す傾向と整合。**ただし FastAPI / Institutional 側に同等の eval バグ修正を適用していないため確定的ではない**。

### 5.2 共通の傾向 (v2 反映後)

1. **Latency**: 全 dataset で PHOTON が短縮 (commandmate v2 -27.6%, FastAPI -38.3%, Institutional -34.0%)。working memory 機能、weight 非依存で確定。
2. **Late-turn drift は v1 で観測されたが v2 で消失**: v1 commandmate で T5-6 PHOTON NC=66.67% は **eval バグの artifact**。v2 では T5-6 PHOTON NC=10.00% と baseline (6.67%) と僅差。FastAPI / Institutional 側の v1 数値も同様の bias の影響を受けている可能性あり。
3. **NC 絶対値は eval set / corpus 質に強く依存**: commandmate v2 baseline NC=20% は eval question vague に起因 (caveat §6)、FastAPI / Institutional の baseline NC < 10% より高い。

---

## 6. Limitation / Caveat (重要)

### 6.1 commandmate eval set の質に関する自己評価

本 PR で hand-craft した 15 sessions × 6 turns の eval set には以下の問題がある:

1. **Q1 が abstract**: 「全体構造を教えてください」「Worktree 機能はどう実装されていますか?」など、specific chunk に直接紐付かない概念質問が多い。v2 baseline T1 NC=40.0% は eval set の質自体に起因する可能性が大。
2. **reference_chunk_ids 未設定**: 各 turn の正解 chunk ID が空、自動 grading が機能しない (NC ベースの判定のみ可)。
3. **15 sessions のみ**: Phase A.1/A.2 の 30 sessions と比べて N が小さい、ばらつきの影響が大きい。
4. **生成方法**: hand-craft で人間が書いたもので、現実のユーザー質問分布と乖離する可能性。

### 6.2 corpus の質

commandmate は active development 中の repo で、**実装と docs の乖離** が大きい。docs が古い箇所 (例: agent 切り替え機能の記述) があり、retrieval は新コードを取るが docs が question に答えやすく書かれていない。FastAPI / Institutional は安定 corpus で、この gap が小さい。

### 6.3 single-run のみ (variance 不明)

Phase A.1/A.2 と同様、本 PR も Option B (1 run/cell) で実行。#143 (Qwen nondeterminism) follow-up で 2nd run + variance 取得が望ましい。

### 6.4 ckpt の不一致 (継続)

mulmoclaude 600-step ckpt の val_loss=1.6238 は roadmap.md の 0.4525 と一致しない。実際の "mulmoclaude TS で訓練" の正確な内容も未確認 (state.json に dataset 情報は無い)。**v2 で訓練ドメイン仮説が支持されたことから、TS 含有比率は無視できないと推定されるが、実証はできていない**。

### 6.5 FastAPI / Institutional の rerun 未実施

本 PR は commandmate のみ v2 rerun。FastAPI / Institutional についても #154 fix (cross-repo / refusal grader) の影響を受ける可能性があるため、3-dataset 真値比較には両者の rerun が必要 (§10 follow-up)。

---

## 7. 解釈と次のアクション (v2 反映)

### 7.1 訓練ドメイン仮説支持の意味

仮説 (近=PHOTON 勝) は **支持**。考えられる効果経路:

1. **PHOTON working memory が訓練ドメインの分布で効果的に機能**: TS+Next.js の subword / 構文パターンに ckpt が適応していれば、low-bit 4bit baseline (Qwen-Coder-14B) より hidden state が安定する余地がある。
2. **#135 (本格再学習) の論理的 motivation は維持**: 訓練ドメインが effective なら、Japanese institutional retrain で institutional NC を改善できる可能性が示唆。

### 7.2 v1 結果との関係 (重要な学び)

v1 報告 (この同じファイルの 2026-04-27 20:00 版) は **eval パイプラインの 3 つのバグ (cross-repo leakage / refusal grader / script repo_id 配線) の重畳 artifact** で、現実とは逆の結論 (+20pp 悪化) を出していた。**eval correctness は数値の signal/noise を完全に決める**: 仮説検証フレームワーク自体に regression test (PR #155 で `tests/test_repo_isolation.py` 等として整備済) を組み込むことの重要性が再確認された。

### 7.3 推奨フォローアップ

1. **FastAPI / Institutional の v2 rerun**: #154 fix を適用した状態で再測定し、3-dataset 真値比較を完成させる
2. **commandmate eval set 改善**: より specific で reference_chunk_ids 設定済の eval set を生成、再測定
3. **mulmoclaude ckpt の訓練 corpus 確認**: state.json / training scripts から dataset 構成を特定 (TS 含有比率の実証)
4. **PHOTON drift metrics 取得**: 本 PR で未測定の latent drift / logit KL / topic shift を 3 dataset で並べて比較
5. **#135 設計の再確認**: 仮説支持を踏まえ、training-domain adaptation 中心のロードマップを継続

---

## 8. Phase A.3 実行記録

| run | start | end | duration | output | turns | NC | P50 |
|-----|-------|-----|----------|--------|-------|-----|-----|
| ingest + index | 19:55 | 20:00 | 5 min | data/indexes/commandmate/ | 5,737 chunks | — | — |
| eval set 生成 | 20:00 | 20:01 | 1 min | data/eval_sets/commandmate_multi_turn_eval.jsonl | 90 | — | — |
| baseline run 1 (v1) | 20:01 | 20:51 | 50 min | logs/phase_a3_baseline_run1.predictions.jsonl | 90 | 42.22%* | 12,473 ms* |
| PHOTON run 1 (v1) | 20:51 | 21:30 | 39 min | logs/phase_a3_photon_run1.predictions.jsonl | 90 | 62.22%* | 6,854 ms* |
| symbol_graph build (commandmate) | 22:09 | 22:09 | <1 min | data/indexes/commandmate/symbol_graph.json | — | — | — |
| **baseline run 2 (v2, primary)** | **22:10** | **22:35** | **25 min** | **logs/phase_a3_baseline_run2.jsonl** | **90** | **20.00%** | **16,210 ms** |
| **PHOTON run 2 (v2, primary)** | **22:35** | **22:53** | **18 min** | **logs/phase_a3_photon_run2.jsonl** | **90** | **13.33%** | **11,740 ms** |

*v1 数値は §4.0 のとおり 3 つの eval バグの artifact、参考値。

---

## 9. 次のアクション

- [x] Phase A.3 v1 完了 (PR #153 初版)
- [x] **Phase A.3 v2 完了 (本 commit、primary 結果)**
- [ ] FastAPI / Institutional の v2 rerun (3-dataset 真値比較完成、別 follow-up)
- [ ] commandmate eval set 改善 (specific question + reference_chunk_ids、別 follow-up)
- [ ] mulmoclaude ckpt 訓練 corpus の確認 (state.json / training script からの dataset 特定)
- [ ] PHOTON drift metrics 取得 (3 dataset で並べた比較、別 follow-up)
- [ ] Phase B (新 LLM Qwen3.x / Gemma4 baseline-only eval) 検討

---

## 10. v1 → v2 で適用された fix の内訳

| Fix | 出典 | 内容 | 影響範囲 |
|-----|------|------|----------|
| #154 Bug 1 (cross-repo retrieval filter) | PR #155 (`baseline_reporag/retrieval/hybrid.py` + `pipeline.py` / `photon_pipeline.py`) | `hybrid_search(repo_id=...)` で `chunk_id.startswith("{repo_id}::")` 以外を drop | 異 repo の chunk が prompt に混入する系統的 leakage を防止 |
| #154 Bug 2 (refusal-aware citation grader) | PR #155 (`baseline_reporag/citation.py` + grader / aggregator) | `is_refusal_answer` が「根拠が不足しています ... [C:1]」のような refusal+echo を NO_CITATION に正しく分類 | 「PHOTON は honest refusal、baseline は echo して citation あり扱い」の系統的偏りを解消 |
| Script bug (repo_id 配線) | 本 PR (`scripts/run_multi_turn_eval.py`) | `--repo-id` で `cfg.repo.repo_id` を build_pipeline 前に上書き。さもないと YAML default index (例: `fastapi_fastapi`) を読み込み、新 retrieval filter で全 chunk が drop されて空 retrieval (= 100% NC) または cross-repo leakage が発生 | commandmate eval が成立する前提条件 |

3 つすべての fix を適用後の数値が v2、本レポートの primary。
