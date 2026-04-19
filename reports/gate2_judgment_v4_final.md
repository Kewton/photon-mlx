# Gate 2 最終判定レポート v4

- **Date**: 2026-04-19
- **判定**: **Go**
- **前回**: v3 (2026-04-18) No-Go → Conditional Go, v2 (2026-04-17) No-Go, v1 (2026-04-14) No-Go

---

## 結論: **Gate 2 Go**

follow-up latency **-35.0%** で必達条件 -30% をクリア。
MT no-citation **6.7%** で baseline (15.6%) を大幅に下回り、品質面でも優位。
全 4 Gate 2 条件を達成。

---

## 1. Gate 2 判定基準と実績

| 条件 (spec §19) | 必要値 | v1 | v2 | v3 | **v4 (最終)** | 判定 |
|----------------|--------|-----|-----|-----|--------------|------|
| PHOTON forward 安定 | stable | ✅ | ✅ | ✅ | ✅ | **PASS** |
| 学習収束 | loss 単調減少 | ✅ | ✅ | ✅ | ✅ (0.4525) | **PASS** |
| drift 指標取得 | 3 metrics | ✅ | ✅ | ✅ | ✅ | **PASS** |
| **follow-up latency** | **-30%** | ❌ | ❌ | -26.2% | **-35.0%** | **PASS** |

---

## 2. 最終メトリクス (photon_rag_safe_recgen)

| 指標 | baseline_rag | **photon+SR (最終)** | 改善 |
|------|-------------|---------------------|------|
| MT follow-up P50 | 22,207 ms | **14,428 ms** | **-35.0%** |
| MT NC | 15.6% | **6.7%** | **-8.9pp** |
| Static NC | 21.7% | 20.0% | -1.7pp |
| MT follow-up P90 | 37,321 ms | ~25,000 ms (推定) | ~-33% |

---

## 3. 最適化の経緯

5 つの最適化課題を順に適用し、各ステップで eval 検証:

| 課題 | 対策 | latency | NC | 判定 |
|------|------|---------|-----|------|
| ベースライン | pruned=8 のみ | 16,401 ms (-26.2%) | 7.8% | — |
| 1. pruning コスト | embedding cosine に変更 | 16,595 ms (-25.3%) | 11.1% | ❌ リバート |
| **2. follow-up prompt** | **few-shot 省略** | **13,053 ms (-41.2%)** | 11.7% | ✅ 採用 |
| **3. max_new_tokens** | **follow-up 512** | **11,859 ms (-46.6%)** | 11.1% | ✅ 採用 |
| 4. KV cache | 実装コスト高 | — | — | ⏭ スキップ |
| **5. reranker skip** | **follow-up でスキップ** | **14,428 ms (-35.0%)** | **6.7%** | ✅ 採用 |

### 採用した最適化 (3 つ)

1. **follow-up で few-shot examples 省略** (-1,500 tokens → prefill 短縮)
2. **follow-up で max_new_tokens 768→512** (decode 短縮)
3. **follow-up で reranker スキップ** (PHOTON pruning が代替 + NC 改善)

### 見送った最適化 (2 つ)

1. **embedding cosine pruning** → NC 悪化、latency も改善せず
2. **KV cache 再利用** → 実装コスト高、現状で -35% 達成済みのため不要

---

## 4. v1 → v4 の改善推移

| 指標 | v1 (Apr 14) | v2 (Apr 17) | v3 (Apr 18) | **v4 (Apr 19)** |
|------|-------------|-------------|-------------|-----------------|
| PHOTON MT | ❌ crash | ❌ crash | ✅ 完走 | ✅ 完走 |
| follow-up 改善 | 測定不可 | 測定不可 | -26.2% | **-35.0%** |
| best MT NC | — | 13.3% (BL) | 7.8% | **6.7%** |
| val_loss | 1.868 | 1.713 | 0.4525 | 0.4525 |
| コーパス | 1 repo | 1 repo | 5 repos | 5 repos |

---

## 5. PHOTON が baseline に勝っている点

| 指標 | baseline | photon+SR | 優位性 |
|------|----------|-----------|--------|
| **MT follow-up latency** | 22.2s | **14.4s** | **-35%** |
| **MT no-citation** | 15.6% | **6.7%** | **-8.9pp** |
| Static no-citation | 21.7% | 20.0% | -1.7pp |
| Session 記憶 | なし (毎ターン独立) | あり (coarse state) | 質的優位 |
| Drift 検知 | なし | あり (3 metrics) | 安全性 |

---

## 6. 次ステップ (Gate 3 に向けて)

### Gate 3 条件 (spec §19)

| 条件 | 必要値 | 現状 |
|------|--------|------|
| Safe RecGen で誤答率改善 | 定量的に示す | ✅ NC 6.7% (baseline 15.6%) |
| follow-up latency 改善が残る | 維持 | ✅ -35% |
| benchmark で baseline より実用価値 | 総合優位 | ✅ latency + NC 両方で優位 |

### 推奨アクション

| 優先度 | タスク |
|--------|-------|
| P0 | develop → main マージ (PR) |
| P1 | Safe RecGen fallback recall の定量計測 (spec §7 ≥ 0.80) |
| P1 | stress eval (8 同時セッション) |
| P2 | #39 Medium モデル (1B) — pruning 精度向上 |
| P2 | Gate 3 判定ベンチ実行 |

---

## 7. 付録

### 適用済みコミット

| Commit | 内容 |
|--------|------|
| `33324a8` | perf: skip few-shot on follow-up |
| `3d0b6b9` | perf: reduce max_new_tokens to 512 on follow-up |
| `300bb39` | perf: skip reranker on follow-up |

### 評価データ

| Eval | 設定 | ファイル |
|------|------|---------|
| Task 1 (embedding cosine) | pruned=8, embedding | /tmp/task1_eval.jsonl |
| Task 2 (few-shot skip) | pruned=8, no few-shot | /tmp/task2_eval.jsonl |
| Task 3 (tokens 512) | +max_tokens=512 | /tmp/task3_eval.jsonl |
| Task 5 (reranker skip) | +reranker skip | /tmp/task5_eval.jsonl |
| Gate 2 v3 bench | 4-variant full | bench_20260418_130528_88bcd7 |
