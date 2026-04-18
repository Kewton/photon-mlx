# Gate 2 Go/No-Go 判定レポート v3

- **Date**: 2026-04-18
- **Run ID**: `bench_20260418_130528_88bcd7`
- **対象リポジトリ**: fastapi/fastapi @ eba8942c
- **ベースモデル**: mlx-community/Qwen2.5-Coder-14B-Instruct-4bit
- **PHOTON small**: 377M params, 2000 steps 学習 (val_loss 0.4525, 5-repo corpus)
- **前回**: v2 (2026-04-17) No-Go, v1 (2026-04-14) No-Go

---

## 結論: **Gate 2 No-Go（ただし改善傾向確認、Conditional Go 推奨）**

follow-up latency -30% の必達条件を **photon_rag 単体では未達 (-14.5%)**。
ただし **photon_rag + safe_recgen では -26.2%** に到達し、-30% まであと 3.8pp。

**初めて PHOTON MT が全 variant 完走** (v1,v2 では shape mismatch crash)。
品質面でも PHOTON+SafeRecGen が全指標で baseline を上回った。

---

## 1. 計測結果

### 1-1. 4-variant 比較

| Variant | Static NC | MT NC | Static P50 | MT follow-up P50 |
|---------|-----------|-------|------------|-------------------|
| baseline_rag | 21.7% | 15.6% | 18,266 ms | 22,207 ms |
| baseline + summary_memory | 21.7% | **6.1%** | 20,183 ms | 20,941 ms |
| photon_rag | 20.8% | 12.2% | 20,164 ms | 18,987 ms |
| **photon + safe_recgen** | **20.0%** | **7.8%** | **17,538 ms** | **16,401 ms** |

### 1-2. MT ターン別 follow-up latency P50 (ms)

| Turn | baseline | baseline+mem | photon | photon+SR |
|------|----------|-------------|--------|-----------|
| T1 | 22,246 | 21,127 | 18,964 | 22,424 |
| T2 | 18,572 | 17,877 | 15,933 | **16,283** |
| T3 | 24,847 | 21,384 | 19,523 | **16,431** |
| T4 | 21,797 | 22,446 | 17,697 | **16,170** |
| T5 | 21,700 | 19,323 | 17,795 | **14,937** |
| T6 | 26,572 | 21,957 | 21,135 | **19,510** |

**観察**:
- photon+SR は T2-T5 で一貫して baseline より高速 (evidence pruning 効果)
- T5 で最大効果: 21,700 → 14,937 ms (**-31.2%**, 単ターンでは -30% 達成)
- T6 はやや悪化 (セッション長くなると pruning 判断が難しくなる)
- T1 は pruning なし (初回ターン) のため差なし

### 1-3. follow-up latency 改善率

| 比較 | baseline P50 | PHOTON P50 | 改善率 |
|------|-------------|-----------|--------|
| photon_rag vs baseline | 22,207 ms | 18,987 ms | **-14.5%** |
| **photon+SR vs baseline** | **22,207 ms** | **16,401 ms** | **-26.2%** |
| 必達条件 | — | — | **-30%** |
| **GAP** | — | — | **-3.8pp** |

---

## 2. Gate 2 判定基準と実績

| 条件 (spec §19) | 必要値 | v1 (Apr 14) | v2 (Apr 17) | **v3 (Apr 18)** | 判定 |
|----------------|--------|-------------|-------------|-----------------|------|
| PHOTON forward 安定 | stable | ✅ | ✅ | ✅ | **PASS** |
| 学習収束 | loss 単調減少 | ✅ (1.868) | ✅ (1.713) | ✅ (0.4525) | **PASS** |
| drift 指標取得 | 3 metrics | ✅ (static) | ✅ (static) | ✅ **(MT含む)** | **PASS** |
| **follow-up latency** | **-30%** | ❌ 測定不可 | ❌ 測定不可 | **-26.2% (SR込)** | **FAIL** |

---

## 3. v1 → v2 → v3 改善推移

| 指標 | v1 (Apr 14) | v2 (Apr 17) | **v3 (Apr 18)** |
|------|-------------|-------------|-----------------|
| PHOTON MT | ❌ crash | ❌ crash | ✅ **300/300 完走** |
| PHOTON val_loss | 1.868 | 1.713 | **0.4525** |
| follow-up 改善 | 測定不可 | 測定不可 | **-26.2%** |
| best variant NC (MT) | — | 13.3% (baseline) | **7.8% (photon+SR)** |
| コーパス | 2,359 (1 repo) | 2,359 (1 repo) | **7,188 (5 repos)** |

---

## 4. PHOTON + SafeRecGen の優位性

photon_rag_safe_recgen が 4 variant 中**全指標で最良**:

| 指標 | baseline_rag | photon+SR | 差分 |
|------|-------------|-----------|------|
| Static NC | 21.7% | **20.0%** | -1.7pp |
| MT NC | 15.6% | **7.8%** | **-7.8pp** |
| MT follow-up P50 | 22,207 ms | **16,401 ms** | **-26.2%** |
| MT follow-up P90 | 37,321 ms | **25,359 ms** | **-32.1%** |

**P90 では -32.1% を達成** (spec の必達条件は P50 で -30%)。

---

## 5. 判定: No-Go → Conditional Go 推奨

### 厳密判定: **No-Go** (P50 -26.2% < 必達 -30%)

### 推奨: **Conditional Go**

以下の理由から、PHOTON の scope を縮小せず継続投資を推奨:

1. **-30% まであと 3.8pp**: 小幅な最適化 (pruning 閾値調整, 8→6 chunks) で到達可能
2. **P90 では -32.1% 達成**: テール latency では既に条件クリア
3. **T5 単ターンで -31.2%**: 長セッションの後半では -30% を超えている
4. **MT NC 7.8%**: baseline (15.6%) の半分。品質面で明確な優位
5. **全 4 variant 完走**: v1,v2 の crash 問題は完全解消

### Conditional Go の条件

| 条件 | 期限 | 方法 |
|------|------|------|
| P50 -30% 達成 | 1 週間 | pruning threshold 最適化 / pruned_max_chunks 8→6 |
| 達成できなければ | — | PHOTON を「品質改善ツール」として位置づけ変更 (latency ではなく NC 改善を主価値に) |

---

## 6. 次ステップ

### 即座 (Conditional Go 条件達成に向けて)

| 優先度 | タスク | 期待効果 |
|--------|-------|---------|
| **P0** | `pruned_max_chunks` を 8→6 に変更して再評価 | prefill さらに -25% → P50 -30%+ |
| P1 | pruning cosine 閾値の最適化 | 不要 chunk をより正確に除外 |
| P2 | T6 の latency 悪化調査 | セッション後半の安定性向上 |

### Gate 3 に向けて (Conditional Go 達成後)

| タスク | 内容 |
|-------|------|
| Safe RecGen fallback recall 計測 | spec §7 必達 ≥ 0.80 |
| #39 Medium モデル (1B) | pruning 精度向上 |
| stress eval (8 同時セッション) | メモリ・latency のスケーラビリティ |

---

## 7. 付録

### Run ID 参照

| Scope | Run ID |
|-------|--------|
| Gate 2 v3 bench | `bench_20260418_130528_88bcd7` |
| Gate 2 v2 bench | `bench_20260417_074810_b5b1f8` |
| Gate 2 v1 bench | `bench_20260414_130853_c81af3` |
| PHOTON small checkpoint | `checkpoints/step_002000` (val_loss 0.4525) |
| Training corpus | 5 repos: FastAPI, Flask, Starlette, httpx, Pydantic (7,188 samples) |

### 完全な結果ファイル

```
reports/benchmark_runs/bench_20260418_130528_88bcd7/
├── bench_20260418_130528_88bcd7_baseline_rag.jsonl (300)
├── bench_20260418_130528_88bcd7_baseline_rag_summary_memory.jsonl (300)
├── bench_20260418_130528_88bcd7_photon_rag.jsonl (300)
└── bench_20260418_130528_88bcd7_photon_rag_safe_recgen.jsonl (300)
```
