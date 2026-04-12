# Benchmark Report

- Date: 2026-04-12
- Repo: fastapi/fastapi @ eba8942c
- Model: mlx-community/Qwen2.5-Coder-14B-Instruct-4bit
- Platform: Mac Studio M3 Ultra
- Eval set: 120 static + 30 multi-turn sessions + 8 stress sessions
- Freeze: reports/benchmark_freeze.json

---

## Baseline Sample Results (4 questions)

| Metric | Value |
|---|---|
| Total turns | 4 |
| Latency P50 | 20,760 ms |
| Latency P90 | 26,179 ms |
| Latency mean | 21,137 ms |
| Retrieval P50 | 47 ms |
| Generation P50 | 20,704 ms |
| Memory peak (first turn) | 91.0 MB |
| Memory peak (mean) | 37.0 MB |
| No-citation rate | 25% (1/4) |
| Wrong citation count | 0 |

### Observations

- **Generation dominates latency**: retrieval is <1% of total after warmup (P50 47ms vs 20.7s generation)
- **First turn is slowest**: model loading + embedding model warmup adds ~6s
- **No-citation rate is high** in the sample (1/4): the onboarding question about dependency injection got no citations. This is a known weakness for broad structural questions where the retrieval may not surface the right chunks.
- **Wrong citation is 0**: when citations are produced, they point to valid chunks

### Latency Breakdown

| Phase | P50 (ms) | % of total |
|---|---|---|
| Retrieval | 47 | 0.2% |
| Graph expansion | ~1 | 0.0% |
| Evidence pack | ~0.3 | 0.0% |
| Generation | 20,704 | 99.7% |
| Citation resolve | ~0.3 | 0.0% |

### Key Takeaways

1. **PHOTON 最適化の主要ターゲットは generation フェーズ** — retrieval は既に十分高速
2. **Follow-up ターンでの retrieval 高速化は確認済み** (6.2s → 47ms: embedding warmup 効果)
3. **No-citation rate の改善** が品質上の最優先課題
4. **Full 120 問の評価** で no_citation_rate と citation_precision を確定させる必要あり

---

## Comparison Matrix (placeholder)

| Variant | Latency P50 | Memory P50 | Citation Precision | Task Score |
|---|---|---|---|---|
| Baseline-RAG | 20,760 ms | 19 MB | TBD | TBD |
| Baseline-RAG + summary memory | - | - | - | - |
| PHOTON-RAG | - | - | - | - |
| PHOTON-RAG + Safe RecGen | - | - | - | - |

---

## Next Steps

1. Run full 120-question static eval
2. Run 30-session multi-turn eval
3. Run stress eval (8 concurrent sessions)
4. Fill comparison matrix
5. Gate 2 判定
