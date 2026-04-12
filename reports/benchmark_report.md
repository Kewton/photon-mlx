# Benchmark Report

- Date: 2026-04-12
- Repo: fastapi/fastapi @ eba8942c
- Model: mlx-community/Qwen2.5-Coder-14B-Instruct-4bit
- Platform: Mac Studio M3 Ultra
- Eval set: 120 static + 30 multi-turn sessions + 8 stress sessions
- Freeze: reports/benchmark_freeze.json

---

## Baseline Results (20 questions, onboarding category)

| Metric | 4q sample | 20q run |
|---|---|---|
| Latency P50 | 20,760 ms | **17,585 ms** |
| Latency P90 | 26,179 ms | **26,585 ms** |
| Latency mean | 21,137 ms | **19,162 ms** |
| Retrieval P50 | 47 ms | **41 ms** |
| Generation P50 | 20,704 ms | **17,544 ms** |
| Memory peak (max) | 91.0 MB | **90.9 MB** |
| Memory peak P50 | 19.0 MB | **19.0 MB** |
| No-citation rate | 25% (1/4) | **35% (7/20)** |
| Wrong citation count | 0 | **0** |

### Observations

- **Generation dominates latency**: retrieval P50 is 41ms vs generation P50 17.5s (99.8% of total)
- **No-citation rate 35%** is concerning — model answers correctly but omits `[C:N]` references in about 1/3 of cases
- **Wrong citation remains 0**: when the model does cite, it points to valid chunks
- **Latency range is wide**: 9.8s to 29.2s depending on answer length
- **Memory is stable** after warmup: P50 ~19 MB per turn

### Latency Breakdown

| Phase | P50 (ms) | % of total |
|---|---|---|
| Retrieval | 41 | 0.2% |
| Graph expansion | ~1 | 0.0% |
| Evidence pack | ~0.3 | 0.0% |
| Generation | 17,544 | 99.8% |
| Citation resolve | ~0.3 | 0.0% |

### Key Takeaways

1. **PHOTON 最適化の主要ターゲットは generation フェーズ**
2. **No-citation rate 35% の改善が最優先** — prompt engineering or post-processing
3. **Wrong citation 0 は堅い** — citation の precision は高い
4. **Follow-up ターンの retrieval は高速** (warmup 後 41ms)

---

## PHOTON Tiny Training Results

| Metric | Value |
|---|---|
| Parameters | 78,970,240 (~79M) |
| Config | paper-conformal downscale |
| Steps | 100 (quick test) |
| Initial loss | 10.56 |
| Final loss | 3.13 |
| Reduction | 70.4% |
| Val loss | 3.16 |
| Training time | 23.4s |

---

## Comparison Matrix

| Variant | Latency P50 | Memory P50 | No-cite rate | Wrong cite |
|---|---|---|---|---|
| Baseline-RAG | 17,585 ms | 19 MB | 35% | 0% |
| Baseline-RAG + summary memory | TBD | TBD | TBD | TBD |
| PHOTON-RAG | TBD | TBD | TBD | TBD |
| PHOTON-RAG + Safe RecGen | TBD | TBD | TBD | TBD |

---

## Next Steps

1. Full 120 問 static eval
2. Multi-turn 30 session eval
3. No-citation rate 改善（prompt 強化）
4. PHOTON-RAG end-to-end 統合
5. Comparison matrix 完成 → Gate 2 判定
