# Benchmark Report

- Date: 2026-04-13 (updated)
- Repo: fastapi/fastapi @ eba8942c
- Model: mlx-community/Qwen2.5-Coder-14B-Instruct-4bit
- Platform: Mac Studio M3 Ultra
- Eval set: 120 static + 30 multi-turn sessions + 8 stress sessions
- Freeze: reports/benchmark_freeze.json
- Index: 1,223 files / 2,707 chunks (翻訳ドキュメント除外後, Issue #7)

---

## Full Baseline Results (2026-04-13, Issue #7 修正後)

### Static 120問

| Metric | Value |
|---|---|
| Total questions | 120 |
| No-citation rate | **54.2% (65/120)** |
| Wrong citation | **0** |
| Latency P50 | 18,774 ms |
| Latency P90 | 40,064 ms |
| Latency mean | 23,256 ms |
| Retrieval P50 | 25 ms |
| Generation P50 | 18,378 ms |
| Memory P50 | 18.8 MB |

#### Per-category

| Category | No-cite rate | Questions |
|---|---|---|
| onboarding | **30.0%** (9/30) | 最も良好 |
| bug_localization | **40.0%** (12/30) | |
| impact_analysis | **63.3%** (19/30) | |
| change_planning | **83.3%** (25/30) | 最も悪い |

### Multi-turn 30 Sessions (180ターン)

| Metric | Value |
|---|---|
| Total turns | 180 |
| No-citation rate | **43.3% (78/180)** |
| Wrong citation | **2** |
| First turn no-cite | 30.0% (9/30) |
| Follow-up no-cite | 46.0% (69/150) |
| Sessions all-cited | 1/30 |
| Latency P50 | 17,066 ms |
| Latency mean | 20,744 ms |

#### Per-turn

| Turn | No-cite rate |
|---|---|
| T1 | 30.0% (9/30) |
| T2 | 43.3% (13/30) |
| T3 | 56.7% (17/30) |
| T4 | 30.0% (9/30) |
| T5 | 50.0% (15/30) |
| T6 | 50.0% (15/30) |

### Observations

- **No-citation rate は質問カテゴリに強く依存**: onboarding 30% vs change_planning 83%
- **Generation が latency の 99.8% を占める**: retrieval P50 25ms vs generation P50 18.4s
- **Wrong citation は極めて低い** (0/120 static, 2/180 MT): citation の precision は高い
- **Multi-turn の follow-up は citation 率が低下** (30% → 46%): session memory の蓄積に伴い citation が省略される傾向
- **change_planning カテゴリは抽象度が高い質問が多く** citation なし回答になりやすい

---

## Historical Comparison (20問 onboarding)

| Metric | Initial (pre-#7) | Post-#7 | Full 120q (onb) |
|---|---|---|---|
| No-citation rate | 35% (7/20) | 25% (5/20) | 30% (9/30) |
| Latency P50 | 17,585 ms | 11,497 ms | 30,466 ms* |
| Wrong citation | 0 | 0 | 0 |

*Full 120問は他カテゴリと同一セッションで逐次実行のため、model warmup 効果が異なる

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

## Comparison Matrix — 4-Variant Final (2026-04-14)

**Run ID**: `bench_20260414_130853_c81af3`

### Static Eval 120問

| Variant | No-citation | P50 (ms) | Mean (ms) | P90 (ms) |
|---------|-------------|----------|-----------|----------|
| baseline_rag | 44.2% | **11,291** | 12,939 | 20,795 |
| baseline_rag + summary_memory | 40.0% | 12,168 | 13,068 | 21,773 |
| photon_rag | 42.5% | 11,738 | 13,325 | 22,380 |
| photon_rag + safe_recgen | 40.8% | 12,566 | 13,255 | 21,013 |

### Multi-turn Eval (180 turns)

| Variant | No-citation | Follow-up P50 | 状態 |
|---------|-------------|---------------|------|
| baseline_rag | 30.6% | **13,727 ms** | ✅ 完了 |
| baseline_rag + summary_memory | 32.2% | 13,857 ms | ✅ 完了 |
| photon_rag | — | — | ❌ MT バグで実行不可 |
| photon_rag + safe_recgen | — | — | ❌ MT バグで実行不可 |

### Gate 2 判定: **No-Go**

- 必須条件「follow-up latency baseline 比 −30%」は測定不可
- Static では 4 variants 間で実効差なし（P50 +4%, no-citation +/-2pt）
- **詳細**: [gate2_judgment.md](gate2_judgment.md)

---

## Run ID Reference

| Run | ID | Scope |
|-----|-----|-------|
| Static 20q (pre-#7) | (initial, no run_id) | onboarding 20q |
| Static 20q (post-#7) | `baseline_eval_fastapi_fastapi_20260413_093254` | onboarding 20q |
| MT 30sess (post-#7) | `mt_eval_fastapi_fastapi_20260413_094621` | 30 sessions |
| **Static 120q (full)** | `baseline_eval_fastapi_fastapi_20260413_135703` | **full 120q** |
| **MT 30sess (full)** | `mt_eval_fastapi_fastapi_20260413_135708` | **full 30 sessions** |

---

## Next Steps

1. ~~Full 120 問 static eval~~ ✅ 完了
2. ~~Multi-turn 30 session eval~~ ✅ 完了
3. Failure cases 分析 (Issue #6) — no-citation の根本原因分類
4. No-citation rate 改善 (Issue #1) — データ駆動で再設計
5. PHOTON-RAG end-to-end 統合
6. Comparison matrix 完成 → Gate 2 判定
