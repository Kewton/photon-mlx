# 制度文書 PHOTON MT 測定レポート (Issue #113)

**実測対象 corpus**: `institutional_documents` (4228 .md, post-#125+#126 状態)
**eval set**: `data/eval_sets/institutional_multi_turn_eval.jsonl` (30 sessions × 6 turns = 180 turns)
**実測日**: 2026-04-26
**Issue**: https://github.com/Kewton/photon-mlx/issues/113
**Branch**: `feature/issue-113-photon-mt-eval`

> **判定: 仮説 C（本格再学習が必要）に該当**。Turn 5-6 NC = 10.83% (PHOTON avg) は基準値 < 3% から大きく乖離。
> ただし PHOTON working memory の **follow-up latency -44.9% 改善** は明確で、再学習後の有望性を示唆。

---

## 1. 実行環境

| 項目 | 値 |
|------|-----|
| マシン | M3 Ultra (Mac Studio, 128GB) |
| OS | macOS Darwin 25.4.0 |
| Python | 3.12.3 |
| mlx-lm | 0.31.2 |
| sentence-transformers | 5.4.0 |
| ブランチ | `feature/issue-113-photon-mt-eval` |
| 実行時刻 | 2026-04-26 00:45–04:00 JST (約 3 時間) |

### Phase 2 ステップ別所要時間

| step | start | end | duration | output |
|------|-------|-----|----------|--------|
| MT eval set 生成 | 24:04 | 24:44 | 41 min | `institutional_multi_turn_eval.jsonl` (30 sessions, 180 turns) |
| baseline run 1 | 00:45 | 01:43 | 58 min | `baseline_mt_run1_*.jsonl` (180 rows) |
| baseline run 2 | 01:43 | 02:42 | 59 min | `baseline_mt_run2_*.jsonl` (180 rows) |
| PHOTON run 1 | 02:42 | 03:19 | 37 min | `photon_mt_run1_*.jsonl` (180 rows) |
| PHOTON run 2 | 03:19 | ~04:00 | ~41 min | `photon_mt_run2_*.jsonl` (180 rows) |

---

## 2. 設定

### baseline (`configs/institutional_docs.yaml`)
- `model.provider: "baseline"`
- chunker: max_chars=800 (post-#126)
- E5 prefix: on (post-#125)

### PHOTON (`configs/institutional_docs_photon.yaml`)
- `model.provider: "photon"`
- 上記と同じ chunker / E5 prefix
- PHOTON checkpoint: 既存 (本 Issue では再学習しない、現行 PHOTON が制度文書ドメインで動作するか測定)

---

## 3. 全体 NC rate

| 指標 | baseline run 1 | baseline run 2 | baseline avg | PHOTON run 1 | PHOTON run 2 | PHOTON avg | Δ (PHOTON - baseline) |
|------|---------------|---------------|--------------|--------------|--------------|------------|------------------------|
| NC turns / 180 | 11 | 14 | 12.5 | 23 | 18 | 20.5 | +8.0 |
| **NC rate** | 6.11% | 7.78% | **6.94%** | 12.78% | 10.00% | **11.39%** | **+4.44pp** |

**所見**: PHOTON は baseline より **+4.44pp NC 増加**（悪化）。2-run の variance:
- baseline: 6.11% / 7.78% (Δ 1.67pp)
- PHOTON: 12.78% / 10.00% (Δ 2.78pp)
→ いずれも有意な変動内。

---

## 4. Turn 5-6 NC rate（本 Issue の主指標）

| 指標 | baseline avg | PHOTON avg | Δ |
|------|--------------|------------|---|
| **Turn 5-6 NC** | **5.00%** | **10.83%** | **+5.83pp** |

設計 §9 仮説判定:
- 仮説 A: Turn 5-6 NC < 3% → 棄却（PHOTON 10.83%）
- 仮説 B: Turn 5-6 NC 3-6% → baseline 該当、PHOTON 非該当
- 仮説 C: Turn 5-6 NC > 6% → **PHOTON 該当**

**結論: 仮説 C — 本格再学習（10-20K step、JP 50%+）が必要**。

---

## 5. Category 別 NC rate

| Category | baseline NC% | PHOTON NC% | Δ NC% | baseline T5-6 NC% | PHOTON T5-6 NC% | Δ T5-6 |
|----------|-------------|-----------|-------|-------------------|-----------------|--------|
| cross_reference | 5.83% | 11.67% | +5.83pp | 5.00% | 12.50% | +7.50pp |
| drill_down | 5.00% | 10.56% | +5.56pp | 3.33% | 8.33% | +5.00pp |
| real_scenario | 15.00% | 13.33% | -1.67pp | 10.00% | 15.00% | +5.00pp |

**所見**:
- **cross_reference / drill_down で PHOTON が大幅悪化**（+5.5〜7.5pp）。Multi-turn で「前ターンの参照」が必要なシナリオで PHOTON が baseline に追いつけていない
- **real_scenario は overall NC では PHOTON が良い (-1.67pp)** が、Turn 5-6 では悪化 (+5.0pp)
- baseline の cross_reference T5-6 5.0% は妥当な性能。PHOTON 12.5% は再学習の主要改善対象

---

## 6. Latency

### follow-up latency (Turn 2 以降)

| 指標 | baseline avg | PHOTON avg | Δ |
|------|--------------|------------|---|
| **follow-up p50** | **19,426 ms** | **10,707 ms** | **-44.9% 改善** ✅ |

### Turn 5-6 latency

| 指標 | baseline avg | PHOTON avg |
|------|-------------|------------|
| p50 | 21,112 ms | 12,315 ms |
| p95 | 33,092 ms | 19,488 ms |

**所見**: PHOTON の **working memory による latency 改善は明確**（follow-up で約半減）。
これは再学習後の真の baseline + PHOTON 比較で活かされる重要な性能。

---

## 7. 再現性 (2-run の安定性)

| 指標 | baseline 2-run Δ | PHOTON 2-run Δ |
|------|-----------------|----------------|
| NC overall | 1.67pp | 2.78pp |
| NC Turn 5-6 | 3.34pp | 1.67pp |

**所見**: 両者とも 2-run variance は ±3pp 以内で安定。本 §3-5 の判定は変動内に収まる結論。

---

## 8. 判定と次のアクション

### 仮説 C 確定 — 本格再学習が必要

| 仮説 | 基準 | 実測 | 判定 |
|------|------|------|------|
| A | Turn 5-6 NC < 3% | 10.83% | ❌ 棄却 |
| B | Turn 5-6 NC 3-6% | 10.83% | ❌ 棄却 |
| **C** | Turn 5-6 NC > 6% | **10.83%** | ✅ **該当** |

### 推奨アクション

1. **Conditional Issue 起票**: 本格再学習（10-20K step、JP 50%+）の実施計画
2. **学習データ準備**: 制度文書から JP corpus 抽出、既存 mulmoclaude (英語コード) と混合
3. **#117 Epic 更新**: Phase 2 conditional 表の「本格再学習」シナリオを active 化
4. **Δ 評価の保留**: 本実測は **再学習前** の数値。再学習後に再測定すべき

### Δ 改善の見込み

PHOTON working memory が **latency -44.9%** は不変と仮定すると、**精度を baseline 並みまで戻す再学習**で:
- 速度は baseline 比 -45% を維持
- NC は baseline 並み (~7%) または改善

これが達成できれば「制度文書ドメインで PHOTON が baseline を上回る」が初めて成立。

---

## 9. follow-up Issue / 申し送り

### #117 Epic Phase 2 conditional への引き継ぎ

| シナリオ | 該当 | 工数 |
|---------|------|------|
| Turn 5-6 NC < 3% | ❌ | 0 日 |
| Turn 5-6 NC 3-6% | ❌ | 3 日 |
| **Turn 5-6 NC > 6%** | **✅** | **5 日（本格再学習）** |

### 関連 Issue

- 本 Issue: #113（PHOTON MT 測定）
- Epic: #117 Phase 2 制度文書ドメイン検証
- baseline: #112 (closed)
- handicap (a) E5 prefix: #125 (closed, regression あり)
- handicap (b) 多言語 reranker: #114→#133 (open, deferred A/B)
- handicap (c) chunker size: #126 (closed, NC 11.21% に回復)

### 関連リンク

- baseline report: `reports/institutional_baseline_static.md`
- chunker A/B report: `reports/institutional_chunker_size_ab.md`
- predictions JSONL: `logs/institutional/{baseline,photon}_mt_run{1,2}_*.jsonl`
- MT eval set: `data/eval_sets/institutional_multi_turn_eval.jsonl`
- 設計書: `workspace/design/issue-113-photon-mt-eval-design-policy.md`
- 作業計画: `workspace/issues/113/work-plan.md`
