# Qwen2.5 vs Qwen3.5 no-think 評価レポート

作成日時: 2026-04-28 22:27 JST

## 概要

400件の比較評価を実施した。比較対象は以下の4条件。

1. Baseline + Qwen2.5-Coder-14B-Instruct-4bit
2. Baseline + Qwen3.5-9B-MLX-4bit no-think
3. PHOTON + Qwen2.5-Coder-14B-Instruct-4bit
4. PHOTON + Qwen3.5-9B-MLX-4bit no-think

各条件は **100 prediction** で比較した。

- static: 40件
- multi-turn: 60件
- 合計: 400件

`photon_generation_enabled: false` で評価しているため、PHOTON条件は **PHOTON retrieval/memory path + Qwen generation** の比較であり、PHOTONネイティブ生成の比較ではない。

## 結果サマリ

| Variant | Static p50 | Multi-turn p50 | No-citation (Static) | No-citation (MT) | Leak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline + Qwen2.5 | 20,206.80 ms | 20,888.09 ms | 2.50% | 1.67% | 0.00% |
| Baseline + Qwen3.5 no-think | 12,407.34 ms | 11,241.10 ms | 0.00% | 0.00% | 0.00% |
| PHOTON + Qwen2.5 | 21,163.72 ms | 13,177.08 ms | 5.00% | 8.33% | 0.00% |
| PHOTON + Qwen3.5 no-think | 11,781.07 ms | 7,438.02 ms | 0.00% | 0.00% | 0.00% |

## 主な観察

### 1. Qwen3.5 no-think が一貫して優位

- **Baseline / static p50**: 20,206.80 ms → 12,407.34 ms (**-38.6%**)
- **Baseline / multi-turn p50**: 20,888.09 ms → 11,241.10 ms (**-46.2%**)
- **PHOTON / static p50**: 21,163.72 ms → 11,781.07 ms (**-44.3%**)
- **PHOTON / multi-turn p50**: 13,177.08 ms → 7,438.02 ms (**-43.6%**)

### 2. no-citation rate も Qwen3.5 no-think が改善

- **Baseline / static**: 2.50% → 0.00%
- **Baseline / multi-turn**: 1.67% → 0.00%
- **PHOTON / static**: 5.00% → 0.00%
- **PHOTON / multi-turn**: 8.33% → 0.00%

### 3. reasoning leak は今回の比較では観測されず

全条件で leak rate は **0.00%** だった。  
Qwen3.5 no-think 統合後、この評価系では reasoning 出力の漏れは再現していない。

### 4. 今回の最良条件は PHOTON + Qwen3.5 no-think

- static p50: **11,781.07 ms**
- multi-turn p50: **7,438.02 ms**
- no-citation: **0.00% / 0.00%**

特に multi-turn で最速かつ no-citation も 0% で、4条件中もっとも安定していた。

## 補足

- `generator_used_counts` は全件 `qwen`
- `generator_fallback_reasons` は全条件で空
- したがって、今回の評価では fallback や別生成器への切替は発生していない

## 結論

今回の 400件比較では、**Qwen3.5-9B-MLX-4bit no-think が Qwen2.5-Coder-14B-Instruct-4bit を、Baseline/PHOTON の両条件で上回った**。  
運用候補としては **PHOTON + Qwen3.5 no-think** が最有力である。

## 参照ファイル

- Summary JSON: `reports/qwen_model_matrix_20260428_400cmp_summary.json`
- Run log: `reports/qwen_model_matrix_20260428_400cmp.log`
- Progress JSON: `reports/qwen_model_matrix_20260428_400cmp/qwen_model_matrix_20260428_400cmp_progress.json`
- Prediction JSONL:
  - `reports/qwen_model_matrix_20260428_400cmp/qwen_model_matrix_20260428_400cmp_baseline_qwen25.jsonl`
  - `reports/qwen_model_matrix_20260428_400cmp/qwen_model_matrix_20260428_400cmp_baseline_qwen35_nothink.jsonl`
  - `reports/qwen_model_matrix_20260428_400cmp/qwen_model_matrix_20260428_400cmp_photon_qwen25.jsonl`
  - `reports/qwen_model_matrix_20260428_400cmp/qwen_model_matrix_20260428_400cmp_photon_qwen35_nothink.jsonl`
