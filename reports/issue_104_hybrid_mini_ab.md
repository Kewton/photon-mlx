# Issue #104 Hybrid Mini A/B Report

**実行日**: 2026-04-23
**Config**: `configs/photon_small_hybrid.yaml`（`aggregation: dynamic` + `dynamic_strategy: hybrid`）
**目的**: hybrid strategy の default 化判定

## 結果（2-run MT eval）

| Run | NC | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 | Turn 6 |
|-----|-----|--------|--------|--------|--------|--------|--------|
| Run 1 | 7.8% | 0% | 13% | 23% | 10% | 0% | 0% |
| Run 2 | 9.4% | 0% | 20% | 13% | 10% | 0% | 13% |
| **Avg** | **8.6%** | **0%** | 16.7% | 18.3% | 10% | 0% | 6.7% |

### Weighted baseline 比較

| metric | weighted | **hybrid** | delta |
|--------|----------|-----------|-------|
| 全体 MT NC | 7.8% | **8.6%** | **+0.8pp 悪化** |
| Turn 1 | ~3% | **0%** | **-3pp 改善** ✓ |
| Turn 2-4 | 6-15% | 10-18% | +2〜+8pp 悪化 |
| Turn 6 | ~0% | 6.7% | +7pp 悪化 |

## 判定：**default 化推奨せず**

hybrid は Wave 5 分析で予測された通り Turn 1 で attention の強みを活かせる。しかし alpha blending（weighted + attention）の **中間値が Turn 2-6 で weighted を下回る**。全体では +0.8pp の悪化で default 化は非推奨。

## 設計上の原因分析

`hybrid_alpha_base: 0.5`, `hybrid_alpha_per_turn: 0.1` という default:
- Turn 1: alpha=0.5（半々）→ Turn 1 は history 不在で attention fallback → 0%
- Turn 2+: alpha 徐々に増加 → weighted の強みが薄まる
- Turn 4+: alpha 高め → attention 寄り、Turn 4-6 で weighted より悪化

実質、**hybrid の alpha schedule が本 workload に最適化されていない**。alpha_base=0.1、per_turn=0.05 など attention 比率を下げれば改善余地あるが、hyperparameter tuning は別 Issue の範囲。

## 決定

1. **dynamic aggregation (Issue #92) 実装は merged**、opt-in として利用可能
2. **default は `aggregation: weighted` 維持**
3. Issue #104 は「empirical A/B 完了、hybrid strategy default 化は empirical 根拠なし」でクローズ
4. Follow-up：
   - `hybrid_alpha_base` / `hybrid_alpha_per_turn` の grid search（別 Issue、Epic #81 の延長）
   - `turn_position` と `drift_based` strategy の独立 A/B（今回未実施）

## 教訓

Wave 5 分析（`reports/mt_nc_eval_wave5_analysis.md`）で示された turn 別の最適 mode は実測と整合（Turn 1 attention、Turn 2-4 weighted）。しかし **単純な alpha blending** では両モードの strengths を捕捉しきれない。より精度の高い dispatcher（例：`turn_count < 2 ? attention : weighted` のような条件付き）が必要な可能性。

## 成果物

- `reports/mt_nc_issue104_hybrid_run1.jsonl` / `.summary.json`
- `reports/mt_nc_issue104_hybrid_run2.jsonl` / `.summary.json`
- 本レポート
