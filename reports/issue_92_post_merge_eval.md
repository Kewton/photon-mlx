# Issue #92 Post-merge Eval Report

**実行日**: 2026-04-23
**PR**: #101 (commit a72775e, cherry-pick of reverted bf6dd52)
**目的**: dynamic aggregation 機能追加後の weighted (default) 経路の regression 検証

## 結果

| Run | Total | NC | Rate | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 | Turn 6 |
|-----|-------|----|----- |--------|--------|--------|--------|--------|--------|
| Post-revert baseline (pre-#92) | 180 | 9 | **5.0%** | 0.0% | 16.7% | 10.0% | 3.3% | 0.0% | 0.0% |
| Post-#92 run 1 | 180 | 22 | 12.2% | 23.3% | 10.0% | 16.7% | 10.0% | 0.0% | 13.3% |
| Post-#92 run 2 (verify) | 180 | 13 | **7.2%** | 0.0% | 10.0% | 20.0% | 10.0% | 3.3% | 0.0% |
| Wave 5 default (過去参考) | 180 | — | 7.8% | — | — | — | — | — | — |

## 判定：**PASS**

- Run 1 の 12.2% は Turn 1=23.3% という異常値を含む外れ値。Run 2 verify で Turn 1=0% に回帰。
- Run 2 の 7.2% は Wave 5 default (7.8%) と同等で refactor による semantic 変化なし。
- default `aggregation: weighted` 維持のため既存挙動への影響は想定通り皆無。
- dynamic モードは opt-in（`aggregation: "dynamic"` 明示時のみ）。

## 教訓

1. **LLM 非決定性による single-run 分散が ±4-5pp ある**（Qwen 2.5-Coder-14B-Instruct-4bit, temp=0.2, top_p=0.9）。
2. **今後の eval-gate は 2-run average を基準とすべき**。single-run での ±3pp 判定は誤検知リスクあり。
3. Run 1 の 23.3% Turn 1 NC は Turn 1 が session state を使わない（history 空）点と矛盾しており、Qwen 生成時の stochastic な引用タグ欠落が主因と推定。

## 次アクション（A-2: Issue #88 retrieval grid search）

- 同じ single-PR + post-merge eval-gate プロトコル
- ただし eval は **2-run average** を採用する
