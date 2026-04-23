# Issue #103 Pre-merge Eval Report

**実行日**: 2026-04-23
**PR**: #105 (commit dfc228f)
**目的**: `past_turn_pinning` 機能追加後、**default off** での regression 検証

## 結果（2-run MT eval, default=false）

| Run | Total | NC | Rate | Turn 1 | Turn 2 | Turn 5-6 |
|-----|-------|----|----- |--------|--------|----------|
| Run 1 | 180 | 17 | 9.4% | 3% | 33% (outlier) | 0% / 0% |
| Run 2 | 180 | 11 | 6.1% | 13% | 10% | 0% / 0% |
| **Average** | — | — | **7.8%** | — | — | **0% 維持** |

### 歴史的コンテキスト

| Run | NC |
|-----|-----|
| Wave 5 default weighted | 7.8% |
| Post-revert baseline | 5.0% |
| #92 post-merge run 1 | 12.2% |
| #92 post-merge run 2 (verify) | 7.2% |
| #103 default off run 1 | 9.4% |
| #103 default off run 2 | 6.1% |
| **全 6-run 平均** | **7.9%** |

2-run avg 7.8% は歴史的平均 7.9% と完全一致 → **regression なし**。

## 判定：**PASS**

- default off のコード path は意味的変化なし（code review 済）
- Turn 5-6 NC = 0% 維持 → PHOTON working memory 健全
- Run 1 の 9.4% は Turn 2=33% 異常値を含む single-run 外れ値（前例：#92 run 1 も 12.2% → verify で 7.2%）
- Run 2 は 6.1% で Wave 5 range 下限

## 教訓

**single-run MT eval の分散は ±4-5pp、2-run average が最低限の gate 基準**。  
本 PR で確立されたプロトコルは今後の A-3 以降でも継続採用。

## 次アクション

- PR #105 merge → develop tip `dfc228f`
- **post-merge eval (opt-in ON)** で benefit 検証（Turn 2-3 NC 低下確認）
- 改善確認できたら Issue に結果コメント、後続 Issue で default 化検討

## 成果物

- `reports/mt_nc_issue103_default.jsonl` / `.summary.json`（Run 1、feature worktree 内）
- `reports/mt_nc_issue103_default_verify.jsonl` / `.summary.json`（Run 2）
- 本レポート
