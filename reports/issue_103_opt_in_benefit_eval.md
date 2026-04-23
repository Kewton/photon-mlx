# Issue #103 Opt-in Benefit Eval Report

**実行日**: 2026-04-23
**Config**: `configs/photon_small_pinning_on.yaml`（`past_turn_pinning_enabled: true`）
**目的**: PR #105 で追加した opt-in feature の実効性検証

## 結果（2-run MT eval）

| Run | NC | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 | Turn 6 |
|-----|-----|--------|--------|--------|--------|--------|--------|
| Run 1 | 9.4% | 13% | 17% | 17% | 3% | 7% | 0% |
| Run 2 | 8.9% | 10% | 17% | 17% | 10% | 0% | 0% |
| **Avg** | **9.2%** | 11.7% | 16.7% | **16.7%** | 6.7% | 3.3% | 0% |

### Default OFF 2-run 比較

| metric | default OFF | **pinning ON** | delta |
|--------|-------------|---------------|-------|
| 全体 MT NC | 7.8% | **9.2%** | **+1.4pp 悪化** |
| Turn 2 | ~22% | 16.7% | **-5pp 改善** ✓ |
| Turn 3 | ~9% | **16.7%** | **+8pp 悪化** ⚠️ |

## 判定：**opt-in benefit 実証されず**

Hypothesis（Issue #103 本文）は Turn 2-3 NC 低減を予測していたが、実測では **Turn 2 で改善・Turn 3 で悪化**と mixed signal。全体では +1.4pp regression。

## 原因分析

### Turn 2 改善（-5pp）の解釈
- Turn 2 は直前ターン（Turn 1）への follow-up が多く、context 継続性が高い
- pinning された過去ターンの chunks が評価に有効

### Turn 3 悪化（+8pp）の解釈
- **Turn 3 は topic shift が起こりやすい**（例: Turn 2 で話題 A の diff、Turn 3 で話題 B のテスト）
- Turn 2 完了時点で pin された chunks は Turn 3 にとって stale
- evidence pack の 16 chunks 枠を stale pinning が消費し、より関連の高い新 chunks が除外される
- 結果として **retrieval noise** が増加

### 設計上の限界
- `find_relevant_past_turn` は **直前ターン専用**ではなく、**任意の過去ターン**を pin
- しかし evidence pack の枠消費で **今回のクエリに最適な chunks** が削られる trade-off
- threshold (0.7) と `max_pinned_chunks` (3) のデフォルトは本 workload に対して過度に寛容

## 決定

1. **PR #105 は merge 済み維持**（実装は clean、default off のため regression リスクなし）
2. **opt-in feature として残置**（特定 workload で効果が期待できる可能性）
3. **default 化は推奨せず**
4. **Issue #103 は本実測結果で「実装完了、benefit not proven」としてクローズ**
5. Follow-up：threshold/max_pinned_chunks チューニングで Turn 3 悪化を回避できるかの A/B は別 Issue（Epic #81 の延長）で検討

## 教訓

**context 継続性を仮定する feature は、topic shift がある workload で逆効果になる**。将来の類似 feature（#104 dynamic aggregation 含む）では：
- default は opt-in で導入
- 2-run MT eval gate で実効性確認
- Turn 別の詳細ブレークダウン分析を必須化

## 成果物

- `reports/mt_nc_issue103_pinning_on.jsonl` / `.summary.json` (Run 1)
- `reports/mt_nc_issue103_pinning_on_verify.jsonl` / `.summary.json` (Run 2)
- 本レポート
