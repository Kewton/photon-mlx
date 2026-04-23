# Issue #89 bge-reranker Retry — Abort Report

**実行日**: 2026-04-23
**commit**: cherry-pick bcb2f26 → 059305e（merge せず、worktree 廃棄）
**判定**: **Abort（merge せず Issue #89 クローズ推奨）**

## 実測結果（pre-merge Static NC）

| Run | Reranker | Static NC | Delta vs baseline |
|-----|----------|-----------|-------------------|
| Baseline (post-revert) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **17.5%** | — |
| **#89 feature branch** | `BAAI/bge-reranker-base` | **18.3%** | **+0.8pp 悪化** |

### カテゴリ別
| category | NC |
|----------|-----|
| onboarding | 3.3% |
| bug_localization | 13.3% |
| change_planning | 26.7% |
| impact_analysis | 30.0% |

## 判定理由

1. **Issue #89 の目標は Static NC < 16% 改善**、結果は 18.3%（未達かつ悪化方向）
2. Wave 5 分析（`reports/two_pass_search_analysis.md`）で既に
   「MS-MARCO MiniLM (reranker) が bge より強い」と実測済み
3. Wave 6 全 revert 時の regression 要因推定とも整合
4. bge-reranker-base は多言語 IR tuned だが FastAPI（英語 code-heavy）workload では
   MS-MARCO MiniLM の code/NL ranking 適合性が勝る

## 結論

**本 workload において bge-reranker-base による Static NC 改善は得られない**。
MT eval を実行しても改善は期待できず（Wave 6 の MT regression 要因推定と整合）、
3h の eval コストはリターンに見合わない。

## 次アクション

- Issue #89 に本結果をコメントしてクローズ（"bge-reranker does not improve Static NC on this workload"）
- Epic #81 配下の retrieval tuning は以下で進める：
  - **embedding モデル更新（#90）**— 未検証のため別トライ価値あり
  - **retrieval パラメータ grid search（#88 harness 使用）**— 現行 reranker 上での最適化
  - **graph expansion 調整（#91）**— Wave 6 regression 要因の可能性、慎重に

## 成果物

- `reports/static_nc_issue89_pre.jsonl`（feature worktree 内、120Q 予測）
- 本レポート
