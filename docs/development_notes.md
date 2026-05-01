# Development Notes

このドキュメントは、PHOTON-RepoRAG の開発方針、評価観点、Gate、Definition of Done をまとめた開発者向けメモです。README は利用者向けのプロダクト概要と始め方に絞っています。

## 開発モード

### v0: Baseline RepoRAG

- open-weight instruct model
- hybrid retrieval
- lexical retrieval
- embedding retrieval
- symbol / graph expansion
- citation 付き回答
- simple session memory
- benchmark 可能

### v1: PHOTON-RAG

- hierarchical encoder
- converter
- local decoder
- session-level working memory
- follow-up ターン高速化
- drift 記録

### v1.1: Safe RecGen

- drift 検知
- topic shift 検知
- exact quote / patch 生成時の強制再読
- fallback reason の記録
- baseline path への一時退避

## 評価の考え方

このプロジェクトの成功条件は、単発回答ではなく multi-turn 実用性に置きます。

### 品質指標

- task correctness
- citation precision
- citation recall
- session consistency
- hallucination rate
- patch / diff grounding quality

### 性能指標

- P50 / P90 latency
- tokens/sec
- peak memory
- memory / active session
- retrieval time
- prefill time
- decode time
- fallback rate

### 安全指標

- no-citation assertion rate
- wrong citation rate
- stale memory rate
- missed fallback rate

## 成功条件

詳細は `spec.md` を参照しますが、実務上の成功条件は次の通りです。

- follow-up レイテンシが baseline 比で明確に改善する
- session あたりメモリが baseline より減る
- citation precision を落とさない
- multi-turn の前提保持が向上する
- Safe RecGen によって危険ケースで堅い経路へ戻れる

## 比較対象

以下を同じ harness で比較します。

1. Baseline-RAG
2. Baseline-RAG + summary memory
3. PHOTON-RAG
4. PHOTON-RAG + Safe RecGen

## ログ方針

最低限、以下は全 run で保存します。

- run_id
- session_id
- turn_id
- repo_id
- repo_commit
- model_id
- retrieval chunk IDs
- evidence pack IDs
- cited chunk IDs
- fallback flag
- fallback reason
- latency breakdown
- memory peak
- answer text
- grader score

ログは後から failure analysis できる粒度で残します。

## Gate

### Gate 1

baseline RepoRAG が安定稼働し、benchmark が再現可能か。

### Gate 2

PHOTON 系 forward / eval が安定し、tiny/small で改善の兆候があるか。

### Gate 3

Safe RecGen によって誤答率改善とレイテンシ改善が両立するか。

### Gate 4

baseline 統合、限定ベータ、研究枝公開のどれに進むか判断できるか。

## 実装ルール

- benchmark を先に凍結する
- repo snapshot を固定する
- exact quote / diff / patch は必ず局所再読する
- retrieval と generation の失敗を分離して分析する
- すべての比較は同じ harness 上で行う
- 「動いた」ではなく「比較できる」を Done とする

## 主要ディレクトリの役割

### `baseline_reporag/`

最初に価値を出すプロダクト線です。比較対象であり、最後まで control として維持します。

### `photon_mlx/`

本命実装です。Mac 向け最適化と PHOTON 系 working memory を入れます。

### `torch_ref/`

正しさ確認用です。mask、shape、forward、teacher-forced eval の検証に使います。

### `bench/`

公平比較の中心です。評価条件、run、集計、グラフ出力をまとめます。

### `evals/`

静的問題、multi-turn セッション、stress eval を管理します。

### `reports/`

benchmark report と failure cases を蓄積します。

## 最初にやること

- `spec.md` を確定する
- 対象 repo を 1 つに固定する
- benchmark を freeze する
- baseline RepoRAG を動かす
- citation が出るところまで最短で到達する

## 今後の拡張候補

- adaptive chunking
- topic-aware memory refresh
- CPU/GPU 分業スケジューリング
- Agentic RAG
- patch planning
- internal beta / public demo

## Definition of Done

このリポジトリは、次を満たしたら 1 つの区切りとします。

1. baseline RepoRAG が安定して動く
2. PHOTON-RAG が同じ harness 上で比較できる
3. Safe RecGen の発火条件と効果が測定できる
4. multi-turn で baseline より価値があると示せる
5. report と failure cases が第三者に共有できる

## 参考ドキュメント

- `spec.md`: 開発仕様と意思決定の基準
- `tasks.md`: 実行タスク一覧
- `reports/`: 評価レポートと失敗事例
