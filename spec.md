# PHOTON-RepoRAG 開発仕様書

- Version: 0.1
- Status: Draft
- Date: 2026-04-12
- Owner: <your_name>
- Primary Platform: Mac Studio M3 Ultra / 256GB
- Primary Goal: RepoRAG における長文脈・多ターン応答の高速化と省メモリ化
- Secondary Goal: PHOTON系アーキテクチャの実験基盤を構築し、Safe RecGen まで実装・評価する

---

## 1. 概要

本プロジェクトは、巨大コードベースを対象とした **セッション型 RepoRAG** を構築する。  
狙いは、通常の RAG が苦手とする「同じ証拠集合に対する連続質問」で、**追撃ターンのレイテンシとセッションあたりメモリを大きく下げること**にある。

本プロジェクトは 2 本立てで進める。

1. **プロダクト線**  
   既存の open-weight instruct model を使った baseline RepoRAG を先に成立させる。

2. **研究開発線**  
   PHOTON系の階層 working memory を実装し、RepoRAG に統合する。  
   さらに drift 検知と fallback を備えた **Safe RecGen** を実装する。

本計画は、frontier 級の基盤モデルを一から学習することを目標にしない。  
まずは **プロダクト価値が出る baseline** と **比較可能な実験基盤** を作り、その上で PHOTON 系の優位性を検証する。

---

## 2. 背景と問題定義

通常の RAG は 1 問目には強いが、次の問題を持つ。

- 同じ資料束・同じ repo に対する follow-up で、毎ターン大量の文脈を再読しやすい
- 会話履歴と evidence pack が伸びるほど、レイテンシとメモリ消費が悪化する
- 連続質問で「何を前提に話しているか」の作業状態を保つのが苦手
- repo 全体の構造理解と局所コード読解を何度も往復する作業に弱い

本プロジェクトが解く問題は、次の 1 文に集約される。

**「Repo 全体の粗い作業状態は継続的に持ち、最終回答では必要な局所証拠だけを再参照する」仕組みを作る。**

---

## 3. 仮説

次の仮説を検証する。

> retrieved evidence を session pack と hierarchical working memory に変換し、follow-up ターンでは coarse state のみを更新しつつ、回答直前に局所証拠を再読する構成にすれば、baseline RAG と比べて multi-turn セッションのレイテンシとメモリを大きく削減しながら、grounded な回答品質を維持できる。

補助仮説は以下の通り。

- PHOTON系 working memory は、repo 全体構造のような粗い文脈保持に向く
- exact quote や patch 生成のような局所精度が重要な場面では、局所再読を強制することで品質低下を抑えられる
- drift 検知と fallback を組み込んだ Safe RecGen は、速度と堅牢性のバランスを改善する

---

## 4. 対象ユーザー

本プロジェクトの対象ユーザーは以下。

- 巨大 repo に新しく入った開発者
- 影響範囲分析を行うテックリード
- 障害解析を行う SRE / プロダクトエンジニア
- repo の設計意図や依存関係を素早く把握したい開発者

---

## 5. 主要ユースケース

### 5.1 対象ユースケース

1. **Repo オンボーディング**
   - この repo の主要モジュールは何か
   - 認証処理はどこからどこへ流れるか
   - 新しい機能を足すならどこから読むべきか

2. **影響範囲分析**
   - この API を変えるとどこが壊れる可能性があるか
   - DB schema 変更の波及先はどこか
   - 認可ロジック修正の影響対象は何か

3. **障害解析**
   - この障害の原因候補を 3 つ出す
   - 最も怪しいモジュールと根拠を示す
   - 再現に必要なログや設定を提案する

4. **変更計画の比較**
   - 修正案 A/B のメリット・リスク比較
   - 既存の抽象化を崩さずに変更する方法
   - 最小修正と根本修正の比較

### 5.2 非対象ユースケース

以下は v1 の対象外とする。

- repo 全体に対する大規模自動リライト
- main ブランチへの自律コミット
- 出典なしの断定回答
- 一般チャット用途への横展開
- frontier 級基盤モデルのフル事前学習

---

## 6. スコープ

### 6.1 In Scope

- baseline RepoRAG の実装
- repo ingestion / indexing / retrieval
- multi-turn session 管理
- PHOTON系 working memory の実装
- Safe RecGen の実装
- bench / eval / profiling / failure analysis
- Mac Studio 上での最適化

### 6.2 Out of Scope

- 大規模分散学習
- 数十 B 級モデルの一から学習
- 汎用 AI agent プラットフォーム化
- CI/CD 自動修正ループの本番投入

---

## 7. 成功条件

baseline は Week 2 で固定し、それに対する相対評価で判定する。

| 指標 | 定義 | 必達 | ストレッチ |
|---|---|---:|---:|
| Follow-up レイテンシ P50 | 同一セッション 2〜6 ターン目の応答時間 | baseline 比 -30% | baseline 比 -45% |
| Peak memory / session | 同時 8 セッション時のセッションあたりピークメモリ | baseline 比 -40% | baseline 比 -60% |
| Citation precision | 回答中の引用が正しい根拠を指している割合 | 0.90 以上 | 0.95 以上 |
| タスク正答率 | 評価セット総合スコア | baseline -3pt 以内 | baseline +2pt |
| 無根拠回答率 | 根拠なし断定回答の割合 | 5% 以下 | 2% 以下 |
| Safe fallback recall | 人手で「再読が必要」と判定したケースで fallback が発火する割合 | 0.80 以上 | 0.90 以上 |
| セッション一貫性 | multi-turn で前提を維持できた割合 | 0.85 以上 | 0.92 以上 |

補足:
- `pt` は percentage points を意味する
- baseline と比較するときは、同じ retrieval 条件・同じ generation 条件を使う

---

## 8. 開発方針

### 8.1 基本方針

- **先に価値が出る baseline を作る**
- **次に比較可能な実験器を作る**
- **その上で PHOTON 系の改良を入れる**
- **最後に Mac 向け最適化を行う**

### 8.2 設計原則

- 検索は **Source of Truth**
- PHOTON系 memory は **Working Memory**
- 最終回答は **局所証拠に再着地** してから生成する
- drift が疑われるときは **Safe RecGen** で再検索・再読に戻る
- exact quote, diff, patch 生成は **必ず局所再読を伴う**

---

## 9. システム全体像

### 9.1 論理構成

```text
[Repo Ingestion]
    -> Chunk Store
    -> Symbol Graph
    -> Lexical Index
    -> Embedding Index
    -> Metadata Store

[User Query]
    -> Query Router
    -> Hybrid Retrieval
    -> Graph Expansion
    -> Evidence Pack Builder
    -> Session Memory Manager
        -> v0: Flat Context Memory
        -> v1: PHOTON Working Memory
        -> v1.1: PHOTON + Safe RecGen
    -> Answer Generator
    -> Citation Resolver
    -> Logger / Evaluator
```

### 9.2 メモリの 3 層

1. **Source of Truth**
   - 元コード
   - README / ADR / 設計資料
   - test / log / issue / incident note
   - commit metadata

2. **Session Pack**
   - そのセッションで重要な chunk 群
   - retrieval の結果と近傍 chunk
   - 直近で引用された証拠

3. **Working Memory**
   - repo 全体の粗い理解
   - その会話で何を追っているか
   - 途中の仮説
   - 重要な前提条件

---

## 10. v0 Baseline 仕様

### 10.1 目的

最短でプロダクト価値を出し、比較対象を固定する。

### 10.2 構成

- open-weight instruct model
- hybrid retrieval
  - lexical
  - embedding
  - symbol / dependency graph expansion
- evidence pack
- citation 付き回答
- simple session memory
  - 直近会話要約
  - 引用済み chunk 優先
  - 重要 chunk pinning

### 10.3 初期設定

- retrieve 候補: 12〜20 chunks
- evidence pack: 12〜24 chunks
- answer-time local refresh: 3〜5 chunks
- max context budget は baseline 時点で固定する

### 10.4 v0 の Done 条件

- repo ingest が安定して動く
- hybrid retrieval の再現性がある
- citation 付き回答が出る
- 単一セッション / 複数セッションのレイテンシが測定できる
- 比較用 benchmark が 1 コマンドで回る

---

## 11. v1 PHOTON系 仕様

### 11.1 目的

multi-turn セッションでの working memory 保持を階層的に実装し、follow-up コストを下げる。

### 11.2 実装対象

- chunker
- hierarchical encoder
- converter
- local decoder
- hierarchical prefill
- coarse state update
- teacher-forced evaluation path
- autoregressive session update path

### 11.3 役割分担

- 粗い全体文脈: hierarchical latent
- 最終回答直前の精密確認: local evidence refresh
- 回答文生成: grounding 済み evidence を優先

### 11.4 Done 条件

- forward が安定して通る
- teacher-forced eval が動く
- multi-turn session で state を保持できる
- drift 指標を記録できる
- baseline と公平比較できる

---

## 12. v1.1 Safe RecGen 仕様

### 12.1 目的

高速な state 更新を維持しつつ、drift や局所精度不足による誤答を抑える。

### 12.2 発火条件

以下のいずれかで fallback を発火する。

- latent drift が閾値を超える
- topic shift が大きい
- exact quote 要求
- diff / patch 生成要求
- answer confidence が低い
- 引用候補が不安定
- 高リスク質問
  - 認可
  - 課金
  - セキュリティ
  - データ破壊

### 12.3 fallback 動作

- 再 retrieval
- local evidence refresh の強化
- hierarchical prefill のやり直し
- 必要に応じて baseline path に一時退避

### 12.4 Done 条件

- fallback reason がログに残る
- fallback の再現率を評価できる
- 誤答率が改善する
- レイテンシ悪化が許容範囲に収まる

---

## 13. モデル戦略

### 13.1 基本戦略

本プロジェクトの「LLM 開発」は 2 段階で定義する。

1. **プロダクト用 LLM**
   - 既存の open-weight instruct model を利用
   - RepoRAG の baseline として使う

2. **研究用 LLM**
   - PHOTON系の small〜mid size model を実装
   - hierarchy / RecGen / drift の効果を検証する

### 13.2 やらないこと

- 数十 B 級モデルのフル事前学習
- 初手から paper-scale 再現に全振り
- ベンチ未整備のまま独自モデルを増やすこと

### 13.3 研究モデルの学習段階

#### Stage A: Tiny 検証
- 目的: architecture correctness
- 目安: 50M〜100M 級
- 成功条件: overfit / loss 低下 / state 一貫性

#### Stage B: Small 検証
- 目的: multi-turn working memory の有効性確認
- 目安: 150M〜300M 級
- 成功条件: baseline 比で follow-up 改善

#### Stage C: Optional Mid
- 目的: 研究結果の強化
- 目安: 400M〜600M 級
- 条件: Gate 2 を通過した場合のみ着手

### 13.4 学習目標

- next-token loss
- hierarchical consistency / recursive consistency
- grounding を意識した SFT
- citation 付き回答フォーマット
- repo reasoning 用 synthetic QA / teacher distillation

---

## 14. データ戦略

### 14.1 利用データ

- target repo の source code
- README / docs / ADR
- test code / fixtures
- build config / deployment config
- issue / PR / commit message
- incident note / runbook / log sample
- public code/text corpus（研究モデルの初期検証用）

### 14.2 データ生成物

- chunked corpus
- lexical index
- embedding index
- symbol graph
- dependency graph
- session benchmark set
- citation benchmark set
- synthetic SFT data
- failure case catalog

### 14.3 データ原則

- repo snapshot を commit hash で固定する
- benchmark 中は同一 snapshot を使う
- private repo はローカルで閉じる
- 外部 API なしでも benchmark が再現できることを優先する

### 14.4 repo snapshot 固定ルール

- `repo_commit` には必ず固定 SHA を使う（`HEAD` は開発中のみ許可）
- benchmark freeze 前に SHA を確定し、`configs/` に記録する
- snapshot を変更する場合は run_id を新規発番し、旧 run と混在させない

---

## 15. 評価計画

### 15.1 比較対象

以下の 4 系統を比較する。

1. **Baseline-RAG**
2. **Baseline-RAG + summary memory**
3. **PHOTON-RAG**
4. **PHOTON-RAG + Safe RecGen**

### 15.2 評価セット

#### Static Eval
- 120 問
- カテゴリ:
  - onboarding: 30
  - impact analysis: 30
  - bug localization: 30
  - change planning: 30

#### Multi-turn Session Eval
- 30 セッション
- 1 セッションあたり 6 ターン
- 合計 180 ターン

#### Stress Eval
- 8 同時セッション
- 各 10 ターン
- memory / latency / fallback を計測

### 15.3 評価指標

#### 品質
- task correctness
- citation precision
- citation recall
- session consistency
- hallucination rate
- patch / diff grounding quality

#### 性能
- P50 / P90 latency
- tokens/sec
- peak memory
- memory / active session
- retrieval time
- prefill time
- decode time
- fallback rate

#### 安全性
- no-citation assertion rate
- wrong citation rate
- stale memory rate
- missed fallback rate

### 15.4 採点ルーブリック

各回答を以下で採点する。

- Correctness: 0〜2
- Grounding: 0〜2
- Usefulness: 0〜1

合計 5 点満点。  
集計時は 100 点換算も併記する。

### 15.5 公平比較ルール

- retrieval 条件を固定する
- prompt テンプレート差分は明示する
- generation パラメータを固定する
- repo snapshot を固定する
- ハードウェアは同一条件で測る
- benchmark 実行順による偏りを避ける

---

## 16. 観測・ログ要件

以下は必須ログとする。

### 16.1 run_id 命名規則

```
{mode}_{repo_id}_{YYYYMMDD}_{short_sha}
```

- `mode`: `baseline` / `photon_tiny` / `photon_small` / `photon_600m_paper`
- `repo_id`: リポジトリ識別子（例: `fastapi_fastapi`）
- `YYYYMMDD`: run 開始日
- `short_sha`: `repo_commit` の先頭 7 桁

例: `baseline_fastapi_fastapi_20260412_9430409`

### 16.2 必須フィールド

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
- confidence score
- latency breakdown
- memory peak
- answer text
- grader score
- human notes

ログは後から failure analysis できる形で保存する。

---

## 17. 技術スタック

### 17.1 実装方針

- **PyTorch/MPS**: 正しさ確認用
- **MLX**: 本命実装 / Mac 最適化
- **Repo pipeline**: Python 主体
- **Serving**: ローカル API + CLI から開始

### 17.2 要件

- Mac Studio 単機で主要 benchmark が回る
- ローカル実行を基本とする
- 1 コマンドで benchmark を再実行できる
- baseline と experimental path が同一 harness 上で動く

---

## 18. 開発マイルストーン

| 期間 | マイルストーン | 成果物 |
|---|---|---|
| Week 1-2 | 仕様確定・baseline 起動 | `spec.md`, baseline RepoRAG, first eval set |
| Week 3-4 | ingest / retrieval / benchmark 固定 | indexer, retriever, benchmark harness |
| Week 5-8 | PHOTON系 forward 実装 | `photon_mlx/`, `torch_ref/`, mask/shape tests |
| Week 9-12 | 学習パス・teacher-forced eval | tiny/small training, drift metrics |
| Week 13-16 | session inference 実装 | PHOTON-RAG end-to-end |
| Week 17-20 | Safe RecGen 実装 | fallback policy, fallback logs, stress eval |
| Week 21-24 | Mac 最適化・公開準備 | report, failure cases, demo, decision memo |

---

## 19. Go / No-Go Gate

### Gate 1 (Week 4)
条件:
- baseline RepoRAG が安定稼働
- benchmark が再現可能
- citation 付き回答が出る

未達時:
- PHOTON 実装を始めず baseline を立て直す

### Gate 2 (Week 12)
条件:
- PHOTON系 forward と eval が安定
- tiny/small で follow-up 改善の兆候がある
- drift 指標が取得できる

未達時:
- PHOTON の scope を縮小
- summary memory 強化案を control として比較継続

### Gate 3 (Week 20)
条件:
- Safe RecGen により誤答率改善
- follow-up latency 改善が残る
- benchmark で baseline より実用価値がある

未達時:
- プロダクト投入は baseline 維持
- PHOTON は研究枝として公開

### Gate 4 (Week 24)
判断:
- baseline に統合する
- 限定ベータで運用する
- 研究成果として公開する
- 次フェーズに進める

---

## 20. リスクと対策

### リスク 1: drift による誤答
対策:
- Safe RecGen
- exact quote 時の強制再読
- high-risk query の fallback
- failure case catalog の継続更新

### リスク 2: retrieval が弱く model 改善が見えない
対策:
- baseline retrieval を先に強化
- graph expansion の改善
- citation error を retrieval / generation に分離して分析

### リスク 3: benchmark が曖昧で比較不能
対策:
- Week 2 で benchmark freeze
- repo snapshot 固定
- 採点ルーブリック明文化
- control 実装を 1 本置く

### リスク 4: scope が広がりすぎる
対策:
- Primary use case を RepoRAG に固定
- Agentic RAG は次フェーズへ送る
- 大規模自動修正は対象外にする

### リスク 5: Mac 上での学習コストが高い
対策:
- tiny/small を先に回す
- optional mid は Gate 2 通過後のみ
- baseline は既存 open-weight model で進める

---

## 21. 成果物

最低限の成果物は以下。

- `spec.md`
- `baseline_reporag/`
- `photon_mlx/`
- `torch_ref/`
- `bench/`
- `evals/`
- `reports/benchmark_report.md`
- `reports/failure_cases.md`
- `demo/`

---

## 22. リポジトリ構成案

```text
project-root/
├─ spec.md
├─ README.md
├─ configs/
│  ├─ baseline.yaml
│  ├─ photon_tiny.yaml
│  ├─ photon_small.yaml
│  └─ eval.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ indexes/
│  └─ eval_sets/
├─ baseline_reporag/
├─ photon_mlx/
├─ torch_ref/
├─ bench/
├─ evals/
├─ scripts/
├─ reports/
└─ demo/
```

---

## 23. Definition of Done

本プロジェクトは、以下を満たしたとき Done とみなす。

1. baseline RepoRAG が安定して動く  
2. PHOTON-RAG が同じ harness 上で比較できる  
3. Safe RecGen の発火条件と効果が測定できる  
4. multi-turn で baseline より価値があると示せる  
5. benchmark / report / failure cases が第三者に共有できる  
6. 次フェーズの意思決定ができる材料が揃う  

---

## 24. 次フェーズ候補

本仕様の次に検討する候補は以下。

- adaptive chunking
- topic-aware memory refresh
- CPU/GPU 分業スケジューリング
- Agentic RAG への拡張
- repo patch planning への拡張
- public demo / internal beta 展開

---

## 25. この仕様の意思決定

この仕様で固定すること。

- 最初の対象は **RepoRAG**
- まず作るのは **baseline**
- PHOTON は **比較可能な研究枝** として実装する
- 最終回答は **必ず局所証拠に再着地** させる
- 成功判定は **multi-turn 実用性** を中心に行う

この仕様で先送りすること。

- frontier 級基盤モデル学習
- 大規模 agent 化
- 自律コード修正の本番投入
- 1 本目からの大規模 public release

---