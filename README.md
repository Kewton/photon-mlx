# PHOTON-RepoRAG

PHOTON 系の階層 working memory を使って、**巨大 repo に対する multi-turn RepoRAG** を高速化・省メモリ化するための開発リポジトリです。

本プロジェクトは 2 本立てで進めます。

1. **Baseline RepoRAG**
   - まず動く、比較可能なプロダクト線
   - hybrid retrieval + citation 付き回答 + session memory

2. **PHOTON-RAG**
   - 階層 working memory を導入した研究開発線
   - multi-turn follow-up のレイテンシとメモリを改善
   - drift を検知して fallback する **Safe RecGen** を実装

---

## 目的

このリポジトリの目的は次の 3 つです。

- **RepoRAG の baseline を最短で成立**させる
- **PHOTON 系の working memory を比較可能な形で実装**する
- **Safe RecGen を含む評価基盤**を作り、multi-turn 実用性で勝負する

---

## 何を解くか

通常の RAG は 1 問目には強い一方で、同じ repo に対する連続質問では次の問題が起きやすいです。

- 毎ターン大量の文脈を再読する
- 会話が伸びるほどレイテンシとメモリが悪化する
- 「いま何を調べているか」という作業状態を維持しにくい
- repo 全体の構造理解と局所コード読解を何度も往復するのが重い

このプロジェクトでは、次の役割分担を採用します。

- **検索**: Source of Truth
- **Session Pack**: セッションで重要な証拠束
- **PHOTON Working Memory**: 粗い全体状態と作業仮説
- **最終回答**: 必ず局所証拠に再着地して生成

---

## 対象ユースケース

- repo オンボーディング
- 影響範囲分析
- 障害解析
- 変更計画の比較
- 関連モジュール探索
- 設計意図の把握

---

## 非対象

v1 では以下は扱いません。

- repo 全体の大規模自動リライト
- 自律コミット
- 出典なしの断定回答
- 汎用チャットボット化
- frontier 級基盤モデルのフル事前学習

---

## 全体アーキテクチャ

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
        -> v0: Flat Session Memory
        -> v1: PHOTON Working Memory
        -> v1.1: PHOTON + Safe RecGen
    -> Answer Generator
    -> Citation Resolver
    -> Logger / Evaluator
```

⸻

リポジトリ構成

```
project-root/
├─ spec.md
├─ README.md
├─ tasks.md
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

⸻

開発方針

このプロジェクトでは、次の順番を崩しません。
1.	まず baseline を作る
2.	次に評価器を固める
3.	その上で PHOTON を実装する
4.	最後に Safe RecGen と最適化を入れる

最初から新アーキテクチャ一本に賭けないことが重要です。

⸻

開発ステータス
 - v0 Baseline RepoRAG
 - v1 PHOTON Working Memory
 - v1.1 Safe RecGen
 - Benchmark Harness
 - Evaluation Set Freeze
 - Benchmark Report
 - Failure Case Catalog

⸻

クイックスタート

> **採用構成 (2026-04-28)**: PHOTON + Qwen3.5-9B-MLX-4bit no-think モード。
> 詳細は [`reports/qwen_model_matrix_20260428_400cmp_report.md`](reports/qwen_model_matrix_20260428_400cmp_report.md) と [`docs/playground.md`](docs/playground.md) 参照。

### Makefile を使う最短手順

```bash
# 1. 環境作成 (.venv が無い場合のみ)
make setup
source .venv/bin/activate

# 2. 対象 repo を ingest + index 構築 (一括)
make prepare REPO=/path/to/target-repo REPO_ID=target_repo

# 3. CLI で 1 問
make ask REPO_ID=target_repo Q="認証処理の入口はどこですか？"

# 4. server を起動
make serve

# 5. 評価ベンチマーク
make eval

# 6. 利用可能な target 一覧
make help
```

PHOTON pipeline で動かす場合は `CONFIG` を切替えるだけ:

```bash
# PHOTON 採用 checkpoint で問い合わせ
export PHOTON_CHECKPOINT_ROOT=/path/to/checkpoints  # 採用 checkpoint の親ディレクトリ
make ask CONFIG=configs/photon_small.yaml REPO_ID=target_repo Q="..."
```

### 手動コマンド (Makefile を使わない場合)

1. 環境作成
```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2. 設定ファイル作成
```
cp .env.example .env
cp configs/baseline.yaml configs/local.baseline.yaml
```

3. 対象 repo を ingest
```
python scripts/ingest_repo.py \
  --repo /path/to/target-repo \
  --repo-id target_repo \
  --commit HEAD
```

4. index を構築
```
python scripts/build_indexes.py --repo-id target_repo
python scripts/build_symbol_graph.py --repo-id target_repo
```

> Symbol graph の構築は **optional** です（Issue #109）。Python ソース中心の repo ではデフォルト（`indexing.symbol_graph.enabled: true`）で build/load されますが、制度文書の markdown など Python シンボルが存在しない repo では YAML で `indexing.symbol_graph.enabled: false` を指定すると `build_symbol_graph.py` は早期 return し、runtime でも `SymbolGraph.load` は呼ばれず `graph=None` で pipeline が組み立てられます。`expand_with_graph` は file-neighbors のみで動作するため retrieval は壊れません。

5. baseline RepoRAG を起動
```
python -m baseline_reporag.server --config configs/local.baseline.yaml
```

6. CLI で問い合わせ
```
python -m baseline_reporag.cli \
  --repo-id target_repo \
  --question "認証処理の入口はどこですか？"
```

7. benchmark を実行
```
python bench/run_all.py --config configs/eval.yaml
```

8. レポート出力
```
python scripts/export_report.py --run-id latest
```

⸻

開発モード

v0: Baseline RepoRAG
 - open-weight instruct model
 - hybrid retrieval
 - lexical
 - embedding
 - symbol / graph expansion
 - citation 付き回答
 - simple session memory
 - benchmark 可能

v1: PHOTON-RAG
 - hierarchical encoder
 - converter
 - local decoder
 - session-level working memory
 - follow-up ターン高速化
 - drift 記録

v1.1: Safe RecGen
 - drift 検知
 - topic shift 検知
 - exact quote / patch 生成時の強制再読
 - fallback reason の記録
 - baseline path への一時退避

⸻

評価の考え方

このプロジェクトの成功条件は、単発回答ではなく multi-turn 実用性 に置きます。

品質指標
 - task correctness
 - citation precision
 - citation recall
 - session consistency
 - hallucination rate
 - patch / diff grounding quality

性能指標
 - P50 / P90 latency
 - tokens/sec
 - peak memory
 - memory / active session
 - retrieval time
 - prefill time
 - decode time
 - fallback rate

安全指標
 - no-citation assertion rate
 - wrong citation rate
 - stale memory rate
 - missed fallback rate

⸻

成功条件

詳細は spec.md を参照しますが、実務上の成功条件は次の通りです。
 - follow-up レイテンシが baseline 比で明確に改善する
 - session あたりメモリが baseline より減る
 - citation precision を落とさない
 - multi-turn の前提保持が向上する
 - Safe RecGen によって危険ケースで堅い経路へ戻れる

⸻

比較対象

以下を同じ harness で比較します。
1.	Baseline-RAG
2.	Baseline-RAG + summary memory
3.	PHOTON-RAG
4.	PHOTON-RAG + Safe RecGen

⸻

ログ方針

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

⸻

Gate

Gate 1

baseline RepoRAG が安定稼働し、benchmark が再現可能か。

Gate 2

PHOTON 系 forward / eval が安定し、tiny/small で改善の兆候があるか。

Gate 3

Safe RecGen によって誤答率改善とレイテンシ改善が両立するか。

Gate 4

baseline 統合、限定ベータ、研究枝公開のどれに進むか判断できるか。

⸻

実装ルール
 - benchmark を先に凍結する
 - repo snapshot を固定する
 - exact quote / diff / patch は必ず局所再読する
 - retrieval と generation の失敗を分離して分析する
 - すべての比較は同じ harness 上で行う
 - 「動いた」ではなく「比較できる」を Done とする

⸻

主要ディレクトリの役割

baseline_reporag/

最初に価値を出すプロダクト線。
比較対象であり、最後まで control として維持する。

photon_mlx/

本命実装。
Mac 向け最適化と PHOTON 系 working memory を入れる。

torch_ref/

正しさ確認用。
mask、shape、forward、teacher-forced eval の検証に使う。

bench/

公平比較の中心。
評価条件、run、集計、グラフ出力をまとめる。

evals/

静的問題、multi-turn セッション、stress eval を管理する。

reports/

benchmark report と failure cases を蓄積する。

⸻

最初にやること
 - spec.md を確定する
 - 対象 repo を 1 つに固定する
 - benchmark を freeze する
 - baseline RepoRAG を動かす
 - citation が出るところまで最短で到達する

⸻

今後の拡張候補
 - adaptive chunking
 - topic-aware memory refresh
 - CPU/GPU 分業スケジューリング
 - Agentic RAG
 - patch planning
 - internal beta / public demo

⸻

Definition of Done

このリポジトリは、次を満たしたら 1 つの区切りとします。
1.	baseline RepoRAG が安定して動く
2.	PHOTON-RAG が同じ harness 上で比較できる
3.	Safe RecGen の発火条件と効果が測定できる
4.	multi-turn で baseline より価値があると示せる
5.	report と failure cases が第三者に共有できる

⸻

参考ドキュメント
 - spec.md: 開発仕様と意思決定の基準
 - tasks.md: 実行タスク一覧
 - reports/: 評価レポートと失敗事例