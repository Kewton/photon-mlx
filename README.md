# PHOTON-RepoRAG

PHOTON-RepoRAG は、業務文書・制度文書・コードリポジトリに対する **Multi-turn RAG** です。単発の文書検索ではなく、会話の流れを踏まえて、前の質問で扱った対象・条件・比較軸を引き継ぎながら回答します。

例えば、1 ターン目で「セーフティネット保証1号の認定基準」を聞いたあと、2 ターン目で「2号との違いは？」と質問するケースがあります。従来の RAG は現在の質問だけで検索しがちですが、業務利用ではこのような省略質問・比較質問・条件追加が頻繁に発生します。

このプロダクトは、現在の質問だけでは不足する文脈を会話履歴から補い、回答に使う根拠を選び直すことで、実務に近い対話型 RAG を実現することを目的にしています。

## 解決する業務課題

- **省略質問への対応**: 「それは？」「2号との違いは？」「必要書類も同じ？」のような質問でも、前の会話文脈を踏まえて回答する
- **条件引き継ぎ**: 対象制度、対象部門、前提条件、比較軸をターン間で維持する
- **根拠の欠落防止**: 現在質問だけでは拾えない evidence を、関連する過去質問から補完する
- **不要文脈の混入抑制**: 会話履歴を全部渡すのではなく、現在質問に関係する履歴だけを使う
- **説明可能な回答**: 回答、引用、retrieval debug、比較メトリクスを確認できる形で出力する
- **業務導入前の比較検証**: baseline RAG と multi-turn RAG の回答・引用・メトリクスを Streamlit 上で比較する

## 従来の RAG / Agentic RAG との違い

| 方式 | 得意なこと | 課題 | このプロダクトとの違い |
|---|---|---|---|
| 従来の RAG | 現在質問に対する文書検索と回答 | 省略質問、条件引き継ぎ、比較質問に弱い | 現在質問だけでなく、関連する過去質問と evidence を使って回答する |
| 会話履歴を全部入れる RAG | 実装が簡単 | 不要な履歴が混ざり、長い会話ほど精度・コスト・レイテンシが悪化する | 必要な過去文脈だけを選び、回答に使う evidence を絞る |
| 質問書き換え型 RAG | 「2号との違いは？」を完全な検索クエリに補正できる | 書き換え品質に依存し、誤った前提を補う可能性がある | 書き換えだけに頼らず、関連過去質問と evidence を明示的に扱う |
| Agentic RAG | 複雑な調査、複数回検索、ツール利用 | 遅い、コストが高い、挙動が不安定、評価しにくい | 業務 Q&A に必要な文脈引き継ぎと根拠選別を、制御されたパイプラインで扱う |

Agentic RAG は、何を調べるべきか自体を探索するタスクに向いています。一方、このプロダクトは、既存の文書・規程・制度・コードを対象に、ユーザーとの会話を引き継ぎながら正しい根拠へ着地する業務 Q&A に向いています。

## 主な特徴

- **Multi-turn chat**: 過去質問を踏まえた follow-up 質問に対応
- **関連過去質問の利用**: 現在質問と関係する過去質問を回答生成時の参考情報として渡す
- **関連 evidence 補完**: 関連過去質問から取得した evidence を現在質問の evidence に追加
- **根拠チャンク保護**: retrieval / reranker 上位チャンクを保護し、必要情報の欠落を抑える
- **比較モード**: baseline と multi-turn variant の回答、引用、メトリクス、差分を比較
- **Retrieval debug**: 使用されたチャンク、source、rank、score、引用差分を確認
- **Streamlit 管理 UI**: ingest / index / training / project 登録 / chat / 比較確認を画面上で実行
- **CLI 再利用**: Streamlit で作成した config と checkpoint を CLI から利用

## 適したユースケース

- 制度文書・社内規程・業務マニュアルに対する問い合わせ対応
- 自治体・行政文書の条件確認、制度比較、必要書類確認
- 法務・契約・金融領域の根拠付き Q&A
- コールセンターやヘルプデスクでの対話型ナレッジ検索
- コードリポジトリのオンボーディング、影響範囲分析、関連モジュール探索

## 技術詳細

README ではプロダクトの目的と始め方に絞っています。内部で使っている PHOTON の役割、working memory、hierarchical prefill、scoring、evidence pruning の詳細は [PHOTON technical overview](docs/photon_technical_overview.md) を参照してください。

## クイックスタート

> **推奨ローカル LLM 構成 (2026-04-28)**: Qwen3.5-9B-MLX-4bit no-think モード。詳細は [`reports/qwen_model_matrix_20260428_400cmp_report.md`](reports/qwen_model_matrix_20260428_400cmp_report.md) と [`docs/playground.md`](docs/playground.md) 参照。

### MVP 手順: Streamlit アプリで確認してから CLI で使う

MVP では、まず Streamlit アプリで ingest / index / チャット確認を行い、必要に応じて multi-turn 用の設定や checkpoint を作成します。その後、Streamlit で作成された config と checkpoint を CLI から再利用する流れを推奨します。

#### 1. GitHub から clone して環境を作る

```bash
git clone https://github.com/Kewton/photon-mlx.git
cd photon-mlx

python -m venv .venv
source .venv/bin/activate
pip install -U pip

# ローカル検証では repository root から wheel / editable install する。
# PyPI 公開後は `pip install photon-rag` に置き換える。
pip install -e .

# Streamlit 管理アプリを使う場合は追加で入れる。
pip install streamlit
```

#### 2. Streamlit アプリを起動して動作確認を始める

```bash
streamlit run app/photon_app.py --server.port 8501
```

ブラウザで表示された Streamlit 画面から、次の順で進めます。

1. **ベクトルデータベース作成**
   - 「対象リポジトリのディレクトリ」に RAG 対象 repo のローカルパスを入力
   - `repo_id` を入力
   - Config と Embedding モデルを選び、`作成開始`
   - status が `completed` になるまで待つ

2. **Training**
   - 同じ対象 repo を選び、最大ステップ数などを設定して `学習開始`
   - 完了すると checkpoint が `checkpoints/<repo_id>/<train_job_id>/` 配下に作成される
   - 代表的には `best/`, `final/`, `step_XXXXXX/` のいずれかを CLI で使う

3. **RAG プロジェクト登録**
   - 作成済みの `repo_id` を選択
   - `Config ファイル` には baseline 側の YAML を選択
     - 通常のコード repo: `configs/baseline.yaml`
     - Markdown / 制度文書 corpus: `configs/institutional_docs.yaml`
   - PHOTON を使う場合は `PHOTON モデル (checkpoint)` を選択し、PHOTON settings から PHOTON 側 YAML を生成
   - 回答生成モデル、temperature、retrieval/reranker 上位保護 N 件、PHOTON score 選別 M 件、関連過去質問/evidence 件数を必要に応じて設定
   - `登録`

4. **チャット**
   - 登録したプロジェクトを選び、質問して Streamlit 上で回答、citation、Retrieval debug、PHOTON score、drift 表示を確認する
   - 比較モードを使う場合は、baseline 側 config と PHOTON 側 config が別々に登録されている必要がある

#### Streamlit での比較モード

比較モードでは、同じ質問を baseline pipeline と PHOTON pipeline に投げ、回答、引用、メトリクス、Retrieval debug をターンごとに比較します。

登録時の考え方:

| 項目 | 選ぶもの |
|---|---|
| Config ファイル | baseline 側 YAML。例: `configs/baseline.yaml` または `configs/institutional_docs.yaml` |
| PHOTON config | `model.provider: photon` の YAML。例: PHOTON settings で生成した `projects/<project_name>/photon.yaml` |
| 回答生成モデル | 既定は `mlx-community/Qwen3.5-9B-MLX-4bit` |
| Temperature | 既定は `0.0` |
| retrieval/reranker 上位保護 N 件 | 既定は `4` |
| PHOTON score 選別 M 件 | 既定は `4` |

比較結果では、以下を確認します。

- **回答差分**: baseline と PHOTON の回答内容の違い
- **回答中に引用マーカーとして使われたチャンク差分**: 実際に回答本文で `[C:N]` として使われた chunk の差分
- **Retrieval debug 比較**: `source`, `PHOTON score`, `PHOTON current`, `PHOTON session`, `Used`, `Citation` の違い
- **PHOTON の効き方**: PHOTON score が付いた件数、除外された候補、working memory 由来の根拠

#### 3. Streamlit で作成したモデルを CLI から使う

CLI は Streamlit の `.cache/photon_app_state.json` を直接読むのではなく、同じ `repo_id`、config、checkpoint を明示して使います。

Streamlit の Training で作成された config は通常 `configs/photon_<repo_id>.yaml`、checkpoint は `checkpoints/<repo_id>/<train_job_id>/` 配下にあります。CLI で使う checkpoint に合わせて config の `model.checkpoint_path` を `final`、`best`、または `step_XXXXXX` に設定してください。

```bash
# 例: Streamlit training job の final checkpoint を使う
export PHOTON_CHECKPOINT_ROOT="$(pwd)/checkpoints/<repo_id>/<train_job_id>"

photon-rag ask \
  --config configs/photon_<repo_id>.yaml \
  --repo-id <repo_id> \
  --question "このリポジトリの主要な入口はどこですか？"
```

`PHOTON_CHECKPOINT_ROOT` は `model.checkpoint_path` の親ディレクトリとして解決されます。たとえば `model.checkpoint_path: "final"` なら、上の例では `checkpoints/<repo_id>/<train_job_id>/final` が読み込まれます。

Streamlit の **PHOTON settings** で `projects/<project_name>/photon.yaml` を生成した場合は、その YAML を CLI に渡せます。

```bash
export PHOTON_CHECKPOINT_ROOT="$(pwd)/checkpoints/<repo_id>/<train_job_id>"

photon-rag ask \
  --config projects/<project_name>/photon.yaml \
  --repo-id <repo_id> \
  --question "前回の質問を踏まえて関連モジュールを教えてください"
```

PHOTON checkpoint を GitHub / Hugging Face Hub などから自動取得する場合は、公開先を `PHOTON_CHECKPOINT_REPO_ID` または `model.checkpoint_repo_id` に設定します。既に配置済みの場合は、従来通り `PHOTON_CHECKPOINT_ROOT` 配下の `model.checkpoint_path` が使われます。

```bash
export PHOTON_CHECKPOINT_ROOT="$HOME/.cache/photon-rag/checkpoints"
export PHOTON_CHECKPOINT_REPO_ID="<org>/<checkpoint-repo>"

photon-rag ask \
  --config configs/institutional_docs_photon.yaml \
  --repo-id <repo_id> \
  --question "..."
```

#### 4. CLI だけで ingest / index する場合

Streamlit を使わず CLI だけで準備する場合は以下を実行します。

```bash
# 対象 repo / markdown corpus を ingest
photon-rag ingest \
  --repo /path/to/target-repo \
  --repo-id target_repo \
  --commit HEAD

# index を構築
photon-rag index --repo-id target_repo

# Python repo では symbol graph、markdown corpus では heading graph を必要に応じて構築
photon-rag symbol-graph --repo-id target_repo
photon-rag heading-graph --repo-id target_repo --config configs/institutional_docs.yaml

# 1 問問い合わせ
photon-rag ask --repo-id target_repo --question "認証処理の入口はどこですか？"

# server 起動
photon-rag serve --config configs/baseline.yaml
```

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

multi-turn variant で動かす場合は `CONFIG` を切替えます。

```bash
# checkpoint を使って問い合わせ
export PHOTON_CHECKPOINT_ROOT=/path/to/checkpoints  # checkpoint の親ディレクトリ
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

## プロジェクト構成

```text
project-root/
├─ app/                  # Streamlit 管理アプリ
├─ baseline_reporag/     # ingest / index / retrieval / generation / comparison
├─ photon_mlx/           # multi-turn variant で使うローカル推論・学習実装
├─ configs/              # baseline / multi-turn / institutional docs 用 config
├─ bench/                # 評価ベンチマーク
├─ docs/                 # 技術詳細・運用メモ
├─ reports/              # 評価レポート
└─ tests/                # unit / component tests
```

## 評価で見るポイント

このプロダクトの評価軸は、単発回答の正しさだけではありません。multi-turn で前提を引き継げているか、回答根拠が妥当か、引用が説明可能かを確認します。

- **回答品質**: task correctness、session consistency、hallucination rate
- **根拠品質**: citation precision、citation recall、evidence の過不足
- **運用品質**: P50 / P90 latency、memory peak、fallback rate
- **説明可能性**: retrieval debug、引用差分、比較メトリクス

`workspace/テストシナリオ2.md` を使った baseline vs PHOTON の再現可能なスコア評価は [Evaluation guide](docs/evaluation.md) を参照してください。

## ライセンス

このリポジトリのコードとドキュメントは MIT License で公開します。詳細は [LICENSE](LICENSE) を参照してください。

依存ライブラリ、外部モデル、学習済み checkpoint、配布先から取得する重みは、それぞれの提供元ライセンスに従います。MVP リリース前の確認項目は [Release checklist](docs/release_checklist.md) を参照してください。

## 関連ドキュメント

- [PHOTON technical overview](docs/photon_technical_overview.md): PHOTON の役割、技術要素、multi-turn RAG への転用方法
- [Development notes](docs/development_notes.md): 開発モード、評価観点、Gate、Definition of Done
- [Release checklist](docs/release_checklist.md): MIT MVP リリース前の確認項目
- [Deployment guide](docs/deployment.md): checkpoint 配布とデプロイ運用
- [Evaluation guide](docs/evaluation.md): multi-turn シナリオ評価とスコア化
- [Playground guide](docs/playground.md): ローカル検証の補足
- [Troubleshooting](docs/troubleshooting.md): エラー時の確認ポイント
- [Tutorial](docs/tutorial.md): 操作手順の補足
