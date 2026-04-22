# PHOTON-RepoRAG チュートリアル

自分のリポジトリで PHOTON-RepoRAG を使う手順。
所要時間: セットアップ約 10 分 + ingest/index 約 5 分 + PHOTON 学習 約 2 時間。

---

## 前提条件

- macOS (Apple Silicon M1 以上)
- Python 3.12+
- RAM 32 GB 以上 (64 GB 推奨)
- 対象リポジトリがローカルにクローン済み

---

## Step 1: インストール

```bash
cd /path/to/photon-mlx
pip install -r requirements.txt
```

初回のみ。Qwen 14B モデル (~8 GB) と cross-encoder reranker `BAAI/bge-reranker-base` (~550 MB) は初回実行時に自動 DL されます。

---

## Step 2: 対象リポジトリの設定

### 2-1. config ファイルをコピー

```bash
cp configs/baseline.yaml configs/my_project.yaml
```

### 2-2. config を編集

```yaml
# configs/my_project.yaml

repo:
  repo_id: "my_project"                    # 任意の識別子 (英数字 + underscore)
  repo_path: "/path/to/my/project"         # 対象リポジトリのパス
  repo_commit: "HEAD"                      # 固定する場合は SHA を指定
```

他の設定はデフォルトのままで動作します。

---

## Step 3: Ingest (リポジトリ取り込み)

```bash
python scripts/ingest_repo.py \
  --repo /path/to/my/project \
  --repo-id my_project \
  --commit HEAD \
  --config configs/my_project.yaml
```

出力例:
```
repo_id:     my_project
repo_commit: a1b2c3d4...
  100 files  350 chunks ...
Done: 150 files, 420 chunks -> data/indexes/my_project/chunks.db
```

---

## Step 4: Index 構築

```bash
# BM25 + Embedding インデックス
python scripts/build_indexes.py \
  --repo-id my_project \
  --config configs/my_project.yaml

# Symbol graph (import/call 関係)
python scripts/build_symbol_graph.py \
  --repo-id my_project \
  --config configs/my_project.yaml
```

---

## Step 5: baseline_rag で動作確認

PHOTON なしの baseline で、まず動作を確認します。

```bash
python -m baseline_reporag.cli \
  --config configs/my_project.yaml \
  --repo-id my_project \
  --question "このリポジトリの主要モジュールは何ですか？"
```

回答が `[C:N]` 付きで返ってきたら成功。

### インタラクティブモード

```bash
python -m baseline_reporag.cli \
  --config configs/my_project.yaml \
  --repo-id my_project

# プロンプトが出たら質問を入力:
Q> 認証処理はどこにありますか？
Q> そこを変えたら何が壊れる？
Q> (空行で終了)
```

---

## Step 6: PHOTON 学習 (オプション)

PHOTON を使う場合のみ。baseline_rag だけで十分なら Step 7 へスキップ。

### 6-1. PHOTON config を作成

```bash
cp configs/photon_small.yaml configs/my_project_photon.yaml
```

編集:
```yaml
# configs/my_project_photon.yaml

repo:
  repo_id: "my_project"
  repo_path: "/path/to/my/project"
  repo_commit: "HEAD"

training:
  train_corpus: "./data/processed/train_my_project.jsonl"
  val_corpus: "./data/processed/val_my_project.jsonl"
  max_steps: 1000     # リポジトリが小さければ 500 で十分
```

### 6-2. 学習コーパス生成

```bash
python scripts/generate_training_corpus.py \
  --repo-id my_project \
  --config configs/my_project.yaml \
  --photon-config configs/my_project_photon.yaml \
  --output-dir data/processed \
  --commit HEAD
```

### 6-3. 学習実行

```bash
python scripts/train_photon.py --config configs/my_project_photon.yaml
```

所要時間: 500 steps で約 1 時間、1000 steps で約 2 時間 (M3 Ultra)。

**Early Stopping (Issue #60)**: `training.early_stopping.enabled: true` を設定すると、`patience` 回続けて `val_loss` が改善しない場合に学習を自動停止し、`restore_best: true` なら `final/` に最良チェックポイントを復元します。

学習中の val_loss を確認 (手動 CLI 実行時のデフォルト):
```bash
tail -f logs/train_log.jsonl
```

Streamlit アプリから起動した場合は run 別の log ディレクトリに分離されます:
```bash
tail -f logs/<job_id>/train_log.jsonl
```

---

## 長コンテキスト推論 (Issue #55)

`configs/photon_long_context.yaml` を指定すると、NTK-aware RoPE scaling により 2048 で学習済みのチェックポイントのまま最大 65,536 トークンの入力を受け取れます。**MLX 経路のみ** — `torch_ref` (PyTorch リファレンス) は従来通り 128 位置までしか扱えません（`scaling != "none"` で `NotImplementedError`）。

```yaml
# configs/photon_long_context.yaml (抜粋)
model:
  max_position_embeddings: 65536
  rope_theta: 10000000.0          # YAML 互換な数値リテラル。`10_000_000.0` は不可
  rope_scaling: ntk                # v1 は {"none", "ntk"} のみ
  rope_scale_factor: 32.0          # 2048 → 65536 で 32 倍
training:
  context_length: 32768
```

### ピーク メモリの実測値（ランダム重み、学習済みなしの参考値）

| prompt_len | KV cache あり | KV cache なし | 速度比 (cache / nocache) |
|---|---|---|---|
| 1,024 | 2.9 GB | 3.2 GB | 0.93x |
| 16,384 | **20.8 GB** | **13.1 GB** | **0.89x** |

- 長 prompt では **`use_kv_cache=False` の方が速くメモリも少なく消費**する場合があります（`top_level_increment` と `local_tail_decode` の累積が nocache の prefill を上回るため）
- 設計見積りより実測が大幅に大きいため、小メモリ環境（~32GB）では `use_kv_cache=False` を推奨
- 32,768 / 65,536 の実測は学習済みチェックポイントが整った段階で bench 再実行予定

詳細は `reports/issue-55-long-context.md` を参照。

### トラブルシュート

メモリ不足・速度劣化が起きた場合は `docs/troubleshooting.md` の「長コンテキストで RAM 不足」節を参照。

---

## Step 7: PHOTON-RAG で使う

### サーバモード

```bash
python -m baseline_reporag.server --config configs/my_project_photon.yaml
```

別ターミナルから:
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "認証処理の入口はどこですか？", "session_id": "my-session"}'
```

### CLI モード

```bash
python -m baseline_reporag.cli \
  --config configs/my_project_photon.yaml \
  --repo-id my_project
```

---

## Step 8: 効果を確認する

### baseline vs PHOTON の比較

同じ質問を baseline と PHOTON で試して、follow-up の速度差を体感:

```bash
# ターミナル 1: baseline (毎ターン同じ速度)
python -m baseline_reporag.cli --config configs/my_project.yaml --repo-id my_project

# ターミナル 2: PHOTON (follow-up が速い)
python -m baseline_reporag.cli --config configs/my_project_photon.yaml --repo-id my_project
```

試す質問例:
```
Q> このリポジトリの主要モジュールは？     ← Turn 1 (two_pass_search=false なら両方同じ速度)
Q> その中で一番大きいモジュールは？       ← Turn 2 (PHOTON が速い)
Q> そのモジュールの依存関係は？           ← Turn 3 (PHOTON がさらに速い)
Q> テストカバレッジはどうなっている？     ← Turn 4 (話題変更 → drift 検知)
```

---

## よくある問題

### Q: ingest で「0 files」になる

config の `include` パターンを確認。対象ファイルの拡張子が含まれているか:
```yaml
repo:
  include:
    - "**/*.py"
    - "**/*.ts"
    # 自分のプロジェクトの言語を追加
```

### Q: 回答に引用 [C:N] が付かない

retrieval が正しいファイルを拾えていない可能性。以下を試す:
1. `reranker.enabled: true` を確認
2. `query_expansion.enabled: true` を確認
3. 日本語で質問している場合、英語のコード用語も含めてみる

### Q: PHOTON 学習で val_loss が下がらない

- コーパスが小さすぎる (100 samples 未満) → リポジトリが小さい場合は baseline のみ推奨
- `max_steps` が多すぎて overfitting → `eval_every_steps: 50` で推移を確認

### Q: PHOTON の follow-up が baseline と変わらない

- `inference.evidence_pruning_enabled: true` を確認
- `inference.pruned_max_chunks: 8` を確認
- Turn 2 以降の pruning だけでは不足な場合は `retrieval.two_pass_search.enabled: true` で Turn 1 にも Pass 1 chunk 選別を入れる (Issue #56)

### Q: Turn 1 の検索精度を上げたい (retrieval.two_pass_search)

`retrieval.two_pass_search` を有効化すると、Turn 1 で以下の 2 パス処理を行う:

1. **Pass 1**: `hybrid_search` が `pass1_top_k` (既定 64) 件の候補を取得
2. **Pass 2**: PHOTON が質問と chunk の類似度を計算し、上位 `pass2_top_k` (既定 16) 件に絞り込んで evidence pack に渡す

```yaml
retrieval:
  two_pass_search:
    enabled: true        # Turn 1 にも Pass 1 PHOTON スコアリングを適用
    pass1_top_k: 64      # Pass 1 で取得する候補数 (fused_top_k 以上を推奨)
    pass2_top_k: 16      # Pass 2 (Qwen) に渡す件数 (evidence_pack.max_chunks と同値推奨)
```

- `enabled: false` (既定) で Turn 1 は従来の 1 パス挙動になる (後方互換)
- Turn 2 以降は `inference.evidence_pruning_enabled` が独立で効くため、両方 `true` も可
- Turn 1 TTFT は reranker が 64 件を処理するため +50% 程度増加する見込み
- プロファイラは Pass 1 実行時に `pass1_scoring` フェーズを記録 (Turn 2+ は従来通り `evidence_pruning`)

---

## ファイル構成 (セットアップ後)

```
data/
├── raw/my_project/          # 対象リポジトリ (ingest 元)
├── indexes/my_project/      # インデックス
│   ├── chunks.db            # チャンク DB
│   ├── lexical.pkl          # BM25
│   ├── embedding/           # ベクトル
│   └── symbol_graph.json    # 依存グラフ
├── processed/
│   ├── train_my_project.jsonl  # PHOTON 学習用
│   └── val_my_project.jsonl
│
configs/
├── my_project.yaml          # baseline 用
└── my_project_photon.yaml   # PHOTON 用
│
checkpoints/                 # PHOTON 学習済みモデル
├── best/                    # Early Stopping 有効時に最良 val_loss の重み
│   ├── weights.npz
│   └── state.json
└── final/                   # 学習終了時点の重み (restore_best=true なら best/ と同内容)
    ├── weights.npz
    └── state.json
```

Streamlit アプリから起動した場合は run 単位で名前空間が分離されます:

```
checkpoints/<repo_id>/<job_id>/
├── best/
├── final/
└── step_XXXXXX/
logs/<job_id>/train_log.jsonl
```
