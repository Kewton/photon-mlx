# Playground — 自分用試行錯誤 Cookbook

PHOTON-RepoRAG で「こういう実験をしたい」と思ったときに引く cookbook。
A-1 (自分用 dev 環境) のスコープで書かれている。配布や運用は対象外。

> 採用構成 (2026-04-28): **PHOTON + Qwen3.5-9B-MLX-4bit no-think**
> Evidence: [`reports/qwen_model_matrix_20260428_400cmp_report.md`](../reports/qwen_model_matrix_20260428_400cmp_report.md)

---

## 0. 前提

| 項目 | 値 |
|------|----|
| Python | 3.12+ |
| MLX | Apple Silicon (M1/M2/M3) 必須 |
| メモリ | 32GB+ 推奨 (Qwen3.5-9B + bge-m3 で ~12GB) |
| 主な config | `configs/baseline.yaml`, `configs/photon_small.yaml`, `configs/institutional_docs_photon.yaml` |

PHOTON checkpoint の場所:
```bash
export PHOTON_CHECKPOINT_ROOT=/path/to/checkpoints  # 例: $HOME/photon-checkpoints
# 採用 checkpoint: photon_institutional_retrain_20260428/step_003000
```

---

## 1. 新しい repo を試したい

### Cookbook 1-1: GitHub から repo を ingest して問い合わせ

```bash
# clone (既存なら省略)
git clone https://github.com/<owner>/<repo> /tmp/<repo>

# ingest + indexing 一括
make prepare REPO=/tmp/<repo> REPO_ID=<repo>

# 問い合わせ
make ask REPO_ID=<repo> Q="このリポジトリのエントリポイントは？"
```

### Cookbook 1-2: Python 以外の repo (markdown 中心など)

`indexing.symbol_graph.enabled: false` を含む config を使う:

```bash
make prepare CONFIG=configs/institutional_docs.yaml REPO=/path/to/docs REPO_ID=docs
make ask CONFIG=configs/institutional_docs.yaml REPO_ID=docs Q="..."
```

### Cookbook 1-3: 巨大 repo で chunk 数を絞りたい

`configs/baseline.yaml` の `evidence_pack.max_chunks` / `max_tokens` を下げる。
ローカル試験用に config をコピーして編集:

```bash
cp configs/baseline.yaml configs/local.small.yaml
# editor で max_chunks: 16 → 8 などに変更
make ask CONFIG=configs/local.small.yaml REPO_ID=<repo> Q="..."
```

---

## 2. baseline と PHOTON を比較したい

### Cookbook 2-1: 同じ質問で 2 condition 比較 (CLI レベル)

```bash
# baseline
make ask CONFIG=configs/baseline.yaml REPO_ID=fastapi_fastapi Q="認証処理の入口は？"

# PHOTON
export PHOTON_CHECKPOINT_ROOT=/path/to/checkpoints
make ask CONFIG=configs/photon_small.yaml REPO_ID=fastapi_fastapi Q="認証処理の入口は？"
```

### Cookbook 2-2: bench で N 件まとめて比較

```bash
make eval                      # configs/eval.yaml の variants を全実行
make eval-baseline             # baseline 単独
make eval-photon               # PHOTON 単独
```

`configs/eval.yaml` の `variants` に config を並べると 1 run で複数比較できる。

### Cookbook 2-3: 出力パスをカスタムにしたい

```bash
python bench/run_all.py --config configs/eval_qwen_model_matrix.yaml --run-id "my-test-$(date +%Y%m%d)"
```

成果物は `workspace/temp/<output_dir>/<run_id>/...` に出る。

---

## 3. 別の LLM を試したい

### Cookbook 3-1: 既存 config の model_id だけ差替え

```bash
cp configs/baseline.yaml configs/test_qwen35.yaml
# editor: model.model_id を別モデルに変更
# 例: "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
make ask CONFIG=configs/test_qwen35.yaml REPO_ID=<repo> Q="..."
```

### Cookbook 3-2: 4-variant matrix で評価する (Qwen 2.5 vs 3.5 の例)

`configs/eval_qwen_model_matrix.yaml` (120件) または `eval_qwen_model_matrix_400.yaml` (400件) を参考に書き換え:

```yaml
variants:
  - id: "baseline_modelA"
    config_path: "./configs/baseline.yaml"
    override:
      model:
        model_id: "your/model-A"
  - id: "baseline_modelB"
    ...
```

実行:
```bash
python bench/run_all.py --config configs/eval_qwen_model_matrix.yaml
# resume したいときは --resume を付ける (途中再開)
python bench/run_all.py --config configs/eval_qwen_model_matrix.yaml --resume
```

### Cookbook 3-3: Qwen 3.5 系で `/think` モードを試す

`baseline_reporag/generation/qwen_thinking.py` で `enable_thinking=False` がデフォルト。
明示的に think を試すなら、user メッセージの先頭に `/think` directive を入れて、
`normalize_qwen_thinking` の `enable_thinking` 引数を `True` に切替える。

---

## 4. PHOTON checkpoint を切替えたい

### Cookbook 4-1: 別の training run を試す

```bash
# 採用 checkpoint
export PHOTON_CHECKPOINT_ROOT=/path/to/checkpoints
# configs/institutional_docs_photon.yaml の checkpoint_path:
#   "photon_institutional_retrain_20260428/step_003000" がデフォルト

# 別 step を試す
cp configs/institutional_docs_photon.yaml configs/local.photon_step1k.yaml
# editor: checkpoint_path を photon_institutional_retrain_20260428/step_001000 などに変更
make ask CONFIG=configs/local.photon_step1k.yaml REPO_ID=<repo> Q="..."
```

### Cookbook 4-2: 自分で再学習する

```bash
# corpus 準備
python scripts/generate_institutional_training_corpus.py --config configs/institutional_docs_photon_retrain.yaml

# 学習
python scripts/train_photon.py --config configs/institutional_docs_photon_retrain.yaml

# checkpoint は configs/institutional_docs_photon_retrain.yaml の output_dir に出る
```

詳細: `docs/deployment.md` の "PHOTON Checkpoint Distribution" を参照。

---

## 5. デバッグ・観察

### Cookbook 5-1: 1 question の retrieval 結果を観察したい

`baseline_reporag.cli` は `--debug` で内部状態を吐ける場合あり (要 cli.py 確認)。
直接 Python から:

```python
from baseline_reporag.config import load_config
from baseline_reporag.pipeline_factory import build_pipeline

cfg = load_config("configs/baseline.yaml")
pipeline = build_pipeline(cfg)
result = pipeline.query(question="...", session_id="debug-1", repo_id="fastapi_fastapi")
print(result.cited_chunk_ids)
print(result.evidence_pack)
```

### Cookbook 5-2: latency / memory profile

```bash
python bench/run_all.py --config configs/eval.yaml
# 結果 JSONL の latency_ms / memory_peak_mb を集計
```

`scripts/export_report.py --run-id <id>` でサマリーを出せる。

### Cookbook 5-3: demo シナリオで挙動確認

```bash
make demo-list                          # 利用可能なシナリオ
make demo SCENARIO=demo-01              # 実行
```

---

## 6. クリーンアップ

```bash
make clean                              # __pycache__ など
```

worktree のクリーンアップは:
```bash
git worktree list
git worktree remove <path>
```

---

## 7. よくあるハマりどころ

| 症状 | 対処 |
|------|------|
| `ImportError: mlx_lm is required` | `pip install mlx-lm` |
| PHOTON checkpoint が見つからない | `PHOTON_CHECKPOINT_ROOT` 環境変数を export 確認 |
| evaluation が遅い (> 10 分) | `max_cases` / `max_sessions` を絞る or `--resume` で再開 |
| `CUDA not available` のような警告 | MLX は Apple Silicon 専用、CUDA 関連は無視で OK |
| disk 不足 | model cache (`~/.cache/huggingface/`) と `data/indexes/` を確認 |

---

## 関連ドキュメント

- `README.md` — プロジェクト全体像
- `docs/deployment.md` — 本番運用の手順 (A-1 では参考程度)
- `docs/troubleshooting.md` — よくある障害と対処
- `docs/code_review_checklist.md` — PR レビュー時の観点
- `CLAUDE.md` — Claude Code 用のプロジェクト規約
