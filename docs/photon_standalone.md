# PHOTON Standalone Guide

このドキュメントは、RAG pipeline を使わずに `photon_mlx` を単体利用するための導線です。

## 位置づけ

OSS MVP の主プロダクトは Multi-turn RAG です。一方で、`photon_mlx/` は RAG から独立した PHOTON runtime / training layer として実装されています。

| 用途 | 推奨入口 |
|---|---|
| Multi-turn RAG として使う | `photon-rag` CLI または Streamlit |
| PHOTON checkpoint を学習する | `photon-train` |
| PHOTON checkpoint で greedy decode する | `photon-generate` |
| Python から PHOTON model / inference を直接使う | `photon_mlx` public API |

`photon_mlx` は `baseline_reporag` に依存しません。RAG 統合は `baseline_reporag/photon_pipeline.py` 側で行います。

## インストール

ローカル開発では repository root から editable install します。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

PHOTON-only 用の optional extra は `photon` です。

```bash
pip install -e ".[photon]"
```

現時点の `project.dependencies` は MVP 互換性のため RAG 依存も含みます。将来、PHOTON-only 配布をより軽量にする場合は、既定 dependencies をさらに分割します。

## CLI

### 学習

```bash
photon-train --config configs/photon_tiny.yaml
```

checkpoint と log の出力先を明示する場合:

```bash
photon-train \
  --config configs/photon_tiny.yaml \
  --checkpoint-dir checkpoints/my_photon_run \
  --log-dir logs/my_photon_run
```

### 生成

```bash
photon-generate \
  --config configs/photon_tiny.yaml \
  --checkpoint checkpoints/my_photon_run/final \
  --prompt "def get_current_user(" \
  --max-new-tokens 96
```

`photon-generate` は PHOTON checkpoint から直接 greedy decode します。RAG の retrieval、evidence pack、citation は使いません。

## Python API

```python
from photon_mlx import PhotonInference, PhotonModel, load_checkpoint, load_photon_config

cfg = load_photon_config("configs/photon_tiny.yaml")
model = PhotonModel(cfg)
state = load_checkpoint(model, "checkpoints/my_photon_run/final")

# tokenizer は利用側で用意します。
inference = PhotonInference(model, cfg, tokenizer)
```

主な public API:

| API | 役割 |
|---|---|
| `PhotonModel` | PHOTON 階層 decoder model |
| `PhotonInference` | session-aware inference / evidence scoring runtime |
| `load_photon_config` | PHOTON YAML config loader |
| `load_checkpoint` / `save_checkpoint` | PHOTON checkpoint I/O |
| `CheckpointState` | checkpoint state DTO |

## RAG 統合との違い

PHOTON standalone は、テキスト列を PHOTON model に入力し、生成や scoring を行う層です。以下は扱いません。

- repository ingest
- vector DB / BM25 / embedding index
- retrieval / reranker
- evidence pack
- citation marker
- Streamlit 比較モード

これらを使う場合は `photon-rag` または Streamlit を使います。PHOTON-RAG 統合では、`baseline_reporag/photon_pipeline.py` が retrieval 結果や会話履歴を PHOTON に渡し、evidence selection や citation 制御に score を利用します。

## 現在の制限

- PHOTON standalone API は experimental です。
- tokenizer は利用側で明示的に用意する必要があります。
- `PhotonInference` の session metadata には `repo_id` / `repo_commit` という名前が残っています。PHOTON-only では namespace metadata として扱ってください。
- RAG なしで citation や Retrieval debug は生成されません。
