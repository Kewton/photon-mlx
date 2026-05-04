# Release Checklist

このドキュメントは、PHOTON-RepoRAG を MIT License の MVP として公開する前に確認する項目をまとめます。

## P0: MIT リリース必須項目

- [x] `LICENSE` を追加する
- [x] `pyproject.toml` の `license` を `MIT` に変更する
- [x] README にライセンス節を追加する
- [x] 生成物・ローカル成果物を `.gitignore` に追加する
- [x] 直接依存ライブラリのライセンスメタデータを確認する
- [x] 使用する外部モデルのライセンスを、配布直前のモデルカードで確認する
- [x] 公開する checkpoint の由来、学習データ、再配布可否を確認する
- [x] GitHub Release / PyPI / Hugging Face Hub などの配布対象に、ローカル生成物や非公開データが混ざらないことを確認する

確認日: 2026-05-04

MVP 方針:

- このリポジトリの MIT License は、Git 管理下のコードとドキュメントに適用する。
- 外部モデル、tokenizer、embedding model、reranker、学習済み checkpoint の重みは同梱しない。
- PHOTON checkpoint は MVP の GitHub Release / PyPI package / repository artifact に含めない。
- `configs/*_photon*.yaml` の `model.checkpoint_path` はローカル配置または利用者管理の checkpoint を参照する設定例であり、重みの再配布を意味しない。
- 開発・評価用の `scripts/` package は MVP wheel の package discovery から除外する。source checkout で必要に応じて利用する。
- PyPI sdist では `MANIFEST.in` により `scripts/`, `tests/`, `workspace/`, `checkpoints/`, `projects/`, `.cache/`, generated reports を除外する。
- checkpoint を将来公開する場合は、学習元 corpus、ベースモデル、tokenizer、派生物の再配布可否を別途確認し、配布先の model card / license / data provenance を用意する。

## 直接依存ライブラリの確認結果

以下はローカル環境の Python package metadata から確認した直接依存のライセンス情報です。transitive dependency まで含む最終的な法務判断ではありません。

| Package | Observed license metadata |
|---|---|
| `fastapi` | MIT classifier |
| `huggingface_hub` | Apache-2.0 |
| `httpx` | BSD-3-Clause |
| `mlx` | MIT |
| `mlx-lm` | MIT |
| `numpy` | BSD-3-Clause family metadata; binary wheels may bundle additional compatible libraries |
| `python-dotenv` | BSD-3-Clause |
| `pyyaml` | MIT |
| `rank-bm25` | Apache2.0 |
| `sentence-transformers` | Apache-2.0 |
| `torch` | BSD-3-Clause |
| `tqdm` | MPL-2.0 AND MIT metadata |
| `transformers` | Apache-2.0 |
| `uvicorn` | BSD classifier |

## モデル・checkpoint の扱い

このリポジトリの MIT License は、リポジトリ内のコードとドキュメントに適用します。外部モデル、tokenizer、embedding model、reranker、学習済み checkpoint はそれぞれ別のライセンス・利用条件を持ちます。

MVP リリースでは、モデル重みと checkpoint を同梱しません。利用者の環境で Hugging Face Hub 等から取得される外部モデルは、それぞれのモデルカードと提供元ライセンスに従います。

### 外部モデル確認結果

以下は 2026-05-04 時点の Hugging Face model card 表示に基づく確認です。モデル ID を変更する場合、または checkpoint を公開する場合は再確認してください。

| Model ID | 用途 | 確認したライセンス表示 | 備考 |
|---|---|---|---|
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit) | 既定の回答生成モデル | Apache-2.0 | `Qwen/Qwen3.5-9B` 由来の MLX quantized model。重みは同梱しない |
| [`mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-Coder-14B-Instruct-4bit) | tokenizer 互換 / 代替生成モデル | Apache-2.0 | `Qwen/Qwen2.5-Coder-14B-Instruct` 由来の MLX quantized model。重みは同梱しない |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | baseline embedding | Apache-2.0 | 既定の軽量 embedding。重みは同梱しない |
| [`sentence-transformers/all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | Streamlit UI の embedding 候補 | Apache-2.0 | optional model。重みは同梱しない |
| [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | baseline reranker | Apache-2.0 | query-passage reranking 用。重みは同梱しない |
| [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | institutional profile embedding | MIT | Markdown / 制度文書向け profile で使用。重みは同梱しない |
| [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | institutional profile reranker | Apache-2.0 | Markdown / 制度文書向け profile で使用。重みは同梱しない |
| [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) | PHOTON institutional config の embedding | MIT | 既存 PHOTON institutional config で使用。重みは同梱しない |
| [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) | Streamlit UI の embedding 候補 | MIT | optional model。重みは同梱しない |
| [`mlx-community/Mistral-7B-Instruct-v0.3-4bit`](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3-4bit) | docs/playground の代替例 | Apache-2.0 | optional example。重みは同梱しない |
| [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) | PHOTON tiny / paper template tokenizer | Meta Llama 2 Community License | gated / custom license。MVP の既定導線では推奨しない。重み・tokenizer は同梱しない |

### Checkpoint 方針

MVP では、PHOTON checkpoint を公開 artifact に含めません。

- `configs/photon_small.yaml` の `checkpoint_path: "step_000600"` はローカル checkpoint 参照例です。
- `configs/institutional_docs_photon.yaml` の `checkpoint_path: "photon_institutional_retrain_20260428/step_003000"` はローカル運用向け参照例です。
- どちらも GitHub Release / PyPI package に checkpoint 重みを同梱するものではありません。
- 利用者は `PHOTON_CHECKPOINT_ROOT` または `model.checkpoint_repo_id` / `PHOTON_CHECKPOINT_REPO_ID` を使って、自身が利用権限を持つ checkpoint を配置・取得します。
- checkpoint を Hugging Face Hub 等で公開する場合は、別 Issue で model card、base model、training data、license、再配布可否を確認します。

## 配布対象から除外するもの

`.gitignore` で以下を除外しています。

- `checkpoints/`
- `projects/`
- `.cache/`
- `logs/`
- `data/raw/`
- `data/processed/`
- `data/indexes/`
- `data/training/`
- `build/`
- `dist/`
- `*.egg-info/`
- Streamlit などで生成される `configs/photon_*.yaml`
- 生成レポートの一部
- `workspace/`
- `reports/scenario2_baseline_vs_photon_score_*`
- `scripts/export_agent_training_data.py`

既に Git 管理下にある評価レポートやテンプレート config は、`.gitignore` の対象でも引き続き Git 管理されます。新しいテンプレート config を追加する場合は、必要に応じて `git add -f` または `.gitignore` の例外を追加してください。

### 配布物混入確認

2026-05-04 時点で、リリース対象は Git 管理下の source tree と `python -m build` が生成する package artifact に限定します。Git 管理外のローカル生成物は release artifact に含めません。

P0 対応として、`pyproject.toml` の package discovery から `scripts*` を除外し、`MANIFEST.in` で sdist から `scripts/` と `tests/` を除外しました。これにより、個人環境向け exporter や未採用の評価補助スクリプトが package artifact に混入することを防ぎます。開発・評価用スクリプトは source checkout から実行します。

`python -m build` で `dist/photon_rag-0.1.0.tar.gz` と `dist/photon_rag-0.1.0-py3-none-any.whl` を生成し、次の除外対象が package artifact に含まれないことを確認しました。

- `scripts/`
- `tests/`
- `checkpoints/`
- `projects/`
- `workspace/`
- `.cache/`
- ローカル `reports/scenario2_baseline_vs_photon_score_*`

確認時点の未追跡ファイルの扱い:

| Path | 扱い |
|---|---|
| `reports/scenario2_baseline_vs_photon_score_20260502.csv` | ローカル評価レポート。配布対象外 |
| `reports/scenario2_baseline_vs_photon_score_20260502.json` | ローカル評価レポート。配布対象外 |
| `reports/scenario2_baseline_vs_photon_score_20260502.md` | ローカル評価レポート。配布対象外 |
| `scripts/export_agent_training_data.py` | 個人環境の SQLite log exporter。配布対象外 |
| `scripts/summarize_eval_matrix.py` | P1 の評価ゲートで採用可否を判断する。未採用の間は配布対象外 |
| `tests/test_summarize_eval_matrix.py` | `scripts/summarize_eval_matrix.py` と同時に P1 で採用可否を判断する。未採用の間は配布対象外 |
