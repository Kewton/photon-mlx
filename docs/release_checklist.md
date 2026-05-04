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
- `scripts/` package は `photon-rag ingest/index/symbol-graph/heading-graph` と `photon-train` / `photon-generate` が利用するため wheel に含める。
- 個人環境向け exporter (`scripts/export_agent_training_data.py`) は `.gitignore` と `MANIFEST.in` で配布対象外にする。
- PyPI sdist では `MANIFEST.in` により `tests/`, `workspace/`, `checkpoints/`, `projects/`, `.cache/`, generated reports を除外する。
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

P1 対応で、`scripts/` package は CLI entrypoint が依存する runtime package として wheel に含める方針へ戻しました。一方で、個人環境向け exporter は `.gitignore` と `MANIFEST.in` により package artifact から除外します。テストコードとローカル生成物は引き続き sdist / wheel に含めません。

`python -m build` で `dist/photon_rag-0.1.0.tar.gz` と `dist/photon_rag-0.1.0-py3-none-any.whl` を生成し、次の除外対象が package artifact に含まれないことを確認しました。

- `scripts/export_agent_training_data.py`
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
| `scripts/summarize_eval_matrix.py` | P1 の評価集計補助として採用 |
| `tests/test_summarize_eval_matrix.py` | `scripts/summarize_eval_matrix.py` の regression test として採用。sdist / wheel には含めない |

## P1: 品質ゲート・動作確認

確認日: 2026-05-04

- [x] `python -m build` で wheel / sdist を作成できることを確認する
- [x] package artifact にローカル生成物、checkpoint、workspace、tests が混入しないことを確認する
- [x] wheel install smoke を実行する
- [x] `photon-rag --help` を確認する
- [x] `photon-rag ask --help` を確認する
- [x] `photon-rag ingest --help` を確認する
- [x] `photon-rag index --help` / `symbol-graph --help` / `heading-graph --help` を確認する
- [x] `baseline-reporag --help` を確認する
- [x] `photon-train --help` を確認する
- [x] `photon-generate --help` を確認する
- [x] release smoke tests を実行する
- [x] 主要 unit tests (`torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/`) を実行する
- [x] Streamlit 関連 unit tests を実行する
- [x] PHOTON standalone API smoke tests を実行する
- [x] 評価集計補助スクリプトの unit tests を追加・実行する
- [x] scenario-2 scorer / eval job / CI gate tests を実行する

P1 での判断:

- `scripts/` package は CLI entrypoint が利用するため wheel に含める。
- `scripts/export_agent_training_data.py` は個人環境の SQLite log exporter なので配布対象外を維持する。
- `scripts/summarize_eval_matrix.py` は個人環境依存がなく、評価結果の集計に使えるため採用する。
- 主要 unit tests は権限付き実行で `1672 passed`。通常 sandbox では `ps` を使う zombie 検知テストが権限不足で失敗するため、release gate では権限付き実行結果を採用する。
- `workspace/テストシナリオ2.md` ベースの baseline vs PHOTON 評価結果と既知制約は `docs/evaluation.md` に整理済み。P1 では scorer / eval job の regression tests で再現性を確認した。
- Streamlit の実データ ingest / index / chat / comparison の完全な手動 E2E は、対象 corpus と checkpoint が必要なため P1 では unit / helper tests と起動導線の確認に留める。実データ E2E は release candidate 確認で実施する。

## P2: MVP リリース手順・公開準備

確認日: 2026-05-05

- [x] `pyproject.toml` の `version = "0.1.0"` を MVP 初回公開版として採用する
- [x] MVP の公開範囲を GitHub repository + GitHub Release 中心に決定する
- [x] PyPI metadata / classifiers / optional dependencies を確認する
- [x] GitHub Release に含める artifact と含めない artifact を明確化する
- [x] Hugging Face Hub での checkpoint 公開は MVP v0.1.0 の範囲外にする
- [x] README の install 手順を GitHub clone + Streamlit 起動の公開導線に合わせる
- [x] tag 作成方針を決める
- [x] release notes を作成する
- [ ] main ブランチ向け PR を作成する
- [ ] PR の CI 通過を確認する

P2 での判断:

- `0.1.0` は、Multi-turn RAG の MVP として「初期公開だが API / UX は今後変わり得る」状態を示すため妥当。`Development Status :: 3 - Alpha` を package metadata に追加する。
- MVP v0.1.0 の主導線は GitHub clone からの Streamlit アプリ起動とする。Streamlit 管理アプリは repository checkout 前提なので、README では PyPI install 単独ではなく source checkout を推奨する。
- GitHub Release には GitHub が自動生成する source archive に加え、必要に応じて `python -m build` で生成した `dist/photon_rag-0.1.0.tar.gz` と `dist/photon_rag-0.1.0-py3-none-any.whl` を添付できる。
- GitHub Release / package artifact には、checkpoint、外部モデル重み、tokenizer、embedding / reranker 重み、`workspace/`、`projects/`、`checkpoints/`、`.cache/`、ローカル生成レポートを含めない。
- PyPI 公開は metadata と wheel smoke の準備済み。ただし MVP の最初の公開では GitHub Release を優先し、PyPI publish は Streamlit アプリの package entrypoint / app asset packaging 方針を別途確認してから実施する。
- Hugging Face Hub への checkpoint 公開は v0.1.0 では行わない。公開する場合は、別 Issue で checkpoint repo、revision、model card、base model、training data、license、再配布可否を確認する。
- tag は main merge 後に `v0.1.0` の annotated tag として作成する。develop や PR head には release tag を打たない。
- release notes は `docs/release_notes_v0.1.0.md` を草案として管理する。
