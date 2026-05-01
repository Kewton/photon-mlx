# Release Checklist

このドキュメントは、PHOTON-RepoRAG を MIT License の MVP として公開する前に確認する項目をまとめます。

## P0: MIT リリース必須項目

- [x] `LICENSE` を追加する
- [x] `pyproject.toml` の `license` を `MIT` に変更する
- [x] README にライセンス節を追加する
- [x] 生成物・ローカル成果物を `.gitignore` に追加する
- [x] 直接依存ライブラリのライセンスメタデータを確認する
- [ ] 使用する外部モデルのライセンスを、配布直前のモデルカードで確認する
- [ ] 公開する checkpoint の由来、学習データ、再配布可否を確認する
- [ ] GitHub Release / PyPI / Hugging Face Hub などの配布対象に、ローカル生成物や非公開データが混ざらないことを確認する

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

MVP リリースでは、少なくとも次を確認してください。

- README / config に記載している model ID のモデルカードを確認する
- モデル重みを release artifact に同梱しない場合でも、利用条件を README または docs に明記する
- checkpoint を公開する場合は、学習元データとベースモデルのライセンスが再配布を許可していることを確認する
- private corpus、ローカル `projects/`、`checkpoints/`、`.cache/`、生成レポートを release artifact に含めない

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

既に Git 管理下にある評価レポートやテンプレート config は、`.gitignore` の対象でも引き続き Git 管理されます。新しいテンプレート config を追加する場合は、必要に応じて `git add -f` または `.gitignore` の例外を追加してください。
