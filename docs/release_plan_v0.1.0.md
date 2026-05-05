# v0.1.0 Release Plan

このドキュメントは、PHOTON-RepoRAG を MIT License の MVP として公開するための P2 手順をまとめます。

## Release Scope

v0.1.0 は Multi-turn RAG の MVP 初回公開版です。主な公開範囲は次の通りです。

- GitHub repository の source code / documentation
- GitHub Release の source archive
- `python -m build` で生成できる sdist / wheel artifact
- Streamlit 管理アプリを使った ingest / index / project registration / chat / comparison mode の導線
- `photon-rag` / `baseline-reporag` / `photon-train` / `photon-generate` の CLI entrypoints
- baseline vs PHOTON の評価手順と既存評価レポート

次は v0.1.0 の公開範囲外です。

- PHOTON checkpoint の同梱
- 外部 LLM / tokenizer / embedding model / reranker の重み同梱
- Hugging Face Hub での checkpoint 公開
- PyPI への即時 publish
- Hosted demo / managed service

## Distribution Decision

MVP v0.1.0 の主導線は GitHub Release と source checkout です。

理由:

- README の推奨フローが GitHub clone から Streamlit 管理アプリを起動する形である
- Streamlit 管理アプリは `app/photon_app.py` を repository checkout から実行する前提である
- checkpoint と外部モデル重みは利用者の権限で取得・配置する必要がある
- PyPI package は CLI smoke 済みだが、Streamlit app entrypoint / app assets packaging は別途整理した方がよい

PyPI は v0.1.0 時点で metadata / wheel smoke まで準備済みとし、publish は後続判断にします。

## GitHub Release Artifacts

含めるもの:

- GitHub auto-generated source archive
- `dist/photon_rag-0.1.0.tar.gz`
- `dist/photon_rag-0.1.0-py3-none-any.whl`
- Release notes

含めないもの:

- `checkpoints/`
- `projects/`
- `.cache/`
- `workspace/`
- local logs
- generated local reports
- external model weights
- tokenizer / embedding / reranker weights
- PHOTON checkpoint weights

## Tag Policy

- main merge 後に `v0.1.0` の annotated tag を作成する
- tag message は `PHOTON-RepoRAG v0.1.0 MVP` とする
- develop branch や PR head には release tag を打たない
- GitHub Release は `v0.1.0` tag から作成する

想定コマンド:

```bash
git checkout main
git pull origin main
git tag -a v0.1.0 -m "PHOTON-RepoRAG v0.1.0 MVP"
git push origin v0.1.0
```

## Pre-release Checks

P0 / P1 で実施済み:

- MIT license / dependency license metadata / external model license 表示確認
- `python -m build`
- wheel install smoke
- CLI help smoke
- package artifact exclusion checks
- main unit tests
- Streamlit helper / component tests
- scenario scorer / eval job / CI gate tests

P2 で実施すること:

- main 向け PR を作成する
- PR CI の通過を確認する
- release notes を PR description または GitHub Release notes に転記できる状態にする

## Known MVP Limits

- Streamlit の完全な実データ E2E は対象 corpus と checkpoint に依存するため、release candidate 確認で実施する
- PHOTON checkpoint は同梱しないため、PHOTON mode の利用者は checkpoint を自身で用意する必要がある
- PyPI publish は後続判断。CLI package としては smoke 済みだが、Streamlit app の package entrypoint は未提供
- PHOTON standalone API は experimental 扱い
