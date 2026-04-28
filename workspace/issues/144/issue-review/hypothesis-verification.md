# Issue #144 仮説検証レポート

**対象 Issue**: test(retrieval): ruri-small-v2 (V2) build failure follow-up — manual spiece.model download workaround
**検証日**: 2026-04-28
**検証者**: Claude Code (Phase 0.5)

---

## 抽出された仮説一覧

| # | カテゴリ | 仮説/主張 |
|---|---------|----------|
| H1 | 真因 | ruri-small-v2 リポジトリの `spiece.model` が HF cache に漏れている (cache 不在) |
| H2 | 真因 | sentence-transformers 5.4.0 + transformers 5.5.3 の組合せでは DistilBert architecture の SentencePiece tokenizer (`spiece.model`) を自動 fetch できていない |
| H3 | 真因 | AutoTokenizer fallback も失敗し、AutoProcessor fallback も failed → "Unrecognized processing class" として最終的に raise する |
| H4 | 真因 | ruri-base (同じ vendor) も同じ問題を抱える |
| H5 | 前提 | sentence-transformers 5.4.0 / transformers 5.5.3 が現在の環境にインストールされている |
| H6 | 前提 | `baseline_reporag.indexing.embedding.EmbeddingIndex` は `SentenceTransformer(model_id)` を直接呼び出して embedding を build する |
| H7 | 前提 | configs/institutional_docs.yaml の embedding は現在 V4 (BAAI/bge-m3) が採用されており、V2 計測は別 variant config 切替で実行する |
| H8 | スコープ | 5-variant A/B (#137) において V0/V1/V3/V4 のメトリクスは確定済みで、V2 のみ補完すれば 5-variant 完走となる |
| H9 | 影響 | V4 採用判定は V2 が補完されても変わらない (低確率で V2 が V4 を上回る場合のみ再判定が必要) |

---

## 各仮説の検証

### H1: ruri-small-v2 の spiece.model cache 漏れ

**判定**: **Unverifiable**（実行時環境依存 — コードベース照合では確認不可）

**根拠**:
- HF cache (`~/.cache/huggingface/hub/...`) はユーザー実行時のローカル状態であり、リポジトリには含まれない
- Issue 内の `find ~/.cache/huggingface/hub/models--cl-nagoya--ruri-small-v2 -name "*spiece*"` 結果 (空) は当該マシンでの実測値
- HF Hub 側の存在 (`spiece.model 439391 bytes`) は外部 Web 状態で、コードでは検証不能

**Stage 1 への申し送り**: 真因の最終確定はユーザー環境での実測に依存するため、ワークアラウンド (Option A: 手動 download) は cache 不在仮説が True/False のいずれでも有効に機能することを設計レビューで確認する。

---

### H2: sentence-transformers 5.4.0 + transformers 5.5.3 の互換性問題

**判定**: **Unverifiable**（バージョン依存 — 実環境照合が必要）

**根拠**:
- リポジトリには `pyproject.toml` / `requirements.txt` 等で依存関係を管理しているが、実際にインストールされているバージョンとは別物
- 2026-04 時点で sentence-transformers 5.4.0 の自動 tokenizer fetch ロジックがどの形式に対応しているかは Web 情報に依存

**Stage 1 への申し送り**: バージョン pin による Option B は副作用 (他モデルの動作影響) が大きいため、Option A (個別 file fetch) を優先する設計判断が妥当か、設計フェーズで再評価する。

---

### H3: AutoTokenizer / AutoProcessor fallback も failed

**判定**: **Unverifiable**（実行時 traceback に依存）

**根拠**:
- Issue 内の `ValueError: Couldn't instantiate the backend tokenizer ...` および `Unrecognized processing class in cl-nagoya/ruri-base` はユーザー側で再現したエラー情報
- 当該エラーが sentence-transformers / transformers 内のどの分岐から raise されているかはバージョン固有でコードベース外

---

### H4: ruri-base も同じ問題

**判定**: **Partially Confirmed（部分確認）**

**根拠**:
- ruri-small-v2 と ruri-base は同じ vendor (`cl-nagoya`) で SentencePiece tokenizer を使う点は構造的に類似
- ただし、Issue 内で ruri-base 自体は Phase B 5-variant に含まれておらず、変更対象は ruri-small-v2 のみ
- ruri-base まで対象に含める提案 (Task 1 の `RURI_MODELS = ['cl-nagoya/ruri-small-v2', 'cl-nagoya/ruri-base']`) は **スコープ過大** の可能性 — 設計レビューで確認すべき

**Stage 1 への申し送り**: scripts/_setup_ruri_models.py で `cl-nagoya/ruri-base` を `RURI_MODELS` に含める必要性は薄い (Issue 受入条件でも ruri-base を fix する記載なし)。スコープを `ruri-small-v2` のみに絞ることを検討する。

---

### H5: 環境のライブラリバージョン

**判定**: **Unverifiable**（実環境依存）

**根拠**: `pyproject.toml` 等を読めば pinned version が分かるが、実際に `pip list` した結果との一致は実行時にしか確認できない。

---

### H6: EmbeddingIndex は SentenceTransformer を直接呼ぶ

**判定**: **Confirmed（確認済み）**

**根拠**:
- `baseline_reporag/indexing/embedding.py:9`: `from sentence_transformers import SentenceTransformer`
- `baseline_reporag/indexing/embedding.py:95`: `self._model = SentenceTransformer(self._model_id)`
- `EmbeddingIndex._model_()` の実装で、tokenizer は SentenceTransformer 内部の自動取得に委ねられる構造

**実装インパクト**: ワークアラウンドは「SentenceTransformer 呼び出し前に spiece.model を先に download する」形でスクリプトレベル (build_indexes 実行前) に挿入するのが自然。EmbeddingIndex の改造は不要。

---

### H7: configs/institutional_docs.yaml は V4 採用

**判定**: **Confirmed（確認済み）**

**根拠**:
- `configs/institutional_docs.yaml:84-87`:
  ```yaml
  # #137 Phase B: 5-variant A/B (V0 e5-small, V1 e5-base, V2 ruri 計測不能,
  # V3 bge-reranker swap, V4 bge-m3) で V4 採用 (NC -6.90pt vs V0 12.93%)。
  # 詳細: reports/institutional_retrieval_ab.md
  model_id: "BAAI/bge-m3"
  ```
- V2 計測は **base config 改変ではなく**、別の variant config 切替もしくは CLI 引数 override が前提と読める。

**Stage 1 への申し送り**: V2 build を再実行する具体手順 (どの config / どの CLI 引数で `cl-nagoya/ruri-small-v2` を指定するか) が Issue 本文に記載なし。設計フェーズで明確化が必要。

---

### H8: 4 variant メトリクス確定 + V2 のみ補完

**判定**: **Partially Confirmed（部分確認）**

**根拠**:
- `reports/institutional_retrieval_ab.md` に V0/V1/V3/V4 のメトリクス記載あり (grep 結果)
- ただし、本検証では report の具体的な数値・確定状態は Stage 1 通常レビューで再確認する

---

### H9: V4 採用判定は不変

**判定**: **Confirmed（推測ベース、構造的に妥当）**

**根拠**:
- V4 (bge-m3) は `NC -6.90pt vs V0` で大幅改善している (config コメント記載)
- V2 (ruri-small-v2 = 67M params 級の small model) が V4 (bge-m3 = 568M params 級) を超える可能性は低い (パラメータ数とモデル容量の観点から)
- ただし、この判定は事前期待であり、計測結果次第で覆る可能性も Issue 内に明記されている

---

## 仮説検証サマリー

| # | 仮説/主張 | 判定 |
|---|----------|------|
| H1 | spiece.model cache 漏れ | Unverifiable |
| H2 | sentence-transformers 互換性問題 | Unverifiable |
| H3 | AutoTokenizer/AutoProcessor fallback 失敗 | Unverifiable |
| H4 | ruri-base も同じ問題 | Partially Confirmed |
| H5 | 環境のライブラリバージョン | Unverifiable |
| H6 | EmbeddingIndex は SentenceTransformer を直接呼ぶ | Confirmed |
| H7 | configs/institutional_docs.yaml は V4 採用 | Confirmed |
| H8 | 4 variant メトリクス確定 + V2 のみ補完 | Partially Confirmed |
| H9 | V4 採用判定は不変 | Confirmed (構造的推測) |

---

## Stage 1 通常レビューへの申し送り事項

1. **H4 関連**: scripts/_setup_ruri_models.py で `cl-nagoya/ruri-base` を含める必要性が Issue 受入条件と整合しない (受入条件は ruri-small-v2 のみ)。スコープ縮小を検討すること。
2. **H7 関連**: V2 build を再実行する具体手順 (config / CLI 引数 / variant 切替方法) が Issue 本文に記載なし。Issue 本文補足が必要。
3. **H1/H2/H3 関連**: 真因 (spiece.model cache 漏れ vs sentence-transformers 互換性問題) の最終確定はワークアラウンド成功時の挙動でしか判別不能。Option A 採用後に「真因が cache 不在ではなく互換性問題だった場合」の fallback 戦略 (Option B / Option C) に関する記載が薄い → 設計フェーズで明確化。
4. **`/data/` 等の path 整合性**: 影響ファイルに `data/indexes/institutional_documents_V2/embedding/`、`logs/institutional_V2_*.jsonl` とあるが、既存 build_indexes / run_baseline_eval スクリプトの output path 命名規則と整合しているか Stage 1 でコード照合する。
