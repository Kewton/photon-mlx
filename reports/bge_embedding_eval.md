# bge-small-en-v1.5 Embedding Evaluation Report (Issue #90)

> **Note on continuity**: 本レポート以降の Static NC / MT NC 数値は
> `BAAI/bge-small-en-v1.5` を embedding モデルに採用した状態での測定値です。
> それ以前のレポート（`reports/gate2_judgment_v*.md`, `reports/bge_reranker_eval.md` 等）は
> `sentence-transformers/all-MiniLM-L6-v2` 前提の測定値であり、モデル差し替え前後の
> 値を直接比較する際は本注記に留意してください。

## 1. 変更概要

- **対象 Issue**: #90 feat(retrieval): embedding モデルを bge-small-en に更新して semantic 検索品質向上
- **親 Epic**: #81（Static NC < 15% 達成のための retrieval チューニング）
- **依存 Issue（マージ済み）**: #95 (grid search), #96 (bge-reranker-base)
- **モデル出自**: BAAI (北京智源人工智能研究院), MIT License
- **変更内容**:
  - Embedding モデル: `sentence-transformers/all-MiniLM-L6-v2` (23M, 384 dim, 英語汎用)
    → `BAAI/bge-small-en-v1.5` (33M, 384 dim, code-aware)
  - `EmbeddingIndex.__init__` のデフォルト `model_id` を bge に変更
  - `EmbeddingIndex.load(dir_path, *, expected_model_id=None)` を新設し、config と
    永続化 `model_id.txt` の不一致を `ValueError` で検出（silent な model 継続運用を防止）
  - YAML `indexing.embedding.normalize: true` は dead key のまま維持（コード側で hardcoded、
    dead key 注記コメントを YAML に追加）
  - Streamlit UI (`app/photon_app.py`) のモデル選択肢に bge を追加（既定は
    multilingual-e5-small のまま維持、D4）

## 2. 受入条件と実測値

### 2-1. 本 PR 内で達成（config + コード変更のみ）

| 項目 | 目標 | 実測 | 判定 |
|------|------|------|------|
| 全 6 YAML の `model_id` が `BAAI/bge-small-en-v1.5` | 一致 | **6/6** (baseline, photon_600m_paper, photon_long_context, photon_small, photon_tiny, photon_tiny_recgen) | ✅ |
| `EmbeddingIndex.__init__` デフォルトが bge | 一致 | **bge** | ✅ |
| `load(dir_path, *, expected_model_id=None)` 不一致で `ValueError` | raise | **raise** (T4 pass) | ✅ |
| `pipeline_factory.py` / `app/photon_app.py` が新 API 使用 | 使用 | **使用** (grep 確認済み) | ✅ |
| `baseline_reporag/tests/test_embedding.py` 新設 | 新設 | **新設 (8 tests)** | ✅ |
| `pytest` 全パス | pass | **TBD** (CI で測定) | — |
| `ruff check .` / `ruff format --check .` | clean | **TBD** (CI で測定) | — |

### 2-2. マージ後 eval で測定（post-merge）

| 項目 | 目標 | 実測 | 判定 |
|------|------|------|------|
| 全 repo の index 再構築完了 | 完了 | **TBD** | — |
| Static no-citation | 20.0% → **< 16%** (-4pp 以上) | **TBD** | — |
| MT no-citation | 6.7% を +1pp 以内で維持 | **TBD** | — |
| Index 構築時間 | ~10 分/repo | **TBD** | — |
| Embedding 初回 DL サイズ | 参考値 ~130 MB | **TBD** | — |

> **ベンチマーク未実行**: Static NC / MT NC / index build time の実測は本 PR では
> 実施せず、PR マージ後に `scripts/run_baseline_eval.py` と
> `scripts/run_multi_turn_eval.py` で測定する。測定後、本レポートの TBD セクションを
> 更新する。

## 3. 実測プロトコル（測定時の手順）

### 3-1. 実モデル smoke

```bash
RUN_SLOW_TESTS=1 python -m pytest \
  baseline_reporag/tests/test_embedding.py::test_bge_small_en_v1_5_loads_and_encodes -v
```

期待値: `BAAI/bge-small-en-v1.5` を HuggingFace から初回ダウンロード (~130 MB)
→ `encode(['hello world'])` が `numpy.ndarray` shape `(1, 384)` を返す。

### 3-2. Index 再構築（全 repo）

```bash
# 旧 MiniLM index を削除
rm -rf data/indexes/*/embedding/

# 各 repo について bge で再構築
python scripts/build_indexes.py --repo-id fastapi_fastapi
python scripts/build_indexes.py --repo-id <other-repo>   # 対象 repo 全て
```

### 3-3. Static NC 測定

```bash
python scripts/run_baseline_eval.py --config configs/baseline.yaml
```

**測定条件（交絡除去のため固定）**:
- `retrieval.weights.lexical = 0.45`
- `retrieval.weights.embedding = 0.45`
- `retrieval.weights.graph = 0.10`
- `retrieval.embedding_top_k = 20`
- `retrieval.fused_top_k = 16`
- `retrieval.rerank_top_k = 12`
- `retrieval.reranker.model_id = "BAAI/bge-reranker-base"` (#96 と同)

### 3-4. MT NC 測定

```bash
python scripts/run_multi_turn_eval.py --config configs/baseline.yaml
```

上記と同じ retrieval パラメータ固定。

## 4. 判定

**TBD** — 実測完了後に以下の観点で判定:

1. Static NC < 16% 達成 → ✅ 採用
2. Static NC >= 16% かつ MT NC 劣化 +1pp 以内 → △ 条件付き採用（後続 Issue で追加対策）
3. MT NC 劣化 +1pp 超 → ✗ revert または nomic-embed / multilingual-e5 への切替検討

## 5. 既知の挙動

- `baseline_reporag/indexing/embedding.py:44` で `text[:2048]` 文字数 truncation を実施。
  bge-small-en-v1.5 の token 上限は 512（MiniLM は 256）。文字数ベースのため bge の
  token 上限を超える場合は sentence-transformers 側で内部的に truncate される（壊れは
  しないが情報ロスあり）。token ベース truncation への変更は別 Issue で扱う。
- YAML `indexing.embedding.normalize` 設定は **dead key**。実挙動は
  `embedding.py:50,58` で hardcoded True。将来コード側で制御したくなった時点で別
  Issue として引数化する。
