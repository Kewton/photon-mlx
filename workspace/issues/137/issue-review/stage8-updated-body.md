## 背景

#133 Phase A (PR #136 merged, `96e7b45`) で **EmbeddingIndex.max_input_chars 設定可能化** + guard test まで完了。本 Issue は **Phase B: 実機 A/B eval (5 variant × institutional eval 116Q)**。

**プロファイル区別**: 本 A/B は `configs/institutional_docs.yaml` (institutional プロファイル) 限定。`configs/baseline.yaml` (global default = `sentence-transformers/all-MiniLM-L6-v2`) は #114 invariant test (`tests/test_pipeline_factory_yaml_invariants.py::test_baseline_yaml_reranker_model_id_unchanged`、PR #132 で merge) で保護され不変。

worker artifacts (#133 worker branch / 別 worktree に保存。本 worktree には未取り込み):
- `workspace/issues/133/work-plan.md` (23KB) — A/B 実験計画詳細
- `workspace/design/issue-133-multilingual-retrieval-ab-design-policy.md` (49.8KB)

variant configs は **Phase B で新規作成** (`.gitignore` に `configs/_experiments/` 先行登録済、現時点では未作成):
- `configs/_experiments/institutional_V[0-4].yaml` を 5 本作成
- 各 config の `repo.repo_id` は eval 時の index load 先と一致させる (V0/V3=`institutional_documents`、V1/V2/V4=`institutional_documents_V<N>`)
- 現行 `load_config()` は `extends` / base include を解釈しないため、各 variant config は `configs/institutional_docs.yaml` の full copy を元に必要フィールドだけ変更する (差分 YAML を使う場合は、CLI に渡す前に full YAML へ展開する生成手順が必要)

## 5 Variant 計画

| Variant | Embedding | Reranker | max_input_chars | rationale |
|---------|-----------|----------|-----------------|-----------|
| V0 | `intfloat/multilingual-e5-small` (現 institutional default) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (現 default、global と共通) | 2048 | 比較基準 (NC 11.21%) |
| V1 | `intfloat/multilingual-e5-base` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 2048 | embedding ↑ |
| V2 | `cl-nagoya/ruri-small-v2` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 2048 | 日本語特化 embedding |
| V3 | `intfloat/multilingual-e5-small` | `BAAI/bge-reranker-v2-m3` | 2048 | 多言語 reranker |
| V4 | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` | **8192** (chars ≒ 4096-8192 tokens、bge-m3 の token 上限 8192 とほぼ整合) | 長文 + 多言語の最強級組合せ |

> **chars vs tokens 注記**: `max_input_chars` は文字数 (`text[:N]` 単位) であり、トークナイザ token 上限ではない。日英混在では char:token 比率が 1:1〜4:1 と幅広く、8192 char で英語は token 上限超過の可能性、日本語は余地未活用の可能性がある。token 単位の正確な truncate 化は別 Issue で follow-up 候補。

## 想定 compute

| Variant | repo_id | re-build_indexes? | eval | 所要 |
|---------|---------|-------------------|------|------|
| V0 | `institutional_documents` (既存) | ❌（embedding=e5-small 既存と同一） | ~30 min | ~30 min |
| V1 | `institutional_documents_V1` | ✅（embedding 変更）| + ~30 min | ~60 min |
| V2 | `institutional_documents_V2` | ✅ | ~60 min | ~60 min |
| V3 | `institutional_documents` (V0 と共有) | ❌（embedding=e5-small 同一、reranker は推論時切替）| ~30 min | ~30 min |
| V4 | `institutional_documents_V4` | ✅ + max_input_chars=8192 | ~30 min | ~60 min |

**戦略**: V0/V3 は embedding が同一 (e5-small) なので index (`institutional_documents`) を共有。reranker は推論時切替のため再 build 不要。V1/V2/V4 は embedding 変更のため専用 `repo_id` で別 index を構築する。

**合計**: 約 4-5 時間（M3 Ultra ローカル）。

### リソース見積もり (Phase B 一時占有分)

**追加ディスク容量** (`data/indexes/<repo_id>/`):
- V1 (e5-base 768dim): embeddings ~150MB + SQLite chunks.db ~150MB
- V2 (ruri-small-v2 384dim): embeddings ~75MB + SQLite chunks.db ~150MB
- V4 (bge-m3 1024dim): embeddings ~200MB + SQLite chunks.db ~150MB
- **合計**: 約 900MB-1.2GB (V0/V3 は既存 index 共有のため追加なし)
- **クリーンアップ**: 最良 variant 採用後、未採用 variant の `data/indexes/<repo_id>/` ディレクトリは削除可

**HF model cache 追加 download** (`~/.cache/huggingface/`):
- `intfloat/multilingual-e5-base` ~1.1GB
- `cl-nagoya/ruri-small-v2` ~120MB
- `BAAI/bge-m3` ~2.3GB
- `BAAI/bge-reranker-v2-m3` ~2.3GB
- **合計**: 約 5.8GB
- **前提**: `sentence-transformers >= 2.3.0` (bge-m3 / bge-reranker-v2-m3 サポート version)。実機実行前に `pip show sentence-transformers` で version 確認

**predictions JSONL 出力衝突回避**:
- V0 と V3 は同 repo_id (`institutional_documents`) のため `scripts/run_baseline_eval.py` の `run_id = baseline_eval_<repo_id>_<YYYYMMDD_HHMMSS>` が秒単位で衝突する可能性。`--output logs/institutional_V<N>_<ts>.jsonl` で variant 名を含む明示パスを指定すると aggregator (`scripts/aggregate_institutional_baseline.py --predictions <glob>`) 集約が安全。

## 採用判定基準

主指標: **overall NC rate** (target: V0 比 -2pt 以上改善)。

タイブレーカー (差 ≤ 1pt 時の優先順位):
1. **category 別 NC**: 悪化カテゴリ数が少ない方優先
2. **p95 retrieval+rerank latency**: 低い方優先
3. **index build cost / memory footprint** (運用 RAM 含む): 軽い方優先

`configs/baseline.yaml` の global default は **不変** (#114 invariant test で保護、PR #132 で merge 済)。institutional 用 default のみ更新。

**運用波及 (V4 採用時のみ)**:
- 現状 institutional プロファイルのサーバ常駐 RAM ≈ 1GB (e5-small ~120MB + ms-marco-MiniLM ~90MB + その他)
- V4 採用後: bge-m3 (~2.3GB) + bge-reranker-v2-m3 (~2.3GB) で **+4-5GB 増 → 合計 ~5-6GB**
- `docs/deployment.md:13` の memory 要件記述 (~1GB) も同時更新が必要 (タイブレーカー (3) で考慮)

## 変更内容

### Phase B 実機実行 (worker scope 外、人手 or 主セッション側)

1. **5 variant config 新規作成**: `configs/_experiments/institutional_V[0-4].yaml` を 5 本作成。現行 `baseline_reporag.config.load_config()` は `extends` を解釈しないため、各 file は `configs/institutional_docs.yaml` の full copy として成立させる。各 variant で `indexing.embedding.max_input_chars` を **明示宣言** (V0/V1/V2/V3=2048、V4=**8192**)。さらに `repo.repo_id` を eval/build の対象 index と一致させる (V0/V3=`institutional_documents`、V1/V2/V4=`institutional_documents_V<N>`)。
2. **ingestion + index build**: variant ごとに以下を実行。V1/V2/V4 のみ実施 (V0/V3 は既存 `institutional_documents` を流用)。
   ```bash
   python scripts/ingest_repo.py --config configs/_experiments/institutional_V<N>.yaml --repo-id institutional_documents_V<N> ...
   python scripts/build_indexes.py --config configs/_experiments/institutional_V<N>.yaml --repo-id institutional_documents_V<N>
   ```
3. **eval 実行**: 5 variant 全てで以下を実行。
   ```bash
   python scripts/run_baseline_eval.py --config configs/_experiments/institutional_V<N>.yaml --repo-id <V0/V3 は institutional_documents、その他は institutional_documents_V<N>> --eval-set data/eval_sets/institutional_static_eval.jsonl --output logs/institutional_V<N>_<ts>.jsonl
   ```
   - 注意: `--repo-id` は query 対象 repo_id と run_id/output 名に効くが、pipeline の index load 先は `build_pipeline(cfg)` 内の `cfg.repo.repo_id` で決まる (`pipeline_factory.py:112-116`)。そのため variant config の `repo.repo_id` と `--repo-id` は必ず一致させる。
4. **集計**: `scripts/aggregate_institutional_baseline.py` を variant ごとに個別実行し、`reports/institutional_retrieval_ab.md` の比較表へ転記する。
   ```bash
   python scripts/aggregate_institutional_baseline.py --predictions logs/institutional_V<N>_<ts>.jsonl --output - --section overall,category,latency
   ```
   - 注意: 現行 aggregator は複数 `--predictions` / glob を variant 別に group by せず、全 records を単一集合として合算する。5 variant 比較には単一 glob 集計を使わない。

### コード変更 (最良 variant 採用後)

- **`configs/institutional_docs.yaml`** 更新:
  - `embedding.model_id` を最良値に置換
  - `reranker.model_id` を最良値に置換
  - `embedding.max_input_chars` を **明示宣言** (現状 fallback 2048。最良 variant の値で固定)
- **既存 index の強制再 build (採用ありの場合の必須手順)**:
  - embedding model_id を切り替えた瞬間、既存 `data/indexes/institutional_documents/embedding/embeddings.npy` は新 model の dim と不一致になり実行時 `ValueError` (matmul shape mismatch) が発生する
  - 手順: `rm -rf data/indexes/institutional_documents/embedding/` → `python scripts/build_indexes.py --config configs/institutional_docs.yaml --repo-id institutional_documents` で再 build
- **`tests/test_pipeline_factory_yaml_invariants.py`** invariant test 活性化:
  - `INSTITUTIONAL_RERANKER_MODEL_ID: str | None = None` (line 36) → 採用 reranker model_id に置換
  - `INSTITUTIONAL_EMBEDDING_MODEL_ID: str | None = None` (line 37) → 採用 embedding model_id に置換
  - `test_institutional_yaml_reranker_model_id_pinned` (line 55-68) の `@pytest.mark.skipif` 解除
  - `test_institutional_yaml_embedding_model_id_pinned` (line 71-78) の `@pytest.mark.skipif` 解除
  - **V4 採用時のみ** (条件付き): `INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS = 8192` 定数 + 第 3 invariant test を追加
    ```python
    def test_institutional_yaml_embedding_max_input_chars_pinned():
        cfg = load_config(CONFIGS_DIR / "institutional_docs.yaml")
        # 注: indexing.embedding は dict-like (build_indexes.py:54 の get() 経路と整合)
        assert cfg.indexing.embedding.get("max_input_chars") == INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS
    ```
- **`reports/institutional_retrieval_ab.md`** 新規作成 (5 variant 比較、採用根拠、category 別 NC、p95 latency、採否判断)

## 受入条件

- [ ] 5 variant 全てで institutional eval 完走 (116Q)
- [ ] `reports/institutional_retrieval_ab.md` に 5 variant 比較表 + category 別 NC + p95 latency + 採否判断を記載
- [ ] **採用判定**: 最良 variant が V0 比で NC -2pt 以上改善した場合 → institutional default を更新 / 改善 < 2pt の場合 → 「非採用 (V0 維持)」と明記して報告 (採用ゼロも完走条件として OK)
- [ ] `configs/baseline.yaml` の global default 不変 (#114 invariant test で保護、CI で自動検証)
- [ ] **採用ありの場合**: `configs/institutional_docs.yaml` の `embedding.model_id` / `reranker.model_id` / `embedding.max_input_chars` を採用値に更新 + `data/indexes/institutional_documents/embedding/` 再 build 完了
- [ ] **採用ありの場合**: institutional 用 invariant test 2 件 (`test_institutional_yaml_reranker_model_id_pinned` / `test_institutional_yaml_embedding_model_id_pinned`) を活性化 (採用値で固定)
- [ ] **V4 採用時のみ**: `INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS=8192` 用の第 3 invariant test を追加
- [ ] **採用ありの場合**: `docs/deployment.md` の reranker/embedding 表 (line 88 周辺) と memory 要件 (line 13) を institutional default として更新

## 戦略: Option C (段階的)

1. **先行**: V0 (re-baseline) → V3 (reranker swap only) — 同じ index (`institutional_documents`) を共有するため `data/indexes/institutional_documents/chunks.db` の SQLite ファイル lock 競合回避のため **逐次実行必須** (V0 完了 → V3 開始)。reranker 効果を独立確認。
2. **次**: V1 (E5-base) + V2 (ruri) + V4 (bge-m3) — 専用 `repo_id` で別 index 構築のため SQLite lock 競合なし、ただし HF cache の同時 download 衝突リスクのため逐次推奨
3. **判定**: 最良 variant 確定 → 採用 or 非採用 (V0 維持) を `reports/institutional_retrieval_ab.md` に明記

## PR 戦略

採用判定結果次第で 1 PR 内の commit 構成が変わる (PR は 1 本にまとめる)。

| 判定 | PR タイトル例 | commit 構成 |
|------|-------------|-------------|
| 非採用 (V0 維持) | `docs(institutional): 5-variant A/B 非採用レポート (#137)` | `reports/institutional_retrieval_ab.md` のみ |
| V0/V1/V2/V3 採用 | `feat(institutional): 5-variant A/B 採用反映 V<N> (#137)` | レポート + `configs/institutional_docs.yaml` + invariant 2 件活性化 + index 再 build (commit 対象外) |
| V4 採用 | `feat(institutional): 5-variant A/B 採用反映 V4 bge-m3 (#137)` | 上記 + 第 3 invariant test 追加 + `docs/deployment.md` memory 要件更新 |

## 関連

- Phase A: #133 (PR #136 merged, `96e7b45`)
- 元 Issue: #114 (PR #132 merged、global baseline 保護用 invariant test を導入。本 Issue の institutional 限定 A/B はこの保護下で実施)
- Epic: #117 Phase 2 制度文書ドメイン検証
- 並列: #135 (本格再学習) と GPU 競合あり、直列推奨
- baseline state: post-#125 (E5 prefix on) + post-#126 (chunker 800)

## 並列開発

#135 (本格再学習、5 日工数) と部分並列だが **M3 Ultra GPU 競合**のため、本 Issue を先行実行 → 最良 variant 採用 → #135 学習に組み込む順序が efficient。

**競合の根拠と回避策**:
- 本 A/B 評価対象の `SentenceTransformer` (embedding) / `CrossEncoder` (reranker) は両方とも `device` 引数を plumbing していない (`baseline_reporag/indexing/embedding.py:95`、`baseline_reporag/retrieval/reranker.py:55`)。PyTorch のデフォルトは macOS で MPS が利用可能なら自動選択 → Apple GPU を消費する。
- `configs/institutional_docs.yaml:37` の `device: "auto"` は現状 SentenceTransformer/CrossEncoder 初期化に渡されておらず参照されない (config 経由での CPU 強制不可)
- #135 の MLX 学習も同じ Apple GPU/Metal を使用 → batch_size 64 同時実行で帯域競合
- **回避策**: device plumbing 追加は本 Issue 範囲外。直列実行 (本 Issue → #135) を推奨。CPU 強制が必要な場合は環境変数等での MPS 無効化が必要だが macOS では工数大 → 直列推奨が現実解

## 想定所要時間

- 実機 A/B: 4-5 時間
- レポート + 採用反映 + 配布: 1 時間
- 合計: **半日**
