# Issue #137 設計方針書 — institutional 多言語 embedding/reranker 5-variant 実機 A/B

**Issue**: #137 feat(retrieval): institutional 多言語 embedding/reranker 5-variant 実機 A/B (#133 Phase B)
**作成日**: 2026-04-26
**ブランチ**: feature/issue-137-institutional-ab
**前提**: Phase A (#133 / PR #136, `96e7b45`) で `EmbeddingIndex.max_input_chars` 設定可能化 + #114 guard test を完了済

---

## 1. スコープ宣言

本 Issue は **実装機能追加ではなく、測定 (A/B 実機評価) + その結果に応じた最小限のコード/設定変更** が本体である。

| カテゴリ | 内容 | コード変更? |
|---------|------|------------|
| Phase B 実機実行 | 5 variant config 作成、ingest/build/eval/集計、採否判断レポート | ✗ (gitignore 配下の variant config と `reports/` 文書のみ) |
| 採用ありの場合の反映 | `configs/institutional_docs.yaml` 更新、index 再 build、invariant test 活性化、`docs/deployment.md` 更新 | ✓ (3-4 ファイル、~30 行規模) |
| V4 採用時のみ追加 | `INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS` 定数 + 第 3 invariant test | ✓ (~10 行追加) |
| 非採用 (V0 維持) | `reports/institutional_retrieval_ab.md` のみ | ✓ (1 ファイル) |

このため、本設計方針書は (a) 実験設計の前提と (b) 採否分岐後の具体的コード surgery plan の 2 軸で構成する。

---

## 2. システムアーキテクチャ (本 Issue 関連部分のみ)

```
┌──────────────────────────────────────────────────────────────┐
│  configs/                                                     │
│  ├─ baseline.yaml         # global default (#114 で不変保護)   │
│  ├─ institutional_docs.yaml # institutional プロファイル        │
│  └─ _experiments/          # gitignored (Phase B で 5 本作成)  │
│     └─ institutional_V[0-4].yaml                              │
└────────────────┬─────────────────────────────────────────────┘
                 ↓ baseline_reporag.config.load_config()
┌──────────────────────────────────────────────────────────────┐
│  baseline_reporag/pipeline_factory.py                         │
│  └─ build_pipeline(cfg) → _build_baseline_deps_no_mlx()       │
│     └─ idx_dir = data_root / "indexes" / cfg.repo.repo_id     │
│        ├─ chunks.db (SQLite, ChunkStore)                      │
│        ├─ lexical/ (BM25 pickle)                              │
│        └─ embedding/ (embeddings.npy + model_id.txt           │
│                       + max_input_chars.txt + chunk_ids.json) │
└────────────────┬─────────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────────┐
│  baseline_reporag/indexing/embedding.py                       │
│  └─ EmbeddingIndex(model_id, max_input_chars=2048)            │
│     └─ SentenceTransformer(model_id) [device 引数なし → MPS auto] │
│                                                               │
│  baseline_reporag/retrieval/reranker.py                       │
│  └─ CrossEncoder(model_id, max_length) [device 引数なし]       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  scripts/                                                     │
│  ├─ ingest_repo.py            (--config / --repo-id / --repo) │
│  ├─ build_indexes.py          (--config / --repo-id)          │
│  ├─ run_baseline_eval.py      (--config / --repo-id /         │
│  │                              --eval-set / --output)         │
│  └─ aggregate_institutional_baseline.py                       │
│     (--predictions / --output / --section)                    │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  tests/test_pipeline_factory_yaml_invariants.py               │
│  ├─ test_baseline_yaml_reranker_model_id_unchanged (#114, ✓)  │
│  ├─ test_institutional_yaml_reranker_model_id_pinned (skipif) │
│  └─ test_institutional_yaml_embedding_model_id_pinned (skipif) │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. レイヤー別影響範囲

| レイヤー | モジュール | 変更種別 | 影響度 |
|---------|----------|---------|--------|
| Config | `configs/_experiments/institutional_V[0-4].yaml` | 新規 (gitignored) | 高 (実験本体) |
| Config | `configs/institutional_docs.yaml` | 採用ありの場合のみ更新 | 中 |
| Indexing | `baseline_reporag/indexing/embedding.py` | 変更なし (Phase A 完了) | - |
| Retrieval | `baseline_reporag/retrieval/reranker.py` | 変更なし | - |
| Test | `tests/test_pipeline_factory_yaml_invariants.py` | 採用ありの場合のみ skipif 解除 + 定数置換 (V4 時は test 追加) | 中 |
| Doc | `docs/deployment.md` | 採用ありの場合のみ memory/model 表更新 | 低 |
| Report | `reports/institutional_retrieval_ab.md` | 新規作成 (必須) | 高 (採否根拠) |
| Data | `data/indexes/institutional_documents{,_V1,_V2,_V4}/` | Phase B で生成 (gitignored) | - |
| Data | `~/.cache/huggingface/` | 4 model 追加 download | - |

---

## 4. 主要設計判断

### 設計判断 #1: variant config の生成方式

**選択肢**:
- **A**: `configs/institutional_docs.yaml` の full copy として 5 本作成
- **B**: 差分 YAML + `extends`/include 機構を新規実装
- **C**: スクリプトで base config から 5 variant を生成

**決定**: **A** (full copy)

**理由**:
- 現行 `baseline_reporag.config.load_config()` は `extends`/include を解釈しない (Codex Stage 7 の S7-002 で確認済)
- B は本 Issue の scope (測定 + 最小コード変更) を逸脱、別 follow-up 候補
- C は便利だが本 Issue 単発の使い捨て扱いで OK、5 ファイルは手作業/エディタで作れる規模
- variant 間の差分は (a) `embedding.model_id` (b) `embedding.max_input_chars` (c) `reranker.model_id` (d) `repo.repo_id` の最大 4 フィールドのみで、full copy でも diff 視認性は高い

**トレードオフ**:
- ✅ メリット: 既存実装ゼロ変更、各 variant config が単独で `load_config()` 可能、debug 容易
- ✅ ISP 副次効果: extends/include 機構不在により institutional プロファイル変更は他 profile config に自動波及しない (interface segregation が config layer で自然に保証される)。将来 extends 機構を導入する場合は、profile 単位の継承境界を明示する API 設計が必要。
- ❌ デメリット: base config の更新が variant 5 本に反映されない (本 A/B 期間中は base 更新しない前提で許容)
- ⚠️ リスク: variant config の他フィールド (paths, chunker など) が誤って変わる可能性 → diff レビューで検出
- ⚠️ DRY 違反の許容前提: 「base 更新凍結」は本 A/B 期間中の運用ルールであり、Phase B 開始時に `configs/institutional_docs.yaml` の HEAD commit を記録し、Phase B 完了まで base 更新 PR を保留する (section 10 リスクマップ参照)

### 設計判断 #2: V0/V3 の index 共有 vs 分離

**選択肢**:
- **A**: V0/V3 は `repo_id=institutional_documents` (既存 index) を共有 → 別 variant では別 repo_id
- **B**: 全 variant で別 repo_id (`institutional_documents_V0` など 5 つ作成)

**決定**: **A** (V0/V3 共有 + V1/V2/V4 別 repo_id)

**理由**:
- V0 と V3 は embedding が同一 (`intfloat/multilingual-e5-small`) のため、index バイナリ (embeddings.npy) のビット比較で同値 → 別途 build する意味がない
- reranker は推論時切替 (CrossEncoder インスタンスは pipeline_factory が呼び出し時生成) のため index 生成は無関係
- 既存 `institutional_documents` index の dim = 384 (e5-small) と V0/V3 が一致 → そのまま流用可
- 追加ディスク容量 ~150-300MB を節約 (V1/V2/V4 用は ~600MB-1.2GB が必要)

**トレードオフ**:
- ✅ メリット: ディスク節約、build 時間節約 (~30 min × 1 variant 分)
- ❌ デメリット: V0 と V3 を **逐次実行必須** (`chunks.db` SQLite ファイル lock 競合回避)
- ⚠️ リスク: `run_baseline_eval.py` の `run_id = baseline_eval_<repo_id>_<YYYYMMDD_HHMMSS>` が秒単位で衝突する可能性 → `--output logs/institutional_V<N>_<ts>.jsonl` で variant 名を含む明示パスを指定 (Issue 本文にも明記済)

### 設計判断 #3: aggregator は variant ごとに個別実行

**選択肢**:
- **A**: 現行 `aggregate_institutional_baseline.py` を variant ごとに個別呼び出し、`reports/institutional_retrieval_ab.md` に手動転記
- **B**: aggregator に `--variant` 引数を追加し、複数 predictions を group by して比較表を直接生成

**決定**: **A** (個別実行 + 手動転記)

**理由**:
- 現行実装 (`scripts/aggregate_institutional_baseline.py:72-87`) は全 records を単一リストに append して合算するため、glob 一発で 5 ファイルを渡すと 580Q 相当の合算レポートになる (Codex Stage 5 S5-002 で確認済)
- B は便利だが本 Issue 範囲外、別 follow-up Issue 候補
- 5 variant × 1 回の手動転記は許容コスト (~30 min)

**トレードオフ**:
- ✅ メリット: aggregator 改修不要、本 Issue は手順書のみで完結
- ❌ デメリット: 手動転記ミスのリスク → レポート内に各 variant の `aggregate` 出力 raw block を残す方針で軽減
- ⚠️ リスク: `--predictions glob` を誤って単一実行すると合算結果になる → Issue 本文に「単一 glob 集計を使わない」と明記済

### 設計判断 #4: invariant test の dual-axis 拡張 (V4 採用時のみ)

**選択肢**:
- **A**: V4 採用時のみ `INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS = 8192` 定数 + 第 3 invariant test を追加
- **B**: 採用 variant に関わらず 3 番目の invariant test を追加 (max_input_chars=2048 を pin)
- **C**: invariant test は活性化のみで第 3 test は追加しない

**決定**: **A** (条件付き追加)

**理由**:
- max_input_chars は global default に対する保護観点が薄い (V0/V1/V2/V3 採用なら 2048 のまま、保護不要)
- V4 のみ 8192 という非標準値が institutional 採用後も維持される必要があり、`fallback 2048` への drift 防止が重要 (build_indexes.py:54 の getattr 経路で気付かず 2048 になるリスク)
- C は V4 採用後の drift 検知ができないため不採用

**トレードオフ**:
- ✅ メリット: 採用結果に応じた最小限の test 追加、V0/V1/V2/V3 採用なら test 構造変更ゼロ
- ❌ デメリット: 条件分岐ロジックが Issue 本文/PR で必要になる
- ⚠️ リスク: V4 採用したのに第 3 test を追加し忘れる → PR レビューチェックリストで防止 (受入条件にも明記済)

### 設計判断 #5: `device` plumbing は本 Issue 範囲外

**選択肢**:
- **A**: 現状の MPS auto 選択のまま、#135 と直列実行を推奨
- **B**: `SentenceTransformer(model_id, device=cfg.device)` / `CrossEncoder(model_id, device=cfg.device)` を plumbing して CPU 強制で並列実行可能化

**決定**: **A** (本 Issue 範囲外 → 直列推奨)

**理由**:
- `configs/institutional_docs.yaml:37` の `device: "auto"` は現状参照されない死フィールド (Codex Stage 3/7 で確認)
- B は別 Issue 範囲 (本 Issue は測定 scope)、device plumbing 自体に embedding/rerank 速度への影響評価が必要
- macOS で MPS 無効化は環境変数経由で工数大、CPU 強制すると latency 5-10x 悪化
- #135 直列実行で本 Issue → #135 の依存順序が最も自然

**トレードオフ**:
- ✅ メリット: 本 Issue は plumbing 不要、scope 明確
- ❌ デメリット: #135 と並列実行できない (5 日 + 半日 → 連続)
- ⚠️ リスク: 将来 device plumbing が必要になった時に別 Issue で対応 → follow-up Issue 候補として明記
- ⚠️ 技術的負債明示: 本 Issue では `device` フィールド死状態を許容するが、後続作業者が config に `device` を書けば効くと誤解しないよう、(a) follow-up Issue を別途登録 (候補: SentenceTransformer/CrossEncoder への device 引数 plumbing、本設計方針書 section 12 で参照) し、(b) 採用反映時に `configs/institutional_docs.yaml:37` 行へ `# TODO(follow-up): currently unused, see follow-up issue` コメントを追加することを推奨

### 設計判断 #6 (LSP 補強): variant model interface 前提

5 variant の embedding/reranker model は以下の **dense 標準 interface** に依存する。bge-m3 が持つ sparse/ColBERT モードは本 Issue では **使用しない** (#114 invariant test と同等の interface contract を維持):

| 種別 | API | 戻り値 | 5 variant 全てで保証 |
|------|-----|--------|---------------------|
| Embedding | `SentenceTransformer.encode(texts, normalize_embeddings=True)` | `np.ndarray[N, dim]` (dense) | ✓ (e5-small: 384, e5-base: 768, ruri-small-v2: 384, bge-m3: 1024 dense) |
| Reranker | `CrossEncoder.predict(pairs)` | `np.ndarray[N]` | ✓ (ms-marco-MiniLM: 標準, bge-reranker-v2-m3: XLMRoberta-based 標準) |

これにより `baseline_reporag/indexing/embedding.py:127` の `self._embeddings @ q_emb` (matmul) と `baseline_reporag/retrieval/reranker.py` の rerank pipeline が variant 切替で動作不良を起こさないことを保証する。bge-m3 で sparse mode を使う必要が出た場合は別 Issue で interface 拡張が必要。

---

## 5. 詳細設計 — 採用ありの場合のコード surgery plan

### 5-1. `configs/institutional_docs.yaml` 更新

```diff
 indexing:
   embedding:
     enabled: true
     provider: "sentence-transformer"
-    model_id: "intfloat/multilingual-e5-small"
+    model_id: "<採用 variant の embedding model_id>"
     batch_size: 64
     normalize: true
+    max_input_chars: <採用値: 2048 または 8192>
   ...
 retrieval:
   reranker:
     enabled: true
-    model_id: "cross-encoder/ms-marco-MiniLM-L-6-v2"
+    model_id: "<採用 variant の reranker model_id>"
     ...
```

### 5-2. 既存 index の強制再 build

```bash
# 必須: embedding model_id を切り替えると embeddings.npy が dim 不一致になる
rm -rf data/indexes/institutional_documents/embedding/

# 再 build
python scripts/build_indexes.py \
  --config configs/institutional_docs.yaml \
  --repo-id institutional_documents
```

`baseline_reporag/indexing/embedding.py:127` の `scores: np.ndarray = self._embeddings @ q_emb` で dim 不一致なら `ValueError: shapes (N,384) and (768,) not aligned` のような実行時エラー → サーバ起動できなくなる。再 build 必須。

### 5-3. `tests/test_pipeline_factory_yaml_invariants.py` 活性化

```diff
 # 採用 variant の値で固定
-INSTITUTIONAL_RERANKER_MODEL_ID: str | None = None
+INSTITUTIONAL_RERANKER_MODEL_ID: str | None = "<採用 reranker model_id>"
-INSTITUTIONAL_EMBEDDING_MODEL_ID: str | None = None
+INSTITUTIONAL_EMBEDDING_MODEL_ID: str | None = "<採用 embedding model_id>"

 @pytest.mark.skipif(
     INSTITUTIONAL_RERANKER_MODEL_ID is None,
     reason="Issue #133: 採用 variant 決定後に有効化 (現在 A/B 評価中)",
 )
 def test_institutional_yaml_reranker_model_id_pinned():
     ...
```

`@pytest.mark.skipif` は条件式 (`is None`) なので **定数置換だけで自動的に活性化される** — skipif デコレータ自体は削除不要。Issue 本文には「skipif 解除」と書いたが実際は「定数置換のみで OK」。これを明確化:

> **修正注**: skipif デコレータは `INSTITUTIONAL_*_MODEL_ID is None` 条件で skip しているため、定数を実値に置換すれば skip 条件が False になり test が自動活性化する。デコレータ自体は **保持** (KISS 観点: 2 件の invariant test を同じ構造で対称的に保ち、コードレビュー時の差分を最小化する。YAGNI 観点では削除も可だが、定数置換のみで活性化/非活性化を切り替えられる現構造は実験/A-B 再実行時にも有用)。

### 5-4. V4 採用時のみ — 第 3 invariant test 追加

```python
# tests/test_pipeline_factory_yaml_invariants.py に追記

INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS: int | None = 8192  # V4 採用時のみ設定


@pytest.mark.skipif(
    INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS is None,
    reason="Issue #137: V4 (bge-m3 8192 chars) 採用時のみ有効化",
)
def test_institutional_yaml_embedding_max_input_chars_pinned():
    """institutional_docs.yaml の embedding.max_input_chars が採用値で固定されていることを検証。

    fallback 2048 への drift (configs を編集して max_input_chars を消す等) を防ぐ。
    """
    cfg = load_config(CONFIGS_DIR / "institutional_docs.yaml")
    # 注: cfg.indexing.embedding は Config オブジェクト (baseline_reporag/config.py) で
    # attribute access と .get() の両 API を提供する。max_input_chars は optional フィールドのため
    # 未宣言時に AttributeError を起こさない .get() 経路を採用 (build_indexes.py:54 の getattr fallback と整合)。
    assert cfg.indexing.embedding.get("max_input_chars") == INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS
```

### 5-5. `docs/deployment.md` 更新 (採用ありの場合)

| 行 | 現状 | 採用後 |
|---|------|--------|
| L13 (memory) | `Sentence-transformers embedding model and cross-encoder reranker require additional ~1 GB.` | **V0/V1/V2/V3 採用: 更新不要 (touch せず、~1GB のまま)** / **V4 採用のみ**: `~5-6 GB (bge-m3 + bge-reranker-v2-m3)` に更新 |
| L88 (reranker model_id) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (global default) | global default 行は保持し、その直下に institutional プロファイル専用 2 行を追加: `retrieval.reranker.model_id (institutional)` \| `<採用 reranker model_id>` \| Used by configs/institutional_docs.yaml / 同形式で `indexing.embedding.model_id (institutional)` 行も追加 |

> **Issue 本文受入条件との整合性注記**: Issue 本文の受入条件 (Stage 4 反映後) は「採用ありの場合: line 13 と line 88」と書かれているが、L13 の更新は **V4 採用時のみ**。V0-V3 採用時は L88 のみ更新で受入条件を満たす。Issue 本文の表現は概要レベルとして許容し、設計方針書の本表を実装ガイドラインとする。

### 5-6. `reports/institutional_retrieval_ab.md` 新規作成

各 variant の aggregator 出力を含む比較レポート。テンプレート:

```markdown
# Institutional 5-variant retrieval A/B (#137 / #133 Phase B)

## 比較表

| Variant | Embedding | Reranker | max_input_chars | NC % | category 別悪化数 | p95 latency (ms) | 採否 |
|---------|-----------|----------|-----------------|------|------------------|------------------|------|
| V0 | e5-small | ms-marco-MiniLM | 2048 | 11.21 | (基準) | XXX | (基準) |
| V1 | e5-base | ms-marco-MiniLM | 2048 | XX.XX | X | XXX | ⭕/❌ |
| ... 同上 5 行 |

## 採用判定

主指標 NC rate, タイブレーカー (category 悪化 → latency → memory) で評価:

- **採用**: V<N>, V0 比 NC -X.XXpt, ...
- **非採用 (V0 維持)**: 改善幅 < 2pt, ...

## 各 variant 集計 raw output

### V0
<!-- aggregate_institutional_baseline.py --predictions ... の出力をそのまま貼付 -->

### V1
... 同上 5 セクション

## 運用波及 (採用後)

- `configs/institutional_docs.yaml` 更新 (model_id 2 件 + max_input_chars 明示宣言): ✓/✗
- `data/indexes/institutional_documents/embedding/` 強制再 build: ✓/✗
- `tests/test_pipeline_factory_yaml_invariants.py` 既存 invariant 2 件活性化: ✓/✗
- `tests/test_pipeline_factory_yaml_invariants.py` 第 3 invariant test 追加 (V4 採用時のみ): ✓/✗/N-A
- `docs/deployment.md` memory 要件 + reranker/embedding 表更新: ✓/✗
```

> **設計意図**: レポートテンプレート末尾の checkbox 化により、採否判定レポートの完成 = 採用反映タスクの抜け漏れチェック完了が機械的に保証される。特に第 3 invariant test 追加 (V4 採用時のみ) は条件分岐があるため checkbox の `N-A` 選択肢を残すことで V0/V1/V2/V3 採用時の誤検出を防ぐ。

### 5-7: 採用反映時の commit 粒度推奨 (SRP)

採用結果別に SRP に基づく commit 構成を推奨。Issue 本文の PR 戦略表 (3 パターン) と整合する形で記載:

| 採用結果 | commit 数 | commit 構成 |
|----------|-----------|-------------|
| **非採用 (V0 維持)** | 1 | (#4) `docs(institutional): add 5-variant retrieval A/B non-adoption report (#137)` — `reports/institutional_retrieval_ab.md` のみ |
| **V0/V1/V2/V3 採用** | 4 | (#1) `feat(institutional): adopt V<N> embedding/reranker in institutional_docs.yaml (#137)` (5-1) + (#2) `test(institutional): activate invariant pinning for V<N> (#137)` (5-3) + (#3) `docs(deployment): add institutional reranker/embedding row (#137)` (5-5: L88 のみ更新、L13 memory は touch せず) + (#4) `docs(institutional): add 5-variant retrieval A/B report (#137)` (5-6) |
| **V4 採用** | 4 | 上記 V0-V3 採用の 4 commit に対し: (#2) で第 3 invariant 追加 (5-4 含む) + (#3) で `docs/deployment.md` の L13 memory 要件も `~5-6 GB (bge-m3 + bge-reranker-v2-m3)` に更新 |

**index 再 build (5-2)** は commit 対象外 (gitignored `data/`)。実機作業として PR description で完了報告するのみ。

> **整合性注記**: Issue 本文の PR 戦略表 (Stage 4 反映後) では V0-V3 採用を 3 commit と表記しているが、deployment.md の L88 institutional 行追加は SRP 観点で別 commit 化することで 4 commit が望ましい。Issue 本文の表現は概要レベルとして許容し、本設計方針書の 4 commit を実装ガイドラインとする。

---

## 6. セキュリティ設計

| 脅威 | 対策 | 優先度 |
|------|------|--------|
| **HF model 経由のサプライチェーン** | sentence-transformers は HF Hub から自動 download。bge-m3, bge-reranker-v2-m3 は BAAI 公式モデル、ruri-small-v2 は cl-nagoya 公式 → 発行元は妥当。ただし現行 `EmbeddingIndex` / `CrossEncoderReranker` は revision pin なしで model_id だけを渡すため、Phase B report に各 model の HF commit/revision と取得日時を記録する。`trust_remote_code=True` は使わない。 | 中 |
| **eval set 漏洩** | `data/eval_sets/institutional_static_eval.jsonl` は tracked file であり、PR に含めない/gitignored という前提は置かない。Institutional 評価用の質問・期待引用・source_document_id が機密/PII を含まないことを PR review で確認し、本 Issue では新規 eval-set rows を追加しない。 | 中 |
| **predictions JSONL 内の機密** | institutional_documents の chunks 自体に機密が含まれる場合あり。`logs/institutional_V<N>_*.jsonl` は `.gitignore` 対象だが、質問・回答・citation を含むため、local 権限・保持期間・削除記録は section 11 cleanup / retention に従う。 | 中 |
| **CLI path traversal / 任意出力** | `pipeline_factory` は `cfg.repo.repo_id` を validate するが、`ingest_repo.py` / `build_indexes.py` の CLI `--repo-id` は `Path(data_root) / "indexes" / args.repo_id` に直接使われる。Phase B 手順では `institutional_documents`, `institutional_documents_V1`, `_V2`, `_V4` の allowlist のみを使用し、`../` や `/tmp/...` を含む値を禁止する。`run_baseline_eval.py --eval-set/--output/--marker-file` も repo 内の既定パスに限定する。恒久対策は `validate_repo_id()` の CLI 適用を follow-up 候補とする。 | 高 |
| **variant config trust** | `configs/_experiments/institutional_V[0-4].yaml` は full copy で gitignored だが、信頼境界は repo 内の `configs/institutional_docs.yaml` から手元で派生した YAML に限定する。外部提供 YAML をそのまま実行せず、`repo.repo_id`, `paths`, `indexing`, `retrieval` の差分をレビューしてから ingest/build/eval に渡す。 | 中 |
| **invariant test の bypass** | 現 worktree に CODEOWNERS は存在しないため CODEOWNERS 監視を前提にしない。採用 PR では required CI (`tests/test_pipeline_factory_yaml_invariants.py`) と reviewer checklist で `INSTITUTIONAL_*` 定数が None に戻っていないことを確認する。CODEOWNERS/branch protection は別 follow-up 候補。 | 中 |

---

## 7. 並列開発・依存関係

```
#114 ─────PR #132──→ tests/invariant_tests (#114 base)
                      ↓
#133 ─Phase A─PR #136→ EmbeddingIndex.max_input_chars (96e7b45)
                      ↓
                     #137 ←(本 Issue, Phase B)
                      ↓
                   採用判定確定
                      ↓
                     #135 (本格再学習) — institutional 採用 model を組み込む
```

- **#135 並列リスク**: M3 Ultra GPU 競合のため直列推奨 (本 Issue → #135)
- **業務影響の扱い**: 本 Issue の実機 A/B 半日 + #135 本格再学習 5 日は直列 5.5 日見込み。#137 の採用 model が #135 の入力条件になるため、この 0.5 日遅延は許容する前提とする。#135 の納期制約が厳しい場合でも、GPU/Metal を使わないデータ整理・レビュー準備のみ並行可とし、MLX 学習ジョブは #137 の採否確定後に開始する。
- **#114 invariant test**: `configs/baseline.yaml` の global default を保護下に置く前提で本 Issue は institutional 限定で動く

---

## 8. 品質基準

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| Phase B 実機 — eval 完走 | 5 variant × 116Q | 全 variant 完走 (タイムアウト/OOM なし) |
| Phase B 実機 — レポート | `reports/institutional_retrieval_ab.md` 存在 + 5 variant 全比較表 | 必須項目記載 |
| 採否判断 | NC rate -2pt 改善 → 採用 / 未満 → 非採用 (V0 維持) | 報告のみで OK |
| invariant 保護 | `python -m pytest tests/test_pipeline_factory_yaml_invariants.py -v` | 全 test pass (採用前: 1 active + 2 skip / V0-V3 採用後: 3 件全 active / V4 採用後: 4 件全 active、第 3 invariant 追加分込み) |
| ruff check | `ruff check .` | 警告 0 件 |
| ruff format | `ruff format --check .` | 差分なし |
| 全 test | `python -m pytest` | 全 pass (既知 pre-existing 2 件除く) |
| index 再 build 検証 | サーバ起動後に institutional プロファイルで 1 query 成功 | ValueError なし |
| Phase B 後 cleanup 検証 | `du -sh data/indexes/institutional_documents* ~/.cache/huggingface` | 未採用 variant index と不要 logs の削除/保持判断を PR description または A/B report に記録 |

---

## 9. 完了条件 (受入条件と同じ)

Issue 本文の「## 受入条件」セクション参照。要約:

1. 5 variant institutional eval 完走 (116Q)
2. `reports/institutional_retrieval_ab.md` に比較表 + 採否判断
3. 採用判定 (NC -2pt 以上改善 → 採用 / 未満 → 非採用)
4. global default 不変 (#114 invariant test で自動保証)
5. 採用ありの場合: institutional_docs.yaml + invariant test 活性化 + index 再 build + deployment.md 更新
6. V4 採用時: 第 3 invariant test 追加

---

## 10. リスクマップ

| リスク | 兆候 | 対処 |
|--------|------|------|
| `run_baseline_eval.py --repo-id` が index load 先を変えない誤解 | V1/V2/V4 で意図せず V0 の index を読む (NC が V0 と完全一致) | variant config の `repo.repo_id` 明示 + `--repo-id` を一致させる手順を Issue 本文に明記済 |
| aggregator が複数 predictions を合算する誤解 | 5 variant 比較表ではなく 580Q 単一集計レポートになる | variant ごとに個別実行する手順を明記済 |
| eval set 取り違え (institutional 116Q vs FastAPI 120Q) | NC が突然乖離 | `--eval-set data/eval_sets/institutional_static_eval.jsonl` 明示を Issue 本文と各実行コマンドに明記済 |
| embedding 切替後に index 再 build 忘れ | サーバ起動時に dim mismatch ValueError | 「採用ありの場合の必須手順」として再 build を明記済 |
| invariant test が skipif のまま CI を通過 | 採用後も institutional default の drift を検知できない | 定数を None 以外にすれば自動活性化 (skipif 条件) |
| HF cache の同時 download 衝突 | V1/V2/V4 を並列実行すると lock file 競合 | 戦略 Option C で逐次実行を推奨済 |
| Phase B 期間中の base config 並行更新 | variant config 5 本が古い base 由来となり比較有効性が損なわれる | Phase B 開始時に `configs/institutional_docs.yaml` の HEAD commit を記録し、Phase B 完了まで base 更新 PR を保留する運用ルールを Issue 本文/PR description に明記 |
| V4 採用したのに第 3 invariant test を追加し忘れる | `INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS` が None のまま CI 通過、採用後 drift 検知不能 | 5-6 レポートテンプレート末尾の checkbox 「第 3 invariant test 追加 (V4 採用時のみ): ✓/✗/N-A」で機械的に検出。さらに 5-7 commit 粒度推奨の commit #2 ("test(...)") に V4 時の追加を含める |
| CLI `--repo-id` / `--output` の path traversal | 実験作業で `../` や絶対パスを渡すと repo 外または想定外 path に index/log/marker を作る | Phase B は allowlist repo_id と repo 内既定 path のみを使う。恒久対策として `ingest_repo.py` / `build_indexes.py` の `validate_repo_id()` 適用を follow-up 化 |
| HF model revision drift | 同じ model_id でも Hub 側更新で Phase B 再実行結果が変わる | A/B report に HF commit/revision と取得日時を記録し、採用判断の再現条件として残す |

---

## 11. 出力先 / 本 Issue で touch する成果物一覧

**設計・計画文書 (本 worktree 内)**:
- 設計方針書 (本ファイル): `workspace/design/issue-137-institutional-multilingual-ab-design-policy.md`
- 作業計画: `workspace/issues/137/work-plan.md` (Phase 4 で作成)

**Phase B 実機作業の成果物 (gitignored、commit 対象外)**:
- variant config 5 本: `configs/_experiments/institutional_V[0-4].yaml`
- index ディレクトリ: `data/indexes/institutional_documents{,_V1,_V2,_V4}/` (build 物)
- predictions JSONL: `logs/institutional_V[0-4]_<ts>.jsonl`
- HF model cache: `~/.cache/huggingface/` (4 model 追加 download)

**Phase B 完了後 cleanup / retention**:
- 採否判定後、canonical index は `data/indexes/institutional_documents/` のみとする。未採用 variant の `data/indexes/institutional_documents_V1/`, `_V2/`, `_V4/` は、A/B report に raw aggregate を転記し終えた後に削除可。
- 採用 variant が V1/V2/V4 の場合も、採用反映では `configs/institutional_docs.yaml` + `repo.repo_id=institutional_documents` で再 build するため、実験用 `_V<N>` index を本番相当として使い続けない。
- `logs/institutional_V[0-4]_<ts>.jsonl` は質問・回答・citation を含むため、report 作成後は必要最小限だけ保持する。機密を含む場合は commit せず、ローカル保存期間と削除完了を PR description または report に記録する。
- HF cache は共有 cache なので `~/.cache/huggingface/` 全体を一括削除しない。ディスク圧迫時のみ `huggingface-cli scan-cache` / `huggingface-cli delete-cache` 等で対象 model repo を確認し、他 Issue が利用中でないことを確認して削除する。
- cleanup 前後に `du -sh data/indexes/institutional_documents* ~/.cache/huggingface` を記録し、Phase B 一時占有 (data/indexes 約 1GB、HF cache 約 5.8GB) の残存を可視化する。

**採用ありシナリオで commit する成果物**:
- `configs/institutional_docs.yaml` (5-1 — 採用反映時)
- `tests/test_pipeline_factory_yaml_invariants.py` (5-3 定数置換 + 5-4 V4 時の第 3 invariant 追加)
- `docs/deployment.md` (5-5 — V0-V3 採用時は L88 のみ、V4 採用時は L13 + L88)
- `reports/institutional_retrieval_ab.md` (5-6 — 必須、本 PR に含む)

**非採用 (V0 維持) シナリオで commit する成果物**:
- `reports/institutional_retrieval_ab.md` のみ (5-6)

---

## 12. 関連ドキュメント

- Issue #137 本文 (Stage 8 反映後の最新)
- Issue #133 (Phase A): PR #136 (`96e7b45`)
- Issue #114: invariant test 導入 (PR #132)
- Issue #135: 本格再学習 (本 Issue 完了後に直列実行)
- Issue #117: Epic / Phase 2 制度文書ドメイン検証

### Follow-up 候補 (本 Issue 範囲外、別 Issue 化推奨)

- **device plumbing**: `SentenceTransformer(model_id, device=cfg.device)` / `CrossEncoder(model_id, device=cfg.device)` を plumbing して `configs/institutional_docs.yaml:37` の `device` フィールドを生かす。CPU 強制での #135 並列実行が可能になる。本 Issue 設計判断 #5 で scope 外と確定。
- **aggregator の variant 別比較サポート**: `scripts/aggregate_institutional_baseline.py` に `--variant` 引数を追加し、複数 predictions JSONL を group by して比較表を直接生成する。本 Issue 設計判断 #3 で scope 外と確定。
- **token 単位の正確な truncate 化**: 現状 `EmbeddingIndex` は `text[: max_input_chars]` で文字数 truncate するが、bge-m3 等の token 上限と整合させるためトークナイザベースの truncate に変更。本 Issue では char/token 比率の注記で運用回避。
- **CLI repo_id validation の横展開**: `pipeline_factory` / `build_symbol_graph.py` と同じ `validate_repo_id()` を `scripts/ingest_repo.py` / `scripts/build_indexes.py` の `--repo-id` にも適用し、手動実験時の path traversal を fail-fast で防ぐ。
- **CODEOWNERS / branch protection**: institutional invariant 定数を None に戻す PR を reviewer checklist だけでなく repository policy でも検知したい場合、CODEOWNERS 追加と required reviewer 設定を別 Issue で検討する。
