# bge-reranker-base Evaluation Report (Issue #89)

> **Note on continuity**: 本レポート以降の no-citation / latency 数値は
> `BAAI/bge-reranker-base` を reranker に採用した状態での測定値です。
> それ以前のレポート（`reports/gate2_judgment_v*.md`, `reports/gate3_judgment.md`,
> `reports/benchmark_report.md` など）は `cross-encoder/ms-marco-MiniLM-L-6-v2`
> 前提の測定値であり、モデル差し替え前後の値を直接比較する際は本注記に留意してください。

## 1. 変更概要

- **対象 Issue**: #89 feat(retrieval): reranker モデルを bge-reranker-base に更新して Static NC 改善
- **親 Epic**: #81（Static NC < 15% 達成のための retrieval チューニング）
- **変更内容**:
  - Reranker モデル: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M, 英語 MS MARCO 特化)
    → `BAAI/bge-reranker-base` (278M, 多言語 IR fine-tuned)
  - `CrossEncoderReranker.__init__` に DI 用 `_model` kw-only 引数を追加（テスト互換性）
  - max_length は 256 のまま（設計判断 #1、`content[:600]` 文字トリムと整合）
  - `rerank_query` の使い分けは現行維持（設計判断 #2、効果分離のため）

## 2. 受入条件と実測値

| 項目 | 目標 | 実測 | 判定 |
|------|------|------|------|
| `BAAI/bge-reranker-base` が sentence-transformers でロード可能 | Yes | **TBD** (`RUN_SLOW_TESTS=1 python -m pytest -k bge_reranker_base`) | — |
| `CrossEncoderReranker.rerank()` の署名後方互換 | 署名維持 | **維持** | ✅ |
| 既存 mock テスト無改変 pass | pass | **182 passed / 1 skipped** (baseline_reporag/tests/) | ✅ |
| `test_reranker.py` unit tests | 全 pass | **7 passed / 1 skipped (slow gate)** | ✅ |
| Static no-citation | < 16% (Epic #81 部分貢献; 全体目標 <15%) | **TBD** (ベンチマーク未実行) | — |
| Static P50 latency 悪化 | warmup 後 +3 s 以内 | **TBD** | — |
| MT no-citation | 5.6% 維持（reranker は Turn 1 のみ適用） | **TBD** | — |
| `ruff check .` / `ruff format --check .` | clean | **clean** | ✅ |

> **ベンチマーク未実行**: Static NC / P50 / MT NC の実測は本 PR では実施せず、
> PR マージ後の別タスクで `bench/` ハーネスを用いて測定する。測定後、本レポート
> の TBD セクションを更新する。

## 3. 実測プロトコル（測定時の手順）

### 3-1. 実モデル smoke

```bash
RUN_SLOW_TESTS=1 python -m pytest baseline_reporag/tests/test_reranker.py::test_bge_reranker_base_loads_and_predicts -v
```

期待値: `BAAI/bge-reranker-base` を HuggingFace から初回ダウンロード (~550 MB)
→ `predict([(query, passage)])` が `numpy.ndarray` shape `(1,)` を返す。

### 3-2. Static NC / P50

```bash
# bench/ ハーネス（具体コマンドは bench/ 配下の run_all.py / freeze_benchmark.py 参照）
python -m bench.run_all --config configs/baseline.yaml
# もしくは
python -m bench.freeze_benchmark --config configs/baseline.yaml
```

**計測条件**:
- candidate 数: `retrieval.rerank_top_k=12` (default)
- warmup: 最初の 1-3 query をスキップ
- P50 集計: 120 Q の静的評価セット

### 3-3. MT NC

```bash
# MT 評価スクリプト（既存の MT eval 手順に準拠）
python -m scripts.run_mt_eval --config configs/baseline.yaml
```

## 4. 既知のリスクと follow-up

1. **実パラメータ数の確認**: `BAAI/bge-reranker-base` の公称 278M は HuggingFace モデルカード値。smoke test 実施時に `sum(p.numel() for p in model.model.parameters())` 相当で実測し、必要なら本レポートに追記する。
2. **`rerank_query` policy 最適化**: 現行は英語 expansion を `rerank_query` として渡す contract を維持（設計判断 #2）。多言語モデル採用後の A/B（英語 expansion vs 日本語原文）は Epic #81 配下の別 Sub-Issue で起票予定。
3. **スコアスケール**: bge-reranker は raw logit を返すため、将来 `file_type_boost` を有効化する際にスケール互換性を再検証する必要あり。現行は `file_type_boost=0.0` で無効化されているため直接の影響はない。

## 5. ロールバック

- `configs/baseline.yaml` / `configs/photon_small.yaml` / `configs/photon_long_context.yaml` の `model_id` を `cross-encoder/ms-marco-MiniLM-L-6-v2` に戻す
- `baseline_reporag/retrieval/reranker.py` / `baseline_reporag/pipeline_factory.py` / `app/photon_app.py` の default を同様に戻す
- これだけで即時復旧可能（新 DI 引数 `_model` は kw-only なので残しても互換）

## 6. 参考

- 設計方針書: `workspace/design/issue-89-bge-reranker-design-policy.md`
- Issue レビュー: `workspace/issues/89/issue-review/summary-report.md`
- 設計レビュー: `workspace/issues/89/multi-stage-design-review/summary-report.md`
