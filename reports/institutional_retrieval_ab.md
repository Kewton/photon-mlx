# Institutional 5-variant retrieval A/B (#137 / #133 Phase B)

**実施日**: 2026-04-26
**eval set**: `data/eval_sets/institutional_static_eval.jsonl` (116 questions)
**baseline reference**: `reports/institutional_baseline_static.md` (V0 prior measurement: 11.21% NC)
**実行環境**: M3 Ultra (MPS auto), sentence-transformers 5.4.0
**前提**: Phase A (#133 / PR #136 / `96e7b45`) で `EmbeddingIndex.max_input_chars` 設定可能化済

---

## 比較表 (overall)

| Variant | Embedding | Reranker | max_input_chars | NC % | Δ vs V0 | p95 latency | Memory peak (max) | 採否 |
|---------|-----------|----------|-----------------|------|---------|-------------|-------------------|------|
| **V0** | `intfloat/multilingual-e5-small` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 2048 | **12.93 %** | (基準) | 24,243 ms | 139.6 MB | (基準) |
| **V1** | `intfloat/multilingual-e5-base` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 2048 | **8.62 %** | **-4.31 pt** | 24,699 ms | 184.2 MB | ❌ 非採用 (V4 が優位) |
| **V2** | `cl-nagoya/ruri-small-v2` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 2048 | **FAILED** | - | - | - | ❌ 計測不能 |
| **V3** | `intfloat/multilingual-e5-small` | `BAAI/bge-reranker-v2-m3` | 2048 | **12.07 %** | -0.86 pt | 24,112 ms | 139.9 MB | ❌ 非採用 (-2pt 未達) |
| **V4** | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` | **8192** | **6.03 %** | **-6.90 pt** | 28,465 ms | 184.5 MB | **⭕ 採用** |

> **V0 baseline 注記**: prior `reports/institutional_baseline_static.md` の baseline NC は 11.21%、本 A/B 再計測で 12.93%。乖離 1.72pt は LLM (Qwen2.5-Coder-14B-Instruct-4bit) の生成揺らぎ範囲内 (`do_sample=False, temperature=0.2` でも nondeterminism は残る)。本 A/B では **同セッション/同コミットで全 variant を再実行した V0=12.93%** を比較基準とする。
>
> **V2 FAILED 注記**: `cl-nagoya/ruri-small-v2` (および同系 `ruri-base`) は sentence-transformers 5.4.0 の `AutoProcessor.from_pretrained` で `Unrecognized processing class` エラーで load 不能。新版 transformers + ruri モデル config の互換性問題で、同じ問題は ruri 系全モデルに該当する模様。トークナイザ config 修正または sentence-transformers の互換層対応が必要 → **本 A/B では V2 を計測不能扱いとし、4-variant 比較で判定**。

---

## 採用判定

### 主指標: overall NC rate (target: V0 比 -2pt 以上改善)

- **採用候補 (≥ -2pt)**: V1 (-4.31pt), V4 (-6.90pt)
- **非採用 (< -2pt)**: V3 (-0.86pt)
- **計測不能**: V2

### 主指標で V1 vs V4 の比較

差は **2.59pt** (V4=6.03% vs V1=8.62%) — 設計方針書 section 4 タイブレーカー (差 ≤ 1pt) を **適用しない** (差が threshold を大きく超過しているため主指標で V4 を選択)。

### 採用: **V4 (BAAI/bge-m3 + BAAI/bge-reranker-v2-m3, max_input_chars=8192)**

採用理由:
- 主指標 NC -6.90pt (V0 12.93% → V4 6.03%) で最大改善
- category 別でも全領域で V0 同等以上、特に `article_lookup` (V0 22.2% → V4 0%) と `penalty` (V0 10.5% → V4 5.3%) の改善が顕著
- 多言語対応 reranker (bge-reranker-v2-m3) と長文対応 embedding (bge-m3, 8192 char) の組合せが institutional 制度文書 (日本語) に最適合
- 既知のトレードオフ:
  - p95 latency +4.2s (24.2s → 28.5s, +17%)
  - サーバ常駐 RAM +4-5GB (~1GB → ~5-6GB) — `docs/deployment.md` 更新で対応

---

## Category 別 NC 比較

| Category | 件数 | V0 NC % | V1 NC % | V3 NC % | V4 NC % |
|----------|------|---------|---------|---------|---------|
| article_lookup | 18 | 22.2 | 5.6 | 50.0 | **0.0** |
| definition | 20 | 5.0 | 5.0 | 0.0 | **0.0** |
| exception | 20 | 35.0 | 30.0 | 10.0 | 30.0 |
| overview | 20 | 5.0 | 0.0 | 0.0 | **0.0** |
| penalty | 19 | 10.5 | 10.5 | 15.8 | **5.3** |
| scope | 19 | 0.0 | 0.0 | 0.0 | **0.0** |
| **合計** | **116** | **12.93** | **8.62** | **12.07** | **6.03** |

**観察**:
- V3 は `article_lookup` で著しく悪化 (V0 22.2% → V3 50.0%) — 多言語 reranker (bge-reranker-v2-m3) が e5-small embedding と相性悪い可能性
- V4 は全 category で V0 同等以上、`exception` は 30% で V0 35% 並 (本質的に難しいカテゴリ、全 variant で高い)
- V4 採用後の残課題は `exception` カテゴリ専用の prompt/retrieval 改善 → 別 follow-up Issue 候補

---

## 各 variant 集計 raw output

### V0 (e5-small + ms-marco-MiniLM-L-6-v2, 2048 chars)

```
| 指標 | 値 |
|------|-----|
| 全質問数 | 116 |
| NC (no-citation) 件数 | 15 |
| **NC rate** | **12.93 %** |

| Category | 件数 | NC 件数 | NC rate |
|----------|------|---------|---------|
| article_lookup | 18 | 4 | 22.2 % |
| definition | 20 | 1 | 5.0 % |
| exception | 20 | 7 | 35.0 % |
| overview | 20 | 1 | 5.0 % |
| penalty | 19 | 2 | 10.5 % |
| scope | 19 | 0 | 0.0 % |
| **合計** | **116** | **15** | **12.93 %** |

| 指標 | 値 |
|------|-----|
| 全体 p50 | 13,818 ms |
| 全体 p95 | 24,243 ms |
| 全体 max | 33,731 ms |
| 全体 mean | 14,766 ms |
| Retrieval p50 | 317 ms |
| Retrieval p95 | 356 ms |
| Generation p50 | 13,399 ms |
| Generation p95 | 23,804 ms |
| Memory peak (p50) | 34.0 MB |
| Memory peak (p95) | 34.0 MB |
| Memory peak (max) | 139.6 MB |
```

predictions: `logs/institutional_V0_20260426_165002.jsonl`

### V1 (e5-base + ms-marco-MiniLM-L-6-v2, 2048 chars)

```
| 指標 | 値 |
|------|-----|
| 全質問数 | 116 |
| NC (no-citation) 件数 | 10 |
| **NC rate** | **8.62 %** |

| Category | 件数 | NC 件数 | NC rate |
|----------|------|---------|---------|
| article_lookup | 18 | 1 | 5.6 % |
| definition | 20 | 1 | 5.0 % |
| exception | 20 | 6 | 30.0 % |
| overview | 20 | 0 | 0.0 % |
| penalty | 19 | 2 | 10.5 % |
| scope | 19 | 0 | 0.0 % |
| **合計** | **116** | **10** | **8.62 %** |

| 指標 | 値 |
|------|-----|
| 全体 p50 | 14,040 ms |
| 全体 p95 | 24,699 ms |
| 全体 max | 30,203 ms |
| 全体 mean | 15,240 ms |
| Retrieval p50 | 326 ms |
| Retrieval p95 | 437 ms |
| Generation p50 | 13,598 ms |
| Generation p95 | 23,120 ms |
| Memory peak (p50) | 34.0 MB |
| Memory peak (p95) | 34.0 MB |
| Memory peak (max) | 184.2 MB |
```

predictions: `logs/institutional_V1_20260426_182847.jsonl`

### V2 (ruri-small-v2 + ms-marco-MiniLM-L-6-v2, 2048 chars) — FAILED

```
ValueError: Unrecognized processing class in cl-nagoya/ruri-small-v2.
Can't instantiate a processor, a tokenizer, an image processor, a video
processor or a feature extractor for this model.
  at sentence_transformers/base/modules/transformer.py:659 (AutoProcessor.from_pretrained)
```

`cl-nagoya/ruri-base` でも同じ error。sentence-transformers 5.4.0 と ruri モデルの tokenizer config 互換性問題のため計測不能。

### V3 (e5-small + bge-reranker-v2-m3, 2048 chars)

```
| 指標 | 値 |
|------|-----|
| 全質問数 | 116 |
| NC (no-citation) 件数 | 14 |
| **NC rate** | **12.07 %** |

| Category | 件数 | NC 件数 | NC rate |
|----------|------|---------|---------|
| article_lookup | 18 | 9 | 50.0 % |
| definition | 20 | 0 | 0.0 % |
| exception | 20 | 2 | 10.0 % |
| overview | 20 | 0 | 0.0 % |
| penalty | 19 | 3 | 15.8 % |
| scope | 19 | 0 | 0.0 % |
| **合計** | **116** | **14** | **12.07 %** |

| 指標 | 値 |
|------|-----|
| 全体 p50 | 13,814 ms |
| 全体 p95 | 24,112 ms |
| 全体 max | 29,551 ms |
| 全体 mean | 14,410 ms |
| Retrieval p50 | 325 ms |
| Retrieval p95 | 360 ms |
| Generation p50 | 13,260 ms |
| Generation p95 | 23,484 ms |
| Memory peak (p50) | 34.0 MB |
| Memory peak (p95) | 34.0 MB |
| Memory peak (max) | 139.9 MB |
```

predictions: `logs/institutional_V3_20260426_171916.jsonl`

### V4 (bge-m3 + bge-reranker-v2-m3, 8192 chars) — **採用**

```
| 指標 | 値 |
|------|-----|
| 全質問数 | 116 |
| NC (no-citation) 件数 | 7 |
| **NC rate** | **6.03 %** |

| Category | 件数 | NC 件数 | NC rate |
|----------|------|---------|---------|
| article_lookup | 18 | 0 | 0.0 % |
| definition | 20 | 0 | 0.0 % |
| exception | 20 | 6 | 30.0 % |
| overview | 20 | 0 | 0.0 % |
| penalty | 19 | 1 | 5.3 % |
| scope | 19 | 0 | 0.0 % |
| **合計** | **116** | **7** | **6.03 %** |

| 指標 | 値 |
|------|-----|
| 全体 p50 | 17,645 ms |
| 全体 p95 | 28,465 ms |
| 全体 max | 36,213 ms |
| 全体 mean | 18,352 ms |
| Retrieval p50 | 368 ms |
| Retrieval p95 | 504 ms |
| Generation p50 | 17,008 ms |
| Generation p95 | 27,750 ms |
| Memory peak (p50) | 34.0 MB |
| Memory peak (p95) | 34.0 MB |
| Memory peak (max) | 184.5 MB |
```

predictions: `logs/institutional_V4_20260426_213604.jsonl`

---

## 運用波及 (採用後 — V4 採用)

- [x] `configs/institutional_docs.yaml` 更新 (model_id 2 件 + max_input_chars=8192 明示宣言)
- [x] `data/indexes/institutional_documents/embedding/` 強制再 build (e5-small → bge-m3 で dim 不一致)
- [x] `tests/test_pipeline_factory_yaml_invariants.py` 既存 invariant 2 件活性化 (`INSTITUTIONAL_RERANKER_MODEL_ID` / `INSTITUTIONAL_EMBEDDING_MODEL_ID` 定数置換)
- [x] `tests/test_pipeline_factory_yaml_invariants.py` 第 3 invariant test 追加 (`INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS = 8192`)
- [x] `docs/deployment.md` memory 要件 (~1GB → ~5-6GB) + reranker/embedding 表更新

---

## Follow-up 候補 (本 Issue 範囲外、別 Issue で対応)

1. **V2 (ruri モデル) sentence-transformers 互換層**: ruri-small-v2/ruri-base が AutoProcessor で load 不能。tokenizer config 修正または `transformers` 側 patch 待ち。pkshatech/GLuCoSE-base-ja 等の代替 Japanese embedding を別 A/B で検証する Issue を切ること。
2. **`exception` カテゴリ NC が高止まり (30%)**: V4 でも改善せず、retrieval 層の問題ではなく prompt/citation/decomposition の問題の可能性 → 別 Issue で原因分析。
3. **bge-m3 + bge-reranker-v2-m3 採用に伴うサーバ常駐 RAM +4-5GB**: deployment 環境の RAM 制約確認、必要に応じて model lazy unload 戦略を検討。
4. **device plumbing**: `SentenceTransformer(device=cfg.device)` / `CrossEncoder(device=cfg.device)` の plumbing 追加で `configs/institutional_docs.yaml:37` の `device` フィールドを生かす (#135 並列実行可能化)。
5. **token-aware truncation**: 現状 `max_input_chars` は char 単位 truncate。bge-m3 の token 上限 (8192 token) に整合する token-based truncation で精度向上の余地あり。
6. **aggregator の variant 別 group by 機能**: 現状複数 predictions JSONL を渡すと合算する仕様。`--variant` 引数で group by 可能にする follow-up Issue。

---

## 関連 Issue / PR

- Phase A: #133 (PR #136 merged, `96e7b45`) — `EmbeddingIndex.max_input_chars` 設定可能化
- 元 Issue: #114 (PR #132 merged) — global baseline 保護用 invariant test 導入。本 Issue は institutional プロファイル限定の A/B として #114 invariant の保護下で実施
- Epic: #117 Phase 2 制度文書ドメイン検証
- 後続: #135 (本格再学習) — V4 採用 model (bge-m3) を組み込んで直列実行
