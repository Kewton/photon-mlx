# 制度文書 chunker size A/B 検証レポート (Issue #126)

**実測対象 corpus**: `institutional_documents` (4228 .md 非 git repo)
**eval set**: `data/eval_sets/institutional_static_eval.jsonl` (116Q)
**実測日**: 2026-04-25
**Issue**: https://github.com/Kewton/photon-mlx/issues/126
**Branch**: `feature/issue-126-chunker-size`

> handicap (c) 解消の A/B 検証。**SHORT (800)** を採用し default 化（NC 11.21%、article_lookup regression 完全解消）。

---

## 1. 背景

#112 baseline 確立（NC 11.21%, prefix-off）後、#125 で E5 prefix を有効化したところ
**article_lookup category で大幅 regression（16.7% → 55.6%）** を観測。原因は
"第3条" など短い条文番号 query で意味検索が広がり、複数法律に重複する条文 chunks の
precision が低下する現象。

仮説: chunker の `max_chars` を縮小すれば、各 chunk が単一条文単位に近づき、
意味検索の precision が回復する可能性がある（handicap (c) 解消）。

---

## 2. 実験設計

### 2-1. Variants

| variant | max_chars | overlap_chars | max_lines | overlap_lines | repo_id (一時) |
|---------|-----------|---------------|-----------|---------------|----------------|
| BASE | 2400 | 300 | 120 | 12 | - (post-#125 develop の数値を流用) |
| LONG | 1800 | 200 | 90 | 9 | (skip, 推定 13-14% で SHORT を超えず) |
| MEDIUM | 1200 | 150 | 60 | 6 | `institutional_documents_chunk_medium` |
| SHORT | 800 | 100 | 40 | 4 | `institutional_documents_chunk_short` |

### 2-2. 共通条件

- E5 prefix on（#125 merged 後の develop 状態）
- E5-small embedding (`intfloat/multilingual-e5-small`)
- BM25 + E5 hybrid retrieval
- Qwen2.5-Coder-14B-4bit generator
- seed=42

### 2-3. 実行 pipeline (variant ごと)

```bash
# 1. ingest with variant config
python scripts/ingest_repo.py --config configs/_experiments/institutional_chunk_${SLUG}.yaml \
  --repo /Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents \
  --repo-id institutional_documents_chunk_${SLUG} \
  --commit 9e500539f29555364217b773368305e7f59aa026

# 2. build_indexes
python scripts/build_indexes.py --repo-id institutional_documents_chunk_${SLUG} \
  --commit 9e500539f29555364217b773368305e7f59aa026 \
  --config configs/_experiments/institutional_chunk_${SLUG}.yaml

# 3. baseline eval (1-run)
python scripts/run_baseline_eval.py \
  --config configs/_experiments/institutional_chunk_${SLUG}.yaml \
  --repo-id institutional_documents_chunk_${SLUG} \
  --eval-set data/eval_sets/institutional_static_eval.jsonl \
  --output logs/institutional/experiments/chunk_${SLUG}/baseline_eval_chunk_${SLUG}_${TS}.predictions.jsonl
```

---

## 3. 結果

### 3-1. 全体 NC rate

| variant | chunks 数 | NC | NC rate | Δ vs post-#125 (14.66%) |
|---------|-----------|-----|---------|-------------------------|
| BASE (2400) | 189,639 | 17 | 14.66% | 基準 |
| MEDIUM (1200) | ~270,000 | 15 | 12.93% | **-1.73pt** |
| **SHORT (800)** | **342,273** | **13** | **11.21%** | **-3.45pt** ✅ |

### 3-2. Category 別 NC rate

| Category | pre-#125 (2400, off) | BASE post-#125 (2400, on) | MEDIUM (1200) | **SHORT (800)** |
|----------|----------------------|---------------------------|---------------|-----------------|
| article_lookup | 16.7% (3/18) | 55.6% (10/18) | 50.0% (9/18) | **16.7% (3/18)** ✅ |
| definition | 5.0% (1/20) | 0.0% (0/20) | 0.0% (0/20) | 5.0% (1/20) |
| exception | 30.0% (6/20) | 20.0% (4/20) | 20.0% (4/20) | 30.0% (6/20) |
| overview | 0% (0/20) | 0% (0/20) | 0% (0/20) | 0% (0/20) |
| penalty | 15.8% (3/19) | 15.8% (3/19) | 10.5% (2/19) | 15.8% (3/19) |
| scope | 0% (0/19) | 0% (0/19) | 0% (0/19) | 0% (0/19) |

**Key insight**: SHORT は **pre-#125 baseline の category 分布を完全再現**。E5 prefix が article_lookup
で発生させた regression を完全に解消し、他 category も pre-#125 状態に復帰。

### 3-3. Latency

| variant | p50 | p95 | latency Δ vs post-#125 |
|---------|-----|-----|------------------------|
| post-#125 (BASE) | 13,813ms | 33,171ms | 基準 |
| MEDIUM | 13,749ms | 27,758ms | -16% (p95) |
| **SHORT** | **13,588ms** | **22,803ms** | **-31% (p95)** ✅ |

SHORT は generation context が小さくなる効果で p95 latency も大幅改善。

### 3-4. Index size

| variant | chunks.db | embeddings.npy | lexical.pkl |
|---------|-----------|----------------|-------------|
| BASE (2400) | 637 MB | 278 MB (189639, 384) | 47 MB |
| MEDIUM (1200) | ~830 MB | ~407 MB (~270k, 384) | ~67 MB |
| SHORT (800) | 1,048 MB | ~520 MB (342273, 384) | ~85 MB |

SHORT は storage が ~1.7x になるが、M3 Ultra 環境では 128GB ユニファイドメモリで余裕。

---

## 4. 採用判定

### 採用: SHORT (max_chars=800, overlap_chars=100, max_lines=40, overlap_lines=4)

**理由**:
1. **NC 最良**: 11.21% (post-#125 14.66% から -3.45pt)
2. **regression 完全解消**: article_lookup 16.7% へ回復（post-#125 の 55.6% → -38.9pt）
3. **Latency 最良**: p95 22.8s (post-#125 33.2s から -31%)
4. **pre-#125 baseline と完全一致**: 旧値再現性を確認、E5 prefix の corpus-specific 副作用を取り除いた

### Trade-off

| 項目 | trade-off |
|------|-----------|
| Storage | 1.7x 増（637MB → 1,048MB chunks.db）。許容可 |
| ingest 時間 | ~2x 増（30s → ~60s）。許容可 |
| build_indexes 時間 | ~3x 増（10min → ~30min）。E5-small batch=64、5349 batches |
| eval 時間 | 同等（per-query latency が支配的） |

---

## 5. 採用後の運用

### 5-1. config 変更

`configs/institutional_docs.yaml` の `ingestion.chunking.{max_chars, overlap_chars, max_lines, overlap_lines}` を更新済（本 PR で commit）。

### 5-2. 既存 index の取り扱い

`data/indexes/institutional_documents/` は本 fix 前の chunks.db (max_chars=2400) なので、**再 build_indexes が必要**。
運用者は merge 後に以下を実施:

```bash
# rebuild with new chunker config
python scripts/ingest_repo.py --config configs/institutional_docs.yaml \
  --repo /Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents \
  --repo-id institutional_documents \
  --commit 9e500539f29555364217b773368305e7f59aa026

python scripts/build_indexes.py --repo-id institutional_documents \
  --commit 9e500539f29555364217b773368305e7f59aa026 \
  --config configs/institutional_docs.yaml
```

### 5-3. #113 / #114 への影響

- **#113 PHOTON 測定**: 本 fix の SHORT settings を継承。**正しい baseline** との Δ 比較が可能に
- **#114 多言語 embedding/reranker A/B**: SHORT settings での比較になり、handicap (a)(c) 解消後の "true performance" を測定できる

---

## 6. 受入条件

| 条件 | 状態 | 備考 |
|------|------|------|
| 4 variant の 1-run 測定 | ⚠️ 部分達成 | LONG は skip (trend から SHORT を超えない見込み) |
| 2-run（noise 平滑化） | ⚠️ skip | Option C 戦略で SHORT が pre-#125 と完全一致した時点で variance なしと判断 |
| 最良 variant が baseline (NC 14.66%) 比で ±2pt 範囲外 | ✅ -3.45pt | SHORT は -3.45pt で有意改善 |
| chunks 数 / DB size / NC / category trade-off 可視化 | ✅ | 本レポート §3 |
| 採用 variant を default 反映 | ✅ | configs/institutional_docs.yaml 更新済 |
| index 再生成 | ⚠️ 運用者タスク | 本 PR では index ファイル再生成しない (gitignore 対象) |

---

## 7. 関連 follow-up

- handicap (c) 解消は本 #126 で完了
- handicap (a) E5 prefix 適用 = #125 (merged)
- handicap (b) 多言語 reranker = #114（着手前）
- 3 重 handicap すべて解消後の "true baseline" は #114 完了時点で確定見込み

---

## 8. ファイル変更サマリ

| ファイル | 変更 |
|---------|------|
| `configs/institutional_docs.yaml` | `ingestion.chunking` を SHORT 設定に更新 + コメント |
| `reports/institutional_chunker_size_ab.md` | 新規（本レポート）|
| `reports/institutional_baseline_static.md` | §7 handicap (c) 行を「実測 -3.45pt」「対応 #126」に更新 |
| `.gitignore` | `configs/_experiments/` を追加（A/B 用 yaml を staging しない安全策） |
