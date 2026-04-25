# 制度文書 baseline Static NC 実測レポート (Issue #112)

**実測対象 corpus**: `institutional_documents` (4228 .md 非 git repo)
**eval set**: `data/eval_sets/institutional_static_eval.jsonl` (116Q, #110 生成)
**config**: `configs/institutional_docs.yaml`
**実測日**: _TBD (ローカル M3 Ultra で実行後に記入)_
**Issue**: https://github.com/Kewton/photon-mlx/issues/112

> **⚠ 3 重 handicap 前提の測定**: 本レポートの数値は以下 3 重 handicap を受容した状態の baseline であり、
> `#113` との Δ 比較以外の用途（絶対値判定）には使えない。詳細は §7 参照。

---

## 1. 実行環境

| 項目 | 値 |
|------|-----|
| マシン | M3 Ultra (Mac Studio, 128GB 想定) |
| OS | macOS Darwin 25.4.0 |
| Python | 3.12+ |
| mlx-lm | _TBD (実測時の version を記入)_ |
| sentence-transformers | _TBD_ |
| rank-bm25 | _TBD_ |
| ブランチ / commit | `feature/issue-112-institutional-config` @ _TBD_ |
| 実行日時 | _TBD_ |

---

## 2. Index 作成メトリクス

### 2-1. Ingest (scripts/ingest_repo.py)

| メトリクス | 値 |
|-----------|-----|
| 入力 md ファイル数 | 4228 |
| 生成 chunks 行数 | _TBD_ |
| SQLite chunks.db サイズ | _TBD MB_ |
| 実行時間 | _TBD 分_ |
| `repo_commit` sentinel | `9e500539f29555364217b773368305e7f59aa026` (= sha1("institutional_v1")) |

### 2-2. Build indexes (scripts/build_indexes.py)

| メトリクス | 値 |
|-----------|-----|
| BM25 lexical.pkl サイズ | _TBD MB_ |
| Embedding embeddings.npy shape | _(N_chunks, 384) = TBD_ |
| Embedding model_id.txt 内容 | `intfloat/multilingual-e5-small` ✅ |
| 実行時間 | _TBD 分_ |
| HuggingFace 初回 download | E5-small ~480MB (`~/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/`) |

---

## 3. 全体 NC rate

| 指標 | 値 |
|------|-----|
| 全質問数 | 116 |
| NC (no-citation) 件数 | _TBD_ |
| NC rate | _TBD %_ |

### 判定結果（設計 §9）

- [ ] `< 50% NC` → 正常完了、#113 着手可 (デフォルト想定)
- [ ] `50-80% NC` → ベースライン取得済として close、follow-up Issue で改善
- [ ] `>= 80% NC` → #113 着手保留、原因分析 Issue 優先

**判定**: _TBD_

---

## 4. Category 別 NC rate

| Category | 件数 | NC 件数 | NC rate | 備考 |
|----------|------|---------|---------|------|
| article_lookup | 18 | _TBD_ | _TBD %_ | 条文番号指定による参照型 |
| definition | 20 | _TBD_ | _TBD %_ | 用語定義 |
| exception | 20 | _TBD_ | _TBD %_ | 例外規定 |
| overview | 20 | _TBD_ | _TBD %_ | 概要・要約 |
| penalty | 19 | _TBD_ | _TBD %_ | 罰則・ペナルティ |
| scope | 19 | _TBD_ | _TBD %_ | 適用範囲 |
| **合計** | **116** | _TBD_ | _TBD %_ | |

---

## 5. Latency (p50 / p95)

| 指標 | 値 |
|------|-----|
| 全体 p50 | _TBD ms_ |
| 全体 p95 | _TBD ms_ |
| Retrieval p50 | _TBD ms_ |
| Retrieval p95 | _TBD ms_ |
| Generation p50 | _TBD ms_ |
| Generation p95 | _TBD ms_ |
| Memory peak | _TBD MB_ |

---

## 6. 代表的な failure 例

各 category から `no_citation=True` または `cited_chunk_ids=[]` のエントリを 1-2 件抜粋。

### 6-1. article_lookup
- **eval_id**: _TBD_
- **question**: _TBD_
- **answer (抜粋)**: _TBD_
- **expected_citation_patterns**: _TBD_
- **コメント**: _TBD_

### 6-2. definition
_TBD_

### 6-3. exception
_TBD_

### 6-4. overview
_TBD_

### 6-5. penalty
_TBD_

### 6-6. scope
_TBD_

---

## 7. 3 重 handicap 注記

本 Issue の baseline 数値は以下 3 重 handicap 前提で測定（設計 §1-4）。各 handicap は個別の
follow-up Issue で是正予定。**baseline の絶対値ではなく #113 PHOTON との Δ を重視する前提**。

| # | handicap | 想定下振れ | 修正ポイント | 対応 follow-up Issue |
|---|----------|----------|-----------|---------------------|
| (a) | E5 prompt prefix 非対応 | −2〜5pt | `baseline_reporag/indexing/embedding.py` の `EmbeddingIndex.build()` / `search()` で `model.encode(texts, ...)` 直前に `"query: "` / `"passage: "` を前置 | _#TBD_ |
| (b) | 英語 cross-encoder reranker 継承 | −5〜10pt | `baseline_reporag/retrieval/reranker.py` の `CrossEncoderReranker` model_id を `jinaai/jina-reranker-v2-base-multilingual` 等の多言語モデルへ切替 | _#TBD_ |
| (c) | 日本語未チューニング chunker size | 未定量 | `baseline_reporag/ingestion/chunker.py` の `_chunk_markdown` で `max_chars=800〜1200` へ縮小実験 | _#TBD_ |
| **合計** | | **−7〜15pt (推定)** | | |

---

## 8. 判定フロー結果

**実測 NC rate**: _TBD %_

**結論**: _TBD_

### #113 への申し送り

- 同じ `configs/institutional_docs.yaml` を base に `model.provider: "photon"` へ切替えた PHOTON 測定を実施
- 同じ 3 重 handicap を PHOTON 側も継承するため、Δ（PHOTON − baseline）は意味があるが絶対値は best-possible を表さない
- 本レポートへのリンクと `embedding/model_id.txt` 確認結果を #113 着手前コメントで明示

---

## 9. 実行コマンドログ

```bash
# 1) E5-small 事前ダウンロード
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# 2) Ingest
python scripts/ingest_repo.py \
    --repo /Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents \
    --repo-id institutional_documents \
    --commit 9e500539f29555364217b773368305e7f59aa026 \
    --config configs/institutional_docs.yaml

# 3) Build indexes
python scripts/build_indexes.py \
    --repo-id institutional_documents \
    --config configs/institutional_docs.yaml

# 4) Run baseline eval
python scripts/run_baseline_eval.py \
    --config configs/institutional_docs.yaml \
    --eval-set data/eval_sets/institutional_static_eval.jsonl \
    --output logs/institutional/baseline_eval_institutional_documents_<YYYYMMDD_HHMMSS>.predictions.jsonl

# 5) Category 別集計 (手動 one-liner、設計 §4-4)
python -c "
import json
from collections import defaultdict
preds = [json.loads(l) for l in open('logs/institutional/<FILENAME>')]
per_cat = defaultdict(lambda: [0, 0])
for p in preds:
    cat = p['category']
    per_cat[cat][1] += 1
    if p.get('no_citation') or not p.get('cited_chunk_ids'):
        per_cat[cat][0] += 1
for cat, (nc, tot) in sorted(per_cat.items()):
    print(f'{cat}: {nc/tot*100:.1f}% ({nc}/{tot})')
"
```

---

## 10. follow-up Issue リスト (本 Issue closing 前に起票)

| # | タイトル案 | 目的 |
|---|-----------|------|
| (i) | `chore(ci): baseline_eval log glob を fastapi_fastapi pattern に狭める` | `scripts/ci_eval_check.py` + `.github/workflows/weekly_eval.yml` |
| (ii) | `feat(indexing): E5 embedding で query:/passage: prefix を付与 (handicap a 対応)` | `baseline_reporag/indexing/embedding.py` |
| (iii) | `feat(retrieval): 多言語 cross-encoder reranker サポート (handicap b 対応)` | `baseline_reporag/retrieval/reranker.py` |
| (iv) | `tune(ingestion): 日本語 markdown chunker size 実験 (handicap c 対応)` | `baseline_reporag/ingestion/chunker.py` |
| (v) | `feat(scripts): aggregate_institutional_baseline.py 新設 (§7-8)` | 手動 one-liner の自動化 |

---

_本レポートは Issue #112 Phase 5 の実測完了後に数値セクションを埋める。実測完了前は全 TBD プレースホルダ。_
