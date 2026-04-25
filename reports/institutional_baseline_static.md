# 制度文書 baseline Static NC 実測レポート (Issue #112)

**実測対象 corpus**: `institutional_documents` (4228 .md 非 git repo)
**eval set**: `data/eval_sets/institutional_static_eval.jsonl` (116Q, #110 生成)
**config**: `configs/institutional_docs.yaml`
**実測日**: 2026-04-25
**Issue**: https://github.com/Kewton/photon-mlx/issues/112

> **⚠ 3 重 handicap 前提の測定**: 本レポートの数値は以下 3 重 handicap を受容した状態の baseline であり、
> `#113` との Δ 比較以外の用途（絶対値判定）には使えない。詳細は §7 参照。

---

## 1. 実行環境

| 項目 | 値 |
|------|-----|
| マシン | M3 Ultra (Mac Studio, 128GB) |
| OS | macOS Darwin 25.4.0 |
| Python | 3.12.3 |
| mlx-lm | 0.31.2 |
| sentence-transformers | 5.4.0 |
| rank-bm25 | 0.2.2 |
| ブランチ / commit | `feature/issue-112-institutional-config` @ `644561c` |
| 実行日時 | 2026-04-25 09:05–09:47 JST |

---

## 2. Index 作成メトリクス

### 2-1. Ingest (scripts/ingest_repo.py)

| メトリクス | 値 |
|-----------|-----|
| 入力 md ファイル数 | 4228 |
| 生成 chunks 行数 | 189,639 |
| SQLite chunks.db サイズ | 637 MB |
| 実行時間 | 約 30 秒 |
| `repo_commit` sentinel | `9e500539f29555364217b773368305e7f59aa026` (= sha1("institutional_v1")) |

### 2-2. Build indexes (scripts/build_indexes.py)

| メトリクス | 値 |
|-----------|-----|
| BM25 lexical.pkl サイズ | 45 MB |
| Embedding embeddings.npy shape | `(189639, 384)` (float32, 278 MB) |
| Embedding model_id.txt 内容 | `intfloat/multilingual-e5-small` ✅ |
| 実行時間 | 約 10 分 (BM25 < 1 分 + E5 ~9 分、2964 batches × 64) |
| HuggingFace E5-small download | 既キャッシュ済（初回 ~480MB / `~/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/`） |

---

## 3. 全体 NC rate

| 指標 | 値 |
|------|-----|
| 全質問数 | 116 |
| NC (no-citation) 件数 | 13 |
| **NC rate** | **11.21 %** |

### 判定結果（設計 §9）

- [x] `< 50% NC` → **正常完了、#113 着手可** (デフォルト想定通り)
- [ ] `50-80% NC` → ベースライン取得済として close、follow-up Issue で改善
- [ ] `>= 80% NC` → #113 着手保留、原因分析 Issue 優先

**判定**: **正常完了**。NC 11.21% は想定 reasonable range（< 50%）に入り、catastrophic ではない。
3 重 handicap 受容前提の baseline として十分使える数値。#113 PHOTON 測定に進む。

---

## 4. Category 別 NC rate

| Category | 件数 | NC 件数 | NC rate | 備考 |
|----------|------|---------|---------|------|
| article_lookup | 18 | 3 | 16.7 % | 条文番号指定による参照型 |
| definition | 20 | 1 | 5.0 % | 用語定義 |
| exception | 20 | 6 | 30.0 % | 例外規定（最も難）|
| overview | 20 | 0 | 0.0 % | 概要・要約 |
| penalty | 19 | 3 | 15.8 % | 罰則・ペナルティ |
| scope | 19 | 0 | 0.0 % | 適用範囲 |
| **合計** | **116** | **13** | **11.21 %** | |

**所見**: `exception` (30.0%) と `article_lookup` (16.7%) が想定通り高 NC。`overview`/`scope` は 0% で
hybrid retrieval が広めの context をうまく拾えている。`exception` の高 NC は条文の例外指定（「ただし
書」「但し」等のパターン）が embedding ベースで類似 chunk を引き寄せにくいためと推測（follow-up 調査）。

---

## 5. Latency (p50 / p95)

| 指標 | 値 |
|------|-----|
| 全体 p50 | 13,813 ms |
| 全体 p95 | 33,171 ms |
| 全体 max | 50,987 ms |
| 全体 mean | 15,476 ms |
| Retrieval p50 | 194 ms |
| Retrieval p95 | 233 ms |
| Generation p50 | 13,579 ms |
| Generation p95 | 32,931 ms |
| Memory peak (p50) | 18.8 MB |
| Memory peak (p95) | 18.9 MB |
| Memory peak (max) | 139.6 MB |
| 116Q 総実時間 | 約 30 分 |

**所見**: latency の支配項は generation_ms（mlx-lm Qwen14B-4bit 推論、Apple Silicon GPU）。
retrieval は BM25 + E5 hybrid で 200ms 弱と高速。memory peak は profile_memory 計測値で
チャンクロードのバースト時のみ 139MB に跳ね、定常時は 19MB 前後。

---

## 6. 代表的な failure 例

各 category から `no_citation=True` を 1 件抜粋（NC=0 の category は successful sample を併記）。

### 6-1. article_lookup（NC 3/18）
- **eval_id**: `INST-ARTICLE-LOOKUP-002`
- **question**: 第4条に規定されている「整備」の内容は？
- **answer (抜粋)**: 「根拠が不足しています。提供されたコードチャンクには『整備』の具体的な内容が明確に記載されていません。」
- **latency_ms**: 10,376（retrieval 214 / generation 10,122）
- **コメント**: 第N条+用語のパターン。条文ジャンプは単純 BM25 では拾えるが、用語が共起する別条文を
  優先することがある。query_expansion が空の影響と推測。

### 6-2. definition（NC 1/20）
- **eval_id**: `INST-DEFINITION-001`
- **question**: 保育所における自己評価ガイドライン（2020年改訂版）の定義は？
- **answer (抜粋)**: 「根拠が不足しています。The provided code chunks do not contain any information about a『保育所における自己評価ガイドライン（2020年改訂版）』or its definition. ...」
- **latency_ms**: 49,485（retrieval 7,297 / generation 42,094）
- **コメント**: source_document_id にハンドブック PDF 由来の見出ししか含まれず、定義本文の chunk が
  retrieval にヒットしなかったケース。retrieval 7秒は外れ値（embedding cache miss + BM25 大候補）。

### 6-3. exception（NC 6/20）
- **eval_id**: `INST-EXCEPTION-008`
- **question**: 法第28条で定める例外は？
- **answer (抜粋)**: 「根拠が不足しています。The provided code chunks do not contain any information about the specific exceptions defined in the second article of the Rental Housing Management Act. ...」
- **latency_ms**: 16,781（retrieval 229 / generation 16,477）
- **コメント**: 「例外」をキーワード化する設問が embedding 距離で広く拾われ、条文番号指定との
  intersection が弱い。category 全体で NC=30% は本 baseline の最大課題。

### 6-4. overview（NC 0/20）
- **eval_id**: `INST-OVERVIEW-001`（successful sample）
- **question**: この政令の目的は？
- **cites**: 9 chunks
- **latency_ms**: 23,140
- **コメント**: 「目的」「概要」型は冒頭・前文 chunk が hybrid retrieval で確実にヒット。NC=0%。

### 6-5. penalty（NC 3/19）
- **eval_id**: `INST-PENALTY-005`
- **question**: 申請者自身が手続きを行うことが要求されていない場合の罰則は？
- **answer (抜粋)**: 「根拠が不足しています。コードチャンクに申請者自身が手続きを行うことが要求されていない場合の罰則に関する具体的な情報がありません。」
- **latency_ms**: 8,195（retrieval 199 / generation 7,950）
- **コメント**: 罰則条文は明示的に存在しない場合がある（任意規定 vs 義務規定の区別）。
  human_verified の精査が必要だが本 Issue スコープ外。

### 6-6. scope（NC 0/19）
- **eval_id**: `INST-SCOPE-001`（successful sample）
- **question**: この法律の適用を受けるのはどのような事業者？
- **cites**: 4 chunks
- **latency_ms**: 13,966
- **コメント**: 適用範囲は法律前段の定義条文で明示されることが多く、retrieval が安定。NC=0%。

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

**実測 NC rate**: **11.21 %** (13/116)

**結論**: 正常完了。NC < 50% の reasonable range に収まり、catastrophic でない。Phase 2 制度文書ドメインの
baseline として #113 PHOTON 測定の比較基準に用いる。

### #113 への申し送り

- 同じ `configs/institutional_docs.yaml` を base に `model.provider: "photon"` へ切替えた PHOTON 測定を実施
- 同じ 3 重 handicap を PHOTON 側も継承するため、Δ（PHOTON − baseline）は意味があるが絶対値は best-possible を表さない
- 本レポートへのリンクと `embedding/model_id.txt` 確認結果を #113 着手前コメントで明示
- baseline の category 別 NC 分布（exception 30% / article_lookup 16.7% / penalty 15.8% / definition 5% / overview 0% / scope 0%）を PHOTON 側 Δ 評価のベースラインに使用

---

## 9. 実行コマンドログ

```bash
# 1) E5-small 事前ダウンロード（既キャッシュ済の場合は確認のみ）
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
    --output logs/institutional/baseline_eval_institutional_documents_20260425_091723.predictions.jsonl

# 5) Category 別集計 (手動 one-liner、設計 §4-4)
python -c "
import json
from collections import defaultdict
preds = [json.loads(l) for l in open('logs/institutional/baseline_eval_institutional_documents_20260425_091723.predictions.jsonl')]
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

_本レポートは Issue #112 Phase 5 の実測完了後に数値セクションを埋めた最終版。_
