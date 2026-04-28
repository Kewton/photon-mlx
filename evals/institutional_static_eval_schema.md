# Institutional Static Eval – JSONL Schema

ファイル: `data/eval_sets/institutional_static_eval.jsonl`
各行が1問。既存 `static_eval.jsonl` の全キーを継承しつつ institutional 用の optional
フィールドを追加する。

## 必須フィールド

| key | 型 | 説明 |
|---|---|---|
| `id` | str | `INST-{CATEGORY}-NNN` (例: `INST-DEFINITION-001`) |
| `category` | str | `definition / article_lookup / overview / scope / penalty / exception` |
| `difficulty` | str | `easy / medium / hard` |
| `question` | str | 日本語 Q |
| `reference_answer` | str | 人手検証時に追記。生成直後は空可 |
| `reference_chunk_ids` | list[str] | Issue #109 chunk_id |
| `grading_notes` | str | grader への指示 |
| `rubric` | obj | `{correctness, grounding, usefulness}` |
| `answerable` | bool | 既存 schema と同じ |

## Optional フィールド（institutional 専用）

| key | 型 | 説明 |
|---|---|---|
| `expected_citation_patterns` | list[str] | `第N条` / `第N条の枝番` / `第N条第M項` / `第N条第M項第K号`。各要素は `^第\d+条(?:の\d+)?(?:第\d+項)?(?:第\d+号)?$` に fullmatch する（枝番 `の\d+` は任意で、`第24条の2` や `第5条の10第2項第3号` 等に対応） |
| `source_document_id` | str | `document.md` の親ディレクトリ名 (例: `0123_rental_mgmt`) |
| `generator_model` | str | 生成モデル名 (例: `gpt-4o-mini-2024-07-18`) |
| `human_verified` | bool | 人手検証済みかどうか (default false) |
| `verified_by` | str \| null | 検証者 |
| `verified_at` | str \| null | ISO8601 |

既存 consumer (`run_baseline_eval.py`) は optional フィールドを無視する。

## カテゴリ定義

| category | 問数 | 適用対象 |
|---|---|---|
| `definition` | 20 | 全 4,228 docs |
| `article_lookup` | 20 | 条文構造 1,707 docs |
| `overview` | 20 | 全 4,228 docs |
| `scope` | 20 | 全 4,228 docs |
| `penalty` | 20 | 条文構造 1,707 docs |
| `exception` | 20 | 条文構造 1,707 docs |
| **計** | **120** | |

## サンプル行

```jsonl
{"id": "INST-ARTICLE-LOOKUP-001", "category": "article_lookup", "difficulty": "medium", "question": "第3条に規定されている登録制度の期間は？", "reference_answer": "5 年", "reference_chunk_ids": ["institutional::0123_rental_mgmt/document.md::80-140"], "source_document_id": "0123_rental_mgmt", "grading_notes": "期間『5 年』を明示し、条文番号を引用していること", "rubric": {"correctness": {"max": 2, "notes": ""}, "grounding": {"max": 2, "notes": ""}, "usefulness": {"max": 1, "notes": ""}}, "answerable": true, "expected_citation_patterns": ["第3条", "第3条第1項"], "generator_model": "gpt-4o-mini-2024-07-18", "human_verified": true}
```
