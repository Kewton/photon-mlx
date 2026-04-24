# Institutional Multi-turn Eval – JSONL Schema

ファイル: `data/eval_sets/institutional_multi_turn_eval.jsonl`
各行が1セッション（6ターン固定）。既存 `multi_turn_eval.jsonl` の骨格を継承。

## 必須フィールド

| key | 型 | 説明 |
|---|---|---|
| `session_id` | str | `INST-MT-NNN` (drill_down 001-015 / cross_reference 016-025 / real_scenario 026-030) |
| `category` | str | `drill_down / cross_reference / real_scenario` |
| `scenario` | str | 自然文のシナリオ記述 |
| `turns` | list[obj] | 長さ 6 固定 |
| `session_tags` | list[str] | institutional + scenario category など |

### turn オブジェクト

| key | 型 | 説明 |
|---|---|---|
| `turn_id` | int | 1-6 |
| `question` | str | 日本語 Q |
| `reference_answer` | str | 人手検証で追記 |
| `reference_chunk_ids` | list[str] | Issue #109 chunk_id |
| `grading_notes` | str | grader 指示 |
| `tags` | list[str] | 例 `local_refresh / follow_up / exact_quote` |
| `expected_citation_patterns` | list[str] | Optional (B-5 用、turn 単位) |

## 6-turn drill_down パターン

| turn | 話題 | 参照箇所 |
|---|---|---|
| 1 | 定義 | metadata.title / 前文 |
| 2 | 適用範囲 | 第1〜2条 |
| 3 | 中核条項 | 任意の条文 |
| 4 | 罰則 | 罰則章 |
| 5 | 例外・但書 | 但書・経過措置条文 |
| 6 | 概観 | 全体要約 |

## Optional フィールド（session 単位）

| key | 型 | 説明 |
|---|---|---|
| `source_document_id` | str | 1 session = 1 source doc (primary) |
| `generator_model` | str | 生成モデル |
| `human_verified` | bool | session-level 検証フラグ |

## シナリオ内訳

| scenario | セッション数 | 割り当て |
|---|---|---|
| `drill_down` | 15 | 1 session = 1 doc |
| `cross_reference` | 10 | 1 session = primary 1 doc + 参照先他条文 |
| `real_scenario` | 5 | 1 session = 1 doc (実務シナリオ) |
| **計** | **30** | |

## サンプル行

```jsonl
{"session_id": "INST-MT-001", "category": "drill_down", "scenario": "賃貸住宅管理業法の定義→適用→罰則ドリルダウン", "source_document_id": "0123_rental_mgmt", "generator_model": "gpt-4o-mini-2024-07-18", "human_verified": false, "turns": [{"turn_id": 1, "question": "賃貸住宅管理業者の定義は？", "reference_answer": "", "reference_chunk_ids": ["institutional::0123_rental_mgmt/document.md::0-40"], "grading_notes": "第2条の定義を引用", "tags": ["definition"], "expected_citation_patterns": ["第2条"]}], "session_tags": ["drill_down", "institutional"]}
```
