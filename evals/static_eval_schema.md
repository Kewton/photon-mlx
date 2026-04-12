# Static Eval – JSONL Schema

ファイル: `data/eval_sets/static_eval.jsonl`  
各行が1問。

## フィールド

```json
{
  "id": "SE-001",
  "category": "onboarding",
  "difficulty": "easy",
  "question": "FastAPI の依存性注入の仕組みを説明してください。",
  "reference_answer": "...",
  "reference_chunk_ids": ["fastapi_fastapi::fastapi/dependencies/...::10-45"],
  "grading_notes": "Depends() の仕組みと依存関係の解決順序に言及していること",
  "rubric": {
    "correctness": {"max": 2, "notes": ""},
    "grounding":   {"max": 2, "notes": ""},
    "usefulness":  {"max": 1, "notes": ""}
  }
}
```

## カテゴリ定義

| category | 問数 | 観点 |
|---|---|---|
| `onboarding` | 30 | repo の構成・主要モジュール・読み始め方 |
| `impact_analysis` | 30 | 変更の波及先・依存関係 |
| `bug_localization` | 30 | 障害原因候補・怪しい箇所の特定 |
| `change_planning` | 30 | 修正案の比較・最小変更・設計意図 |

## difficulty

- `easy`: 単一ファイル・単一関数で答えが得られる
- `medium`: 複数ファイルをまたぐ、または文脈理解が必要
- `hard`: 設計意図・トレードオフ・影響範囲の横断的把握が必要
