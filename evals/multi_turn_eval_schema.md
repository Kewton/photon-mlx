# Multi-turn Session Eval – JSONL Schema

ファイル: `data/eval_sets/multi_turn_eval.jsonl`  
各行が1セッション（6ターン）。

## フィールド

```json
{
  "session_id": "MT-001",
  "category": "onboarding",
  "scenario": "依存性注入の仕組みを段階的に深掘りするセッション",
  "turns": [
    {
      "turn_id": 1,
      "question": "FastAPI の全体構成を教えてください。",
      "reference_answer": "...",
      "reference_chunk_ids": [],
      "grading_notes": "主要ディレクトリと役割を正しく説明していること",
      "tags": []
    },
    {
      "turn_id": 2,
      "question": "依存性注入の仕組みをコードで説明してください。",
      "reference_answer": "...",
      "reference_chunk_ids": [],
      "grading_notes": "Depends() の実装を引用していること",
      "tags": ["local_refresh"]
    }
  ],
  "session_tags": ["topic_narrowing"]
}
```

## session_tags

| tag | 意味 |
|---|---|
| `topic_narrowing` | 話題が段階的に絞り込まれるセッション |
| `topic_shift` | 途中で話題が切り替わるセッション（drift 検知テスト） |
| `exact_quote` | exact quote 要求ターンを含む（Safe RecGen 発火テスト） |
| `diff_or_patch` | diff / patch 要求ターンを含む |
| `high_risk` | 認可・課金・セキュリティ系の質問を含む |

## turn tags

| tag | 意味 |
|---|---|
| `local_refresh` | answer-time local refresh が期待されるターン |
| `follow_up` | 前ターンの内容を前提とした質問 |
| `topic_shift_here` | このターンで話題が切り替わる |
| `exact_quote` | exact quote を明示的に要求 |
