# Stress Eval – JSONL Schema

ファイル: `data/eval_sets/stress_eval.jsonl`  
各行が1セッション。8セッションを同時起動して memory / latency / fallback を計測。

## フィールド

```json
{
  "session_id": "ST-001",
  "concurrency_group": 1,
  "turns": [
    {
      "turn_id": 1,
      "question": "...",
      "tags": []
    }
  ]
}
```

## 計測対象

- peak memory / active session
- P50 / P90 end-to-end latency（全8セッション）
- fallback 発火率
- session consistency（前ターンの前提を維持できたか）

## 構成ルール

- 8セッション × 10ターン = 80ターン
- 各セッションは独立した話題から開始する
- うち2セッションは意図的に topic shift を含む
- うち2セッションは exact quote / patch 要求を含む
