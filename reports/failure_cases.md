# Failure Cases

このファイルは benchmark 実行中に発見された失敗ケースを記録する。
テンプレートは `reports/failure_case_template.md` を参照。

---

## FC-001: 依存性注入の説明で citation なし

### 基本情報

| 項目 | 値 |
|---|---|
| run_id | baseline_eval_fastapi_fastapi_20260412_232239 |
| eval_id | SE-ONB-003 |
| repo_id | fastapi_fastapi |
| 発生日 | 2026-04-12 |

### カテゴリ

- [x] no-citation assertion（根拠なし断定）

### 質問

```
FastAPI の依存性注入の仕組みを説明してください。
```

### 観察

- モデルは依存性注入の仕組みを概説する回答を生成したが、[C:N] citation が一切含まれていない
- retrieval は実行されており evidence pack にも chunk が含まれている
- generation 時に citation フォーマットを使わずに回答した

### 根本原因の分類

- [ ] retrieval 側の問題
- [x] generation 側の問題（citation フォーマット遵守の不足）
- [ ] memory 側の問題
- [ ] fallback 側の問題

### 対策候補

1. system prompt の citation 要求を強化（few-shot example を追加）
2. evidence pack の先頭に「必ず [C:N] を使って引用してください」を追加
3. post-processing で citation なし回答を検出して再生成

### ステータス

- [x] 未対処
- [ ] 対処中
- [ ] 修正済み
- [ ] 既知の限界として記録

---

## 統計サマリー

| 指標 | 4問サンプル | Full eval |
|---|---|---|
| no-citation 件数 | 1/4 (25%) | TBD |
| wrong citation 件数 | 0/4 (0%) | TBD |
| stale memory 件数 | N/A | TBD |
| missed fallback 件数 | N/A | TBD |
