# Failure Case Template

failure case を記録するときはこのテンプレートをコピーして使う。
ファイル名: `reports/failure_cases/FC-{NNN}.md`

---

## FC-NNN: {一行タイトル}

### 基本情報

| 項目 | 値 |
|---|---|
| run_id | |
| session_id | |
| turn_id | |
| repo_id | |
| repo_commit | |
| model_id | |
| 発生日 | |
| 報告者 | |

### カテゴリ

<!-- 該当するものをチェック -->
- [ ] wrong citation（存在しない / 無関係な箇所を引用）
- [ ] no-citation assertion（根拠なし断定）
- [ ] stale memory（古い仮説を引き継いで誤答）
- [ ] missed fallback（再読すべきで発火しなかった）
- [ ] false fallback（不要な fallback が発火した）
- [ ] retrieval failure（そもそも正しい chunk が取れていない）
- [ ] generation failure（chunk は正しいが回答が誤り）
- [ ] other:

### 質問（ターン全文）

```
Q1: 
Q2: 
...
```

### 実際の回答（問題のターン）

```

```

### 期待される回答

```

```

### 引用状況

| cited chunk ID | 内容の妥当性 | 備考 |
|---|---|---|
| | ○ / × | |

### 根本原因の分類

- [ ] retrieval 側の問題
- [ ] generation 側の問題
- [ ] memory 側の問題（stale / drift）
- [ ] fallback 側の問題

### 再現手順

```bash
# 再現コマンド
```

### 対策候補

1. 
2. 

### ステータス

- [ ] 未対処
- [ ] 対処中
- [ ] 修正済み（対応 run_id: ）
- [ ] 既知の限界として記録
