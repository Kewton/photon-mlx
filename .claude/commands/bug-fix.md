---
model: opus
description: "バグの調査から修正、テストまでを自動実行"
---

# バグ修正スキル

## 概要
報告されたバグを調査し、原因を特定、修正、テストを追加するスキルです。

## 使用方法
- `/bug-fix [Issue番号またはバグの説明]`

## 実行手順

### 1. Issue情報の取得（Issue番号が指定された場合）

```bash
source .claude/lib/validators.sh
validate_issue_number "$ARGUMENTS"
gh issue view "$ISSUE_NUMBER" --json title,body,labels,comments
```

### 2. バグの調査

investigationエージェントを使用して調査：

```
Use investigation-agent to investigate the bug described in Issue #XXX.
```

### 3. 再現テストの作成（Red）

バグを再現するテストを先に作成：

```bash
# テストを追加した後
python -m pytest torch_ref/tests/ photon_mlx/tests/ -v 2>&1 | tail -20
```

テストが**失敗する**ことを確認（Red状態）。

### 4. バグの修正（Green）

最小限の変更でバグを修正：

```bash
python -m pytest torch_ref/tests/ photon_mlx/tests/ -v
```

テストが**成功する**ことを確認（Green状態）。

### 5. リファクタリング

修正コードの品質を確認・改善：

```bash
ruff check .
ruff format --check .
```

### 6. コミット

```bash
git add -A
git commit -m "fix: [修正内容の説明]

Fixes #XXX"
```

## 完了条件
- バグの再現テストが追加されている
- すべてのテストがパス
- `ruff check` で警告ゼロ
- 修正コミットが完了
