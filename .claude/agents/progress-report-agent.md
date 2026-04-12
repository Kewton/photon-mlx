---
model: sonnet
description: "進捗レポートを生成するサブエージェント"
---

# 進捗レポートエージェント

## 役割
プロジェクトの現在の状態を分析し、進捗レポートを生成するエージェントです。

## 実行手順

### 1. 共通プロンプトの読み込み

```bash
cat .claude/prompts/progress-report-core.md
```

上記プロンプトの内容に従ってレポートを生成してください。

### 2. データ収集

```bash
# オープンIssue
gh issue list --state open --json number,title,labels,assignees

# 最近クローズしたIssue
gh issue list --state closed --limit 10 --json number,title,closedAt

# オープンPR
gh pr list --state open --json number,title,labels,reviewDecision

# 最近のコミット
git log --oneline -20

# ブランチ一覧
git branch -a

# worktree一覧
git worktree list
```

### 3. コードベース分析

```bash
# コード行数
find . -name "*.py" -not -path "./.venv/*" | xargs wc -l 2>/dev/null | tail -1
find . -path "*/tests/*.py" | xargs wc -l 2>/dev/null | tail -1

# テスト実行
python -m pytest torch_ref/tests/ photon_mlx/tests/ -v 2>&1 | tail -5

# ruff状態
ruff check . 2>&1 | tail -10
```

### 4. レポート生成

以下のフォーマットで出力：

```markdown
# 進捗レポート（YYYY-MM-DD）

## サマリー
- オープンIssue: X件
- オープンPR: X件
- 最近完了: X件

## 直近の成果
- ...

## 進行中の作業
- ...

## 今後の優先事項
- ...

## コードベース状態
- ソースコード: X行
- テストコード: X行
- ruff: 警告X件
- テスト: X件パス
```

## 出力
- マークダウン形式の進捗レポート
