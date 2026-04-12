---
model: sonnet
description: "Issue用のgit worktreeを作成し、作業環境をセットアップ"
---

# Worktreeセットアップスキル

## 概要
指定されたIssueに対応するgit worktreeを作成し、開発環境をセットアップします。

## 使用方法
- `/worktree-setup [Issue番号]`

## 実行手順

### 1. Issue番号の検証

```bash
source .claude/lib/validators.sh
validate_issue_number "$ARGUMENTS"
```

### 2. Issue情報の取得

```bash
gh issue view "$ISSUE_NUMBER" --json title,body,labels
```

### 3. ブランチ名の決定

Issue情報からブランチ名を自動生成：
- フォーマット: `feature/issue-{N}-{短い説明}`（英語、ケバブケース）
- 例: `feature/issue-42-add-streaming-support`

### 4. Worktreeの作成

```bash
WORKTREE_DIR="../photon-mlx-issue-${ISSUE_NUMBER}"
git worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR" develop
```

### 5. 環境セットアップ

```bash
cd "$WORKTREE_DIR"
python -m pytest
python -m pytest torch_ref/tests/ photon_mlx/tests/ -v
```

ビルドが成功することを確認してください。

### 6. 初期コミット（必要に応じて）

作業開始を示す空コミットを作成：

```bash
git commit --allow-empty -m "chore: start work on #${ISSUE_NUMBER}"
```

## 完了条件
- Worktreeが `../photon-mlx-issue-{N}` に作成されている
- ブランチが作成されている
- `python -m pytest` が成功している
- 作業ディレクトリのパスが出力されている

## 出力例
```
Worktree作成完了:
  ディレクトリ: ../photon-mlx-issue-42
  ブランチ: feature/issue-42-add-streaming-support
  ステータス: ビルド成功
```
