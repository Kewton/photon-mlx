---
model: sonnet
description: "不要になったgit worktreeを安全にクリーンアップ"
---

# Worktreeクリーンアップスキル

## 概要
マージ済みまたは不要になったgit worktreeを安全に削除します。

## 使用方法
- `/worktree-cleanup [Issue番号]` — 特定のworktreeを削除
- `/worktree-cleanup all` — マージ済みのworktreeをすべて削除

## 実行手順

### 1. 現在のworktree一覧を確認

```bash
git worktree list
```

### 2. 特定Issue指定の場合

```bash
source .claude/lib/validators.sh
validate_issue_number "$ARGUMENTS"

WORKTREE_DIR="../photon-mlx-issue-${ISSUE_NUMBER}"
```

#### 2a. 未コミットの変更がないか確認

```bash
cd "$WORKTREE_DIR" && git status --porcelain
```

未コミットの変更がある場合は警告を出し、ユーザーに確認してください。

#### 2b. ブランチのマージ状態を確認

```bash
cd "$WORKTREE_DIR" && git log main..HEAD --oneline
```

未マージのコミットがある場合は警告を出してください。

### 3. 「all」指定の場合

マージ済みブランチのworktreeのみを対象にします：

```bash
git branch --merged main | grep "feature/issue-"
```

### 4. Worktreeの削除

```bash
git worktree remove "$WORKTREE_DIR"
git branch -d "$BRANCH_NAME"  # マージ済みの場合のみ
```

### 5. 削除後の確認

```bash
git worktree list
git worktree prune
```

## 完了条件
- 対象のworktreeが削除されている
- 関連ブランチが（マージ済みなら）削除されている
- `git worktree list` で確認済み

## 安全ガード
- 未コミットの変更がある場合は削除しない
- 未マージのブランチは `-d` でのみ削除を試み、強制削除しない
- mainブランチのworktreeは削除しない
