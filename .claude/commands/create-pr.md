---
model: sonnet
description: "PRを自動作成し、CIチェックまで実行"
---

# PR自動作成スキル

## 概要
現在のブランチの変更内容からPull Requestを自動作成します。

## 使用方法
- `/create-pr`

## 前提条件
- 作業ブランチ上にいること（mainブランチではないこと）
- コミット済みの変更があること

## 実行手順

### 1. 事前チェック

```bash
git branch --show-current
git log main..HEAD --oneline
```

現在のブランチがmainの場合はエラーとして中断してください。

### 2. ビルド・テスト確認

PR作成前に以下を実行し、すべてパスすることを確認：

```bash
ruff format --check .
ruff check .
python -m pytest torch_ref/tests/ photon_mlx/tests/ -v
python -m pytest
```

いずれかが失敗した場合は修正してからPR作成に進んでください。

### 3. 変更内容の分析

```bash
git diff main..HEAD --stat
git log main..HEAD --pretty=format:"%s"
```

### 4. PR作成

`gh pr create` を使用してPRを作成：

- **タイトル**: コミット内容を要約（70文字以内）
- **本文**: 以下のテンプレートに従う

```markdown
## 概要
<!-- 変更の目的と内容を簡潔に -->

## 変更内容
<!-- 主な変更点をリスト -->

## テスト
- [ ] `python -m pytest torch_ref/tests/ photon_mlx/tests/ -v` パス
- [ ] `ruff check .` 警告ゼロ
- [ ] `ruff format --check .` パス

## 関連Issue
<!-- Closes #XXX -->
```

### 5. CI確認

PR作成後、CIの状態を確認：

```bash
gh pr checks
```

## 完了条件
- PRが作成されていること
- CIが開始されていること
- PR URLが出力されていること
