---
model: sonnet
description: "Issue単位の具体的な作業計画立案"
---

# 作業計画立案スキル（Issue単位）

## 概要
Issue単位での具体的な作業計画を立案し、実装タスクの詳細化を策定するスキルです。

## 使用方法
- `/work-plan [Issue番号または概要]`

## 前提条件
- 対象Issueの概要と要件が明確
- GitHubリポジトリにアクセス可能

## 実行内容

あなたはテックリードです。1つのIssue実装のための具体的な作業計画を立案してください：

### 1. Issue概要の確認

まずIssue情報を取得します：

```bash
gh issue view {issue_number} --json number,title,body,labels,assignees
```

以下の形式で概要をまとめます：

```markdown
## Issue: [タイトル]
**Issue番号**: #XXX
**サイズ**: S/M/L
**優先度**: High/Medium/Low
**依存Issue**: #YYY（あれば）
```

### 2. 詳細タスク分解

#### 実装タスク（Phase 1）
- [ ] **Task 1.1**: データモデル・型定義
  - 成果物: `baseline_reporag/xxx/__init__.py` or `baseline_reporag/xxx.py`
  - 依存: なし

- [ ] **Task 1.2**: コアロジック実装
  - 成果物: `baseline_reporag/xxx/__init__.py`
  - 依存: Task 1.1

- [ ] **Task 1.3**: 既存モジュールとの統合
  - 成果物: `baseline_reporag/app/__init__.py` 等
  - 依存: Task 1.2

#### テストタスク（Phase 2）
- [ ] **Task 2.1**: ユニットテスト
  - 成果物: `photon_mlx/tests/xxx.py`
  - カバレッジ目標: 主要パスを網羅

- [ ] **Task 2.2**: 統合テスト
  - 成果物: `photon_mlx/tests/xxx_integration.py`

#### ドキュメントタスク（Phase 3）
- [ ] **Task 3.1**: README更新（必要な場合）
- [ ] **Task 3.2**: CLAUDE.md更新（モジュール構成変更時）

### 3. 品質チェック項目

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| ビルド | `python -m pytest` | エラー0件 |
| Clippy | `ruff check .` | 警告0件 |
| テスト | `python -m pytest` | 全テストパス |
| フォーマット | `ruff format --check .` | 差分なし |

### 4. Definition of Done

Issue完了条件：
- [ ] すべてのタスクが完了
- [ ] テストが全パス
- [ ] ruff警告ゼロ
- [ ] CIチェック全パス
- [ ] コードレビュー承認
- [ ] ドキュメント更新完了

### 5. 次のアクション

作業計画承認後：
1. **ブランチ作成**: `feature/{issue_number}-[feature-name]`
2. **タスク実行**: 計画に従って実装
3. **進捗報告**: `/progress-report` で定期報告
4. **PR作成**: `/create-pr` で自動作成

## 出力先

`workspace/issues/{issue_number}/work-plan.md`
