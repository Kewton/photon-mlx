---
model: sonnet
description: "Issue開発を完全自動化（TDD→テスト→リファクタリング→報告）"
---

# PM自動開発スキル

## 概要
Issue開発（TDD実装 → 受入テスト → リファクタリング → 進捗報告）を**完全自動化**するプロジェクトマネージャースキルです。

**アーキテクチャ**: サブエージェント方式を採用し、各フェーズを専門エージェントに委譲します。

## 使用方法
- `/pm-auto-dev [Issue番号]`
- `/pm-auto-dev [Issue番号] --max-iterations=5`

## 実行内容

あなたはプロジェクトマネージャーとして、Issue開発を統括します。

### パラメータ
- **issue_number**: 開発対象のIssue番号（必須）
- **max_iterations**: 最大イテレーション回数（デフォルト: 3）

### サブエージェント/レビュアー指定

| エージェント | モデル/ツール | 理由 |
|-------------|-------------|------|
| tdd-impl-agent | **Claude opus** | コード生成にOpus必要 |
| **codex-reviewer** | **Codex**（commandmatedev経由） | 潜在バグ・セキュリティ脆弱性の独立レビュー |
| acceptance-test-agent | **Claude opus** | テスト品質にOpus必要 |
| refactoring-agent | **Claude opus** | コード改善にOpus必要 |
| progress-report-agent | sonnet | テンプレート埋め込み程度 |

---

## 実行フェーズ

### Phase 0: 初期設定

TodoWriteツールで作業計画を作成：

```
- [ ] Phase 1: Issue情報収集
- [ ] Phase 2: TDD実装 (イテレーション 0/3)
- [ ] Phase 2.5: Codexコードレビュー（潜在バグ・セキュリティ）
- [ ] Phase 3: 受入テスト
- [ ] Phase 4: リファクタリング
- [ ] Phase 5: ドキュメント最新化
- [ ] Phase 6: 進捗報告
```

### Phase 1: Issue情報収集

```bash
gh issue view {issue_number} --json number,title,body,labels,assignees
```

ディレクトリ構造作成：
```bash
BRANCH=$(git branch --show-current)
ISSUE_NUM=$(echo "$BRANCH" | grep -oE '[0-9]+')
BASE_DIR="workspace/issues/${ISSUE_NUM}/pm-auto-dev/iteration-1"
mkdir -p "$BASE_DIR"
```

### Phase 2: TDD実装（イテレーション可能）

TDDコンテキストファイル作成後、サブエージェント呼び出し：
```
Use tdd-impl-agent (model: opus) to implement Issue #{issue_number} with TDD approach.
Context file: workspace/issues/{issue_number}/pm-auto-dev/iteration-1/tdd-context.json
Output file: workspace/issues/{issue_number}/pm-auto-dev/iteration-1/tdd-result.json
```

### Phase 2.5: Codexコードレビュー（潜在バグ・セキュリティ脆弱性）

TDD実装完了後、受入テスト前に**Codex**による独立コードレビューを実施します。
Claude（実装者）とは異なるモデルの視点で、潜在バグとセキュリティ脆弱性を検出します。

#### 2.5-1. 変更ファイルの特定

```bash
git diff develop --name-only | grep '.py$' > /tmp/changed_files.txt
CHANGED_FILES=$(cat /tmp/changed_files.txt | tr '\n' ', ')
```

#### 2.5-2. Codexへのレビュー依頼

```bash
WORKTREE_ID=$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]' | sed 's/^photon-mlx-/photon-mlx-/')

commandmatedev send "$WORKTREE_ID" \
  "Issue #{issue_number} のTDD実装コードに対して、以下の観点でコードレビューを実施してください。

## レビュー観点
1. **潜在バグ**: ロジックエラー、エッジケースの未処理、off-by-one、NULL/None未チェック、リソースリーク
2. **セキュリティ脆弱性**: コマンドインジェクション、パストラバーサル、eval使用、入力検証不足、情報漏洩

## 対象ファイル
${CHANGED_FILES}

## 指示
1. 上記の変更ファイルを全て読み込んでください
2. 各ファイルに対して上記2観点でレビュー
3. 結果をJSON形式で workspace/issues/{issue_number}/pm-auto-dev/iteration-1/codex-review-result.json に出力

## 出力フォーマット
{
  \"reviewer\": \"codex\",
  \"issue_number\": {issue_number},
  \"review_focus\": [\"潜在バグ\", \"セキュリティ脆弱性\"],
  \"files_reviewed\": [\"baseline_reporag/...\", ...],
  \"findings\": [
    {
      \"id\": \"CB-001\",
      \"severity\": \"critical|high|medium|low\",
      \"category\": \"潜在バグ|セキュリティ脆弱性\",
      \"file\": \"baseline_reporag/...\",
      \"line\": 123,
      \"title\": \"問題の要約\",
      \"description\": \"詳細説明\",
      \"suggestion\": \"修正案\"
    }
  ],
  \"summary\": {
    \"critical\": 0,
    \"high\": 0,
    \"medium\": 0,
    \"low\": 0,
    \"total\": 0
  },
  \"verdict\": \"pass|needs_fix\"
}" \
  --agent codex --auto-yes --duration 1h
```

#### 2.5-3. Codex完了待機

```bash
commandmatedev wait "$WORKTREE_ID" --timeout 3600 --on-prompt agent
```

#### 2.5-4. レビュー結果確認

Codexのレビュー結果ファイルを読み込み、判定を行う：

```bash
cat workspace/issues/{issue_number}/pm-auto-dev/iteration-1/codex-review-result.json
```

- **verdict: "pass"** → Phase 3（受入テスト）に進む
- **verdict: "needs_fix"** で critical/high 指摘あり → 指摘内容を確認し、Phase 2に戻ってTDD修正を実施。修正後に再度Phase 2.5を実行（最大2回まで）
- **verdict: "needs_fix"** で medium/low のみ → Phase 3に進む（Phase 4リファクタリングで対応）

---

### Phase 3: 受入テスト

```
Use acceptance-test-agent (model: opus) to verify Issue #{issue_number}.
Context file: workspace/issues/{issue_number}/pm-auto-dev/iteration-1/acceptance-context.json
Output file: workspace/issues/{issue_number}/pm-auto-dev/iteration-1/acceptance-result.json
```

### Phase 4: リファクタリング

```
Use refactoring-agent (model: opus) to improve code quality for Issue #{issue_number}.
Context file: workspace/issues/{issue_number}/pm-auto-dev/iteration-1/refactor-context.json
Output file: workspace/issues/{issue_number}/pm-auto-dev/iteration-1/refactor-result.json
```

### Phase 5: ドキュメント最新化

- **README.md**: 機能一覧の更新
- **CLAUDE.md**: モジュール構成の更新

### Phase 6: 進捗報告

```
Use progress-report-agent to generate progress report for Issue #{issue_number}.
Context file: workspace/issues/{issue_number}/pm-auto-dev/iteration-1/progress-context.json
Output file: workspace/issues/{issue_number}/pm-auto-dev/iteration-1/progress-report.md
```

## ファイル構造

```
workspace/issues/{issue_number}/
├── work-plan.md
└── pm-auto-dev/
    └── iteration-1/
        ├── tdd-context.json
        ├── tdd-result.json
        ├── codex-review-result.json    ← Codexコードレビュー結果
        ├── acceptance-context.json
        ├── acceptance-result.json
        ├── refactor-context.json
        ├── refactor-result.json
        ├── progress-context.json
        └── progress-report.md
```

## 完了条件

- Phase 2: TDD実装成功（全テストパス、ruff警告0件）
- Phase 2.5: Codexコードレビュー完了（critical/high指摘なし、またはTDD修正済み）
- Phase 3: 受入テスト成功
- Phase 4: リファクタリング完了
- Phase 5: ドキュメント最新化完了
- Phase 6: 進捗レポート作成完了
