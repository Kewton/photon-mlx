---
model: sonnet
description: "複数Issueの実行計画を策定（優先度・依存関係・スケジュール）"
---

# Issues実行計画スキル

## 概要
複数のIssueに対する実行計画を策定するスキルです。優先度、依存関係、リソース制約を考慮し、最適な実行順序とスケジュールを生成します。

## 使用方法
- `/issues-exec-plan`
- `/issues-exec-plan [ラベルフィルター]`
- 「未対応Issueの実行計画を作成してください」

## 前提条件
- GitHubリポジトリ（https://github.com/Kewton/photon-mlx）にアクセス可能
- 対象Issueが存在すること

## 実行内容

あなたはプロジェクトマネージャーとして、複数Issueの実行計画を策定します。

### 1. Issue一覧の取得

```bash
# 全オープンIssueの取得
gh issue list --repo Kewton/photon-mlx --state open --json number,title,labels,assignees,milestone

# ラベルフィルター（指定時）
gh issue list --repo Kewton/photon-mlx --state open --label "{label}" --json number,title,labels,assignees,milestone
```

### 2. Issue分析

各Issueについて以下を分析します：

| 項目 | 説明 |
|------|------|
| **サイズ** | S (1-2h), M (3-4h), L (5-8h), XL (8h+) |
| **優先度** | P0 (緊急), P1 (高), P2 (中), P3 (低) |
| **種別** | feature, bug, refactor, docs |
| **影響モジュール** | baseline_reporag/xxx/ |
| **依存関係** | 他Issueとの前後関係 |

### 3. 依存関係マッピング

```markdown
## 依存関係グラフ

#{issue_a} (型定義)
    │
    ├──▶ #{issue_b} (コアロジック)
    │       │
    │       ▶ #{issue_d} (統合テスト)
    │
    ▶ #{issue_c} (設定管理)
```

### 4. 実行順序の決定

#### 優先度マトリクス

| | 影響度 高 | 影響度 中 | 影響度 低 |
|---|---------|---------|---------|
| **緊急度 高** | P0: 即座対応 | P1: 優先対応 | P2: 計画対応 |
| **緊急度 中** | P1: 優先対応 | P2: 計画対応 | P3: バックログ |
| **緊急度 低** | P2: 計画対応 | P3: バックログ | P3: バックログ |

### 5. 実行計画の策定

```markdown
## 実行計画

### Sprint 1 (Week 1)

| 順序 | Issue | タイトル | サイズ | 優先度 | 依存 |
|------|-------|---------|--------|--------|------|
| 1 | #A | 型定義の追加 | S | P1 | なし |
| 2 | #B | Provider抽象化 | M | P1 | #A |

### Sprint 2 (Week 2)

| 順序 | Issue | タイトル | サイズ | 優先度 | 依存 |
|------|-------|---------|--------|--------|------|
| 3 | #C | ツール実行改善 | M | P2 | #A |
| 4 | #D | TUI更新 | S | P2 | #B |
```

### 6. リスク評価

| リスク | 影響 | 対策 |
|--------|------|------|
| Issue間の依存関係による遅延 | スケジュール遅延 | クリティカルパスを優先 |
| 技術的不確実性 | 見積もり超過 | Spike Issueで事前調査 |
| モジュール間の整合性 | 統合時の不具合 | 早期統合テスト |

### 7. 品質ゲート

各Issue完了時に以下を確認：

```bash
python -m pytest
ruff check .
python -m pytest
ruff format --check .
```

### 8. 実行計画サマリー

```markdown
## Issues実行計画サマリー

### 概要
- 対象Issue数: N件
- 総見積もり: XXh
- スプリント数: N

### 実行順序（クリティカルパス）

1. #{issue_a} (S) → 2. #{issue_b} (M) → 3. #{issue_d} (M)

### 並行実行可能

- #{issue_c} は #{issue_a} 完了後に並行実行可能

### 次のアクション

- [ ] Sprint 1 の最初のIssueから `/work-plan` を実行
- [ ] `/pm-auto-issue2dev` で自動開発を開始
```

## 出力先

`workspace/issues-exec-plan.md`

## 完了条件

- 全オープンIssueが分析されている
- 依存関係が明確に記載されている
- 優先度・サイズが判定されている
- 実行順序が決定されている
- リスク評価が行われている

## 関連コマンド

- `/issue-create`: Issue作成
- `/issue-split`: Issue分割
- `/issue-enhance`: Issue内容の補完
- `/work-plan`: 作業計画立案
- `/pm-auto-issue2dev`: Issue補完から開発まで一括実行
