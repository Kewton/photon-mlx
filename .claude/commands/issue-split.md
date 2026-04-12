---
model: opus
description: "大規模Issueを実装可能な粒度に分割"
---

# Issue分割スキル

## 概要
大規模なIssueを実装可能な適切な粒度に分割するスキルです。依存関係を考慮し、Phase分けされた実行計画を生成します。

## 使用方法
- `/issue-split [Issue番号]`
- 「Issue #XXXを分割してください」

## 前提条件
- 対象Issueが存在すること
- Issue内容が十分に記述されていること（不足時は `/issue-enhance` を先に実行）

## 実行内容

あなたはテックリードとして、Issueを適切な粒度に分割します。

### 1. Issue情報の取得

```bash
gh issue view {issue_number} --json number,title,body,labels,assignees
```

### 2. 分割基準

以下の基準で分割を判断します：

| 基準 | 閾値 | 判定 |
|------|------|------|
| 変更ファイル数 | 5ファイル以上 | 分割推奨 |
| 変更モジュール数 | 3モジュール以上 | 分割推奨 |
| 実装時間見積もり | 4時間以上 | 分割推奨 |
| テストケース数 | 10ケース以上 | 分割推奨 |
| 要件数 | 5つ以上 | 分割推奨 |

### 3. 分割方針

#### 3-1. レイヤー別分割

PHOTON-RepoRAGのモジュール構成に基づいて分割します：

| レイヤー | モジュール | 分割単位 |
|---------|-----------|---------|
| データモデル | `baseline_reporag/contracts/` | 型定義・共通インターフェース |
| プロバイダー | `baseline_reporag/provider/` | LLMバックエンド実装 |
| ツール | `baseline_reporag/tooling/` | ツール定義・実行 |
| エージェント | `baseline_reporag/agent/` | プロトコル・ループ |
| アプリ | `baseline_reporag/app/` | オーケストレーション |
| UI | `baseline_reporag/tui/` | ターミナルUI |
| 設定 | `baseline_reporag/config/` | 設定管理 |
| セッション | `baseline_reporag/session/` | 永続化 |

#### 3-2. Phase分け

```markdown
## Phase 1: 基盤（依存なし）
- Sub-issue #A: 型定義・protocol追加
  - 対象: baseline_reporag/contracts/mod.rs
  - 依存: なし

## Phase 2: コアロジック（Phase 1に依存）
- Sub-issue #B: コアロジック実装
  - 対象: baseline_reporag/xxx/mod.rs
  - 依存: Sub-issue #A

## Phase 3: 統合（Phase 2に依存）
- Sub-issue #C: 既存モジュールとの統合
  - 対象: baseline_reporag/app/mod.rs
  - 依存: Sub-issue #B

## Phase 4: テスト・ドキュメント
- Sub-issue #D: 統合テスト
  - 対象: photon_mlx/tests/xxx.rs
  - 依存: Sub-issue #C
```

### 4. Sub-issue品質基準

各Sub-issueは以下の品質基準を満たすこと：

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| ビルド | `python -m pytest` | エラー0件 |
| Clippy | `ruff check .` | 警告0件 |
| テスト | `python -m pytest` | 全テストパス |
| フォーマット | `ruff format --check .` | 差分なし |

### 5. 依存関係グラフ

```
Sub-issue #A (型定義)
    │
    ▼
Sub-issue #B (コアロジック)
    │
    ▼
Sub-issue #C (統合)
    │
    ▼
Sub-issue #D (テスト・ドキュメント)
```

### 6. Sub-issue作成

各Sub-issueをGitHubに作成します：

```bash
gh issue create \
  --repo Kewton/photon-mlx \
  --title "feat: {sub-issue title} (part of #{issue_number})" \
  --body "{sub-issue body}" \
  --label "feature"
```

親Issueに分割結果を記録：

```bash
gh issue comment {issue_number} --body "## Issue分割結果

このIssueは以下のSub-issueに分割されました：

### Phase 1: 基盤
- [ ] #{sub_a} - 型定義・protocol追加

### Phase 2: コアロジック
- [ ] #{sub_b} - コアロジック実装

### Phase 3: 統合
- [ ] #{sub_c} - 既存モジュールとの統合

### Phase 4: テスト
- [ ] #{sub_d} - 統合テスト

### 実行順序
Sub-issue #A → #B → #C → #D"
```

### 7. 分割サマリー

```markdown
## Issue分割サマリー

### 親Issue: #{issue_number}
### 分割数: N個

| Phase | Sub-issue | タイトル | 依存 | 見積もり |
|-------|-----------|---------|------|---------|
| 1 | #A | 型定義 | なし | S |
| 2 | #B | コアロジック | #A | M |
| 3 | #C | 統合 | #B | M |
| 4 | #D | テスト | #C | S |

### 担当
- 全Sub-issue: Rust開発者

### 次のアクション
- [ ] 各Sub-issueの `/work-plan` を作成
- [ ] Phase 1 から順次実装開始
```

## 出力先

分割結果は親Issueのコメントとして記録されます。

## 完了条件

- Sub-issueがすべてGitHubに作成されている
- 依存関係が明確に記載されている
- Phase分けが適切である
- 各Sub-issueが独立してビルド・テスト可能な粒度である
- 親Issueに分割結果が記録されている

## 関連コマンド

- `/issue-create`: Issue作成
- `/issue-enhance`: Issue内容の補完
- `/work-plan`: 作業計画立案
- `/pm-auto-issue2dev`: Issue補完から開発まで一括実行
