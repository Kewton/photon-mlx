---
model: sonnet
description: "GitHub Issueを作成（テンプレート準拠・品質チェック付き）"
---

# Issue作成スキル

## 概要
PHOTON-RepoRAGプロジェクトの規約に準拠したGitHub Issueを作成するスキルです。Issue種別に応じたテンプレートを使用し、品質基準を満たすIssueを生成します。

## 使用方法
- `/issue-create [概要説明]`
- 「新しいIssueを作成してください: ○○機能の追加」

## 前提条件
- GitHubリポジトリ（https://github.com/Kewton/photon-mlx）にアクセス可能
- `gh` CLIが認証済み

## 実行内容

あなたはプロジェクトマネージャーとして、高品質なIssueを作成します。

### 1. Issue種別の判定

ユーザーの説明から以下の種別を判定します：

| 種別 | ラベル | 説明 |
|------|--------|------|
| 機能追加 | `feature` | 新しい機能の追加 |
| バグ修正 | `bug` | 既存機能の不具合修正 |
| リファクタリング | `refactor` | コード品質の改善 |
| ドキュメント | `documentation` | ドキュメントの追加・更新 |
| パフォーマンス | `performance` | パフォーマンス改善 |

### 2. コードベース調査

対象領域のコードを調査し、Issueに必要な情報を収集します：

```bash
# プロジェクト構造の確認
ls baseline_reporag/

# 関連モジュールの確認
cat baseline_reporag/{関連モジュール}/mod.rs
```

#### PHOTON-RepoRAGのモジュール構成

```
baseline_reporag/
├── agent/       # エージェントループ・プロトコル
├── app/         # アプリケーションオーケストレータ
├── config/      # 設定管理
├── contracts/   # 共通型定義
├── extensions/  # スラッシュコマンド・拡張
├── generation/    # 回答生成 (generator.py, prompt.py, evidence_pack.py)
├── tooling/     # ツール実行・検証
├── tui/         # TUI描画
├── session/     # セッション永続化
└── state/       # 状態マシン
```

### 3. Issue本文の作成

#### 機能追加テンプレート

```markdown
## 概要
[機能の簡潔な説明]

## 背景・動機
[なぜこの機能が必要か]

## 技術スタック
- Python 3.12+, pip
- 関連モジュール: baseline_reporag/xxx/

## 要件
### 機能要件
- [ ] [要件1]
- [ ] [要件2]

### 非機能要件
- [ ] `ruff check .` 警告0件
- [ ] `python -m pytest` 全テストパス
- [ ] `ruff format --check .` 差分なし

## 影響範囲
- 変更対象: `baseline_reporag/xxx/__init__.py`
- テスト: `photon_mlx/tests/xxx.py`

## 受入条件
- [ ] [条件1]
- [ ] [条件2]
- [ ] 品質チェック全パス

## 参考情報
- [関連Issue/PR]
```

#### バグ修正テンプレート

```markdown
## 概要
[バグの簡潔な説明]

## 再現手順
1. [手順1]
2. [手順2]
3. [手順3]

## 期待動作
[正しい動作の説明]

## 実際の動作
[現在の不正な動作の説明]

## 環境
- OS: [OS名]
- Rust: [バージョン]
- PHOTON-RepoRAG: [バージョン/コミット]

## 技術的調査
- 原因推定: [推定される原因]
- 関連コード: `baseline_reporag/xxx/__init__.py`

## 修正方針
- [ ] [修正内容1]
- [ ] テスト追加

## 受入条件
- [ ] バグが再現しなくなること
- [ ] 回帰テストが追加されていること
- [ ] 品質チェック全パス
```

### 4. Issue作成

```bash
gh issue create \
  --repo Kewton/photon-mlx \
  --title "{type}: {description}" \
  --body "{issue_body}" \
  --label "{label}"
```

### 5. 品質チェック

作成前に以下を確認します：

- [ ] タイトルが `<type>: <description>` 形式になっている
- [ ] 概要が明確で簡潔である
- [ ] 要件・受入条件が具体的である
- [ ] 影響範囲が特定されている
- [ ] 関連するモジュールが記載されている
- [ ] 品質基準（python -m pytest/ruff/test/fmt）が含まれている

### 6. 作成完了報告

```markdown
## Issue作成完了

- **Issue**: #{issue_number}
- **タイトル**: {title}
- **種別**: {type}
- **ラベル**: {label}
- **URL**: https://github.com/Kewton/photon-mlx/issues/{issue_number}

### 次のアクション
- [ ] `/issue-enhance` でIssue内容を補完
- [ ] `/work-plan` で作業計画を立案
- [ ] `/worktree-setup` でWorktree環境を構築
```

## 完了条件

- GitHub Issueが作成されている
- テンプレートに準拠した本文が記載されている
- 適切なラベルが付与されている
- 品質基準が含まれている

## 関連コマンド

- `/issue-enhance`: Issue内容の補完
- `/issue-split`: 大規模Issueの分割
- `/work-plan`: 作業計画立案
- `/pm-auto-issue2dev`: Issue補完から開発まで一括実行
