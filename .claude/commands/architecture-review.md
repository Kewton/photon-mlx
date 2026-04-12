---
model: opus
description: "アーキテクチャレビュー（SOLID/KISS/YAGNI/DRY・セキュリティ・リスク評価）"
---

# アーキテクチャレビュースキル

## 概要
Issueの実装方針に対してアーキテクチャレビューを実施するスキルです。設計原則（SOLID, KISS, YAGNI, DRY）の遵守、セキュリティ、リスク評価を包括的に行います。

## 使用方法
- `/architecture-review [Issue番号]`
- 「Issue #XXXのアーキテクチャレビューを実施してください」

## 前提条件
- 対象Issueの内容が明確であること
- 設計方針書（`/design-policy`）が作成済みであることが望ましい

## 実行内容

あなたはシニアアーキテクトとして、以下の観点でレビューを実施します。

### 1. Issue情報の取得

```bash
gh issue view {issue_number} --json number,title,body,labels,assignees
```

### 2. 対象コードの調査

影響範囲のコードを読み込み、現状のアーキテクチャを把握します：

```
baseline_reporag/
├── agent/__init__.py         # エージェントループ・プロトコル
├── app/                 # アプリケーションオーケストレータ
│   ├── pipeline.py, cli.py, server.py, profiler.py
├── config/__init__.py        # 設定管理
├── contracts/__init__.py     # 共通型定義
├── extensions/__init__.py    # スラッシュコマンド・拡張
├── provider/            # LLMプロバイダー
│   ├── generator.py, prompt.py, evidence_pack.py
├── tooling/__init__.py       # ツール実行・検証
├── tui/__init__.py           # TUI描画
├── session/__init__.py       # セッション永続化
└── state/__init__.py         # 状態マシン
```

### 3. 設計原則レビュー

#### 3-1. SOLID原則

| 原則 | チェック項目 | 評価 |
|------|-------------|------|
| **S** - 単一責任 | 各モジュール・構造体が単一の責務を持つか | - |
| **O** - 開放閉鎖 | protocolで拡張可能か、既存コード変更不要か | - |
| **L** - リスコフ置換 | protocol実装が契約を守るか | - |
| **I** - インターフェース分離 | protocolが肥大化していないか | - |
| **D** - 依存性逆転 | 具象型に直接依存していないか | - |

```python
// Good: protocol による依存性逆転
protocol LlmProvider {
    def chat(self, messages: &[Message]) -> Response, ProviderError>;
}

class Agent<P: LlmProvider> {
    provider: P,
}

// Bad: 具象型への直接依存
class Agent {
    provider: OllamaClient, // 特定実装に依存
}
```

#### 3-2. KISS原則

- 不必要に複雑な設計になっていないか
- シンプルな実装で十分な箇所はないか
- 過度なジェネリクスやマクロの使用はないか

#### 3-3. YAGNI原則

- 現時点で不要な機能を先行実装していないか
- 将来の拡張のための過度な抽象化はないか

#### 3-4. DRY原則

- コードの重複はないか
- 共通ロジックが `contracts/` や共通モジュールに抽出されているか

### 4. 技術スタックの適合性

| 項目 | 基準 | 評価 |
|------|------|------|
| Python 3.12+ | 言語機能の適切な活用 | - |
| pip | 依存クレートの妥当性 | - |
| Ollama/OpenAI互換API | プロバイダー抽象化の適切さ | - |
| curl subprocess | HTTPトランスポートの安全性 | - |
| rustyline | CLI入力の堅牢性 | - |

### 5. セキュリティレビュー

| チェック項目 | リスク | 評価 |
|-------------|--------|------|
| **eval使用** | メモリ安全性の破壊 | - |
| **コマンドインジェクション** | shell exec経由の任意コマンド実行 | - |
| **サンドボックス脱出** | ツール実行時のファイルアクセス制限 | - |
| **パストラバーサル** | ファイル操作時のディレクトリ脱出 | - |
| **シークレット漏洩** | APIキーのログ出力・永続化 | - |
| **入力バリデーション** | LLM応答の不正入力処理 | - |
| **権限昇格** | ツール実行権限の適切な制限 | - |

```python
// Bad: コマンドインジェクション脆弱性
def execute_command(user_input: &str) -> String, Error> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(user_input)  // 未検証の入力を直接実行
        .output()?;
    // ...
}

// Good: 入力を検証し、許可リストで制限
def execute_command(tool_name: &str, args: &[&str]) -> String, ToolError> {
    if !ALLOWED_TOOLS.contains(&tool_name) {
        return Err(ToolError::NotAllowed(tool_name.to_string()));
    }
    // ...
}
```

### 6. リスク評価

#### リスクマトリクス

| リスク項目 | 影響度 | 発生確率 | 対策 |
|-----------|--------|---------|------|
| ... | 高/中/低 | 高/中/低 | ... |

#### リスクレベル判定

- **高リスク**: 即座に対応が必要（セキュリティ問題、データ損失リスク）
- **中リスク**: 実装前に設計見直し推奨
- **低リスク**: 注意事項として記録

### 7. レビュー総合判定

```markdown
## アーキテクチャレビュー総合判定

### 判定: APPROVED / CONDITIONAL / REJECTED

### サマリー

| 観点 | 評価 | コメント |
|------|------|---------|
| SOLID原則 | A/B/C | ... |
| KISS原則 | A/B/C | ... |
| YAGNI原則 | A/B/C | ... |
| DRY原則 | A/B/C | ... |
| セキュリティ | A/B/C | ... |
| リスク | 高/中/低 | ... |

### 指摘事項

1. [高] ...
2. [中] ...
3. [低] ...

### 推奨アクション

- [ ] ...
- [ ] ...
```

## 出力先

`workspace/issues/{issue_number}/architecture-review.md`

## 完了条件

- 全チェック項目が評価されている
- 指摘事項が優先度付きで一覧化されている
- 総合判定が出されている
- 推奨アクションが明確である

## 関連コマンド

- `/design-policy`: 設計方針書作成
- `/apply-review`: レビュー結果を設計方針書に反映
- `/multi-stage-design-review`: マルチステージ設計レビュー
- `/work-plan`: 作業計画立案
