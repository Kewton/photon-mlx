---
model: opus
description: "Issue単位の設計方針書を作成"
---

# 設計方針書作成スキル

## 概要
Issue単位での設計方針書を作成するスキルです。PHOTON-RepoRAGプロジェクトのアーキテクチャに沿った設計判断を文書化し、実装前の合意形成を支援します。

## 使用方法
- `/design-policy [Issue番号]`
- 「Issue #XXXの設計方針書を作成してください」

## 前提条件
- 対象Issueの内容が明確であること
- GitHubリポジトリにアクセス可能

## 実行内容

あなたはソフトウェアアーキテクトとして、以下の設計方針書を作成します。

### 1. Issue情報の取得

```bash
gh issue view {issue_number} --json number,title,body,labels,assignees
```

### 2. システムアーキテクチャ概要

PHOTON-RepoRAGの全体アーキテクチャを踏まえた設計を行います：

```
┌─────────────────────────────────────────────────┐
│                    CLI (rustyline)                │
│                   baseline_reporag/app/cli.py                  │
├─────────────────────────────────────────────────┤
│              Application Orchestrator             │
│                  baseline_reporag/app/__init__.py                    │
│         ┌──────────┬──────────┬────────┐         │
│         │ agentic  │  plan    │ render │         │
│         └──────────┴──────────┴────────┘         │
├─────────────────────────────────────────────────┤
│               Agent Loop / Protocol               │
│                baseline_reporag/agent/__init__.py                    │
├──────────────┬──────────────┬───────────────────┤
│   Provider   │   Tooling    │    Extensions      │
│ baseline_reporag/provider/│ baseline_reporag/tooling/ │  baseline_reporag/extensions/   │
│  ┌────────┐  │              │                    │
│  │ ollama │  │              │                    │
│  │ openai │  │              │                    │
│  │transport│ │              │                    │
│  └────────┘  │              │                    │
├──────────────┴──────────────┴───────────────────┤
│  State Machine  │  Session   │  Config  │  TUI   │
│  baseline_reporag/state/     │ baseline_reporag/session│baseline_reporag/config│baseline_reporag/tui │
├─────────────────┴────────────┴─────────┴────────┤
│              Contracts (共通型定義)                 │
│               baseline_reporag/contracts/__init__.py                │
└─────────────────────────────────────────────────┘
```

### 3. レイヤー構成と責務

| レイヤー | モジュール | 責務 |
|---------|-----------|------|
| **CLI** | `baseline_reporag/app/cli.py` | ユーザー入力の受付、rustylineによる対話 |
| **App** | `baseline_reporag/app/` | アプリケーションロジックの統合、ツール実行ループ |
| **Agent** | `baseline_reporag/agent/` | LLMとの対話プロトコル、エージェントループ |
| **Provider** | `baseline_reporag/provider/` | LLMバックエンド抽象化（Ollama/OpenAI互換） |
| **Tooling** | `baseline_reporag/tooling/` | ツール定義・実行・結果検証 |
| **Extensions** | `baseline_reporag/extensions/` | スラッシュコマンド、拡張機能 |
| **State** | `baseline_reporag/state/` | 状態マシン、状態遷移管理 |
| **Session** | `baseline_reporag/session/` | セッション永続化（JSONファイル） |
| **Config** | `baseline_reporag/config/` | 設定管理（TOML/環境変数） |
| **TUI** | `baseline_reporag/tui/` | ターミナルUI描画 |
| **Contracts** | `baseline_reporag/contracts/` | 共通型定義（Message, Tool, Response等） |

### 4. 技術選定

| カテゴリ | 選定技術 | 選定理由 |
|---------|---------|---------|
| 言語 | Python 3.12+ | メモリ安全性、パフォーマンス |
| ビルド | pip | Rust標準ビルドシステム |
| LLMバックエンド | Ollama, OpenAI互換API | ローカル/クラウド両対応 |
| HTTP | curl subprocess | 外部依存最小化 |
| CLI入力 | rustyline | 行編集・履歴サポート |
| データ永続化 | JSONファイル | シンプル、外部DB不要 |
| テスト | python -m pytest | 統合テスト中心 |

### 5. 設計パターン

#### 5-1. Provider抽象化（protocol + enum dispatch）

```python
/// LLMプロバイダーの共通インターフェース
protocol LlmProvider {
    def chat(self, messages: &[Message]) -> Response, ProviderError>;
    def list_models(self) -> Vec<String>, ProviderError>;
}

/// Ollamaプロバイダー実装
class OllamaProvider {
    endpoint: String,
    model: String,
}

class LlmProvider for OllamaProvider {
    def chat(self, messages: &[Message]) -> Response, ProviderError> {
        // curl subprocess でOllama APIを呼び出し
    }
    // ...
}
```

#### 5-2. 状態マシンパターン

```python
/// エージェントの状態
enum AgentState {
    Idle,
    WaitingForInput,
    Processing,
    ToolExecution(ToolCall),
    Error(AgentError),
}

/// 状態遷移
class AgentState {
    def transition(self, event: AgentEvent) -> Self {
        match (self, event) {
            (AgentState::Idle, AgentEvent::UserInput(msg)) => AgentState::Processing,
            (AgentState::Processing, AgentEvent::ToolCall(call)) => AgentState::ToolExecution(call),
            // ...
        }
    }
}
```

#### 5-3. エラー型（構造化enum）

```python
/// プロバイダーエラー
#[derive(Debug, thiserror::Error)]
enum ProviderError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}
```

### 6. データモデル

PHOTON-RepoRAGはセッションデータをJSON形式でファイルに永続化します：

```
~/.photon-mlx/
├── config.toml          # ユーザー設定
└── sessions/
    └── {session_id}.json # セッションデータ
```

#### セッションデータ構造

```python
class Session {
    id: String,
    created_at: DateTime,
    messages: Vec<Message>,
}

class Message {
    role: Role,       // User, Assistant, System, Tool
    content: String,
    tool_calls: Option<Vec<ToolCall>>,
    tool_result: Option<ToolResult>,
}
```

### 7. セキュリティ設計

| 脅威 | 対策 | 優先度 |
|------|------|--------|
| **サンドボックス脱出** | ツール実行時のパス検証、許可ディレクトリ制限 | 高 |
| **コマンドインジェクション** | shell exec時の入力サニタイズ、許可コマンドリスト | 高 |
| **パストラバーサル** | ファイル操作時の正規化とベースディレクトリチェック | 高 |
| **APIキー漏洩** | ログ出力からのマスキング、環境変数での管理 | 高 |
| **eval使用** | 原則禁止、必要時はレビュー必須 | 中 |
| **大量リソース消費** | ツール実行のタイムアウト設定 | 中 |

### 8. 設計判断とトレードオフ

Issue #{issue_number} に関する設計判断を記録します：

```markdown
### 設計判断 #1: [判断タイトル]

**選択肢**:
- A: [選択肢Aの説明]
- B: [選択肢Bの説明]

**決定**: 選択肢 A

**理由**:
- [理由1]
- [理由2]

**トレードオフ**:
- メリット: [メリット]
- デメリット: [デメリット]
- リスク: [リスク]
```

### 9. 影響範囲

変更対象のモジュールと影響範囲を明記します：

| モジュール | 変更種別 | 影響度 |
|-----------|---------|--------|
| `baseline_reporag/xxx/` | 新規追加/変更 | 高/中/低 |
| `photon_mlx/tests/xxx.py` | テスト追加 | - |

### 10. 品質基準

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| ビルド | `python -m pytest` | エラー0件 |
| Clippy | `ruff check .` | 警告0件 |
| テスト | `python -m pytest` | 全テストパス |
| フォーマット | `ruff format --check .` | 差分なし |

## 出力先

`workspace/issues/{issue_number}/design-policy.md`

## 完了条件

- アーキテクチャ図が作成されている
- レイヤー構成と責務が明確である
- 設計パターンが具体的なPythonコードで示されている
- セキュリティ要件が記載されている
- 設計判断とトレードオフが記録されている
- 影響範囲が明確である

## 関連コマンド

- `/architecture-review`: アーキテクチャレビュー実行
- `/apply-review`: レビュー結果を設計方針書に反映
- `/multi-stage-design-review`: マルチステージ設計レビュー
- `/work-plan`: 作業計画立案
- `/pm-auto-design2dev`: 設計から開発まで一括実行
