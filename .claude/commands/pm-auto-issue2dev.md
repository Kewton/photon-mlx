---
model: sonnet
description: "Issueレビューから実装完了まで完全自動化（Issueレビュー→設計→設計レビュー→作業計画→TDD実装）"
---

# PM自動 Issue→開発スキル

## 概要
Issueレビューから実装完了までの全工程（Issueレビュー → 設計方針策定 → 設計レビュー → 作業計画立案 → TDD実装）を**完全自動化**するプロジェクトマネージャースキルです。ユーザーはIssue番号を指定するだけで、Issueの品質向上から開発完了まで自律的に実行します。

**アーキテクチャ**: 5つの既存コマンドを順次実行し、各フェーズの成果物を次フェーズに引き継ぎます。

## 使用方法
- `/pm-auto-issue2dev [Issue番号]`
- 「Issue #XXXをIssueレビューから開発まで自動実行してください」

## 実行内容

あなたはプロジェクトマネージャーとして、Issueレビューから開発までの全工程を統括します。以下のフェーズを順次実行し、各フェーズの完了を確認しながら進めてください。

### パラメータ

- **issue_number**: 開発対象のIssue番号（必須）

### サブエージェントモデル指定

各サブコマンド内で個別にモデル指定されています（レビュー・TDD系=opus、反映・報告系=sonnet継承）。

---

## 実行フェーズ

### Phase 0: 初期設定とTodoリスト作成

まず、TodoWriteツールで作業計画を作成してください：

```
- [ ] Phase 1: マルチステージIssueレビュー
- [ ] Phase 2: 設計方針書確認・作成
- [ ] Phase 3: マルチステージ設計レビュー
- [ ] Phase 4: 作業計画立案
- [ ] Phase 5: TDD自動開発
- [ ] Phase 6: 完了報告
```

---

### Phase 1: マルチステージIssueレビュー

#### 1-1. Issueレビュー実行

`/multi-stage-issue-review` コマンドを実行：

```
/multi-stage-issue-review {issue_number}
```

**このフェーズで行われること**:
- 仮説検証（コードベース照合）
- 1st Iteration: 通常レビュー → 指摘反映 → 影響範囲レビュー → 指摘反映
- 2nd Iteration: 通常レビュー → 指摘反映 → 影響範囲レビュー → 指摘反映
- GitHubのIssue本文が更新される

#### 1-2. 完了確認

- サマリーレポートが生成されていること
- GitHubのIssueが更新されていること

**出力ファイル**: `workspace/issues/{issue_number}/issue-review/summary-report.md`

#### 1-3. Codex stage 結果ファイル + `reviewer="codex"` 検証 (Issue #140 / S7-001 follow-up)

Codex 担当 Stage 5/7 の結果ファイルが存在し `reviewer="codex"` が記録されていることを確認する。Claude による不正な上書きを検出する WARNING 機構。

```bash
# REVIEWER_VERIFICATION_SNIPPET_BEGIN (issue-review)
ISSUE="${ISSUE:-{issue_number}}"
case "$ISSUE" in
  ''|*[!0-9]*)
    safe_issue=$(printf '%s' "$ISSUE" | LC_ALL=C tr -c '[:alnum:]_.@:-' '?')
    printf 'WARNING: ISSUE=%s is not a numeric issue number; skipping reviewer verification\n' "$safe_issue"
    ISSUE=""
    ;;
esac
if [ -n "$ISSUE" ]; then
  for stage in 5 7; do
    f="workspace/issues/$ISSUE/issue-review/stage${stage}-review-result.json"
    if [ ! -f "$f" ]; then
      printf 'WARNING: %s missing (Codex stage skipped or not yet run)\n' "$f"
      continue
    fi
    reviewer=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reviewer",""))' "$f" 2>/dev/null || echo "")
    safe_reviewer=$(printf '%s' "$reviewer" | tr -c 'A-Za-z0-9_-' '?')
    if [ "$reviewer" != "codex" ]; then
      printf 'WARNING: %s reviewer=%s (expected codex)\n' "$f" "$safe_reviewer"
    fi
  done
fi
# REVIEWER_VERIFICATION_SNIPPET_END
```

判定: 結果ファイルが存在し `reviewer="codex"` であれば WARNING なし。それ以外 (欠落 / claude / 文字化け) は WARNING を出して **completion report に記録** する (raise / exit 1 はしない — 段階的厳格化の第 1 段階)。

---

### Phase 2: 設計方針書の確認・作成

#### 2-1. 設計方針書の存在確認

```bash
ls workspace/design/issue-{issue_number}-*-design-policy.md 2>/dev/null
```

#### 2-2. 設計方針書がない場合

設計方針書が存在しない場合は、`/design-policy` コマンドを実行して作成：

```
/design-policy {issue_number}
```

---

### Phase 3: マルチステージ設計レビュー

#### 3-1. 設計レビュー実行

`/multi-stage-design-review` コマンドを実行：

```
/multi-stage-design-review {issue_number}
```

**このフェーズで行われること**:
- Stage 1: 通常レビュー（設計原則）
- Stage 2: 整合性レビュー
- Stage 3: 影響分析レビュー
- Stage 4: セキュリティレビュー
- 各ステージの指摘事項を設計方針書に反映

#### 3-2. 完了確認

- サマリーレポートが生成されていること
- 設計方針書が更新されていること

**出力ファイル**: `workspace/issues/{issue_number}/multi-stage-design-review/summary-report.md`

#### 3-3. Codex stage 結果ファイル + `reviewer="codex"` 検証 (Issue #140 / S7-001 follow-up)

Codex 担当 Stage 3/4 の結果ファイルが存在し `reviewer="codex"` が記録されていることを確認する。

```bash
# REVIEWER_VERIFICATION_SNIPPET_BEGIN (design-review)
ISSUE="${ISSUE:-{issue_number}}"
case "$ISSUE" in
  ''|*[!0-9]*)
    safe_issue=$(printf '%s' "$ISSUE" | LC_ALL=C tr -c '[:alnum:]_.@:-' '?')
    printf 'WARNING: ISSUE=%s is not a numeric issue number; skipping reviewer verification\n' "$safe_issue"
    ISSUE=""
    ;;
esac
if [ -n "$ISSUE" ]; then
  for stage in 3 4; do
    f="workspace/issues/$ISSUE/multi-stage-design-review/stage${stage}-review-result.json"
    if [ ! -f "$f" ]; then
      printf 'WARNING: %s missing (Codex stage skipped or not yet run)\n' "$f"
      continue
    fi
    reviewer=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reviewer",""))' "$f" 2>/dev/null || echo "")
    safe_reviewer=$(printf '%s' "$reviewer" | tr -c 'A-Za-z0-9_-' '?')
    if [ "$reviewer" != "codex" ]; then
      printf 'WARNING: %s reviewer=%s (expected codex)\n' "$f" "$safe_reviewer"
    fi
  done
fi
# REVIEWER_VERIFICATION_SNIPPET_END
```

判定: Phase 1-3 と同方針 — 結果ファイルが存在し `reviewer="codex"` であれば WARNING なし。それ以外は WARNING を出して **completion report に記録** する。

---

### Phase 4: 作業計画立案

#### 4-1. 作業計画作成

`/work-plan` コマンドを実行：

```
/work-plan {issue_number}
```

**このフェーズで行われること**:
- 設計方針書に基づいたタスク分解
- 依存関係の整理
- 実装順序の決定

#### 4-2. 完了確認

- 作業計画書が生成されていること

**出力ファイル**: `workspace/issues/{issue_number}/work-plan.md`

---

### Phase 5: TDD自動開発

#### 5-1. TDD実装実行

`/pm-auto-dev` コマンドを実行：

```
/pm-auto-dev {issue_number}
```

**このフェーズで行われること**:
- TDD実装（Red-Green-Refactor）
- 受入テスト
- リファクタリング
- ドキュメント更新
- 進捗報告

#### 5-2. 完了確認

- `python -m pytest` エラー0件
- `ruff check .` 警告0件
- `python -m pytest` 全テストパス
- 進捗レポートが生成されていること

**出力ファイル**: `workspace/issues/{issue_number}/pm-auto-dev/iteration-1/progress-report.md`

---

### Phase 6: 完了報告

#### 6-1. 最終検証

```bash
python -m pytest
ruff check .
python -m pytest
ruff format --check .
```

#### 6-2. 成果物サマリー

完了時に以下を報告：

```markdown
## PM Auto Issue2Dev 完了報告

### Issue #{issue_number}

#### 実行フェーズ結果

| Phase | 内容 | ステータス |
|-------|------|-----------|
| 1 | マルチステージIssueレビュー | 完了 |
| 2 | 設計方針書確認・作成 | 完了 |
| 3 | マルチステージ設計レビュー | 完了 |
| 4 | 作業計画立案 | 完了 |
| 5 | TDD自動開発 | 完了 |

#### 品質チェック

| チェック項目 | コマンド | 結果 |
|-------------|----------|------|
| ビルド | python -m pytest | Pass |
| Clippy | ruff check . | Pass |
| テスト | python -m pytest | Pass |
| フォーマット | ruff format --check . | Pass |

#### 生成ファイル

- Issueレビュー: `workspace/issues/{issue_number}/issue-review/summary-report.md`
- 設計方針書: `workspace/design/issue-{issue_number}-*-design-policy.md`
- 設計レビュー: `workspace/issues/{issue_number}/multi-stage-design-review/summary-report.md`
- 作業計画: `workspace/issues/{issue_number}/work-plan.md`
- 進捗報告: `workspace/issues/{issue_number}/pm-auto-dev/iteration-1/progress-report.md`

#### 次のアクション

- [ ] コミット確認
- [ ] PR作成（`/create-pr`）
```

---

## ファイル構造

```
workspace/
├── design/
│   └── issue-{issue_number}-*-design-policy.md
└── issue/{issue_number}/
    ├── issue-review/
    │   ├── original-issue.json
    │   ├── hypothesis-verification.md
    │   ├── stage1-*.json ~ stage8-*.json
    │   └── summary-report.md
    ├── multi-stage-design-review/
    │   ├── stage1-*.json ~ stage4-*.json
    │   └── summary-report.md
    ├── work-plan.md
    └── pm-auto-dev/
        └── iteration-1/
            ├── tdd-*.json
            ├── acceptance-*.json
            ├── refactor-*.json
            └── progress-report.md
```

---

## 完了条件

以下をすべて満たすこと：

- Phase 1: マルチステージIssueレビュー完了（Issue本文が更新されている）
- Phase 2: 設計方針書が存在する
- Phase 3: マルチステージ設計レビュー完了（4ステージすべて）
- Phase 4: 作業計画書が作成されている
- Phase 5: TDD自動開発完了（テスト全パス、ruff警告0件）
- Phase 6: 完了報告

---

## 関連コマンド

- `/multi-stage-issue-review`: マルチステージIssueレビュー
- `/design-policy`: 設計方針書作成
- `/multi-stage-design-review`: マルチステージ設計レビュー
- `/work-plan`: 作業計画立案
- `/pm-auto-dev`: TDD自動開発
- `/create-pr`: PR作成
- `/pm-auto-design2dev`: 設計レビューから実装完了まで（Issueレビューなし版）
