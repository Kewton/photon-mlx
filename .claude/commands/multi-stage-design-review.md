---
model: sonnet
description: "設計書の4段階レビュー（通常→整合性→影響分析→セキュリティ）と指摘対応を自動実行"
---

# マルチステージ設計レビューコマンド

## 概要
4段階のアーキテクチャレビューとその指摘事項対応を自動で実行するコマンドです。各段階でレビュー→対応のサイクルを回し、**設計方針書の品質**を段階的に向上させます。

> **重要**: このコマンドは**設計方針書のレビューと改善**を目的としています。
> ソースコードの実装は行いません。レビュー結果は設計方針書に反映されます。

## 使用方法
```bash
/multi-stage-design-review [Issue番号]
/multi-stage-design-review [Issue番号] --skip-stage=3,4
```

**例**:
```bash
/multi-stage-design-review 2              # 全4段階を実行
/multi-stage-design-review 2 --skip-stage=4  # セキュリティレビューをスキップ
```

## 実行内容

あなたはマルチステージレビューの統括者です。4段階のレビューサイクルを順次実行し、各段階で指摘事項を対応してから次の段階に進みます。

### パラメータ

- **issue_number**: 対象Issue番号（必須）
- **skip_stage**: スキップするステージ番号（カンマ区切り）

### サブエージェント/レビュアー指定

| ステージ | レビュアー | 理由 |
|---------|----------|------|
| Stage 1-2（通常・整合性） | **Claude opus**（サブエージェント） | 品質判断にOpus必要 |
| Stage 3-4（影響分析・セキュリティ） | **Codex**（commandmatedev `--agent codex` 経由） | 異なるモデルによるクロスレビュー |
| 指摘反映（全ステージ） | sonnet（継承） / Codex内で直接反映 | 設計方針書更新 |

> **⚠ 重要**: Stage 3-4 は必ず `commandmatedev send ... --agent codex` でCodexに委譲すること。
> Claude サブエージェント（Agent tool）で代替実行してはならない。
> commandmatedev が利用不可の場合はユーザーに報告して中断すること。

> **Codex 担当 Stage は必須** (Issue #140 / S7-001 follow-up):
> Stage 3-4 は **必須** ステージ。`--skip-stage=3` `--skip-stage=4` `--skip-stage=3,4` 等で skip した場合は **WARNING** を出し、最終 summary report に skipped 状態と reviewer 検証結果 (`reviewer="codex"` 確認) を **completion report に記録** すること。本フェーズは段階的厳格化の **第 1 段階** であり、現時点で skip 自体は禁止しない (raise / exit 1 への昇格は次回 Issue で扱う)。
>
> skill 完了報告には以下を必ず含める:
> - Stage 別 finding 数 (Must Fix / Should Fix / Nice to Have)
> - reviewer フィールド検証結果 (Stage 3, 4 の各 review-result.json で `reviewer="codex"` であること)

---

## レビューステージ

| Stage | レビュー種別 | フォーカス | 目的 |
|-------|------------|----------|------|
| 1 | 通常レビュー | 設計原則 | SOLID/KISS/YAGNI/DRY準拠確認 |
| 2 | 整合性レビュー | 整合性 | 設計書と実装の整合性確認 |
| 3 | 影響分析レビュー | 影響範囲 | 変更の波及効果分析 |
| 4 | セキュリティレビュー | セキュリティ | eval使用・サンドボックス・コマンドインジェクション確認 |

---

## 実行フェーズ

### Phase 0: 初期設定

#### 0-1. TodoWriteで作業計画作成

```
- [ ] Stage 1: 通常レビュー
- [ ] Stage 1: 指摘事項対応
- [ ] Stage 2: 整合性レビュー
- [ ] Stage 2: 指摘事項対応
- [ ] Stage 3: 影響分析レビュー
- [ ] Stage 3: 指摘事項対応
- [ ] Stage 4: セキュリティレビュー
- [ ] Stage 4: 指摘事項対応
- [ ] 最終確認
```

#### 0-2. ディレクトリ構造作成

```bash
mkdir -p workspace/issues/{issue_number}/multi-stage-design-review
```

---

### Stage 1: 通常レビュー（設計原則）

#### 1-1. レビュー実行

```
Use architecture-review-agent (model: opus) to review Issue #{issue_number} with focus on 設計原則.

Context file: workspace/issues/{issue_number}/multi-stage-design-review/stage1-review-context.json
Output file: workspace/issues/{issue_number}/multi-stage-design-review/stage1-review-result.json
```

**コンテキスト内容**:
```json
{
  "issue_number": "{issue_number}",
  "focus_area": "設計原則",
  "stage": 1,
  "stage_name": "通常レビュー"
}
```

#### 1-2. 指摘事項対応（設計方針のみ）

> **重要**: このステップでは**設計方針書のみ**を更新します。ソースコードは変更しません。

レビュー結果にMust Fix/Should Fix項目がある場合：

```
Use apply-review-agent to update design policy for Issue #{issue_number} Stage 1.
Target: workspace/design/issue-{issue_number}-*-design-policy.md

Context file: workspace/issues/{issue_number}/multi-stage-design-review/stage1-apply-context.json
Output file: workspace/issues/{issue_number}/multi-stage-design-review/stage1-apply-result.json

IMPORTANT: Only update design policy documents. Do NOT modify source code.
```

#### 1-3. Stage 1完了確認

- Must Fix項目すべて対応済み
- 設計方針書が更新されている

---

### Stage 2: 整合性レビュー

#### 2-1. レビュー実行

```
Use architecture-review-agent (model: opus) to review Issue #{issue_number} with focus on 整合性.

Context file: workspace/issues/{issue_number}/multi-stage-design-review/stage2-review-context.json
Output file: workspace/issues/{issue_number}/multi-stage-design-review/stage2-review-result.json
```

**コンテキスト内容**:
```json
{
  "issue_number": "{issue_number}",
  "focus_area": "整合性",
  "stage": 2,
  "stage_name": "整合性レビュー",
  "design_doc_path": "workspace/design/issue-{issue_number}-*-design-policy.md"
}
```

#### 2-2. 指摘事項対応（設計方針のみ）

> **重要**: このステップでは**設計方針書のみ**を更新します。ソースコードは変更しません。

```
Use apply-review-agent to update design policy for Issue #{issue_number} Stage 2.
Target: workspace/design/issue-{issue_number}-*-design-policy.md

Context file: workspace/issues/{issue_number}/multi-stage-design-review/stage2-apply-context.json
Output file: workspace/issues/{issue_number}/multi-stage-design-review/stage2-apply-result.json

IMPORTANT: Only update design policy documents. Do NOT modify source code.
```

#### 2-3. Stage 2完了確認

- 設計書と実装の整合性確認完了

---

### Stage 3-4: Codexによるクロスレビュー（影響分析 + セキュリティ）

Stage 3（影響分析）とStage 4（セキュリティ）は**Codex**（`--agent codex`）に委譲し、異なるモデルによるクロスレビューで品質を向上させます。

> **⚠ 禁止事項（厳守）**:
> 1. このステージをClaudeサブエージェント（Agent tool）で代替実行しないこと。必ず `commandmatedev send --agent codex` でCodexに送信すること。
> 2. Codexが生成した結果ファイルを Claude 側で再実行・上書きしないこと。Codex が findings 0件を返した場合も「問題なし」という正当な結果であり、件数が少ないことを理由に opus で再レビューすることを禁止する。
> 3. 結果ファイルの `reviewer` フィールドが `"codex"` でない場合は、不正な上書きが発生しているため即座にユーザーに報告すること。

#### 3-0. commandmatedev 利用可能性確認

```bash
# commandmatedev が利用可能か確認（利用不可の場合はユーザーに報告して中断）
which commandmatedev || { echo "ERROR: commandmatedev not available. Stage 3-4 requires Codex via commandmatedev."; exit 1; }
```

#### 3-1. Codexへの影響分析レビュー依頼

commandmatedev CLIを使い、同一worktreeの**Codex**セッション（`--agent codex`）にレビューを依頼します。

**worktree IDの取得**:
```bash
WORKTREE_ID=$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]' | sed 's/^photon-mlx-/photon-mlx-/')
```

**Stage 3: 影響分析レビュー + 反映**:
```bash
commandmatedev send "$WORKTREE_ID" \
  "Issue #{issue_number} の設計方針書に対する影響分析レビューを実施してください。

## 対象
- 設計方針書: workspace/design/issue-{issue_number}-*-design-policy.md
- Stage 1レビュー結果: workspace/issues/{issue_number}/multi-stage-design-review/stage1-review-result.json
- Stage 2レビュー結果: workspace/issues/{issue_number}/multi-stage-design-review/stage2-review-result.json

## 指示
1. 設計方針書を読み込む
2. 影響範囲の観点でレビュー（変更の波及効果、既存モジュールへの影響、テスト範囲）
3. レビュー結果をJSON形式で workspace/issues/{issue_number}/multi-stage-design-review/stage3-review-result.json に出力
4. Must Fix/Should Fix指摘があれば設計方針書を直接更新
5. 反映結果を workspace/issues/{issue_number}/multi-stage-design-review/stage3-apply-result.json に出力

## 重要
- 設計方針書のみを更新すること。ソースコードは変更しない。

## 出力フォーマット
stage3-review-result.json:
{
  \"stage\": 3,
  \"focus\": \"影響範囲\",
  \"reviewer\": \"codex\",
  \"findings\": [{\"id\": \"DR3-NNN\", \"severity\": \"Must Fix|Should Fix|Nice to Have\", \"title\": \"...\", \"description\": \"...\"}],
  \"must_fix_count\": N,
  \"should_fix_count\": N,
  \"nice_to_have_count\": N
}" \
  --agent codex --auto-yes --duration 1h
```

Codexの完了を待機:
```bash
commandmatedev wait "$WORKTREE_ID" --timeout 3600 --on-prompt agent
```

#### 3-2. Stage 3完了確認

- `stage3-review-result.json` が生成されている
- `reviewer` フィールドが `"codex"` であることを確認する

> **⚠ 上書き禁止**: Codexの結果ファイルが既に存在する場合、Claude側で再実行・上書きしてはならない。
> Codexが findings 0件を返した場合も、それは「問題なし」という正当なレビュー結果である。
> findings の件数が少ないことを理由に Agent tool (opus) で再実行することを禁止する。

---

**Stage 4: セキュリティレビュー + 反映**:
```bash
commandmatedev send "$WORKTREE_ID" \
  "Issue #{issue_number} の設計方針書に対するセキュリティレビューを実施してください。

## 対象
- 設計方針書: workspace/design/issue-{issue_number}-*-design-policy.md
- Stage 1-3レビュー結果: workspace/issues/{issue_number}/multi-stage-design-review/stage*-review-result.json

## 指示
1. 設計方針書を読み込む
2. セキュリティの観点でレビュー（eval使用、サンドボックス、コマンドインジェクション、入力検証、権限管理）
3. レビュー結果をJSON形式で workspace/issues/{issue_number}/multi-stage-design-review/stage4-review-result.json に出力
4. Must Fix/Should Fix指摘があれば設計方針書を直接更新
5. 反映結果を workspace/issues/{issue_number}/multi-stage-design-review/stage4-apply-result.json に出力

## 重要
- 設計方針書のみを更新すること。ソースコードは変更しない。

## 出力フォーマット
stage3と同じJSON形式（stageを4に変更、focusを「セキュリティ」に変更）" \
  --agent codex --auto-yes --duration 1h
```

Codexの完了を待機:
```bash
commandmatedev wait "$WORKTREE_ID" --timeout 3600 --on-prompt agent
```

#### 4-1. Stage 4完了確認

- `stage4-review-result.json` が生成されている
- `reviewer` フィールドが `"codex"` であることを確認する
- セキュリティ上の問題がすべて解消
- eval使用の正当性確認
- サンドボックス・コマンドインジェクション対策確認

> **⚠ 上書き禁止**: Stage 3-2 と同様、Codexの結果を Claude 側で再実行・上書きしてはならない。

---

### Phase Final: 最終確認と報告

#### 最終検証

```bash
python -m pytest
ruff check .
python -m pytest
ruff format --check .
```

#### サマリーレポート作成

**ファイルパス**: `workspace/issues/{issue_number}/multi-stage-design-review/summary-report.md`

```markdown
# マルチステージレビュー完了報告

## Issue #{issue_number}

### ステージ別結果

| Stage | レビュー種別 | 指摘数 | 対応数 | ステータス |
|-------|------------|-------|-------|----------|
| 1 | 通常レビュー | X | X | 完了 |
| 2 | 整合性レビュー | X | X | 完了 |
| 3 | 影響分析レビュー | X | X | 完了 |
| 4 | セキュリティレビュー | X | X | 完了 |

### 最終検証結果

| チェック項目 | コマンド | 結果 |
|-------------|----------|------|
| ビルド | python -m pytest | Pass |
| Clippy | ruff check . | Pass |
| テスト | python -m pytest | Pass |
| フォーマット | ruff format --check . | Pass |

### 次のアクション

- [ ] 設計方針書の最終確認
- [ ] /tdd-class または /pm-auto-dev で実装を開始
```

---

## ファイル構造

```
workspace/issues/{issue_number}/
└── multi-stage-design-review/
    ├── stage1-review-context.json
    ├── stage1-review-result.json
    ├── stage1-apply-context.json
    ├── stage1-apply-result.json
    ├── stage2-review-context.json
    ├── stage2-review-result.json
    ├── stage2-apply-context.json
    ├── stage2-apply-result.json
    ├── stage3-review-context.json
    ├── stage3-review-result.json
    ├── stage3-apply-context.json
    ├── stage3-apply-result.json
    ├── stage4-review-context.json
    ├── stage4-review-result.json
    ├── stage4-apply-context.json
    ├── stage4-apply-result.json
    └── summary-report.md
```

---

## 完了条件

以下をすべて満たすこと：

- 全4ステージのレビュー完了
- 各ステージの指摘事項が**設計方針書に反映**完了
- **設計方針書が最新の状態に更新**されている
- サマリーレポート作成完了

> **Note**: このコマンドではソースコードの変更・テスト実行は行いません。
> 設計方針書のレビューと改善のみを実施します。

---

## 関連コマンド

- `/architecture-review`: 単体アーキテクチャレビュー
- `/apply-review`: レビュー指摘事項の反映
- `/pm-auto-dev`: 自動開発フロー
- `/create-pr`: PR作成
