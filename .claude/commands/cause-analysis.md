---
model: sonnet
description: "Issue記載の事象から根本原因を分析し原因と対策案をIssueに追記"
---

# 根本原因分析（Cause Analysis）

## 概要
Issueに記載されている不具合事象を元に、Opus 4.6（commandmatedev経由）を活用して根本原因を深く分析し、原因と対策案をIssue本文に追記します。

## 使用方法
```
/cause-analysis [Issue番号]
```

例：
```
/cause-analysis 184
/cause-analysis 185
```

## 前提条件
- developブランチ上で実行すること
- CommandMateサーバーが稼働していること
- 対象Issueが存在し、事象が記載されていること
- GitHubリポジトリ（https://github.com/Kewton/photon-mlx）にアクセス可能

## 実行内容

あなたはシニアデバッグエンジニアとして、不具合の根本原因を分析します。

### Step 1: Issue情報の取得

```bash
ISSUE_NUM="$ARGUMENTS"
gh issue view "$ISSUE_NUM" --repo Kewton/PHOTON-RepoRAG --json number,title,body,labels,comments
```

Issue本文から以下を抽出：
- 事象の要約
- 再現手順
- 影響範囲
- 関連するソースコード

### Step 2: 事前調査（オーケストレーター側）

Issue本文に記載された関連ソースコードを確認し、分析の初期仮説を立てます。

```bash
# 関連ファイルの確認
# Issue本文から抽出したファイルパスを読み取る
```

以下の観点で初期仮説を整理：

| 観点 | 内容 |
|------|------|
| **直接原因** | コードのどの部分が事象を引き起こしているか |
| **根本原因の候補** | なぜその状態になるのか（設計問題、仕様漏れ、エッジケース等） |
| **影響パス** | データフローのどの経路で問題が発生するか |

### Step 3: Opus 4.6 による深層分析

commandmatedevを使用して、develop worktreeのOpus 4.6エージェントに根本原因分析を依頼します。

```bash
WORKTREE_ID="photon-mlx-develop"

ANALYSIS_PROMPT="Issue #${ISSUE_NUM} の根本原因分析を実施してください。

## Issue内容
$(gh issue view ${ISSUE_NUM} --repo Kewton/PHOTON-RepoRAG --json body -q '.body')

## 分析要求

### Phase 1: 事象の再現パス特定
1. Issue記載の再現手順に基づき、コード上の実行パスを特定してください
2. 関連するソースファイルを読み取り、処理フローを追跡してください
3. 事象が発生する条件（トリガー条件）を具体的に特定してください

### Phase 2: 根本原因の特定
以下の観点で根本原因を分析してください：
- **直接原因**: どのコードが事象を直接引き起こしているか（ファイル名:行番号）
- **根本原因**: なぜそのコードがその状態になるのか
- **設計上の問題**: アーキテクチャや設計レベルで改善すべき点があるか
- **類似リスク**: 同じパターンが他の箇所にも存在するか

### Phase 3: 対策案の策定
以下の粒度で対策案を提示してください：
- **即座対策（Quick Fix）**: 最小限の変更で事象を解消する方法
- **恒久対策（Permanent Fix）**: 根本原因を解消する方法
- **予防策（Prevention）**: 再発を防ぐための設計改善・テスト追加

### 出力形式
以下のJSON形式で結果を出力してください：
\`\`\`json
{
  \"issue_number\": ${ISSUE_NUM},
  \"reproduction_path\": {
    \"trigger_condition\": \"トリガー条件の説明\",
    \"execution_flow\": [\"ステップ1\", \"ステップ2\", ...],
    \"key_files\": [\"ファイル:行番号\", ...]
  },
  \"root_cause\": {
    \"direct_cause\": \"直接原因の説明\",
    \"root_cause\": \"根本原因の説明\",
    \"design_issue\": \"設計上の問題（ある場合）\",
    \"similar_risks\": [\"類似リスク1\", ...]
  },
  \"countermeasures\": {
    \"quick_fix\": {
      \"description\": \"即座対策の説明\",
      \"files_to_change\": [\"対象ファイル\"],
      \"estimated_effort\": \"小/中/大\"
    },
    \"permanent_fix\": {
      \"description\": \"恒久対策の説明\",
      \"files_to_change\": [\"対象ファイル\"],
      \"estimated_effort\": \"小/中/大\"
    },
    \"prevention\": {
      \"description\": \"予防策の説明\",
      \"tests_to_add\": [\"追加すべきテスト\"]
    }
  },
  \"recommendation\": \"推奨する対策とその理由\"
}
\`\`\`"

commandmatedev send "$WORKTREE_ID" "$ANALYSIS_PROMPT" --agent copilot --model claude-opus-4.6 --auto-yes --duration 1h
```

### Step 4: 分析結果の待機と取得

```bash
commandmatedev wait "$WORKTREE_ID" --timeout 3600 --on-prompt agent
commandmatedev capture "$WORKTREE_ID"
```

### Step 5: 結果の検証

オーケストレーター自身で分析結果を検証します：

- [ ] 特定された実行パスがコード上で確認できるか
- [ ] 根本原因の説明が論理的に整合しているか
- [ ] 対策案が実現可能で副作用のリスクが考慮されているか
- [ ] 類似リスクの指摘が妥当か

### Step 6: Issue本文の更新

分析結果をIssue本文に追記します。

```bash
# 現在のIssue本文を取得
CURRENT_BODY=$(gh issue view "$ISSUE_NUM" --repo Kewton/PHOTON-RepoRAG --json body -q '.body')

# 根本原因分析セクションを追記
UPDATED_BODY="${CURRENT_BODY}

---

## 根本原因分析

### 再現パス
${再現パスの説明}

**トリガー条件**: ${トリガー条件}

**実行フロー**:
1. ${ステップ1}
2. ${ステップ2}
...

**関連ファイル**:
${ファイル:行番号のリスト}

### 直接原因
${直接原因の説明}

### 根本原因
${根本原因の説明}

### 設計上の問題
${設計上の問題があれば記載}

### 類似リスク
${同じパターンが他にも存在する箇所}

### 対策案

#### 即座対策（Quick Fix）
${即座対策の説明}
- 対象ファイル: ${ファイルリスト}
- 工数見積: ${小/中/大}

#### 恒久対策（Permanent Fix）
${恒久対策の説明}
- 対象ファイル: ${ファイルリスト}
- 工数見積: ${小/中/大}

#### 予防策（Prevention）
${予防策の説明}
- 追加テスト: ${テストリスト}

### 推奨対策
${推奨する対策とその理由}

---
🔍 分析実施: Claude Code + Opus 4.6 (commandmatedev)
📅 分析日時: $(date +%Y-%m-%d)
"

gh issue edit "$ISSUE_NUM" --repo Kewton/PHOTON-RepoRAG --body "$UPDATED_BODY"
```

### Step 7: 結果報告

```
Issue #${ISSUE_NUM} の根本原因分析が完了しました。

## 分析結果サマリー

| 項目 | 内容 |
|------|------|
| 直接原因 | ${直接原因の要約} |
| 根本原因 | ${根本原因の要約} |
| 推奨対策 | ${推奨対策の要約} |
| 工数見積 | ${見積} |

Issue本文を更新しました: https://github.com/Kewton/PHOTON-RepoRAG/issues/${ISSUE_NUM}

次のアクション:
- /bug-fix ${ISSUE_NUM} でバグ修正を開始
- /tdd-class で対策を実装
```

## Opus 4.6 エージェント連携の注意事項

- commandmatedevの `--agent copilot --model claude-opus-4.6` でOpus 4.6を指定
- `--agent copilot` のみではSonnetが使用されるため、必ず `--model claude-opus-4.6` を併記する
- develop worktree（photon-mlx-develop）上で分析を実行
- 分析結果は `commandmatedev capture --agent copilot --json` で取得し、オーケストレーターが検証・整形する
- Opusがコードを変更しないよう、分析のみを依頼する（read-only分析）

## エラーハンドリング

| エラー | 対応 |
|--------|------|
| Issue番号が無効 | エラー表示し中断 |
| Issueに事象記載なし | ユーザーに /current-situation の実行を案内 |
| CommandMateサーバー未起動 | 起動手順を案内 |
| Opus分析タイムアウト | captureで途中結果を取得し、不足分をオーケストレーターが補完 |
| 分析結果の整合性不足 | オーケストレーターが追加調査し補完 |

## 完了条件

- [ ] 再現パスがコード上で特定されている
- [ ] 根本原因が論理的に説明されている
- [ ] 対策案（即座・恒久・予防）が具体的に記載されている
- [ ] Issue本文が更新されている
- [ ] 結果サマリーがユーザーに報告されている
