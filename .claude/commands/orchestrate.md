---
model: sonnet
description: "複数Issueを並列オーケストレーション（準備→開発→PR→マージ→UAT→修正ループ→完了）"
---

# 並列Issueオーケストレーション

## 概要
developブランチをオーケストレーターとして、複数Issueの並列開発からUAT合格までの全ライフサイクルを統括します。各Issueはfeatureブランチのworktreeで並列に開発され、commandmatedev CLIで制御します。

**原則**: オーケストレーターはコードに触れない。制御と判断のみ。

## 使用方法
- `/orchestrate [Issue番号1] [Issue番号2] ...`
- `/orchestrate [Issue番号1] [Issue番号2] --phase design` （設計フェーズまで）
- `/orchestrate [Issue番号1] [Issue番号2] --phase impl` （実装まで）
- `/orchestrate [Issue番号1] [Issue番号2] --full` （UAT合格まで全自動）

## 前提条件
- developブランチ上で実行すること
- CommandMateサーバーが稼働していること（`commandmatedev ls` で確認）
- GitHubリポジトリ（https://github.com/Kewton/photon-mlx）にアクセス可能

## 実行内容

あなたはプロジェクトマネージャーとして、複数Issueの並列開発を統括します。

### パラメータ
- **issue_numbers**: 開発対象のIssue番号（スペース区切り、2つ以上）
- **--phase**: 実行範囲の制限（design, impl, pr, uat）。省略時はPRマージまで
- **--full**: UAT合格まで全自動で実行

---

## Phase 0: 初期設定

TodoWriteツールで作業計画を作成：

```
- [ ] Phase 1: 依存関係分析・実行計画（ラベル分類含む）
- [ ] Phase 2: Worktree準備
- [ ] Phase 2.5: 根本原因分析（バグIssueのみ、Opus 4.6）
- [ ] Phase 3: 並列開発（バグ: /bug-fix、機能: /pm-auto-issue2dev）
- [ ] Phase 4: 設計突合（バリア）
- [ ] Phase 5: 品質確認
- [ ] Phase 6: PR作成・マージ（/pr-merge-pipeline）
- [ ] Phase 7: UAT（--full時のみ）
- [ ] Phase 8: 完了報告
```

---

## Phase 1: 依存関係分析・実行計画

### 1-1. Issue情報の取得

各Issueの詳細を取得：

```bash
for issue_num in {issue_numbers}; do
  gh issue view "$issue_num" --repo Kewton/PHOTON-RepoRAG --json number,title,body,labels
done
```

### 1-2. Issue種別の分類

各Issueのラベルを確認し、バグと機能追加を分類する：

```bash
for issue_num in {issue_numbers}; do
  labels=$(gh issue view "$issue_num" --repo Kewton/PHOTON-RepoRAG --json labels -q '[.labels[].name] | join(",")')
  if echo "$labels" | grep -q "bug"; then
    echo "BUG: #${issue_num}"
  else
    echo "FEATURE: #${issue_num}"
  fi
done
```

分類結果を記録し、Phase 2.5 と Phase 3 で使用する：
- **BUG_ISSUES**: `bug` ラベルを持つIssue → Phase 2.5（根本原因分析）+ Phase 3（/bug-fix）
- **FEATURE_ISSUES**: それ以外 → Phase 3（/pm-auto-issue2dev）

### 1-3. 依存関係の分析

各Issueについて以下を分析：
- **影響ファイル**: Issue本文の「影響ファイル」セクションから抽出
- **共通ファイル**: 複数Issueが同じファイルを変更する場合のコンフリクトリスク
- **依存関係**: Issue間の前後関係（A の成果物が B の入力になるか）

### 1-4. 並列実行可否の判定

```
独立:     共通ファイルなし → 完全並列
弱依存:   共通ファイルあるが変更箇所が異なる → 並列可（設計突合で確認）
強依存:   A の出力が B の入力 → 直列実行（A完了後にB開始）
```

### 1-5. 実行計画の記録

```bash
DATE=$(date +%Y-%m-%d)
mkdir -p workspace/orchestration/runs/$DATE
```

実行計画を `workspace/orchestration/runs/$DATE/plan.md` に出力：
- 対象Issue一覧
- 依存関係グラフ
- 並列実行グループ
- マージ推奨順序

---

## Phase 2: Worktree準備

### 2-1. 既存worktreeの確認

```bash
commandmatedev ls --branch feature/issue-
```

### 2-2. 不足worktreeの作成

各Issueについて、対応するworktreeが存在しない場合は作成：

```bash
# Issue情報からブランチ名を決定（feature/issue-{N}-{短い説明}）
git worktree add -b "feature/issue-{N}-{description}" "../PHOTON-RepoRAG-feature-issue-{N}-{description}" develop
```

### 2-3. CommandMateへの登録確認

```bash
curl -s -X POST http://localhost:3000/api/repositories/sync
commandmatedev ls --branch feature/issue-
```

全worktreeが表示されることを確認。

---

## Phase 2.5: 根本原因分析（バグIssueのみ）

Phase 1-2 で `bug` ラベルと分類されたIssueに対して、Opus 4.6 による根本原因分析を実行する。
機能Issue（FEATURE_ISSUES）はこのフェーズをスキップする。

### 2.5-1. Opus 4.6 に分析依頼

develop worktree上でバグIssueごとに根本原因分析を実行する：

```bash
WORKTREE_ID="photon-mlx-develop"

for bug_issue in $BUG_ISSUES; do
  ISSUE_BODY=$(gh issue view "$bug_issue" --repo Kewton/PHOTON-RepoRAG --json body -q '.body')

  commandmatedev send "$WORKTREE_ID" "Issue #${bug_issue} の根本原因分析を実施してください。コードを変更せず分析のみ行い、結果をテキストで出力してください。

## Issue内容
${ISSUE_BODY}

## 分析要求
1. 事象の再現パスをコード上で特定
2. 根本原因を特定（直接原因、設計上の問題、類似リスク）
3. 対策案を策定（即座対策、恒久対策、予防策）" \
    --agent copilot --model claude-opus-4.6 --auto-yes --duration 1h

  # Opus完了をポーリング
  for i in $(seq 1 40); do
    processing=$(curl -s "http://localhost:3000/api/worktrees" | \
      python3 -c "import sys,json; d=json.load(sys.stdin); print([w for w in d['worktrees'] if w['id']=='${WORKTREE_ID}'][0]['sessionStatusByCli']['copilot']['isProcessing'])" 2>/dev/null)
    [ "$processing" = "False" ] && break
    sleep 15
  done

  # 結果を取得
  commandmatedev capture "$WORKTREE_ID" --agent copilot --json
done
```

### 2.5-2. 分析結果のIssue追記

Opus 4.6 の分析結果をオーケストレーターが検証し、Issue本文に追記する：

```bash
gh issue edit "$bug_issue" --repo Kewton/PHOTON-RepoRAG --body "${CURRENT_BODY}${ANALYSIS_SECTION}"
```

### 2.5-3. 複数バグIssueの場合

バグIssueが複数ある場合は**順次実行**する（Opus 4.6はdevelop worktree 1つで共有するため）。

---

## Phase 3: 並列開発

### 3-0. ワーカー起動シーケンス（重要）

**commandmatedev でタスクを確実に処理させるには、以下の起動シーケンスを厳守すること。**
過去の実行で多数のケースでワーカーがタスクを処理せず `ready` のまま停止する問題が発生した。

#### 起動手順（各ワーカーごと）

```bash
# Step 1: セッションが idle/ready の場合、hello で起動
commandmatedev send <worktree-id> "hello"
sleep 5

# Step 2: running 状態を確認（最大30秒待機）
for i in $(seq 1 6); do
  status=$(commandmatedev ls 2>&1 | grep "<worktree-id>" | awk '{print $3}')
  [ "$status" = "running" ] && break
  sleep 5
done

# Step 3: running を確認できたらタスクを送信
commandmatedev send <worktree-id> "<command>" --auto-yes --duration 3h

# Step 4: タスク送信後、processing 状態を API で確認
sleep 10
processing=$(curl -s "http://localhost:3000/api/worktrees" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); [print(w['isProcessing']) for w in d['worktrees'] if w['id']=='<worktree-id>']" 2>/dev/null)

# Step 5: processing=False の場合、"a" を送信して再開
if [ "$processing" = "False" ]; then
  commandmatedev send <worktree-id> "a"
  sleep 10
fi
```

#### ワーカーが処理しない場合のリカバリー

> **⚠ 重要**: リカバリー時も必ずスラッシュコマンド（`/pm-auto-issue2dev`, `/bug-fix`）を再送信すること。
> 日本語の直接指示に切り替えると、Issueレビュー・設計・設計レビュー・作業計画のフェーズがスキップされ、品質が低下する。

1. **"a" 送信**: まず "a" を送信してプロンプト応答で再開を試みる
2. **スラッシュコマンド再送信**: "a" で再開しない場合、同じスラッシュコマンドを再送信する
3. **セッションクリア→再送信**: それでも処理しない場合、ユーザーに CommandMate UI からセッション削除を依頼し、idle 状態から Step 1 を再実行してスラッシュコマンドを再送信する

```bash
# リカバリー Step 1: "a" で再開を試みる
commandmatedev send <worktree-id> "a"
sleep 15

# リカバリー Step 2: スラッシュコマンドを再送信（直接指示に切り替えない）
commandmatedev send <worktree-id> "/pm-auto-issue2dev ${N}" --auto-yes --duration 3h
# または
commandmatedev send <worktree-id> "/bug-fix ${N}" --auto-yes --duration 3h
```

### 3-1. 各ワーカーにタスク送信

**Issue種別に応じて異なるコマンドを送信する。起動シーケンス（3-0）を各ワーカーに適用すること。**

```bash
# バグIssue: /bug-fix を送信（Phase 2.5の分析結果を参照して修正）
for bug_issue in $BUG_ISSUES:
  # 起動シーケンス実行後:
  commandmatedev send <worktree-id> "/bug-fix ${bug_issue}" \
    --auto-yes --duration 3h

# 機能Issue: /pm-auto-issue2dev を送信（従来通り）
for feature_issue in $FEATURE_ISSUES:
  # 起動シーケンス実行後:
  commandmatedev send <worktree-id> "/pm-auto-issue2dev ${feature_issue}" \
    --auto-yes --duration 3h
```

### 3-2. 進捗監視と処理確認

タスク送信後、**APIで processing 状態を必ず確認する**：

```bash
# 30秒後に全ワーカーの処理状態を確認
sleep 30
curl -s "http://localhost:3000/api/worktrees" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for wt in data.get('worktrees', []):
    if 'issue-' in wt.get('id', ''):
        sid = wt['id']
        processing = wt.get('isProcessing', False)
        running = wt.get('isSessionRunning', False)
        print(f'{sid}: running={running} processing={processing}')
        if running and not processing:
            print(f'  WARNING: {sid} is not processing! Send \"a\" to resume.')
"
```

**processing=False のワーカーには "a" を送信して再開する。**

### 3-3. 完了待機

全ワーカーの完了を待つ：

```bash
commandmatedev wait <worktree-ids...> --timeout 10800 --on-prompt agent
```

### 3-4. プロンプト対応

`commandmatedev wait` が exit code 10 を返した場合、プロンプト内容を確認して応答：

```bash
# --on-prompt agent が自動応答するが、それでも残るプロンプトには手動対応
for each worktree:
  commandmatedev respond <worktree-id> "2"  # "Yes, allow all" を選択
```

### 3-5. 完了後の成果物確認

**wait が完了しても、コミットや dev-reports が存在しない場合がある。**
必ず以下を確認し、成果物がない場合はリカバリーする：

```bash
for each worktree:
  commits=$(git -C "$worktree_path" log develop..HEAD --oneline)
  dirty=$(git -C "$worktree_path" status --short | wc -l)

  if [ -z "$commits" ] && [ "$dirty" -eq 0 ]; then
    # 成果物なし → 直接指示で再送信（3-0 のリカバリー手順）
  elif [ -z "$commits" ] && [ "$dirty" -gt 0 ]; then
    # 未コミット変更あり → 手動で品質チェック＆コミット
    cd "$worktree_path"
    ruff format && ruff check . && python -m pytest
    git add -A && git commit -m "feat(issue-${N}): ..."
  fi
done
```

**`--phase design` 指定時**: 全ワーカーの設計フェーズ完了を確認して終了。

---

## Phase 4: 設計突合（バリア）

弱依存のIssueがある場合、設計書をクロスチェックする。

### 4-1. 各ワーカーの設計書を取得

```bash
commandmatedev capture <worktree-id>
```

各worktreeの `workspace/design/issue-{N}-*-design-policy.md` を確認。

### 4-2. クロスチェック観点

- **影響ファイルの重複**: 同じファイルを変更する場合のコンフリクトリスク
- **型定義の整合性**: 共通型への変更が矛盾しないか
- **アーキテクチャの一貫性**: 設計方針が相反しないか
- **モジュール境界**: 新規モジュールの責務が重複しないか

### 4-3. 問題がある場合

該当ワーカーに修正指示を送信：

```bash
commandmatedev send <worktree-id> "設計書の以下の点を修正してください: {具体的な指摘}" \
  --auto-yes --duration 1h
```

**`--phase impl` 指定時**: 全ワーカーの実装完了を確認して終了。

---

## Phase 5: 品質確認

### 5-1. 各ワーカーに品質チェック送信

```bash
QUALITY_CMD="以下を順に実行し結果を報告してください:
1. ruff format --check .
2. ruff check .
3. python -m pytest
最後に Pass/Fail のサマリーを出力してください。"

for each worktree:
  commandmatedev send <worktree-id> "$QUALITY_CMD" --auto-yes --duration 1h
```

### 5-2. 結果収集

```bash
for each worktree:
  commandmatedev wait <worktree-id> --timeout 600
  commandmatedev capture <worktree-id>
```

### 5-3. 品質NGの場合

ワーカーに修正を指示し、再度品質チェック。最大3回まで自動リトライ。

---

## Phase 6: PR作成・マージ

`/pr-merge-pipeline` コマンドの内容を実行する：

```
/pr-merge-pipeline {issue_numbers}
```

詳細は `/pr-merge-pipeline` コマンドを参照。

**`--phase pr` 指定時**: PR作成・マージ完了を確認して終了。

---

## Phase 7: UAT（--full時のみ）

### 7-1. 受入テスト実行

developブランチ（オーケストレーター自身）で実行：

```bash
git pull origin develop
/uat {issue_numbers}
```

### 7-2. UAT結果判定

- **全PASS**: Phase 8（完了）へ
- **FAILあり**: `/uat-fix-loop` を実行

```
/uat-fix-loop {fail_issue_numbers}
```

詳細は `/uat-fix-loop` コマンドを参照。

---

## Phase 8: 完了報告

### 8-1. 最終検証

```bash
python -m pytest
ruff check .
python -m pytest
ruff format --check .
```

### 8-2. 結果レポート

`workspace/orchestration/runs/$DATE/summary.md` に統合サマリーを出力：

```markdown
## オーケストレーション完了報告

### 対象Issue

| Issue | タイトル | ステータス |
|-------|---------|-----------|
| #{N} | {title} | 完了 |
| #{M} | {title} | 完了 |

### 実行フェーズ結果

| Phase | 内容 | ステータス |
|-------|------|-----------|
| 1 | 依存関係分析 | 完了 |
| 2 | Worktree準備 | 完了 |
| 3 | 並列開発 | 完了 |
| 4 | 設計突合 | 完了（問題なし） |
| 5 | 品質確認 | 完了（全Pass） |
| 6 | PR・マージ | 完了（PR #XX, #YY） |
| 7 | UAT | 完了（全PASS） |

### 品質チェック

| チェック項目 | 結果 |
|-------------|------|
| python -m pytest | Pass |
| ruff check . | Pass |
| python -m pytest | Pass |
| ruff format --check . | Pass |

### 成果物

- 設計書: workspace/design/issue-{N}-*-design-policy.md
- 作業計画: workspace/issues/{N}/work-plan.md
- 進捗報告: workspace/issues/{N}/pm-auto-dev/iteration-1/progress-report.md
- UATレポート: sandbox/{N}/latest/report.html
- 統合サマリー: workspace/orchestration/runs/{DATE}/summary.md
```

---

## エラーハンドリング

| エラー | 対応 |
|--------|------|
| developブランチでない | エラー表示し中断 |
| CommandMateサーバー未起動 | `commandmatedev start --daemon` を案内 |
| worktree作成失敗 | エラー表示、手動作成を案内 |
| ワーカーのタイムアウト | captureで状況確認→追加指示 or ユーザーに報告 |
| 品質チェック3回連続失敗 | ユーザーに報告して中断 |
| コンフリクト解消失敗 | ユーザーに報告して中断 |
| UAT 4回連続FAIL | ユーザーに判断を仰ぐ |

---

## 完了条件

- [ ] 全Issueの開発が完了している
- [ ] 品質チェック全パス（ruff警告0件、テスト全パス）
- [ ] 全IssueのPRがdevelopにマージ済み
- [ ] developブランチでの統合ビルド・テストが全パス
- [ ] （--full時）UAT全テストPASS
- [ ] 統合サマリーが出力されている

## 関連コマンド

- `/pm-auto-issue2dev`: Issue単位の全自動開発（機能Issueに送信）
- `/bug-fix`: バグ調査→修正→テスト（バグIssueに送信）
- `/cause-analysis`: 根本原因分析（Opus 4.6、バグIssueのPhase 2.5で使用）
- `/current-situation`: 不具合事象の整理とIssue登録
- `/pr-merge-pipeline`: PR作成からマージ完了まで
- `/uat`: 受入テスト
- `/uat-fix-loop`: UAT不合格時の修正ループ
- `/issues-exec-plan`: 複数Issueの実行計画策定
- `/worktree-setup`: worktree個別作成
- `/worktree-cleanup`: worktree個別削除
