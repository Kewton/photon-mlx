---
model: opus
description: "Issueの受入テストをdevelopブランチ上でPHOTON-RepoRAGを実際に起動して実施し、HTMLレポートを生成"
---

# ユーザー受入テスト（UAT）

## 概要
developブランチでIssueの受入テストを実施します。PHOTON-RepoRAGを実際に起動して動作確認を行い、結果をHTMLレポートとして出力します。複数Issueの一括テストに対応しています。

**重要**: コンテキストを綺麗に保つため、テスト実行はサブエージェントで行います。

## 使用方法
```
/uat [Issue番号]              # 単一Issue
/uat [Issue番号1] [Issue番号2] ...  # 複数Issue（スペース区切り）
```

例：
```
/uat 8
/uat 8 9 10
```

## 実行手順

### 1. 事前チェック

```bash
# developブランチであることを確認
current_branch=$(git branch --show-current)
if [ "$current_branch" != "develop" ]; then
  echo "ERROR: developブランチで実行してください（現在: $current_branch）"
  exit 1
fi

# 未コミットの変更がないことを確認
git status --porcelain
```

### 2. 引数の解析

`$ARGUMENTS` をスペースで分割し、Issue番号のリストを生成する。

```
入力例: "8 9 10"
→ ISSUE_LIST = [8, 9, 10]
```

各Issue番号に対して `gh issue view` で存在を確認する。無効な番号があれば警告を表示し、有効なIssueのみ続行する。

```bash
for issue_num in $ARGUMENTS; do
  gh issue view "$issue_num" --repo Kewton/PHOTON-RepoRAG --json title -q '.title' 2>/dev/null
done
```

### 3. Issue情報の取得（Issueごと）

各Issueについて以下を取得：

```bash
gh issue view $issue_num --json title,body,labels
```

Issue本文から以下を抽出：
- **受け入れ基準**（「受け入れ基準」「Acceptance Criteria」セクション）
- **提案する解決策**（期待される機能の概要）
- **関連するファイル・モジュール**

### 4. 受入テスト計画の作成

Issueの受け入れ基準と提案する解決策に基づき、**具体的なテスト計画**を作成する。

**重要**: 各テスト項目には必ず **PHOTON-RepoRAGバイナリを実際に起動する具体的なコマンド** を含めること。コードリーディングやユニットテストの確認のみで完結するテスト項目は不可。

#### テスト計画のフォーマット

各テスト項目を以下の形式で記述する：

```
AT-${ISSUE_NUM}-${N}: ${テスト項目タイトル}
  対応する受け入れ基準: ${Issueの受け入れ基準の該当項目}
  前提条件:
    - ${セットアップ手順（ファイル作成、環境変数設定等）}
  実行コマンド:
    - ${前提条件のセットアップコマンド}
    - echo "${具体的なテスト入力プロンプト}" | ./target/release/photon-mlx --model qwen3.5:35b --no-approval --oneshot 2>"$RUN_DIR/AT-${N}/stderr.log" >"$RUN_DIR/AT-${N}/stdout.log"
    または対話的テストの場合: 具体的な手順を記述
  期待結果:
    - ${stdout/stderrに期待される具体的な出力内容}
  PASS/FAIL判定基準:
    - PASS: ${具体的な条件}
    - FAIL: ${具体的な条件}
  ログ出力先:
    - stdout: $RUN_DIR/AT-${N}/stdout.log
    - stderr: $RUN_DIR/AT-${N}/stderr.log
```

#### テスト計画の必須要件

1. **Issueの受け入れ基準との対応**: 各受け入れ基準に最低1つのテスト項目が対応すること
2. **E2E実行**: 全テスト項目に `./target/release/photon-mlx --model qwen3.5:35b` を使った実行コマンドを含むこと
3. **具体的な入力値**: `echo "${テスト入力}"` の内容が具体的に定義されていること（「テスト入力」のようなプレースホルダは不可）
4. **具体的な期待結果**: stdout/stderrに期待される出力が具体的に記述されていること
5. **エビデンスの保存先**: 全テスト項目に `stdout.log` / `stderr.log` の出力パスが指定されていること
6. **補完テスト**: E2Eで検証困難な項目（エッジケース等）は、`python -m pytest` による補完を明記した上で許容する。ただしその場合も可能な限りE2E実行を併用すること

### 5. テスト計画のレビュー（第1回）

作成したテスト計画を**サブエージェント（general-purpose）**でレビューする。

#### レビュー観点

以下の2つの観点で漏れ・不備がないか検証する：

**観点A: UAT方針への準拠**
- [ ] 全テスト項目にPHOTON-RepoRAGバイナリ起動コマンド（`./target/release/photon-mlx --model qwen3.5:35b`）が含まれているか
- [ ] `echo "${テスト入力}"` の内容が具体的に定義されているか（プレースホルダではないか）
- [ ] 各テスト項目に `stdout.log` / `stderr.log` の出力パスが指定されているか
- [ ] PASS/FAIL判定基準が明確に定義されているか
- [ ] 前提条件のセットアップ手順が具体的か

**観点B: Issue記載内容の網羅性**
- [ ] Issueの「受け入れ基準」の全項目にテスト項目が対応しているか（対応表を作成して確認）
- [ ] Issueの「提案する解決策」に記載された主要機能がテストされているか
- [ ] Issueの「エッジケースハンドリング方針」に記載されたケースがカバーされているか
- [ ] 正常系だけでなく異常系（エラーケース）のテストが含まれているか

#### レビュー結果の出力形式

```
## テスト計画レビュー（第1回）

### 観点A: UAT方針への準拠
| チェック項目 | 結果 | 指摘事項 |
|------------|------|---------|
| PHOTON-RepoRAGバイナリ起動コマンド | OK/NG | ${詳細} |
| テスト入力の具体性 | OK/NG | ${詳細} |
| ログ出力パス指定 | OK/NG | ${詳細} |
| PASS/FAIL判定基準 | OK/NG | ${詳細} |
| 前提条件の具体性 | OK/NG | ${詳細} |

### 観点B: Issue記載内容の網羅性
| 受け入れ基準 | 対応テスト項目 | カバー状況 |
|------------|-------------|----------|
| ${基準1} | AT-N-1 | OK/不足 |
| ${基準2} | AT-N-2, AT-N-3 | OK/不足 |

### 指摘事項一覧
1. [重要度: 高/中/低] ${指摘内容}
2. ...
```

### 6. テスト計画の修正（第1回）

ステップ5のレビュー指摘事項に基づき、テスト計画を修正する。修正内容を明示する。

### 7. テスト計画のレビュー（第2回）

修正後のテスト計画を**同じレビュー観点**で再度レビューする（ステップ5と同じ手順）。第1回の指摘事項が解消されているかも確認する。

### 8. テスト計画の修正（第2回）

ステップ7のレビュー指摘事項に基づき、テスト計画を最終修正する。

### 9. テスト計画のユーザー確認

最終テスト計画を一覧化し、**ユーザーに確認**する：

```
以下のテスト計画で受入テストを実施します。追加・修正はありますか？

Issue #10: ANVIL.mdプロジェクト指示ファイル対応
  AT-10-1: .photon-mlx/ANVIL.md の自動注入
    コマンド: echo "ANVIL.mdの内容を表示して" | ./target/release/photon-mlx --model qwen3.5:35b --no-approval --oneshot
    期待結果: LLMの応答にANVIL.mdの指示内容が反映されている
  AT-10-2: ...

(y: 続行 / 追加・修正があれば入力してください)
```

Issueに受け入れ基準が明記されている場合はそれに従う。明記されていない場合はIssue本文から推定する。不明確なテスト項目がある場合は、追加で質問してから進める。

### 10. 作業環境の準備（ラン管理）

同一Issueに対して複数回テストを実施することを想定し、**ラン（実行回）ごとにディレクトリを分離**する。

```bash
DATE=$(date +%Y-%m-%d)

for issue_num in $ISSUE_LIST; do
  # 既存ランの連番を取得して次の番号を決定
  EXISTING=$(ls -d "./sandbox/${issue_num}/${DATE}_"* 2>/dev/null | wc -l | tr -d ' ')
  SEQ=$(printf "%03d" $((EXISTING + 1)))
  RUN_DIR="./sandbox/${issue_num}/${DATE}_${SEQ}"

  mkdir -p "$RUN_DIR"

  # latest シンボリックリンクを更新
  ln -sdef "${DATE}_${SEQ}" "./sandbox/${issue_num}/latest"
done
```

ディレクトリ構成:
```
sandbox/
├── 9/
│   ├── 2026-03-17_001/       # 1回目
│   │   ├── report.html
│   │   ├── AT-1/
│   │   └── AT-2/
│   ├── 2026-03-17_002/       # 差し戻し後の2回目
│   │   ├── report.html
│   │   ├── AT-1/
│   │   └── AT-2/
│   ├── latest -> 2026-03-17_002/  # 最新ランへのシンボリックリンク
│   └── history.html          # 全ランの履歴一覧
```

以降の手順で参照するパスは `./sandbox/${ISSUE_NUM}/` ではなく **`$RUN_DIR`（= `./sandbox/${ISSUE_NUM}/${DATE}_${SEQ}/`）** を使用する。

### 11. サブエージェントによるテスト実行

**各Issueのテスト項目をサブエージェント（general-purpose）で実行します。**

独立したIssueのテストは並列にサブエージェントを起動して効率化する。

#### サブエージェントへの指示テンプレート

**重要**: このテンプレートをそのまま使用すること。独自の指示文に書き換えないこと。`${...}` 部分のみをテスト計画の内容で置換する。

```
Issue #${ISSUE_NUM} の受入テスト AT-${ISSUE_NUM}-${N} を実施してください。

## 必須事項
- PHOTON-RepoRAGバイナリを**実際に起動**して動作確認すること（コードリーディングやユニットテスト確認だけでは不可）
- テスト結果のエビデンスとして stdout.log / stderr.log を必ず保存すること
- コードリーディングのみで PASS 判定しないこと

## テスト内容
${テスト計画から転記: テスト項目タイトルと説明}

## 前提条件のセットアップ
${テスト計画から転記: 具体的なセットアップ手順}

## 実行手順
1. python -m pytest（ビルド済みの場合はスキップ可）
2. 前提条件をセットアップ:
   ${テスト計画から転記: セットアップコマンド}
3. PHOTON-RepoRAGバイナリを実行し、出力をログファイルに保存:
   ${テスト計画から転記: 実行コマンド（stdout.log/stderr.logへのリダイレクト付き）}
4. 出力結果を期待値と照合
5. テスト結果（PASS/FAIL）と詳細を報告
6. （補完テストがある場合）python -m pytest で補完検証を実施

## 期待結果
${テスト計画から転記: 具体的な期待結果}

## PASS/FAIL判定基準
${テスト計画から転記: 判定基準}

## 作業ディレクトリ
$RUN_DIR/AT-${N}/

## エビデンス保存（必須）
以下のファイルを必ず作成すること:
- $RUN_DIR/AT-${N}/stdout.log  （PHOTON-RepoRAGの標準出力）
- $RUN_DIR/AT-${N}/stderr.log  （PHOTON-RepoRAGの標準エラー出力）
- $RUN_DIR/AT-${N}/result.json （テスト結果）

## 出力形式
以下をJSON形式で $RUN_DIR/AT-${N}/result.json に保存し、内容を報告：
{
  "test_id": "AT-${ISSUE_NUM}-${N}",
  "issue_number": ${ISSUE_NUM},
  "title": "テスト項目名",
  "status": "PASS" or "FAIL",
  "description": "テスト内容の説明",
  "steps": ["実際に実行した手順1", "実際に実行した手順2", ...],
  "command": "実際に実行したコマンド",
  "expected": "期待結果",
  "actual": "実際の結果（stdout/stderrの内容を含む）",
  "evidence_files": ["stdout.log", "stderr.log"],
  "evidence": "根拠となるコマンド出力やログ（stdout.logの主要部分を引用）",
  "notes": "補足事項"
}
```

#### テスト実行の品質基準

サブエージェントの結果を受け取った後、以下を確認する。満たさない場合はサブエージェントに再実行を指示する：

- [ ] `stdout.log` / `stderr.log` が実際に作成されているか
- [ ] `result.json` の `command` フィールドに実際に実行したコマンドが記録されているか
- [ ] `result.json` の `actual` フィールドにPHOTON-RepoRAGの出力内容が含まれているか（コードの行番号参照ではなく）
- [ ] エビデンスがコードリーディング結果ではなく、PHOTON-RepoRAGバイナリの実行結果に基づいているか

### 12. スクリーンショット取得（可能な場合）

macOS環境の場合、TUI出力をキャプチャ：

```bash
# macOSの場合、screencaptureでターミナルのスクリーンショットを取得可能
# screencapture -x "$RUN_DIR/AT-${N}/screenshot.png"
```

スクリーンショット取得が難しい場合は、ステップ11で保存した stdout.log / stderr.log をエビデンスとして使用する。

### 13. HTMLレポート生成

#### Issue単位のレポート（ラン別）

各ランのテスト結果を `$RUN_DIR/report.html` に生成する。レポートにはラン番号（第N回）を明記する。

#### 履歴レポート（Issue単位）

各Issueの全ランの履歴を `./sandbox/${ISSUE_NUM}/history.html` に生成・更新する。新しいランの結果を既存の履歴に追記する形式。

#### 全体サマリーレポート（複数Issue時）

複数Issueが指定された場合、全Issueの結果を集約した `./sandbox/uat-summary.html` を追加で生成する。

#### HTMLレポートの構成

```html
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>受入テストレポート - Issue #${ISSUE_NUM}</title>
</head>
<body>
  <!-- ヘッダー: Issue情報 -->
  <!-- サマリー: PASS/FAIL数、合格率 -->
  <!-- テスト結果テーブル: 各テスト項目の詳細 -->
  <!-- エビデンス: コマンド出力やスクリーンショット -->
  <!-- フッター: 実行環境情報 -->
</body>
</html>
```

#### 全体サマリーHTMLの構成（複数Issue時）

```html
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>UAT全体サマリー</title>
</head>
<body>
  <!-- ヘッダー: テスト実施日時、対象Issue一覧 -->
  <!-- 全体サマリー: 全Issue合算のPASS/FAIL数、合格率 -->
  <!-- Issue別サマリーテーブル -->
  <!--   Issue番号 | タイトル | テスト数 | PASS | FAIL | 判定 | レポートリンク -->
  <!-- フッター: 実行環境情報 -->
</body>
</html>
```

#### HTMLデザイン要件

- **モダンなデザイン**: ダークテーマベース、カード型レイアウト
- **色分け**: PASS=緑（#22c55e）、FAIL=赤（#ef4444）、SKIP=グレー
- **レスポンシブ**: ブラウザで見やすいレイアウト
- **情報量**: テスト手順、期待値、実際の結果、エビデンスを各テストに表示
- **サマリーセクション**: 合格率表示（パーセンテージ + 視覚的バー）、全テスト数/PASS数/FAIL数
- **エビデンス折りたたみ**: 長いログ出力は `<details>` タグで折りたたみ
- **実行環境情報**: 日時、ブランチ、コミットハッシュ、Rustバージョン、OS情報
- **Issue別レポートへのリンク**: 全体サマリーから各Issueのレポートへリンク

#### HTMLテンプレートの主要セクション（Issue単位レポート）

1. **ヘッダー**
   - Issue番号・タイトル
   - テスト実施日時
   - ブランチ名・コミットハッシュ
   - **ラン番号**（第N回）

2. **サマリーダッシュボード**
   - 合格率（パーセンテージ + 視覚的バー）
   - PASS / FAIL / SKIP の件数
   - 全体の判定（ALL PASS → ACCEPTED / それ以外 → REJECTED）

3. **テスト結果一覧**
   - 各テスト項目をカード形式で表示
   - テストID、タイトル、ステータスバッジ
   - テスト手順（番号付きリスト）
   - 期待結果 vs 実際の結果
   - エビデンス（`<details>` で折りたたみ）

4. **フッター**
   - 実行環境: OS, Rustバージョン, PHOTON-RepoRAGバージョン
   - 生成ツール: Claude Code

#### 履歴HTMLの主要セクション（Issue単位）

`./sandbox/${ISSUE_NUM}/history.html` に全ランの履歴を表示する。

1. **ヘッダー**: Issue番号・タイトル
2. **履歴テーブル**:

| # | 日時 | コミット | テスト数 | PASS | FAIL | 判定 | レポートリンク |
|---|------|---------|---------|------|------|------|--------------|
| 1 | 2026-03-17 10:00 | 07adb94 | 8 | 6 | 2 | REJECTED | [レポート](./2026-03-17_001/report.html) |
| 2 | 2026-03-17 15:00 | 758a524 | 8 | 8 | 0 | ACCEPTED | [レポート](./2026-03-17_002/report.html) |

3. **前回との差分**（2回目以降）: 前回FAILだった項目が今回PASSになったか等の変化を表示

### 14. 結果報告

全Issue の結果サマリーをユーザーに報告。ラン番号を明記する：

```
受入テスト完了（第2回）

  Issue #8:  7/7 PASS (100%) → ACCEPTED  [第2回]
  Issue #9:  3/4 PASS ( 75%) → REJECTED  [第1回]

  全体: 10/11 PASS (91%)

  レポート:
    ./sandbox/8/2026-03-17_002/report.html
    ./sandbox/9/2026-03-17_001/report.html
    ./sandbox/8/history.html  (履歴)
    ./sandbox/uat-summary.html  (全体サマリー)
```

単一Issueの場合は全体サマリーは生成せず、Issue単体のレポートと履歴のみ出力する。

### 15. テスト結果のIssueへの記録（既存コメント更新方式）

テスト実施後、各IssueのコメントにUAT結果を記録する。**2回目以降は既存のUATコメントを更新（編集）**し、履歴を追記する形式にする。

#### 既存UATコメントの検索

```bash
# 既存の「受入テスト結果 (UAT)」コメントを検索
COMMENT_ID=$(gh api repos/Kewton/PHOTON-RepoRAG/issues/${ISSUE_NUM}/comments \
  --jq '[.[] | select(.body | startswith("## 受入テスト結果 (UAT)"))] | last | .id' 2>/dev/null)
```

#### 初回: 新規コメント作成

```bash
gh issue comment $ISSUE_NUM --repo Kewton/PHOTON-RepoRAG --body "$(cat <<'COMMENT_EOF'
## 受入テスト結果 (UAT)

### 履歴

| # | 日時 | コミット | テスト数 | PASS | FAIL | 判定 |
|---|------|---------|---------|------|------|------|
| 1 | ${実施日時} | ${COMMIT_HASH} | ${TOTAL} | ${PASS_COUNT} | ${FAIL_COUNT} | ${ACCEPTED or REJECTED} |

### 最新結果（第1回 - ${COMMIT_HASH}）

| ID | テスト項目 | 結果 |
|-----|-----------|------|
| AT-N-1 | テスト項目名 | ✅ PASS / ❌ FAIL |
| ... | ... | ... |

### FAIL詳細（FAILがある場合）

各FAIL項目の原因と詳細を記載する。

---
📋 HTMLレポート: `sandbox/${ISSUE_NUM}/latest/report.html`
📊 履歴レポート: `sandbox/${ISSUE_NUM}/history.html`
🤖 Generated with Claude Code
COMMENT_EOF
)"
```

#### 2回目以降: 既存コメントを更新

既存のUATコメントを `gh api --method PATCH` で編集し、履歴テーブルに行を追記、最新結果セクションを更新する。

```bash
# 新しい履歴行を追加し、最新結果セクションを更新
gh api repos/Kewton/PHOTON-RepoRAG/issues/comments/${COMMENT_ID} \
  --method PATCH \
  --field body="$(cat <<'COMMENT_EOF'
## 受入テスト結果 (UAT)

### 履歴

| # | 日時 | コミット | テスト数 | PASS | FAIL | 判定 |
|---|------|---------|---------|------|------|------|
| 1 | 2026-03-17 10:00 | 07adb94 | 8 | 6 | 2 | REJECTED |
| 2 | 2026-03-17 15:00 | 758a524 | 8 | 8 | 0 | ACCEPTED |

### 最新結果（第2回 - 758a524）

| ID | テスト項目 | 結果 | 前回比 |
|-----|-----------|------|--------|
| AT-N-1 | テスト項目名 | ✅ PASS | 🔄 FAIL→PASS |
| AT-N-2 | テスト項目名 | ✅ PASS | — (変化なし) |
| ... | ... | ... | ... |

---
📋 HTMLレポート: `sandbox/${ISSUE_NUM}/latest/report.html`
📊 履歴レポート: `sandbox/${ISSUE_NUM}/history.html`
🤖 Generated with Claude Code
COMMENT_EOF
)"
```

これにより、Issue上でテストの実施履歴と結果を一つのコメントで追跡できる。

## 完了条件

- [ ] テスト計画が2回のレビューサイクルを経て確定している
- [ ] テスト計画の全項目にPHOTON-RepoRAGバイナリ起動コマンドが含まれている
- [ ] 全Issueの全テスト項目が実行されている
- [ ] 各テスト項目で `stdout.log` / `stderr.log` が作成され、PHOTON-RepoRAGバイナリの実行出力が記録されている
- [ ] 各テスト項目で `result.json` が作成され、実行コマンドと実際の結果が記録されている
- [ ] 各ランのHTMLレポートが `./sandbox/${ISSUE_NUM}/${DATE}_${SEQ}/report.html` に生成されている
- [ ] `latest` シンボリックリンクが最新ランを指している
- [ ] 履歴HTMLが `./sandbox/${ISSUE_NUM}/history.html` に生成・更新されている
- [ ] 複数Issue時は全体サマリーが `./sandbox/uat-summary.html` に生成されている
- [ ] レポートがブラウザで正しく表示される
- [ ] 結果サマリー（ラン番号付き）がユーザーに報告されている
- [ ] 各IssueのGitHubコメントにテスト結果が記録されている（2回目以降は既存コメント更新）
- [ ] 過去のランの結果・エビデンスが保持されている（上書きされていない）

## エラーハンドリング

| エラーケース | 対応 |
|-------------|------|
| developブランチでない | エラー表示し中断 |
| `python -m pytest` 失敗 | エラー表示し中断（テスト不可） |
| Issue番号が無効 | 該当Issueをスキップし、有効なIssueのみ続行 |
| Issueに受入基準がない | テスト項目を推定しユーザーに確認 |
| PHOTON-RepoRAGの起動に失敗 | エラーログを記録しFAILとして報告 |
| テスト途中でエラー | 該当テストをFAILとし、残りのテストを続行 |
| 全Issue番号が無効 | エラー表示し中断 |
