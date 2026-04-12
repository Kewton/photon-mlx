---
name: release
description: "Create a new release with version bump, CHANGELOG update, Git tag, and GitHub Release. Use when releasing a new version of the project."
disable-model-invocation: true
allowed-tools: "Bash, Read, Edit, Write"
argument-hint: "[version-type] (major|minor|patch) or [version] (e.g., 1.2.3)"
---

# リリーススキル

新しいバージョンをリリースします。git worktree + commandmatedev でリリースブランチを作成・操作し、PRを作成します。mainへマージ後にタグ・GitHub Releasesを作成します。

## 使用方法

```bash
/release patch      # パッチバージョンアップ (0.1.0 → 0.1.1)
/release minor      # マイナーバージョンアップ (0.1.0 → 0.2.0)
/release major      # メジャーバージョンアップ (0.1.0 → 1.0.0)
/release 1.0.0      # 直接バージョン指定
```

## 前提条件

- `commandmatedev start --daemon` が起動済みであること
- commandmatedev にPHOTON-RepoRAGリポジトリが登録済みであること

## ブランチフロー

```
main (メインworktree) ─── git worktree add ──→ ../photon-mlx-release-v{version} (リリースworktree)
                                                  ↓
                                          commandmatedev send でリリース作業を委譲
                                          (バージョン更新・ビルド・テスト・コミット・PR作成)
                                                  ↓
                                          PR ──→ main マージ
                                                  ↓
                                          メインworktreeでタグ作成 → GitHub Actions 自動ビルド
                                                  ↓
                                          worktreeクリーンアップ
```

CLAUDE.mdの「mainへはPRマージのみ」ルールに準拠しています。

## Phase 1: PR作成まで

### 1. 事前チェック（メインworktreeで実行）

以下を確認してください：

```bash
# 現在のブランチがmainであることを確認
git branch --show-current

# 未コミットの変更がないことを確認
git status

# リモートと同期していることを確認
git fetch origin
git pull origin main
```

**エラーケースの対応:**

| 状況 | 対応 |
|------|------|
| mainブランチでない | `git checkout main` を実行 |
| 未コミットの変更がある | コミットまたはスタッシュを促す |
| リモートと差分がある | `git pull origin main` を促す |

### 2. 現在のバージョン取得

```bash
# pyproject.tomlからバージョンを取得
current_version=$(grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "Current version: $current_version"
```

### 3. 新バージョンの計算

引数に基づいて新バージョンを計算します：

- `patch`: PATCH部分を+1 (例: 0.1.0 → 0.1.1)
- `minor`: MINOR部分を+1、PATCHを0に (例: 0.1.1 → 0.2.0)
- `major`: MAJOR部分を+1、MINOR/PATCHを0に (例: 0.2.0 → 1.0.0)
- 直接指定: 指定されたバージョンをそのまま使用

### 4. タグ存在チェック

```bash
# タグが既に存在しないことを確認
if git rev-parse "v$new_version" >/dev/null 2>&1; then
  echo "Error: Tag v$new_version already exists"
  exit 1
fi
```

### 5. リリース用worktree作成（メインworktreeで実行）

```bash
WORKTREE_DIR="../photon-mlx-release-v$new_version"
git worktree add -b "release/v$new_version" "$WORKTREE_DIR" main
```

### 6. commandmatedevでworktree同期

新しく作成したworktreeをcommandmatedevに認識させます：

```bash
curl -s -X POST http://localhost:3000/api/repositories/sync
```

### 7. worktree IDの取得

```bash
RELEASE_WT=$(commandmatedev ls --branch "release/v$new_version" --quiet)
echo "Worktree ID: $RELEASE_WT"
```

IDが取得できない場合は、同期が完了していない可能性があります。数秒待って再試行してください。

### 8. commandmatedevでリリース作業を実行

リリースworktreeのエージェントに以下の作業を委譲します：

```bash
commandmatedev send "$RELEASE_WT" "以下のリリース作業をすべて実行してください。

## リリースバージョン情報
- 旧バージョン: $current_version
- 新バージョン: $new_version

## 作業手順

### 1. pyproject.toml更新
pyproject.toml の version を \"$new_version\" に変更してください。

### 2. pip.lock更新
pip freeze > requirements.txt を実行してください。

### 3. README.md更新
README.md内のダウンロードURLのバージョンを更新してください：
/releases/download/v$current_version/ → /releases/download/v$new_version/

### 4. CHANGELOG.md更新
- git log v$current_version..HEAD --oneline でv$current_version以降の変更を確認
- [Unreleased]セクションの下に [${new_version}] - $(date +%Y-%m-%d) セクションを追加
- 変更内容をAdded/Fixed/Changedに分類して記載
- 空の[Unreleased]セクションを残す

### 5. 品質チェック
以下をすべて実行し、全パスすることを確認：
- python -m pytest
- python -m pytest
- ruff check .

### 6. コミット・プッシュ
git add pyproject.toml pip.lock CHANGELOG.md README.md
git commit -m 'chore: release v$new_version'
git push origin release/v$new_version

### 7. PR作成
gh pr create --base main --head release/v$new_version --title 'chore: release v$new_version' で作成。
bodyにはCHANGELOG.mdの該当セクションとチェックリストを含めてください。

品質チェックが失敗した場合はリリースを中断し、エラー内容を報告してください。" --auto-yes --duration 1h
```

### 9. 完了待ち・結果確認

```bash
# エージェントの完了を待つ（最大10分）
commandmatedev wait "$RELEASE_WT" --timeout 600

# 結果を確認
commandmatedev capture "$RELEASE_WT"
```

エージェントからの出力でPR URLを確認してください。

## Phase 2: PRマージ後

**ユーザーがPRをマージしたことを確認してから以下を実行します。**

### 10. タグ作成・プッシュ（メインworktreeで実行）

```bash
# メインworktreeでmainを最新化
git pull origin main

# タグ作成・プッシュ
git tag "v$new_version"
git push origin "v$new_version"
```

タグのプッシュにより GitHub Actions（`.github/workflows/release.yml`）が自動で以下を実行します：
- 4プラットフォーム向けバイナリビルド（linux-amd64, linux-arm64, darwin-amd64, darwin-arm64）
- GitHub Release作成とバイナリアップロード

### 11. リリースworktreeのクリーンアップ

```bash
# worktree削除
git worktree remove "../photon-mlx-release-v$new_version"

# ブランチ削除（ローカル・リモート）
git branch -d "release/v$new_version"
git push origin --delete "release/v$new_version"
```

### 12. developへの反映（commandmatedev経由）

developブランチのworktree IDを取得し、反映を委譲します：

```bash
DEVELOP_WT=$(commandmatedev ls --branch develop --quiet)
commandmatedev send "$DEVELOP_WT" "git pull origin develop && git merge main && git push origin develop" --auto-yes
commandmatedev wait "$DEVELOP_WT" --timeout 120
```

developのworktreeがcommandmatedevに登録されていない場合は、メインworktreeから直接実行します：

```bash
# developのworktreeパスを確認
git worktree list | grep develop
# 該当パスで実行
cd <develop-worktree-path> && git pull origin develop && git merge main && git push origin develop
```

## 完了確認

リリース完了後、以下を確認します：

```bash
# タグ一覧
git tag -l

# 最新タグ
git describe --tags --abbrev=0

# GitHub Releases（バイナリがアップロードされていることを確認）
gh release view "v$new_version"

# GitHub Actions のビルド状況
gh run list --limit 3
```

## エラーハンドリング

| エラーケース | 対応 |
|-------------|------|
| 未コミットの変更がある | エラー表示し、コミットまたはスタッシュを促す |
| リモートとの差分がある | `git pull`を促す |
| タグが既に存在する | エラー表示し、別バージョンの指定を促す |
| worktree作成失敗 | 同名ディレクトリが存在しないか確認 |
| commandmatedevサーバー未起動 | `commandmatedev start --daemon` を促す |
| worktree ID取得失敗 | `curl -s -X POST http://localhost:3000/api/repositories/sync` で同期を促す |
| commandmatedev send/waitタイムアウト | `commandmatedev capture` で状況確認 |
| CHANGELOG.mdが存在しない | 新規作成するか確認 |
| [Unreleased]セクションが空 | 警告を表示し、続行するか確認 |
| `python -m pytest` が失敗 | エラー修正を促し、リリースを中断。worktreeを削除 |
| `python -m pytest` が失敗 | テスト修正を促し、リリースを中断。worktreeを削除 |
| `ruff check` に警告がある | 警告修正を促し、リリースを中断。worktreeを削除 |
| PRのCIが失敗 | commandmatedev send で修正コミットを追加 |
| GitHub Actionsのビルド失敗 | ワークフロー修正後にタグを削除して再作成 |
| リリース中断時のクリーンアップ | `git worktree remove` → `git branch -D` → `git push origin --delete` |

## 参考

- [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/)
- [Semantic Versioning](https://semver.org/lang/ja/)
- [GitHub Actions Release Workflow](../../.github/workflows/release.yml)
- [CommandMate Agent Operations](commandmatedev docs --section agent-operations)
