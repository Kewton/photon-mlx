# リリース実行例

このドキュメントでは、`/release` スキルの実行例を示します。

## 例1: パッチリリース (v0.1.0 → v0.1.1)

### コマンド
```
/release patch
```

### 実行結果
```
リリーススキルを開始します

事前チェック
  OK mainブランチで実行中
  OK 未コミットの変更なし
  OK リモートと同期済み

バージョン情報
  現在: 0.1.0
  新規: 0.1.1
  種別: patch

タグチェック
  OK v0.1.1 は存在しません

リリースブランチ作成
  OK release/v0.1.1 を作成

ファイル更新
  OK pyproject.toml を更新 (0.1.0 → 0.1.1)
  OK pip.lock を更新
  OK README.md のダウンロードURLを更新
  OK CHANGELOG.md を更新

品質チェック
  OK python -m pytest 成功
  OK python -m pytest 全テストパス
  OK ruff check 警告ゼロ

コミット & プッシュ
  OK コミット作成: chore: release v0.1.1
  OK release/v0.1.1 をプッシュ

PR作成
  OK PR作成完了
  URL: https://github.com/Kewton/PHOTON-RepoRAG/pull/XX

--- PRマージ後に以下を実行してください ---

タグ作成
  $ git checkout main && git pull origin main
  $ git tag v0.1.1 && git push origin v0.1.1

  タグプッシュにより GitHub Actions が自動で:
  - 4プラットフォーム向けバイナリビルド
  - GitHub Release 作成・バイナリアップロード

クリーンアップ
  $ git branch -d release/v0.1.1
  $ git push origin --delete release/v0.1.1

developへの反映
  $ git checkout develop && git merge main && git push origin develop

確認コマンド:
  gh release view v0.1.1          # リリース詳細・バイナリ確認
  gh run list --limit 3           # GitHub Actions ビルド状況
```

---

## 例2: マイナーリリース (v0.1.1 → v0.2.0)

### コマンド
```
/release minor
```

### 実行結果
```
リリーススキルを開始します

事前チェック
  OK mainブランチで実行中
  OK 未コミットの変更なし
  OK リモートと同期済み

バージョン情報
  現在: 0.1.1
  新規: 0.2.0
  種別: minor

...（以下同様）

PR作成
  OK PR作成完了
  URL: https://github.com/Kewton/PHOTON-RepoRAG/pull/XX

--- PRマージ後にタグ作成・クリーンアップを実行 ---
```

---

## 例3: メジャーリリース (v0.2.0 → v1.0.0)

### コマンド
```
/release major
```

### 実行結果
```
リリーススキルを開始します

事前チェック
  OK mainブランチで実行中
  OK 未コミットの変更なし
  OK リモートと同期済み

バージョン情報
  現在: 0.2.0
  新規: 1.0.0
  種別: major

WARNING: メジャーバージョンアップです。破壊的変更が含まれていることを確認してください。
続行しますか？ (y/n): y

...（以下同様）

PR作成
  OK PR作成完了
  URL: https://github.com/Kewton/PHOTON-RepoRAG/pull/XX

--- PRマージ後にタグ作成・クリーンアップを実行 ---
```

---

## エラー例

### 未コミットの変更がある場合

```
/release patch

リリーススキルを開始します

事前チェック
  NG 未コミットの変更があります

以下のファイルに変更があります:
  M baseline_reporag/tooling/mod.rs
  ?? new-file.rs

対処方法:
  1. 変更をコミット: git add . && git commit -m "..."
  2. 変更をスタッシュ: git stash
  3. 変更を破棄: git checkout .

リリースを中断しました。
```

### タグが既に存在する場合

```
/release patch

リリーススキルを開始します

タグチェック
  NG タグ v0.1.1 は既に存在します

対処方法:
  1. 別のバージョンを指定: /release 0.1.2
  2. 既存タグを削除（非推奨）: git tag -d v0.1.1 && git push origin :refs/tags/v0.1.1

リリースを中断しました。
```

### ビルド・テストが失敗する場合

```
/release patch

リリーススキルを開始します

品質チェック
  OK python -m pytest 成功
  NG python -m pytest 失敗（3 tests failed）

テスト結果:
  FAILED photon_mlx/tests/test_forward.py::web_search_validation_rejects_empty_query
  FAILED photon_mlx/tests/test_training.py::classured_response_parser_handles_web_search
  FAILED photon_mlx/tests/test_session.py::compact_command_summarizes_older_messages

対処方法:
  1. テストを修正してからリリースを再実行してください
  2. python -m pytest で詳細を確認

リリースを中断しました。
リリースブランチを削除: git checkout main && git branch -D release/v0.1.1
```

### PRのCIが失敗する場合

```
PR作成後、CIチェックが失敗した場合:

対処方法:
  1. リリースブランチに修正コミットを追加
     git checkout release/v0.1.1
     # 修正を行う
     git add . && git commit -m "fix: resolve CI failure for release v0.1.1"
     git push origin release/v0.1.1
  2. CIが再実行され、パスしたらPRをマージ
```
