# CHANGELOGエントリテンプレート

このテンプレートは、リリース時にCHANGELOG.mdを更新する際の形式を示します。

## バージョンセクションの形式

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- 新機能の説明 (Issue #XX)

### Changed
- 変更内容の説明 (Issue #XX)

### Deprecated
- 非推奨になった機能の説明

### Removed
- 削除された機能の説明

### Fixed
- バグ修正の説明 (Issue #XX)

### Security
- セキュリティ関連の修正
```

## セクションの説明

| セクション | 内容 |
|-----------|------|
| **Added** | 新機能 |
| **Changed** | 既存機能の変更 |
| **Deprecated** | 将来削除予定の機能 |
| **Removed** | 削除された機能 |
| **Fixed** | バグ修正 |
| **Security** | セキュリティ関連の修正 |

## 記載ルール

1. **過去形で記載**: 「追加した」「修正した」ではなく「追加」「修正」
2. **Issue番号を併記**: 可能な限り関連Issue番号を記載
3. **ユーザー視点で記載**: 技術的な詳細ではなく、ユーザーへの影響を記載
4. **BREAKING CHANGEの明示**: 破壊的変更は `**BREAKING**:` プレフィックスを付与

## 例

### 良い例

```markdown
### Added
- Web検索ツール `web.search` を追加（DuckDuckGo/SerperAPI対応） (Issue #6)
- `file.edit` ツールで部分的なファイル編集が可能に (Issue #9)

### Changed
- **BREAKING**: 設定ファイルのパスを `.photon-mlx/config` から `.photon-mlx/config.toml` に変更

### Fixed
- LLMレスポンスの出力が重複して表示される問題を修正 (Issue #1)
```

### 悪い例

```markdown
### Added
- web search added  <!-- 英語/小文字始まり -->
- baseline_reporag/tooling/__init__.py に WebSearch variant を追加  <!-- 技術的すぎる -->

### Fixed
- バグを修正  <!-- 具体性がない -->
```

## 比較リンクの形式

ファイル末尾に以下の形式で比較リンクを追加：

```markdown
[unreleased]: https://github.com/Kewton/PHOTON-RepoRAG/compare/vX.Y.Z...HEAD
[X.Y.Z]: https://github.com/Kewton/PHOTON-RepoRAG/compare/vA.B.C...vX.Y.Z
[A.B.C]: https://github.com/Kewton/PHOTON-RepoRAG/releases/tag/vA.B.C
```
