---
model: sonnet
description: "アーキテクチャレビュー結果を設計方針書に反映"
---

# レビュー結果反映スキル

## 概要
アーキテクチャレビュー（`/architecture-review`）の指摘事項を設計方針書（`/design-policy`）に反映するスキルです。レビューで発見された改善点を体系的にドキュメントに適用します。

## 使用方法
- `/apply-review [Issue番号]`
- 「Issue #XXXのレビュー結果を設計方針書に反映してください」

## 前提条件
- `/architecture-review` が実行済みであること
- `/design-policy` で設計方針書が作成済みであること
- レビュー結果ファイルが存在すること

## 実行内容

あなたはシニアアーキテクトとして、レビュー結果を設計方針書に反映します。

### 1. レビュー結果の確認

レビュー結果ファイルを読み込みます：

```bash
cat workspace/issues/{issue_number}/architecture-review.md
```

### 2. 設計方針書の確認

現在の設計方針書を確認します：

```bash
cat workspace/issues/{issue_number}/design-policy.md
```

### 3. 反映項目の分類

レビュー指摘事項を以下のカテゴリに分類します：

| カテゴリ | 説明 | 優先度 |
|---------|------|--------|
| **設計原則違反** | SOLID/KISS/YAGNI/DRY違反 | 高 |
| **セキュリティ懸念** | サンドボックス脱出、コマンドインジェクション | 高 |
| **メモリ安全性** | eval使用、ライフタイム問題 | 高 |
| **構造改善** | モジュール構成、protocol設計 | 中 |
| **パフォーマンス** | 不要な clone、アロケーション | 低 |

### 4. 設計方針書への反映

各指摘事項について、設計方針書の該当セクションを更新します：

#### 4-1. アーキテクチャ変更

```markdown
## 変更履歴

### レビュー反映 (Issue #{issue_number})

| 項目 | 変更前 | 変更後 | 理由 |
|------|--------|--------|------|
| ... | ... | ... | レビュー指摘 #N |
```

#### 4-2. 設計判断の追記

新たな設計判断が必要な場合、トレードオフを明記して追記します。

#### 4-3. コーディングガイドライン更新

Python固有の規約変更がある場合、該当セクションを更新します：

```python
// 例: protocol設計の改善
// Before: 巨大なprotocolに全メソッドを集約
protocol Provider {
    def chat(self, ...) -> ...;
    def stream(self, ...) -> ...;
    def models(self) -> ...;
    def health_check(self) -> ...;
}

// After: 責務ごとにprotocolを分離
protocol ChatProvider {
    def chat(self, ...) -> ...;
}

protocol StreamProvider {
    def stream(self, ...) -> ...;
}
```

### 5. 品質チェック

反映後、設計方針書の整合性を確認します：

- [ ] 全セクションが矛盾なく更新されている
- [ ] 変更履歴が正しく記録されている
- [ ] PHOTON-RepoRAGのモジュール構成と整合している
- [ ] セキュリティ要件が適切に反映されている

### 6. 反映サマリー出力

```markdown
## レビュー反映サマリー

### Issue #{issue_number}

#### 反映結果

| カテゴリ | 指摘数 | 反映数 | 保留数 |
|---------|--------|--------|--------|
| 設計原則 | N | N | 0 |
| セキュリティ | N | N | 0 |
| 構造改善 | N | N | 0 |

#### 次のアクション

- [ ] `/work-plan` で作業計画を更新
- [ ] `/tdd-impl` でTDD実装を開始
- [ ] `/pm-auto-dev` で自動開発を実行
```

## 出力先

`workspace/issues/{issue_number}/design-policy.md`（更新）

## 完了条件

- レビュー指摘事項がすべて分類されている
- 設計方針書が更新されている
- 変更履歴が記録されている
- 反映サマリーが出力されている

## 関連コマンド

- `/architecture-review`: アーキテクチャレビュー実行
- `/design-policy`: 設計方針書作成
- `/multi-stage-design-review`: マルチステージ設計レビュー
- `/work-plan`: 作業計画立案
- `/tdd-impl`: TDD実装
- `/pm-auto-dev`: TDD自動開発
