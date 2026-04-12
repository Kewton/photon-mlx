# 受け入れテストコアプロンプト

## 目的
Issueの受け入れ基準（Acceptance Criteria）を検証するテストを作成・実行する。

## 手順

### 1. 受け入れ基準の抽出

Issue本文から受け入れ基準を特定し、テスト可能な形に分解する。

受け入れ基準が明示されていない場合は、Issue本文から推測し、リスト化する。

### 2. テスト設計

各受け入れ基準に対して：

- **正常系テスト**: 期待される動作の確認
- **異常系テスト**: エラーケース、境界値の確認
- **エッジケース**: 空入力、大量データ、特殊文字など

### 3. テストの配置

```
photon_mlx/tests/
  acceptance/
    __init__.py          # モジュール定義
    test_issue_XXX.py    # Issue別のテストファイル
```

`pyproject.toml` にテスト用の依存関係が必要な場合は `[dev-dependencies]` に追加する。

### 4. テスト実装パターン

```python

mod acceptance_tests {
    use super::*;

    /// AC-1: [受け入れ基準の説明]
    @pytest.mark
    def test_ac1_description() {
        // Given: 前提条件のセットアップ
        // When: 操作の実行
        // Then: 結果の検証
    }
}
```

### 5. 実行と検証

```bash
python -m pytest torch_ref/tests/ photon_mlx/tests/ -v
ruff check .
```

### 6. 結果報告

以下のフォーマットで結果をまとめる：

| # | 受け入れ基準 | テスト名 | 結果 |
|---|------------|---------|------|
| AC-1 | ... | `test_ac1_...` | PASS/FAIL |
| AC-2 | ... | `test_ac2_...` | PASS/FAIL |

## 品質基準

- すべての受け入れ基準にテストが対応している
- テストは独立して実行可能（依存関係なし）
- テスト名から何を検証しているか理解できる
- Given-When-Then パターンに従っている
