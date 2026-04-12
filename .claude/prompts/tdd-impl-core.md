# TDD実装コアプロンプト

## 原則: Red-Green-Refactor

テスト駆動開発の3ステップを厳密に守ること。

### Step 1: Red（テストを書く）

1. 実装する機能の仕様を明確にする
2. その仕様を検証するテストを書く
3. `python -m pytest` を実行し、テストが**失敗する**ことを確認する

```bash
python -m pytest torch_ref/tests/ photon_mlx/tests/ -v 2>&1 | tail -20
```

**重要**: テストが失敗しない場合は、テストが正しく書かれているか確認すること。

### Step 2: Green（最小限の実装）

1. テストを通す**最小限**のコードを書く
2. `python -m pytest` を実行し、テストが**成功する**ことを確認する

```bash
python -m pytest torch_ref/tests/ photon_mlx/tests/ -v
```

**重要**: この段階では完璧なコードを目指さない。テストが通ることだけに集中する。

### Step 3: Refactor（リファクタリング）

1. テストが通る状態を維持しながらコードを改善する
2. 以下を実行して品質を確認：

```bash
ruff format .
ruff check .
python -m pytest torch_ref/tests/ photon_mlx/tests/ -v
```

## Rustコーディング規約

### 構造
- モジュールは責務ごとに分割
- `pub` は必要最小限に
- `__init__.py` よりファイル名でのモジュール定義を優先

### エラーハンドリング
- `unwrap()` は本番コードで使用禁止（テストコードは可）
- `thiserror` または独自エラー型を使用
- `Result<T, E>` を適切に返す
- `?` 演算子を活用

### テスト
- ユニットテストは対象モジュール内の ` mod tests` に配置
- 統合テストは `photon_mlx/tests/` ディレクトリに配置
- テスト名は `test_` プレフィックス + 動作を説明する名前
- `#[should_panic]` よりも `Result` ベースのテストを優先
- テストヘルパーは共通モジュールにまとめる

### 命名規則
- 関数・変数: `snake_case`
- 型・トレイト: `PascalCase`
- 定数: `SCREAMING_SNAKE_CASE`
- ライフタイム: 短い小文字（`'a`, `'b`）

### ドキュメント
- 公開APIには `///` ドキュメントコメントを付ける
- 例（`# Examples`）を含める

## サイクルの繰り返し

1機能 = 1サイクル（Red→Green→Refactor）を繰り返す。
1サイクルが完了したら次の機能に進む。

すべてのサイクル完了後に最終検証：

```bash
ruff format --check .
ruff check .
python -m pytest torch_ref/tests/ photon_mlx/tests/ -v
python -m pytest
```
