---
model: opus
description: "受け入れテストを作成・実行"
---

# 受け入れテストスキル

## 概要
Issueの受け入れ基準に基づいてテストを作成し、実行します。

## 使用方法
- `/acceptance-test [Issue番号]`

## 実行内容

**共通プロンプトを読み込んで実行します**:

```bash
cat .claude/prompts/acceptance-test-core.md
```

↑ **このプロンプトの内容に従って、受け入れテストを実行してください。**

## サブエージェントモード

サブエージェントとして呼び出す場合：

```
Use acceptance-test-agent to verify Issue #XXX acceptance criteria.
```

## 完了条件

以下をすべて満たすこと：
- 受け入れ基準ごとにテストが作成されている
- すべてのテストがパス（`python -m pytest torch_ref/tests/ photon_mlx/tests/ -v`）
- `ruff check .` で警告ゼロ
- テスト結果のサマリーが出力されている
