---
model: opus
description: "テスト駆動開発で高品質なPythonコードを実装"
---

# TDD実装スキル

## 概要
テスト駆動開発（Test-Driven Development）の手法に従って、高品質なPythonコードを実装するスキルです。

## 使用方法
- `/tdd-class [機能名]`

## 実行内容

**共通プロンプトを読み込んで実行します**:

```bash
cat .claude/prompts/tdd-impl-core.md
```

↑ **このプロンプトの内容に従って、TDD実装を実行してください。**

## 完了条件

以下をすべて満たすこと：
- すべてのテストが成功（Red → Green サイクル完了）
- `ruff check .` で警告ゼロ
- `python -m pytest` で全テストパス
- コミットが完了

## サブエージェントモード

サブエージェントとして呼び出す場合は、PM Auto-Devが以下のように実行します：

```
Use tdd-impl-agent to implement Issue #XXX with TDD approach.
```
