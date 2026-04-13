# オーケストレーション実行計画

**日付**: 2026-04-13
**対象Issue**: #1

## 対象Issue一覧

| Issue | タイトル | 種別 | ラベル |
|-------|---------|------|--------|
| #1 | No-citation rate 35% の改善 | FEATURE | なし |

## 前提条件

- Issue #7 (retrieval 品質修正) マージ済み
- no-citation rate: 35% → 25% (static 20問) に改善済み
- 残りの no-citation は generation 側（[C:N] フォーマット遵守不足）が主因

## 影響ファイル

| Issue | 影響ファイル |
|-------|-------------|
| #1 | `baseline_reporag/generation/prompt.py`, `baseline_reporag/citation.py`, `reports/failure_cases.md` |

## 依存関係グラフ

```
#7 (retrieval 修正) ✅ 完了
 └─→ #1 (本Issue) ← 今回実行
      └─→ #2 (full eval)
```

## Worktree計画

| Issue | ブランチ名 | Worktreeパス |
|-------|-----------|-------------|
| #1 | `feature/1-no-citation-rate` | `../photon-mlx-feature-1-no-citation-rate` |
