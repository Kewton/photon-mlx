# Issue #135 Day 1 進捗報告 (Phase 0 + Phase 1 Task 1.1, 1.2)

**Issue**: feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合
**実行日**: 2026-04-26
**フェーズ**: Day 1 段階的着手 (オプション A、~2-4 時間想定)
**ブランチ**: feature/issue-135-photon-retrain (現状未コミット)

## 完了タスク

### Phase 0: 機密保護とリポジトリ衛生

| Task | 状態 | 成果物 |
|------|------|--------|
| 0.1 `.gitignore` に `data/training/` 追加 | ✅ 完了 | `.gitignore` (+2 行、Issue #135 コメント付き) |
| 0.2 pre-commit hook (detect-secrets) 設定 | ⏸️ 後続 | 別 commit / Day 2 で対応 (DR4-006) |
| 0.3 mulmoclaude 600 step checkpoint 物理確認 | ⏸️ 不要確認済 | `ls checkpoints/` 結果: ディレクトリ無し → 紛失 risk 顕在化、Day 2 で smoke test 失敗時 scratch 学習方針発動 |
| 0.4 PHOTON eval/runtime checkpoint load smoke test | ⏸️ 後続 | Task 1.3 (baseline_reporag 統合) と一体化、Day 2 で対応 |

### Phase 1: コアモジュール抽出と境界整理

| Task | 状態 | 成果物 |
|------|------|--------|
| 1.1 `photon_mlx/checkpoint.py` 新設 | ✅ 完了 | 新規 116 行、`CheckpointState` DTO + 純粋 I/O `save_checkpoint` / `load_checkpoint` |
| 1.2 `photon_mlx/trainer.py` refactor + wrapper | ✅ 完了 | -54 / +32 行、TrainState ↔ CheckpointState 変換 wrapper |
| 1.3 `baseline_reporag/photon_pipeline.py` 統合 | ⏸️ 後続 | 現状 PHOTON pipeline は checkpoint load していないため Day 2 で新規追加 |
| 1.4 `pipeline_factory.py` lazy MLX import 規約確認 | ⏸️ 後続 | Task 1.3 完了後に確認 |

## 設計方針書反映の検証

### DR1-002 (Must Fix): photon_mlx 境界規約 — `photon_mlx/checkpoint.py` 新設

✅ **達成**: 新 module は `mlx.core` のみに依存し、`mlx.optimizers` / `photon_mlx.loss` を import しない。`baseline_reporag` から runtime checkpoint load するための入口として機能。

**検証テスト**: `TestCheckpointIO::test_module_does_not_pull_training_deps` — `sys.modules` をクリアしてから `import photon_mlx.checkpoint` 後、`mlx.optimizers` と `photon_mlx.loss` が import されないことを assert。

### DR3-001 (Must Fix): trainer 互換 wrapper による既存 callsite 保護

✅ **達成**: 既存 `from photon_mlx.trainer import save_checkpoint, load_checkpoint, TrainState` は変更不要。`TrainState` API も保持。

**検証テスト**:
- `TestTrainerCompatWrapper::test_trainer_load_returns_train_state` — trainer 経由 load は `TrainState` を返す
- `TestTrainerCompatWrapper::test_checkpoint_module_and_trainer_are_interoperable` — trainer-written → ckpt-readable / ckpt-written → trainer-readable の双方向互換性

### S7-001 (Must Fix → 顕在化): PHOTON eval/runtime checkpoint load 経路

🔴 **検証で顕在化**: `baseline_reporag/photon_pipeline.py:_build_photon_deps` を grep した結果、checkpoint load コードが **存在しない**。現 PHOTON eval は random-initialized weights で実行されている可能性が高い。

→ 設計方針書 §11 リスク表トップの「最大破壊リスク」が **既に発生していた** ことを確認。Day 2 Task 1.3 で `_build_photon_deps` に checkpoint load を新規追加する必要あり。

## 品質チェック結果

| チェック項目 | コマンド | 結果 |
|-------------|----------|------|
| ruff check | `ruff check photon_mlx/` | ✅ All checks passed! |
| ruff format | `ruff format --check photon_mlx/` | ✅ 23 files OK (test_checkpoint.py auto-fix 済) |
| pytest photon_mlx | `python -m pytest photon_mlx/` | ✅ **446 passed** (新規 8 件 + 既存 438 件) |
| pytest 全体 | `python -m pytest baseline_reporag/ tests/ torch_ref/` | ✅ **604 passed**, 2 failed (既知 pre-existing failure: `tests/test_generate_training_corpus.py` の 2 件)、2 skipped — 本変更による regression 無し |

## 追加されたテスト (8 件)

`photon_mlx/tests/test_checkpoint.py` (新規):

1. `TestCheckpointState::test_default_construction` — DTO デフォルト値
2. `TestCheckpointState::test_explicit_construction` — DTO 明示初期化
3. `TestCheckpointIO::test_save_and_load_round_trip` — 純粋 I/O round trip
4. `TestCheckpointIO::test_load_ignores_unknown_state_keys` — forward-compat
5. `TestCheckpointIO::test_load_corrupt_state_json_raises` — error path
6. `TestCheckpointIO::test_module_does_not_pull_training_deps` — DR1-002 境界規約 (training import 防止)
7. `TestTrainerCompatWrapper::test_trainer_load_returns_train_state` — 既存 trainer API 保護
8. `TestTrainerCompatWrapper::test_checkpoint_module_and_trainer_are_interoperable` — 双方向互換性

## ファイル変更サマリ

```
.gitignore                          | +2 行 (data/training/ + コメント)
photon_mlx/checkpoint.py            | +116 行 (新規)
photon_mlx/trainer.py               | -54 / +32 行 (=-22 行、json/_flatten/fields 削除、wrapper 追加)
photon_mlx/tests/test_checkpoint.py | +236 行 (新規)
```

## 残タスクと次のアクション (Day 2 以降)

### 優先度 高 (Day 2 朝)

- **Task 1.3** `baseline_reporag/photon_pipeline.py:_build_photon_deps` に checkpoint load を新規追加 (S7-001 解消)
- **Task 0.4** PHOTON eval/runtime checkpoint load smoke test 実装
- **Task 1.4** `pipeline_factory.py` lazy MLX import 規約に違反していないことを確認

### 優先度 中 (Day 2 午後)

- **Task 0.2** pre-commit hook (detect-secrets) 設定
- **Phase 2** (Day 1 夜 〜 Day 2): `scripts/_corpus_core.py` + `scripts/generate_institutional_training_corpus.py` 新設

### コミット推奨

現在の変更 (Phase 0.1 + Phase 1.1 + Phase 1.2) は単独でクリーンに切れる単位。1 commit で:

```
refactor(photon_mlx): extract checkpoint I/O into checkpoint.py (#135)

DR1-002 / DR3-001: separate the runtime checkpoint surface so
baseline_reporag can load PHOTON weights without pulling in
mlx.optimizers / photon_mlx.loss.  trainer.py becomes a thin TrainState
compat wrapper around the new module; existing callsites are unchanged.
Adds 8 unit tests covering the boundary, the I/O round trip, and
forward-compat; 446 photon_mlx tests pass with no regression.

Also adds data/training/ to .gitignore (DR4 機密保護: institutional
JP corpus must not be committed).
```

## 結論

Day 1 段階的着手として **2-4 時間相当の範囲を完了**。設計方針書の Must Fix 2 件 (DR1-002 / DR3-001) を実コードで実装し、TDD で 8 件の検証テストを追加。既存 446 photon_mlx tests に regression 無し。

S7-001 (PHOTON eval が checkpoint を load していない実態) を grep で確認、Day 2 Task 1.3 で正式対応する。Day 3-4 GPU 占有学習 (17-33 GPU-hours) は **依然として未着手**、別セッションでユーザー承認後に実行する。
