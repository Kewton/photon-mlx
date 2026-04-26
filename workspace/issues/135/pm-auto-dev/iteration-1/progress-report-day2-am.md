# Issue #135 Day 2 朝 進捗報告 (Phase 0-1 完了)

**Issue**: feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合
**実行日**: 2026-04-26 (Day 2 AM, Day 1 から継続)
**フェーズ**: Phase 0 + Phase 1 全タスク完了
**ブランチ**: feature/issue-135-photon-retrain (Day 1 + Day 2 = 6 commits)

## 完了タスク (Day 2 朝)

### Task 1.3: S7-001 解消 (commit `2dbf458`)

**問題**: `baseline_reporag/photon_pipeline.py:_build_photon_deps` は checkpoint を load していなかった。**現 PHOTON eval は random-init weights で動作**。Codex Stage 7 が指摘した最大破壊リスク。

**実装**: `model.checkpoint_path` YAML 設定を新規追加。
- 設定あり → `photon_mlx.checkpoint.load_checkpoint(model, path)` で weights load
- 設定なし → WARNING ログ + random-init で継続 (dev/test 互換)
- 不在 path → FileNotFoundError raise

**追加テスト 3 件** (`TestBuildPhotonDepsCheckpointLoad`):
- `test_unset_checkpoint_path_warns` — WARNING 出力検証
- `test_valid_checkpoint_loads_weights` — embedding 一致検証 (max abs diff < 1e-6)
- `test_missing_checkpoint_path_raises` — FileNotFoundError 検証

### Task 0.4: smoke test 実装 (commit `1c920ae`)

`baseline_reporag/tests/test_photon_checkpoint_smoke.py` 新規。`_build_photon_deps` factory path で end-to-end の checkpoint load を検証。Task 1.3 が unit レベル、こちらが factory wiring レベル。

**追加テスト 2 件** (`TestPhotonCheckpointSmoke`):
- `test_factory_loads_checkpoint_from_yaml` — YAML 経由の load 完全一致検証
- `test_factory_warns_when_checkpoint_path_unset` — 警告チャネル疎通

### Task 1.4: lazy MLX import 規約確認 (commit `f17c3a6`)

`pipeline_factory.py` は MLX 不依存を維持していることを直接検証 + boundary regression test 追加。

**追加テスト 1 件** (`test_build_photon_deps_does_not_import_trainer`):
- subprocess で `_build_photon_deps` を呼び、`photon_mlx.trainer` / `photon_mlx.loss` / `mlx.optimizers` が sys.modules に登場しないことを assert
- subprocess 採用理由: in-process だと sys.modules 操作が他テスト (test_training の monkeypatch) と衝突するため

DR1-002 / DR3-001 / Issue #135 Phase 1 の境界規約 (commit `ea2fa57`) を physical 検証で固定。

### Task 0.2: pre-commit hook 設定 (commit `8f1672d`)

`.pre-commit-config.yaml` 新規。DR4-006 (機密漏洩防止) を満たす:
- **detect-secrets v1.5.0**: API キー / token の commit 阻止
- **ruff** (lint + autofix) + **ruff-format**: CLAUDE.md 品質基準と整合

ユーザーが `pip install pre-commit && pre-commit install` で活性化。CI 動作は不変 (purely additive)。

## 品質チェック結果

| チェック項目 | コマンド | 結果 |
|-------------|----------|------|
| ruff | `ruff check baseline_reporag/photon_pipeline.py baseline_reporag/tests/` | ✅ Pass |
| ruff format | `ruff format --check ...` | ✅ Pass |
| pytest 全体 | `pytest baseline_reporag/ photon_mlx/ tests/ torch_ref/tests/` | ✅ **1056 passed**, 2 skipped, 2 failed (既知 pre-existing failure) |
| 新規追加テスト | TestBuildPhotonDepsCheckpointLoad + smoke + boundary | ✅ **6 件すべて pass** |

既知の 2 件失敗 (`tests/test_generate_training_corpus.py`) は CLAUDE.md 明記の pre-existing failure、本変更由来でない。

## コミット履歴 (Day 1 + Day 2)

```
8f1672d chore: add pre-commit config with detect-secrets (#135 / DR4-006)
f17c3a6 test(photon_pipeline): pin DR1-002 boundary in subprocess (#135 / Task 1.4)
1c920ae test(photon_pipeline): add Day 1 checkpoint load smoke test (#135 / Task 0.4)
2dbf458 fix(photon_pipeline): load checkpoint in _build_photon_deps (#135 / S7-001)
57d7742 docs(issue-135): pm-auto-issue2dev 完了報告 — design + work-plan + multi-stage review
ea2fa57 refactor(photon_mlx): extract checkpoint I/O into checkpoint.py (#135)
```

## 重要な発見 (Day 2 朝)

### 1. S7-001 が**実際に発生していた**

設計レビュー Stage 7 で Codex が指摘した「PHOTON eval が random-init weights で動作」は理論上の懸念ではなく**実態**だった。`_build_photon_deps` には checkpoint load 経路が一切無かった。Day 2 で正式対応 (commit `2dbf458`)。

設計方針書 §11 リスクトップ「最大破壊リスク」が予言通り顕在化したケース。Day 1 smoke test で発見できたので、本格再学習着手前に解消できた。

### 2. boundary test の subprocess 化

`sys.modules` から module を pop すると、他テストの `monkeypatch.setattr` が古い module 参照を持ったままになり連鎖失敗する。subprocess 分離が必須だった (commit `f17c3a6`)。

これは photon_mlx ↔ baseline_reporag の境界保護を**実コードレベルで毎 CI 検証**できるようになった意味で重要。将来 trainer / loss を photon_pipeline に再導入する commit は、この test で blocked される。

## 残タスク (本セッション以降)

### CPU で完結 (次セッションで実施可)

- **Phase 2 corpus 生成スクリプト本体作成** (実行 = LLM 呼び出しは保留): `scripts/_corpus_core.py` + `scripts/generate_institutional_training_corpus.py` 新設
- **Phase 3 データロード拡張**: `photon_mlx/data.py` に `iterate_mixed_batches` 追加
- **Phase 4 Config 分離**: `configs/institutional_docs_photon_retrain.yaml` 新設、`TrainingConfig.train_corpora_mix` / `val_split` 追加
- **Phase 5 テスト追加**: 22-30 件
- **Phase 4 DR4-003**: `photon_mlx/checkpoint.py` に integrity.json (SHA-256) 書込・照合追加

### GPU 必須 (#137 完了後 + ユーザー承認)

- **Phase 6 学習実行 (Day 3-4)**: 17-33 GPU-hours、10K/15K/20K step
- **Phase 7 eval / 採用判定 (Day 4-5)**: 2-run × 3 ckpt + 境界帯 4-run
- **Phase 8 ロールアウト (Day 5)**: checkpoint 保存、報告書、CLAUDE.md / roadmap 更新

### Phase 2 corpus 生成 — 実行のみ #137 完了待ち

スクリプト本体 (`scripts/generate_institutional_training_corpus.py`) は CPU で作成可能。**実行 (mlx_lm 呼び出し)** は #137 Phase B eval が GPU 占有中のため保留。

## 結論

Day 1 + Day 2 朝で **設計方針書の Phase 0 + Phase 1 全タスクを完了**。

- DR1-002 / DR3-001 (checkpoint module 抽出 + 境界規約) を実装+テスト固定
- DR4-006 (pre-commit / detect-secrets) を準備
- **S7-001 (PHOTON eval random-init bug) を発見+解消** ← Codex の事前予言が的中、Day 1 smoke test で発覚

合計 6 commits、新規追加テスト 14 件、品質チェック regression 0。次セッションは Phase 2-4 (CPU タスク) を継続するか、#137 完了後に GPU 必須の Phase 6 へ進む。
