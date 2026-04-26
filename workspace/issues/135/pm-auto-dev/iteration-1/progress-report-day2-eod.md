# Issue #135 Day 2 EOD 進捗報告 (CPU 範囲 全完了)

**Issue**: feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合
**実行日**: 2026-04-26 (Day 2 EOD、Day 2 PM から継続)
**フェーズ**: Phase 0-4 + trainer 分岐 + DR4-001 hardening 全完了
**ブランチ**: feature/issue-135-photon-retrain (Day 1 + Day 2 = **15 commits**)
**作業範囲**: CPU only、LLM 実呼び出し / GPU 学習は禁止

## 完了タスク (Day 2 EOD、2 commits)

### Task A: trainer.py 混合経路分岐 (commit `34a833a`)

`photon_mlx/trainer.py::train` に `t_cfg.train_corpora_mix is not None` 分岐を追加。Phase 3 (`iterate_mixed_batches`) と Phase 4-1 (TrainingConfig schema) を結合。

- **設定ありの場合**: `iterate_mixed_batches` 呼び出し、`val_split>0` で tuple unpack、`val_split=0` で legacy `t_cfg.val_corpus` を任意で使用
- **設定なしの場合**: 既存 `iterate_batches` 単一 corpus 経路を維持 (regression なし)
- **`approved_roots` test hook**: 本番では `None` → `data/training/`+`data/processed/` のデフォルト allow-list、テストは tmp パスを explicit 渡し

**検証点**: cfg-side 検証 (`TrainingConfig.__post_init__` DR1-003) と loader-side 検証 (`iterate_mixed_batches` DR4-002) は二重ガード、yaml 経由 → cfg、直接呼出 → loader 共に reject。

**テスト**: 2 件 (`TestTrainerMixedDispatch::test_train_uses_mixed_batches_when_mix_set` + `_legacy_path_when_mix_unset`)

### Task B: DR4-001 generate script CLI hardening (commit `344caae`)

`scripts/generate_institutional_training_corpus.py` に DR4-001 を完全実装。新公開 helper 2 件 + 定数 3 件。

- **`parse_validated_args(argv, *, approved_corpus_roots, approved_output_roots)`**:
  - `--corpus-dir`: `resolve(strict=True)` + approved root 配下確認 (本番: institutional_documents mount)
  - `--output`: parent dir resolve + approved root 配下確認 (本番: `./data/training`)
  - `--sessions`: `[1, MAX_SESSIONS=5000]` 範囲外で `ValueError` (タイポ起因の多日 LLM 暴走防止)
  - `--seed`: `[0, 2**32)` 範囲外で `ValueError`
  - `--val-ratio`: `(0.0, 0.5)` 範囲外で `ValueError`
- **`write_atomic(path, content, *, mode=0o600)`**:
  - `os.open(O_CREAT, mode=0o600)` で tmp file 作成 → fdopen で書込 → `os.replace` で atomic rename
  - 成功時 tmp は残らない、失敗時は cleanup
  - 0o600 = owner read/write only (機密 institutional 文書の transit 中保護)
- **定数**: `MAX_SESSIONS`, `DEFAULT_APPROVED_CORPUS_ROOTS`, `DEFAULT_APPROVED_OUTPUT_ROOTS` (テストが kwarg で上書き可能)
- **`main()`**: `parse_validated_args` + `write_atomic` 経由に書換、対外動作は valid input で不変

**テスト**: 8 件 (TestCLIArgValidation 6 / TestAtomicWrite 2):
- `test_corpus_dir_must_exist` (FileNotFoundError)
- `test_corpus_dir_outside_approved_root_rejected` (ValueError)
- `test_sessions_upper_bound` / `test_sessions_must_be_positive`
- `test_val_ratio_bounds` (4 値 parametrise: -0.1 / 0.0 / 0.5 / 0.9)
- `test_valid_args_pass` (round-trip)
- `test_atomic_write_creates_file_with_restricted_permissions` (0o600 検証)
- `test_atomic_write_does_not_leave_tmp_on_success`

## 品質チェック結果

| チェック項目 | コマンド | 結果 |
|-------------|----------|------|
| ruff | `ruff check` 全変更ファイル | ✅ All checks passed! |
| ruff format | `ruff format --check` 全変更ファイル | ✅ Pass |
| pytest 全体 | `pytest baseline_reporag/ photon_mlx/ tests/ torch_ref/tests/` | ✅ **1114 passed**, 2 既知 pre-existing failure (本変更由来でない) |

| テスト累計 | 件数 |
|----------|------|
| Day 1 終了時 | 1056 passed |
| Day 2 PM 終了時 | 1104 passed (+48) |
| Day 2 EOD 終了時 | **1114 passed** (+10、本セッション追加分) |

## 累計コミット履歴 (Day 1 + Day 2 = 15 commits)

```
344caae feat(scripts): DR4-001 CLI hardening for training corpus generator (#135)
34a833a feat(photon_mlx/trainer): dispatch to iterate_mixed_batches when mix set
d0277e3 docs(issue-135): Day 2 PM 進捗報告
f25022c feat(scripts): training corpus generator scaffolding (#135 / Phase 2)
87802fb feat(configs): institutional_docs_photon_retrain.yaml for #135 Phase 4-3
a2a9d5b feat(photon_mlx/checkpoint): integrity.json SHA-256 verification (DR4-003)
587930c feat(torch_ref/config): add train_corpora_mix + val_split (Phase 4-1)
397b0bb feat(photon_mlx/data): add iterate_mixed_batches for JP/EN corpus mix
ecd1c2a docs(issue-135): Day 2 AM 進捗報告 — Phase 0-1 全タスク完了
8f1672d chore: add pre-commit config with detect-secrets (#135 / DR4-006)
f17c3a6 test(photon_pipeline): pin DR1-002 boundary in subprocess (Task 1.4)
1c920ae test(photon_pipeline): add Day 1 checkpoint load smoke test (Task 0.4)
2dbf458 fix(photon_pipeline): load checkpoint in _build_photon_deps (S7-001)
57d7742 docs(issue-135): pm-auto-issue2dev 完了報告
ea2fa57 refactor(photon_mlx): extract checkpoint I/O into checkpoint.py (#135)
```

## 設計方針書反映の最終状況

**Must Fix / Should Fix / Nice to Have すべて実装+テスト固定済**:

| 項目 | 反映状況 | Commit |
|------|---------|--------|
| **S7-001** PHOTON eval random-init | ✅ Critical fix | `2dbf458` |
| **DR1-001** generate script 3 段分割 | ✅ | `f25022c` |
| **DR1-002** photon_mlx/checkpoint.py 抽出 | ✅ | `ea2fa57` |
| **DR1-003** strict validation | ✅ data + config 二重 | `397b0bb` + `587930c` |
| **DR1-005** val_split (val_corpora_mix 廃止) | ✅ | `587930c` + `87802fb` |
| **DR1-006** list 戻り型 | ✅ | `397b0bb` |
| **DR1-007** sequence-level mixing | ✅ test で固定 | `397b0bb` |
| **DR2-008** train/val tuple unpack | ✅ | `397b0bb` + `34a833a` |
| **DR2-009** min_lr=3e-6 必須 | ✅ yaml で固定 | `87802fb` |
| **DR3-001** trainer.py 互換 wrapper | ✅ | `ea2fa57` |
| **DR3-002** 外部 LLM provider 制限 | ✅ default qwen + comment | `87802fb` + `f25022c` |
| **DR4-001** CLI hardening | ✅ | `344caae` |
| **DR4-002** path security + token validation | ✅ | `397b0bb` |
| **DR4-003** integrity.json SHA-256 | ✅ | `a2a9d5b` |
| **DR4-005** eval set リーク検出 | ✅ | `f25022c` |
| **DR4-006** pre-commit hook | ✅ | `8f1672d` |
| **S3-007 / DR3-007** retrain yaml 分離 | ✅ | `87802fb` |
| **S5-001** cosine_decay 単体経路 | ✅ yaml + comment | `87802fb` |
| **S5-002** 累計 step | ✅ max_steps=10600 | `87802fb` |
| **S5-004** tokens/step 65,536 | ✅ | `87802fb` |
| **S5-005** sequence pool 独立性 | ✅ test で固定 | `397b0bb` |

## Day 2 EOD で残るタスク (GPU 必須)

CPU のみで完結する作業はすべて完了。残るは:

1. **`scripts/generate_institutional_training_corpus.py main()` 実行** (LLM 必須):
   - `--corpus-dir /Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents`
   - `--output ./data/training/institutional/train_jp.jsonl`
   - `--eval-set ./data/eval_sets/institutional_multi_turn_eval.jsonl`
   - `--sessions 2000` `--val-ratio 0.05` `--seed 42`
   - `--provider qwen` (mlx local default)
   - 想定時間: 数時間 (qwen mlx local on M3 Ultra)

2. **mulmoclaude `train_en.jsonl` 取得**: 既存 600 step 学習で使った en corpus を `data/training/mulmoclaude/` へ配置 (見つからない場合は scratch 学習 risk を発動)

3. **Phase 6 学習実行 (Day 3-4)**: 17-33 GPU-hours、`configs/institutional_docs_photon_retrain.yaml` で `python -m photon_mlx.trainer` 実行 (10K/15K/20K step 候補)

4. **Phase 7 eval (Day 4-5)**: 各 checkpoint で `scripts/run_multi_turn_eval.py` MT eval、Turn 5-6 NC < 6% 達成 checkpoint を採用

5. **Phase 8 ロールアウト (Day 5)**:
   - 採用 checkpoint を `checkpoints/photon_institutional_step<NNNNNN>_<yyyymmdd>/` へ保存
   - `configs/institutional_docs_photon_retrain.yaml` の `model.checkpoint_path` 確定 (現在コメントアウト中)
   - `reports/institutional_photon_mt_eval_v2.md` / `gate2_v5_post_retrain.md` 新規作成
   - `CLAUDE.md` / `workspace/mvp/roadmap.md` メトリクス更新

## 結論

**CPU で完結するすべての設計方針書反映タスクが完了**。

- 累計 15 commits、新規追加テスト 72 件 (Day 1 14 + Day 2 PM 48 + Day 2 EOD 10)
- 設計方針書 Must Fix / Should Fix / Nice to Have **全 21 項目** を実装+テスト固定
- 1114 passed、本変更由来の regression 0
- S7-001 (PHOTON eval random-init) を Day 2 AM で発見+解消 — Codex 事前予言が的中
- 旧 `iterate_batches` 経路は完全に維持、既存 configs 非破壊

本セッションは **idle 状態** に移行します。次回再開時は #137 完了 + ユーザー承認後に GPU 必須の Phase 6-8 (corpus 生成実行 → 学習 → eval → ロールアウト) を実施します。
