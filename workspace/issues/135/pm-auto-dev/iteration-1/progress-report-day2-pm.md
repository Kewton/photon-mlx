# Issue #135 Day 2 PM 進捗報告 (Phase 2-5 完了)

**Issue**: feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合
**実行日**: 2026-04-26 (Day 2 PM、Day 2 AM から継続)
**フェーズ**: Phase 2 + Phase 3 + Phase 4 (3 サブ) 完了
**ブランチ**: feature/issue-135-photon-retrain (Day 1 + Day 2 = 12 commits)
**作業範囲**: CPU only、LLM 実呼び出し / GPU 学習は禁止

## 完了タスク (Day 2 PM、5 commits)

### Phase 3: iterate_mixed_batches (commit `397b0bb`)

`photon_mlx/data.py` に新関数を追加。既存 `iterate_batches` の signature/挙動は完全に不変 (S3-001 後方互換)。

- **DR1-003 strict validation**: 空 dict / 重み <= 0 / 重み非有限 / sum(weights) ≠ 1.0 (±1e-6) — すべて `ValueError`
- **DR1-005 / DR2-008 戻り値**: `val_split=0` で `list[mx.array]`、`val_split>0` で `(train, val) tuple`
- **S5-005 pool 独立性**: 各 corpus を独立に pack — `pack_sequences` が corpus 境界をまたがない
- **DR1-007 sequence-level mixing**: 制御目標 = sequence ratio、token ratio は実測のみ
- **DR4-002 path security**: `Path.resolve(strict=True)` + `approved_roots` 配下チェック (default `data/training/` + `data/processed/`)
- **DR4-002 token validation**: 非空 list[int]、`0 <= tok < vocab_size`、長さ ≤ context_length × 4、行サイズ ≤ 2 MiB

**テスト**: 17 件 (TestStrictValidation 6 / TestReturnType 2 / TestSequenceMixing 3 / TestPathSecurity 2 / TestTokenValidation 4)

### Phase 4-1: TrainingConfig train_corpora_mix / val_split (commit `587930c`)

`torch_ref/config.py::TrainingConfig` に 2 フィールド追加。`__post_init__` で **DR1-003 strict validation**。

- `train_corpora_mix: dict[str, float] | None = None` — None なら従来 `train_corpus` 単一 path に fallback
- `val_split: float = 0.0` — DR1-005 簡素化版 (val_corpora_mix dict は廃止)

**検証**: 空 dict / 非数値 weight / 非有限 / <= 0 / sum off-target はすべて `ValueError`。`val_split` も `[0, 1)` 範囲外で raise。

**テスト**: 12 件 (TestTrainingConfigCorporaMix)

### Phase 4-2: DR4-003 integrity.json (commit `a2a9d5b`)

`photon_mlx/checkpoint.py` に SHA-256 ベースの改竄検出を追加。

- `save_checkpoint` が `integrity.json` を atomic write (tmp + os.replace)
- `load_checkpoint` がデフォルトで verify (legacy checkpoint 互換: missing → WARNING + load 継続)
- `verify_integrity=True` 引数で strict mode (missing → raise)
- weights.npz / state.json の hash 不一致は常に `ValueError` (warn-and-load しない)

**注意点**: 既存 `test_load_ignores_unknown_state_keys` (state.json 改変後 load) も integrity check に引っかかったため、forward-compat シナリオの test を「再 stamp 後 load」に更新 (legitimate な未来 trainer の挙動を模倣)。

**テスト**: 6 件 (TestCheckpointIntegrity)

### Phase 4-3: configs/institutional_docs_photon_retrain.yaml (commit `87802fb`)

旧 `configs/institutional_docs_photon.yaml` (`#112/#126` ベースライン保持) をコピーして retrain 専用 yaml を新設。

- `train_corpora_mix`: `{train_jp.jsonl: 0.5, train_en.jsonl: 0.5}` (DR1-003 sum=1.0)
- `val_split: 0.05` (DR1-005)
- `learning_rate: 3e-5`, `min_learning_rate: 3e-6`, `warmup_ratio: 0.0` (S5-001 cosine_decay 単体経路)
- `micro_batch_size: 2`, `gradient_accumulation_steps: 16` (S5-004 effective batch=32, tokens/step=65,536)
- `max_steps: 10600` (S5-002 累計 step、既存 600 + 10K 追加)
- `model.checkpoint_path`: 未確定 (Day 5 採用判定後に埋める)

**テスト**: 1 件 (`test_institutional_retrain_yaml_loads_with_expected_hyperparams` で値ベース固定)

### Phase 2: corpus 生成スクリプト本体 (commit `f25022c`)

`scripts/_corpus_core.py` (pure helpers) + `scripts/generate_institutional_training_corpus.py` (3-layer split per DR1-001) を新設。

- **`scripts/_corpus_core.py`**: `validate_eval_overlap` (DR4-005 リーク検出) + `split_train_val` (DR1-005 deterministic split)
- **`scripts/generate_institutional_training_corpus.py`**:
  - `Session` / `CorpusReport` dataclass
  - `LLMClient` Protocol (Mock 容易な duck-typed interface)
  - `build_sessions(*, corpus_dir, n, scenarios, llm_client, seed, lang, n_turns)` → `Iterator[Session]` (LLM 駆動、I/O 最小)
  - `verify_corpus(sessions, eval_path)` → `CorpusReport` (LLM-free 検証)
  - `main()` (composition only、`pragma: no cover` で unit test 対象外)

**ユーザー指示遵守**:
- ✅ 実機 LLM 呼び出しなし (テストは `_FakeLLMClient` のみ)
- ✅ `main()` は実行禁止 (#137 完了後にユーザー実行)
- ✅ スクリプト本体作成のみ、`#137` GPU 占有中の干渉なし

**テスト**: 12 件 (TestValidateEvalOverlap 3 / TestSplitTrainVal 3 / TestBuildSessions 3 / TestVerifyCorpus 3)

## 品質チェック結果

| チェック項目 | コマンド | 結果 |
|-------------|----------|------|
| ruff | `ruff check` 全変更ファイル | ✅ All checks passed! |
| ruff format | `ruff format --check` 全変更ファイル | ✅ Pass |
| pytest 全体 | `pytest baseline_reporag/ photon_mlx/ tests/ torch_ref/tests/` | ✅ **1104 passed**, 2 既知 pre-existing failure (本変更由来でない) |

| テスト累計 | 件数 |
|----------|------|
| Day 1 終了時 | 1056 passed |
| Day 2 PM 終了時 | **1104 passed** (+48 件、本セッション追加分) |

## コミット履歴 (Day 2 PM)

```
f25022c feat(scripts): add training corpus generator scaffolding (#135 / Phase 2)
87802fb feat(configs): institutional_docs_photon_retrain.yaml for #135 Phase 4-3
a2a9d5b feat(photon_mlx/checkpoint): integrity.json SHA-256 verification (#135 / DR4-003)
587930c feat(torch_ref/config): add train_corpora_mix + val_split (#135 / Phase 4-1)
397b0bb feat(photon_mlx/data): add iterate_mixed_batches for JP/EN corpus mix (#135 / Phase 3)
```

## 累計コミット履歴 (Day 1 + Day 2 = 12 commits)

```
f25022c feat(scripts): add training corpus generator scaffolding (#135 / Phase 2)
87802fb feat(configs): institutional_docs_photon_retrain.yaml for #135 Phase 4-3
a2a9d5b feat(photon_mlx/checkpoint): integrity.json SHA-256 verification (#135 / DR4-003)
587930c feat(torch_ref/config): add train_corpora_mix + val_split (#135 / Phase 4-1)
397b0bb feat(photon_mlx/data): add iterate_mixed_batches for JP/EN corpus mix (#135 / Phase 3)
ecd1c2a docs(issue-135): Day 2 AM 進捗報告 — Phase 0-1 全タスク完了
8f1672d chore: add pre-commit config with detect-secrets (#135 / DR4-006)
f17c3a6 test(photon_pipeline): pin DR1-002 boundary in subprocess (#135 / Task 1.4)
1c920ae test(photon_pipeline): add Day 1 checkpoint load smoke test (#135 / Task 0.4)
2dbf458 fix(photon_pipeline): load checkpoint in _build_photon_deps (#135 / S7-001)
57d7742 docs(issue-135): pm-auto-issue2dev 完了報告
ea2fa57 refactor(photon_mlx): extract checkpoint I/O into checkpoint.py (#135)
```

## 設計方針書反映の検証

| Must Fix / Should Fix | 反映状況 |
|----------------------|----------|
| **S7-001** PHOTON eval random-init | ✅ commit `2dbf458` (Day 2 AM) |
| **DR1-001** 3 段分割 (build_sessions / verify_corpus / _corpus_core) | ✅ commit `f25022c` |
| **DR1-002** photon_mlx/checkpoint.py 抽出 | ✅ commit `ea2fa57` (Day 1) |
| **DR1-003** strict validation (sum=1.0, weight>0) | ✅ commit `397b0bb` + `587930c` |
| **DR1-005** val_split (no val_corpora_mix) | ✅ commit `587930c` + `87802fb` |
| **DR1-006** list 戻り型 | ✅ commit `397b0bb` |
| **DR1-007** sequence-level vs token-level metrics | ✅ commit `397b0bb` (test で固定) |
| **DR2-008** train/val tuple unpack | ✅ commit `397b0bb` |
| **DR2-009** min_lr=3e-6 必須 | ✅ commit `87802fb` (yaml で固定) |
| **DR3-001** trainer.py 互換 wrapper | ✅ commit `ea2fa57` (Day 1) |
| **DR3-002** 外部 LLM provider 制限 | ✅ commit `87802fb` yaml comment + Phase 2 main() default `qwen` |
| **DR4-002** path security + token validation | ✅ commit `397b0bb` |
| **DR4-003** integrity.json (SHA-256) | ✅ commit `a2a9d5b` |
| **DR4-005** eval set リーク検出 | ✅ commit `f25022c` (`validate_eval_overlap`) |
| **DR4-006** pre-commit hook | ✅ commit `8f1672d` (Day 1) |
| **S3-007 / DR3-007** retrain yaml 分離 | ✅ commit `87802fb` |
| **S5-001** cosine_decay 単体経路 | ✅ commit `87802fb` (yaml + DR2-009 comment) |
| **S5-002** 累計 step | ✅ commit `87802fb` (max_steps=10600) |
| **S5-004** tokens/step = 65,536 | ✅ commit `87802fb` (micro_batch=2 × grad_accum=16) |
| **S5-005** sequence pool 独立性 | ✅ commit `397b0bb` (test で固定) |

## 残タスク (本セッション以降)

### CPU で完結 (次セッションで実施可、もしくは次イテレーションで)

- **trainer.py の混合経路分岐**: `t_cfg.train_corpora_mix is not None` のとき `iterate_mixed_batches` を呼ぶ実装を `photon_mlx/trainer.py::train` に追加 (設計方針書 §10 列 148 — 改修最小)
- **DR4-001 generate script の追加 hardening**: `--corpus-dir`/`--output` の `resolve(strict=True)` + approved root 確認、`--sessions` の上限 (1-5000)、`--seed` の int 検証、partial output cleanup、`*.tmp` + 0600 permission + atomic rename — 本セッションでは入れず CLI 機能拡張時に対応

### GPU 必須 (#137 完了 + ユーザー承認後)

- **`scripts/generate_institutional_training_corpus.py main()` 実行**: 実 LLM (qwen mlx local default) で 2,000+ sessions 生成、`data/training/institutional/train_jp.jsonl` 出力 + metadata.json
- **mulmoclaude train_en.jsonl 取得**: 既存 600 step 学習で使った en corpus を `data/training/mulmoclaude/` へ配置 (既存 corpus が見つからない場合は scratch 学習 risk 発動)
- **Phase 6 学習実行**: 17-33 GPU-hours、10K/15K/20K step 候補
- **Phase 7 eval / 採用判定 / Phase 8 ロールアウト**

### Phase 5 (テスト追加) の状況

設計方針書 §8 が要求する 22-30 件の追加テストは、本セッションで以下を充足:

| カテゴリ | 設計目安 | 今回追加 |
|---------|---------|---------|
| iterate_mixed_batches unit | 4-6 | **17** |
| TrainingConfig schema | 含 | **12** |
| checkpoint integrity | 含 | **6** |
| yaml shape | 1-2 | **1** |
| corpus generator (LLM-free) | 4-6 | **12** |
| **合計 (Day 2 PM)** | **22-30** | **48** |

## 結論

Day 1 + Day 2 で **設計方針書の Phase 0 + Phase 1 + Phase 2 (script 本体のみ) + Phase 3 + Phase 4 全タスクを完了**。

- DR1-001 / DR1-002 / DR1-003 / DR1-005 / DR1-007 / DR3-001 / DR3-002 / DR4-002 / DR4-003 / DR4-005 / DR4-006 / S5-001 / S5-002 / S5-004 / S5-005 / S7-001 を実装+テスト固定
- 累計 12 commits、新規追加テスト 62 件 (Day 1 14 件 + Day 2 PM 48 件)
- 品質チェック regression 0、本変更由来の test 失敗 0

次セッションは:
1. trainer.py の混合経路分岐 (CPU)
2. DR4-001 CLI hardening (CPU)
3. #137 完了後 → corpus 生成 (実 LLM、~数時間 CPU/GPU)
4. ユーザー承認 → Phase 6 学習 (17-33 GPU-hours)
