# Issue #135 作業計画書 — PHOTON 本格再学習 (制度文書ドメイン対応)

## 1. メタデータ

## Issue: feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合
**Issue番号**: #135
**サイズ**: L (Large、推定 5 日工数)
**優先度**: High (#117 Epic Phase 2 conditional 表の最重い分岐)
**依存Issue**: #113 (PR #134 merged) / #117 / #134 (closed)
**競合 Issue**: #137 (institutional 多言語 embedding A/B、GPU/config 競合あり)
**ブランチ**: feature/issue-135-photon-retrain
**設計方針書**: workspace/design/issue-135-photon-retrain-design-policy.md (667 行)

| 項目 | 値 |
|------|-----|
| ベースブランチ | `develop` (PR は `develop` → `main`) |
| 設計 stage 累計指摘 | Must Fix 11 / Should Fix 22 / Nice to Have 5 + DR1-001〜DR1-011 (Stage 1) + DR3-001〜DR3-006 (Stage 3) + DR4-001〜DR4-008 (Stage 4) すべて反映済 |
| 仮説検証ステータス | 12 件中 10 件 Confirmed / 1 件 Day 1 物理確認必須 (mulmoclaude 600 step) / 1 件本 Issue 内補足確認済 |
| 想定総工数 | 5 日 (Day 1-2 データ準備 / Day 3-4 学習 / Day 4-5 eval & rollout) |

---

## 2. 全体スケジュール

| Day | フェーズ | 主要タスク | 完了 gate | 主要成果物 |
|-----|---------|-----------|---------|----------|
| **Day 1** | Phase 0-1 | `.gitignore` 更新 + 最初の commit / mulmoclaude/step_600 物理確認 / `_build_photon_deps` の checkpoint load 改修 + smoke test (**S7-001 / DR1-002 最優先**) / `df -h` 残量確認 / `photon_mlx/checkpoint.py` 抽出 / dependency audit / secret scan 方針確定 (DR4-006 / DR4-008) | smoke test pass、checkpoint load 失敗 path も test 済、audit log 添付、security validation 方針確定 | `.gitignore` diff、`photon_mlx/checkpoint.py`、`baseline_reporag/tests/test_photon_pipeline.py` smoke test |
| **Day 2** | Phase 2-5 | `scripts/_corpus_core.py` 新設 / `scripts/generate_institutional_training_corpus.py` 実装 (DR1-001 3 関数分割) + eval リーク検証内蔵 / 2,000 sessions corpus 生成 / `iterate_mixed_batches` + `TrainingConfig.train_corpora_mix` + `val_split` + unit tests (DR3-003: `val_corpora_mix` は追加しない) | corpus metadata.json で全条件達成、unit tests pass、ruff 警告 0 | corpus + metadata.json、`photon_mlx/data.py` 拡張、`torch_ref/config.py` 拡張、unit tests 28+ |
| **Day 3** | Phase 6 前半 | `configs/institutional_docs_photon_retrain.yaml` 作成 / **ユーザー承認確認** / GPU lock 取得 (DR4-007) / **lr-finder 1K step small run** (lr=1e-5/3e-5/1e-4) で sec/optimizer-step 実測 + ETA 再計算 / ETA OK なら本走行 (累計 step ~10K まで) | lr 確定、ETA ≤ 5 日、lock 取得ログあり、1K step 毎 FastAPI MT eval で forgetting 5pp 超なし | step_010000 checkpoint、lr-finder report、forgetting curve |
| **Day 4** | Phase 6 後半 + Phase 7 前半 | 累計 step 15K → 20K 学習継続 / 各 checkpoint で `run_multi_turn_eval` 2-run / 境界帯時 +2 run / FastAPI 系 regression eval / 中間 checkpoint 即時削除 | step_010000/015000/020000 採用候補確定、6 run + 境界帯 +2 run | step_015000/020000 checkpoint、MT eval raw logs |
| **Day 5** | Phase 7 後半 + Phase 8 | `_build_photon_deps` 経由で採用 checkpoint load smoke test / reports v2 / gate2_post_retrain / gate2_v5 作成 / docs 更新 (CLAUDE.md / roadmap.md / metrics.md) / develop rebase + 全テスト再実行 / PR 作成 / publish 判断 (DR4-004) / follow-up Issue (DR1-004 / DR1-011 / DR2-010) 起票 | 全受入条件達成、PR レビュー承認、CI 全パス | reports 3 本、CLAUDE.md / roadmap.md / metrics.md 更新、PR、follow-up Issues |

> **重要**: Day 3-4 は GPU 占有の本学習フェーズ。Day 1 の smoke test 通過 + ユーザー承認後に着手する。承認なしに `/tmp/photon-mlx-gpu.lock` を取得しない。

---

## 3. 詳細タスク分解

> **TDD 並走方針**: Phase 5 (テスト) は Phase 1-4 と並走する (Red → Green → Refactor)。
> 各タスクは「成果物 / 依存タスク / 想定時間 (時)」を明記する。

### Phase 0: 機密保護とリポジトリ衛生 (Day 1 朝、目安 2.5h)

- [ ] **Task 0.1**: `.gitignore` に `data/training/` 追加 + `git ls-files data/training/` で tracked ファイル無確認
  - 成果物: `.gitignore` 更新 commit (本 Issue 最初の commit)、`git ls-files data/training/` 出力ログ
  - 依存タスク: なし
  - 想定時間: 0.5h
  - 参照: 設計方針書 §7 / S3-003 / DR3-005

- [ ] **Task 0.2**: pre-commit hook (`detect-secrets`, `truffleHog` 等) 設定方針確定 + secret scan ローカル実行 + dependency audit (`pip freeze` lock + `pip-audit`/`osv-scanner`)
  - 成果物: secret scan log、`pip freeze` lock file、`pip-audit` report
  - 依存タスク: なし
  - 想定時間: 1.0h
  - 参照: 設計方針書 §7 / DR4-006 / DR4-008

- [ ] **Task 0.3**: mulmoclaude 600 step checkpoint の物理確認 (`ls checkpoints/photon_mulmoclaude/step_600/weights.npz`、`state.json` の sha256 記録)
  - 成果物: 物理確認ログ + sha256 記録
  - 依存タスク: なし
  - 想定時間: 0.5h
  - 参照: 設計方針書 §7 / 仮説検証 #5 / DR3-001

- [ ] **Task 0.4**: PHOTON eval/runtime checkpoint load smoke test 雛形実装 (Phase 1 Task 1.3 完了後に拡充)
  - 成果物: `baseline_reporag/tests/test_photon_pipeline.py` (新規、smoke 1 件で `_build_photon_deps` 経由 load 成功を確認)
  - 依存タスク: 0.3
  - 想定時間: 0.5h
  - 参照: 設計方針書 §11 / S7-001 / DR1-002 / DR3-001

### Phase 1: コアモジュール抽出と境界整理 (Day 1 午後、目安 4.0h)

- [ ] **Task 1.1**: `photon_mlx/checkpoint.py` 新設 — `CheckpointState` (TrainState 不依存) DTO、`save_checkpoint(model, state, path)` / `load_checkpoint(model, path)` の純粋 I/O 分離
  - 成果物: `photon_mlx/checkpoint.py` (新規)、`CheckpointState` dataclass
  - 依存タスク: 0.3
  - 想定時間: 1.5h
  - 参照: 設計方針書 §4 (DR1-002 行) / DR3-001 / DR4-003 (manifest hash)

- [ ] **Task 1.2**: `photon_mlx/trainer.py` の checkpoint 関数を新 module 経由に refactor
  - 成果物: `photon_mlx/trainer.py` の `load_checkpoint` / `save_checkpoint` を `photon_mlx.checkpoint` から再 export、`TrainState` 互換 wrapper を維持
  - 依存タスク: 1.1
  - 想定時間: 1.0h
  - 参照: 設計方針書 §4 / 既存 tests `photon_mlx/tests/test_training.py::TestCheckpoint` pass 必須

- [ ] **Task 1.3**: `baseline_reporag/photon_pipeline.py::_build_photon_deps` を `photon_mlx.checkpoint` 経由に書換 (現行 line 308 直後に追加)
  - 成果物: `baseline_reporag/photon_pipeline.py` の checkpoint load 分岐 (`checkpoint_path` 設定時のみ load、missing/corrupt は `FileNotFoundError`)
  - 依存タスク: 1.1
  - 想定時間: 1.0h
  - 参照: 設計方針書 §6.5 (lines 425-456) / S7-001 / DR1-002

- [ ] **Task 1.4**: `pipeline_factory.py` の lazy MLX import 規約に違反していないことを確認 (smoke test pass、`baseline_reporag.trainer` direct import 0)
  - 成果物: `baseline_reporag/tests/test_pipeline_factory_lazy_mlx.py` 拡張 (1-2 件、import 表面 snapshot test)
  - 依存タスク: 1.3
  - 想定時間: 0.5h
  - 参照: 設計方針書 §6.5 境界規約 / S3-006

### Phase 2: データ生成 corpus 構築 (Day 1 夜 〜 Day 2、目安 7.5h)

- [ ] **Task 2.1**: `scripts/_corpus_core.py` 新設 — `tokenize` / `pack_sequences` 互換 / `val_split` / `validate_tokenizer_id` / `resolve_tokenizer_id` / `validate_eval_overlap(session_ids, eval_path) -> int` を抽出
  - 成果物: `scripts/_corpus_core.py` (private module)
  - 依存タスク: なし (Phase 1 と並列可)
  - 想定時間: 1.5h
  - 参照: 設計方針書 §4 / S3-004 / DR1-001 / DR1-004

- [ ] **Task 2.2**: `scripts/generate_institutional_training_corpus.py` 新設 — `Session` / `CorpusReport` dataclass + `build_sessions` / `verify_corpus` / `main` の 3 関数分割 (DR1-001)
  - 成果物: `scripts/generate_institutional_training_corpus.py` (新規)
  - 依存タスク: 2.1
  - 想定時間: 2.0h
  - 参照: 設計方針書 §6.6 (lines 460-515) / DR1-001

- [ ] **Task 2.3**: CLI 入力検証実装 — `--corpus-dir` `resolve(strict=True)` + approved root check / `--sessions` 1〜5000 / `--seed` 0〜2^32-1 / `--val-ratio` 0.0〜0.5 / atomic write (`*.tmp` + 0600 + rename) / LLM provider default `qwen` (local) / `--provider openai` 等は明示承認なしで使用禁止
  - 成果物: CLI argparse + 検証関数
  - 依存タスク: 2.2
  - 想定時間: 1.0h
  - 参照: 設計方針書 §6.6 CLI/I/O security contract / DR4-001 / DR3-002

- [ ] **Task 2.4**: eval リーク検証 (`verify_corpus`) を session_id 重複 + content near-duplicate (normalized hash + 5-gram Jaccard) で実装
  - 成果物: `verify_corpus` 内 `validate_eval_overlap` + near-duplicate check、リーク非 0 で exit code 1
  - 依存タスク: 2.2 / 2.1
  - 想定時間: 1.0h
  - 参照: 設計方針書 §7 / DR4-005

- [ ] **Task 2.5**: 2,000+ sessions × 4-6 turns 生成 (cross_reference / drill_down 30% 以上、scenario distribution 集計)
  - 成果物: `data/training/institutional/{train_jp.jsonl, val_jp.jsonl, metadata.json}` (機密、commit 対象外)
  - 依存タスク: 2.3 / 2.4
  - 想定時間: 1.5h (LLM 呼び出し時間中心、CPU 寄り)
  - 参照: 設計方針書 §6.1 / 受入条件 #2

- [ ] **Task 2.6**: 出力 schema 検証 — 1 session = 1 doc、turn delimiter `<|turn_sep|>` 通常文字列、metadata に `jp_sequence_ratio` (制御目標) + `jp_token_ratio` (実測値) 両記録 + `n_sessions_requested / n_sessions_succeeded / failure_breakdown`
  - 成果物: metadata.json 検証 PASS ログ (sessions ≥ 2,000、JP sequence 50%、JP token 50%±5pp、cross_reference/drill_down 30%+、eval_overlap=0)
  - 依存タスク: 2.5
  - 想定時間: 0.5h
  - 参照: 設計方針書 §6.1 / DR1-007 / DR1-009 / 受入条件 #2

### Phase 3: データロード拡張 (Day 2、目安 3.0h)

- [ ] **Task 3.1**: `photon_mlx/data.py` に `iterate_mixed_batches(corpus_paths: dict[str, float], *, context_length, batch_size, seed=42, shuffle=True, val_split=0.0)` 新関数追加 (既存 `iterate_batches` は不変)
  - 成果物: `photon_mlx/data.py` 拡張、`iterate_mixed_batches` 実装
  - 依存タスク: 2.1
  - 想定時間: 1.5h
  - 参照: 設計方針書 §6.2 (lines 290-357) / S3-001 / DR1-006

- [ ] **Task 3.2**: 入力検証 — sum(weights) ≠ 1.0 (±1e-6) で strict ValueError、tokens int / 0 ≤ id < vocab_size / 長さ ≤ context_length×4 / 0 < line ≤ 2 MiB、corpus_paths キーは approved root 配下 (`data/training/` または `data/processed/`) のみ allowlist
  - 成果物: `iterate_mixed_batches` 内 strict validation 関数
  - 依存タスク: 3.1
  - 想定時間: 1.0h
  - 参照: 設計方針書 §6.2 / DR1-003 / DR4-002

- [ ] **Task 3.3**: `pack_sequences` / `create_batches` 再利用 + sequence pool weighted sampling 実装 (corpus 間を跨がない pool 構築 + sequence-level weighted sampling)
  - 成果物: `iterate_mixed_batches` の sampling ロジック、val_split > 0 で `(train_batches, val_batches)` tuple 返却
  - 依存タスク: 3.1
  - 想定時間: 0.5h
  - 参照: 設計方針書 §5 (設計判断 #1) / §6.2 / S5-005 / DR2-008

### Phase 4: Config 分離と TrainingConfig 拡張 (Day 2、目安 3.0h)

- [ ] **Task 4.1**: `configs/institutional_docs_photon_retrain.yaml` 新設 (旧 yaml は #112/#126 ベースライン保持、不変)
  - 成果物: `configs/institutional_docs_photon_retrain.yaml` (`model.provider: photon` + `model.checkpoint_path` leaf path + 学習 hyperparam: `lr=3e-5`, `min_lr=3e-6`, `warmup_ratio=0.0`, `max_steps=10000-20000`, `train_corpora_mix` (sum=1.0 strict), `val_split=0.05`, `micro_batch_size=2`, `grad_accum=16`)
  - 依存タスク: なし (Phase 1-3 と並列可)
  - 想定時間: 1.0h
  - 参照: 設計方針書 §4 / 設計判断 #7 / S3-007 / S7-001 / DR1-005

- [ ] **Task 4.2**: `TrainingConfig` に `train_corpora_mix: dict[str, float] | None = None` / `val_split: float = 0.05` を追加 (`__post_init__` で strict 検証)
  - 成果物: `torch_ref/config.py` 拡張 + `__post_init__` validation (空 dict / 負 weight / 全 weight 0 / 非数値 weight / sum != 1.0 (±1e-6) を reject)
  - 依存タスク: 3.1 (signature 揃え)
  - 想定時間: 1.0h
  - 参照: 設計方針書 §6.3 (lines 360-398) / S7-003 / DR1-003 / DR1-005

- [ ] **Task 4.3**: `photon_mlx/checkpoint.py` の `manifest.json` (sha256) 書込・照合実装 + `load_photon_config` の relative path 絶対化 (YAML ファイルディレクトリ基準)
  - 成果物: `manifest.json` schema (`weights.npz / state.json / config / corpus metadata` の sha256、`tokenizer_id`、`resume_base_step`、`cumulative_step`) + load 時 hash 不一致で raise
  - 依存タスク: 1.1
  - 想定時間: 1.0h
  - 参照: 設計方針書 §7 / DR4-003

### Phase 5: テスト追加 (Day 2 〜 Day 3 朝、Phase 1-4 と並走、目安 6.0h、計 28+ 件)

設計方針書 §8 に従い、以下を追加。Phase 1-4 のタスクに対する Red → Green → Refactor を並走。

- [ ] **Task 5.1**: `iterate_mixed_batches` unit tests (5-7 件): 比率正確性 (sequence ratio 0.5±0.02 / token ratio metadata 記録)、シード再現性、空 corpus → ValueError、単一 corpus、weight sum != 1.0 → ValueError、val_split > 0 で tuple 返却 + 比率維持
  - 成果物: `photon_mlx/tests/test_data.py` (新規)
  - 依存タスク: 3.1 / 3.2 / 3.3
  - 想定時間: 1.5h
  - 参照: 設計方針書 §8 行 549-550 / DR1-003 / DR1-005 / DR1-007 / DR2-003

- [ ] **Task 5.2**: `test_resume_from_continual_learning` (2-3 件): `lr=3e-5` 起点 → cosine decay → `min_lr=3e-6` 検証 (S5-001)、累計 step (S5-002)、既存 600 step 重みが恒等でない (= 学習が走った)
  - 成果物: `photon_mlx/tests/test_training.py::TestResume` 新規
  - 依存タスク: 4.1 / 4.2
  - 想定時間: 1.0h
  - 参照: 設計方針書 §8 行 551 / S3-002 / S5-001 / S5-002 / DR2-005

- [ ] **Task 5.3**: `generate_institutional_training_corpus.py` 検証 unit tests (5-7 件): eval リーク検出 (= 0 / > 0 で exit 1)、JP 比率 (sequence-level / token-level 両方)、scenario 分布、`build_sessions` を `llm_client` Mock 注入 (1-2 件)、dry-run / resume / 失敗率 5% 超 exit 1
  - 成果物: `tests/test_generate_institutional_training_corpus.py` (新規)
  - 依存タスク: 2.2 / 2.4
  - 想定時間: 1.5h
  - 参照: 設計方針書 §8 行 554 / DR1-001 / DR1-009

- [ ] **Task 5.4**: `photon_mlx/checkpoint.py` `manifest.json` テスト (3-5 件): sha256 不一致 reject、symlink escape reject、approved root 外 reject、state.json schema validation
  - 成果物: `photon_mlx/tests/test_checkpoint.py` (新規)
  - 依存タスク: 1.1 / 4.3
  - 想定時間: 0.5h
  - 参照: 設計方針書 §8 行 552 / DR3-001 / DR4-003

- [ ] **Task 5.5**: PHOTON eval/runtime checkpoint load smoke test 拡充 (3-4 件): `_build_photon_deps` で `model.checkpoint_path` 設定時の load / missing で `FileNotFoundError` / corrupt state.json で raise / `checkpoint_path` 未設定時の従来挙動
  - 成果物: `baseline_reporag/tests/test_photon_pipeline.py` 拡張 (Day 1 Task 0.4 を本実装に拡充)
  - 依存タスク: 1.3 / 1.4
  - 想定時間: 0.5h
  - 参照: 設計方針書 §8 行 556-557 / S7-001 / DR4-003

- [ ] **Task 5.6**: 小規模 corpus + 100 step での dry-run integration test (1-2 件、`corpora_mix` 経路を含む)
  - 成果物: `photon_mlx/tests/test_training_integration.py` (新規 or 既存拡張)
  - 依存タスク: 3.1 / 3.2 / 4.2
  - 想定時間: 0.5h
  - 参照: 設計方針書 §8 行 559

- [ ] **Task 5.7**: Security validation 追加 (4-6 件): `--corpus-dir` traversal reject、symlink escape reject、`--sessions` / `--seed` / `--val-ratio` 境界値、malformed JSONL token (負数 / bool / float / vocab 範囲外 / 巨大 line)、eval content near-duplicate/hash overlap reject
  - 成果物: `tests/test_generate_institutional_training_corpus.py` + `photon_mlx/tests/test_data.py` 拡張
  - 依存タスク: 2.3 / 2.4 / 3.2
  - 想定時間: 0.5h
  - 参照: 設計方針書 §8 行 555 / DR4-001 / DR4-002 / DR4-005

### Phase 6: 学習実行 (Day 3-4) — **GPU 占有、ユーザー承認必須、目安 17-33 GPU-hours = 0.7-1.4 日**

- [ ] **Task 6.1**: ユーザー承認確認 + GPU lock 取得 (`/tmp/photon-mlx-gpu.lock` または `.locks/issue-135-train.lock` を `flock` で serialize) + `git status --short` が expected files のみ + `gh issue view 137 --repo Kewton/photon-mlx --json state` で #137 close 確認
  - 成果物: lock 取得ログ、承認記録、#137 status 出力
  - 依存タスク: Phase 5 全件 PASS / Day 1 smoke test PASS
  - 想定時間: 0.5h
  - 参照: 設計方針書 §11 / S7-005 / DR3-006 / DR4-007

- [ ] **Task 6.2**: 10K step 学習実行 (累計 state.step、`micro_batch_size=2` / `grad_accum=16` (effective batch=32) / `lr=3e-5` / `min_lr=3e-6` / `warmup_ratio=0.0` / `context_length=2048`)。事前に **lr-finder 1K step small run** (lr=1e-5/3e-5/1e-4) で sec/optimizer-step 実測 + ETA 再計算
  - 成果物: `checkpoints/photon_institutional_retrain_<yyyymmdd>/step_010000/{weights.npz, state.json, manifest.json}`、lr-finder report
  - 依存タスク: 6.1
  - 想定時間: 8-17 GPU-hours (= 0.4-0.7 日)
  - 参照: 設計方針書 §補足 Day 3 / DR1-010 / DR2-004

- [ ] **Task 6.3**: 1K step 毎の英語コード eval (FastAPI MT) で forgetting 早期検出 (5pp 超で停止)
  - 成果物: forgetting curve log (1K step 毎 NC)
  - 依存タスク: 6.2
  - 想定時間: 6.2 と並列 (1K step 毎 5min)
  - 参照: 設計方針書 §11 リスク表 catastrophic forgetting

- [ ] **Task 6.4**: 15K step 学習継続
  - 成果物: `step_015000/` checkpoint
  - 依存タスク: 6.2 / 6.3
  - 想定時間: 4-9 GPU-hours
  - 参照: 設計方針書 §2.2 / 設計判断 #2

- [ ] **Task 6.5**: 20K step 学習継続
  - 成果物: `step_020000/` checkpoint
  - 依存タスク: 6.4
  - 想定時間: 4-9 GPU-hours
  - 参照: 設計方針書 §2.2 / 設計判断 #2

- [ ] **Task 6.6**: 中間 checkpoint (1K step 単位) の即時削除、最終 3 ポイント (10K/15K/20K) のみ保持
  - 成果物: `df -h` 残量モニタログ、削除ログ
  - 依存タスク: 6.2 / 6.4 / 6.5
  - 想定時間: 0.5h (バックグラウンド)
  - 参照: 設計方針書 §11 リスク表 SSD 逼迫 / S7-006 / DR1-010

### Phase 7: Eval 実行と採用判定 (Day 4-5、目安 5.0h)

- [ ] **Task 7.1**: 各 checkpoint で `scripts/run_multi_turn_eval.py` で 2-run × 3 ckpt = 6 run の MT eval (NC / latency / category 別)
  - 成果物: 6 run の raw eval log
  - 依存タスク: 6.2 / 6.4 / 6.5
  - 想定時間: 4-8h (eval phase)
  - 参照: 設計方針書 §2.2 / §8 行 560 / 設計判断 #6

- [ ] **Task 7.2**: 境界帯 (Turn 5-6 NC 5-7%) の追加 +2 run (4-run 平均で再判定)
  - 成果物: 境界帯 4-run 平均 NC
  - 依存タスク: 7.1
  - 想定時間: 1-2h (条件発動時のみ)
  - 参照: 設計方針書 §2.2 / 設計判断 #6

- [ ] **Task 7.3**: FastAPI 系 retrieval eval (gate2 v4 PHOTON+SR ベースの NC 6.7% +5pp 以内 = ≤ 11.7%)
  - 成果物: `reports/gate2_post_retrain_eval.md` (新規、fastapi_fastapi 1 run、provider=photon)
  - 依存タスク: 6.2 / 6.4 / 6.5
  - 想定時間: 40min × 3 ckpt = 2h
  - 参照: 設計方針書 §2.2 / §8 行 561 / C-3 受入条件

- [ ] **Task 7.4**: 採用判定 (最低条件: Turn 5-6 NC < 6% AND latency ≤ 13.6s AND FastAPI MT NC ≤ 11.7%; その中で Turn 5-6 NC 最小)
  - 成果物: 採用 checkpoint 指定、`gate2_judgment_v5_post_retrain.md` 草稿
  - 依存タスク: 7.1 / 7.2 / 7.3
  - 想定時間: 0.5h
  - 参照: 設計方針書 §5 設計判断 #6 / 受入条件 #11-#14

- [ ] **Task 7.5**: 不採用時 pivot (3-6% 帯なら軽量 fine-tune 5K step、>6% なら本格再学習継続 or Phase 2 完了基準再定義 別 Issue 起票)
  - 成果物: pivot 計画 + 別 Issue (該当時)
  - 依存タスク: 7.4
  - 想定時間: 条件発動時のみ
  - 参照: 設計方針書 §11 リスク表 / DR2-011 / S3-012

### Phase 8: ロールアウトとドキュメント更新 (Day 5、目安 4.5h)

- [ ] **Task 8.1**: 採用 checkpoint を `checkpoints/photon_institutional_step<NNNNN>_<yyyymmdd>/` に保存 + git LFS or external storage (DR4-003 `manifest.json` 含む)
  - 成果物: 採用 checkpoint leaf + LFS / external storage 配置
  - 依存タスク: 7.4
  - 想定時間: 1.0h
  - 参照: 設計方針書 §10 / 受入条件 #16

- [ ] **Task 8.2**: `configs/institutional_docs_photon_retrain.yaml` の `checkpoint_path` 確定
  - 成果物: yaml の `model.checkpoint_path` 更新
  - 依存タスク: 8.1
  - 想定時間: 0.2h
  - 参照: 設計方針書 §3.1 / 受入条件 #17

- [ ] **Task 8.3**: `reports/institutional_photon_mt_eval_v2.md` 新規作成 (旧 / 新 / baseline 比較表 + checkpoint metadata 列として `resume_base_step` / `additional_updates` / `cumulative_step` の 3 列必須、DR1-008 強制)
  - 成果物: `reports/institutional_photon_mt_eval_v2.md` (新規、旧 v1 への cross-link、DR2-007)
  - 依存タスク: 7.1 / 7.4
  - 想定時間: 1.0h
  - 参照: 設計方針書 §10 行 622 / S7-002 / DR1-008 / DR2-007

- [ ] **Task 8.4**: `reports/gate2_judgment_v5_post_retrain.md` 新規作成 (v4 → v5 cross-link)
  - 成果物: `reports/gate2_judgment_v5_post_retrain.md` (新規)
  - 依存タスク: 7.4 / 8.3
  - 想定時間: 0.5h
  - 参照: 設計方針書 §10 / 受入条件 #15 / S3-005

- [ ] **Task 8.5**: `CLAUDE.md` 現メトリクス更新 (L156-160) / `workspace/mvp/roadmap.md` Phase 2 conditional 表 (L88-94) >6% 分岐実行済マーキング (DR2-011) / `workspace/mvp/metrics.md` 反映
  - 成果物: 3 ファイル更新 commit
  - 依存タスク: 7.4 / 8.3 / 8.4
  - 想定時間: 0.5h
  - 参照: 設計方針書 §10 / 受入条件 #21

- [ ] **Task 8.6**: `docs/deployment.md` / `docs/troubleshooting.md` 採用 checkpoint 取得手順追記 (Nice to Have、follow-up Issue でも可)
  - 成果物: docs 2 ファイル更新 or follow-up Issue
  - 依存タスク: 8.1
  - 想定時間: 0.5h
  - 参照: 設計方針書 §10 / 受入条件 #22 / S3-011

- [ ] **Task 8.7**: 公開判断 (DR4-004 5 軸チェック): corpus 規模 / 一意性 / canary 検出 / k-anonymity / membership inference test。非公開とする場合は private/ACL storage 配置を確認
  - 成果物: 公開判断 record (private/public + 根拠)
  - 依存タスク: 8.1
  - 想定時間: 0.5h
  - 参照: 設計方針書 §7 / 受入条件 #25 / DR4-004

- [ ] **Task 8.8 (補足)**: develop rebase + 全テスト再実行 + `/create-pr` で PR 作成 + follow-up Issue (DR1-004 `generate_training_corpus.py` 書換 / DR1-011 yaml diff test / DR2-010 pyproject pytest 設定) 起票
  - 成果物: PR、follow-up 3 Issues
  - 依存タスク: 8.1 〜 8.7
  - 想定時間: 0.3h
  - 参照: 設計方針書 §11 リスク表 / DR1-004 / DR1-011 / DR2-010

---

## 4. 依存関係マップ

```
Day 1 朝             Day 1 午後             Day 2                              Day 3-4 (GPU 占有)              Day 4-5
─────────            ─────────              ────                              ──────────────────             ────────

[Phase 0]            [Phase 1]              [Phase 2-5 並走 (TDD)]            [Phase 6]                       [Phase 7-8]
 ├─0.1 .gitignore    ├─1.1 checkpoint.py   ┌─────────────────────────┐      ┌─ユーザー承認──┐               ┌─7.1 MT eval
 ├─0.2 secret/audit  │   ↓                 │ Phase 2 corpus           │      │6.1 GPU lock + │              │   (6 run)
 ├─0.3 mulmoclaude   ├─1.2 trainer refactor│ ├─2.1 _corpus_core ──┐  │      │   #137 close   │              ├─7.2 境界帯
 │   ckpt 物理確認   │   ↓                 │ ├─2.2 generate_*.py  │  │      └────┬───────┘                  │   +2 run
 │   ↓               ├─1.3 photon_pipeline │ ├─2.3 CLI 検証       │  │           ↓                          ├─7.3 FastAPI eval
 └─0.4 smoke test    │   ↓                 │ ├─2.4 eval リーク    │  │      ┌─6.2 lr-finder──┐              │   ↓
     雛形 ───────────┴─1.4 lazy MLX 確認   │ ├─2.5 corpus 生成    │  │      │   1K step       │              ├─7.4 採用判定
                          ↓                │ └─2.6 schema 検証 ───┤  │      │   ETA 再計算    │              │   ↓
                          (Phase 1 done)   │                       │  │      └────┬───────┘                  │ ┌─8.1 ckpt LFS
                                           │ Phase 3 data loader   │  │           ↓ OK                       │ ├─8.2 yaml path
                                           │ ├─3.1 mixed_batches   │  │      ┌─6.2 10K step──→─6.4 15K──→─6.5 20K┤
                                           │ ├─3.2 strict valid    │  │      │ + 6.3 1K step                 │ ├─8.3 report v2
                                           │ └─3.3 sampling        │  │      │   FastAPI eval                │ ├─8.4 gate2 v5
                                           │                       │  │      │   forgetting check            │ ├─8.5 docs
                                           │ Phase 4 config        │  │      └─6.6 中間 ckpt 削除─┘          │ ├─8.6 deploy doc
                                           │ ├─4.1 yaml 新規       │  │                                       │ ├─8.7 公開判断
                                           │ ├─4.2 TrainingConfig  │  │                                       │ └─8.8 PR + follow-up
                                           │ └─4.3 manifest.json   │  │
                                           │                       │  │
                                           │ Phase 5 tests (TDD)   │  │
                                           │ ├─5.1〜5.7            │  │
                                           │ (Phase 1-4 と並走)    │  │
                                           └─────────────────────────┘
                                                    ↓ (Phase 1-5 完了)
                                                    Phase 6 着手前提
```

**Critical path**: Phase 0 → Phase 1 → Phase 5 (smoke test 含む) → ユーザー承認 → Phase 6 → Phase 7 → Phase 8。
**並列可能**: Phase 2 / Phase 3 / Phase 4 は Phase 1 完了後に互いに並列。Phase 5 は Phase 1-4 と並走 (TDD)。

---

## 5. 品質チェック項目

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| ビルド | `python -m pytest --collect-only` | エラー 0 件 |
| ruff | `ruff check .` | 警告 0 件 |
| フォーマット | `ruff format --check .` | 差分なし |
| テスト | `python -m pytest --ignore=data/training/ photon_mlx/tests/ baseline_reporag/tests/ tests/ torch_ref/tests/` | 535/537 件パス (既知 2 件 `tests/test_generate_training_corpus.py` の pre-existing failure 維持。本 Issue 起因の新規 failure 0 件) |
| Smoke test (Day 1 必須) | `python -m pytest baseline_reporag/tests/test_photon_pipeline.py` | Pass (Day 1 終了時点で必須、学習開始 gate) |
| Smoke test (Day 5) | 上記 + `_build_photon_deps` 経由で採用 checkpoint load を実機 1 run | Pass |
| Secret scan | `gitleaks detect --no-git --source .` 相当 | Pass、`data/training/` / `checkpoints/` / `.env` / provider logs が staging されていないこと |
| Dependency audit | `pip-audit` または `osv-scanner` | High/Critical CVE は upgrade / pin / risk acceptance 記録のいずれか |
| Manifest 検証 | `python -m pytest photon_mlx/tests/test_checkpoint.py -k manifest` | Pass、sha256 不一致 reject、approved root 外 reject |
| metadata.json 検証 | `verify_corpus` 出力レビュー | sessions ≥ 2,000、JP sequence 50%、JP token 50%±5pp、cross_reference/drill_down 30%+、eval_overlap=0、failure_breakdown の `api_error+json_parse_error` ≤ 5% |

---

## 6. Definition of Done

設計方針書 §9 受入条件 + §10 影響範囲 + §11 リスク のすべてを満たす形で再掲。

- [ ] Phase 0-8 すべて完了
- [ ] sessions 数 ≥ 2,000、JP sequence ratio = 50% (target、ε ≤ 2pp)、JP token ratio ≥ 50%±5pp、eval リーク 0、cross_reference/drill_down 比 30%+
- [ ] Turn 5-6 NC < 6% (最低) / < 3% (理想) AND follow-up p50 latency ≤ 13.6s 維持
- [ ] FastAPI 系 retrieval NC 6.7% +5pp 以内 (= 11.7% 以下)
- [ ] pytest `--ignore=data/training/` 全 pass (約 535/537、既知 2 件 pre-existing failure 維持。本 Issue 追加 28+ 件込みで 535+28=563/565 想定)
- [ ] ruff check . 警告 0、ruff format --check . 差分なし
- [ ] manifest.json sha256 照合 pass、checkpoint load smoke test pass
- [ ] `reports/institutional_photon_mt_eval_v2.md` (DR1-008 3 列必須) / `reports/gate2_post_retrain_eval.md` / `reports/gate2_judgment_v5_post_retrain.md` 作成
- [ ] CLAUDE.md (L156-160 メトリクス) / workspace/mvp/roadmap.md (Phase 2 conditional 表 >6% 行 マーキング) / metrics.md 更新
- [ ] `.gitignore` `data/training/` 追加済 (本 Issue 最初の commit)、`git ls-files data/training/` 出力空
- [ ] secret scan / dependency audit pass、High/Critical CVE 0 件 or 受容記録
- [ ] follow-up Issue 起票 (DR1-004 / DR1-011 / DR2-010、Day 5 close 同時に link)
- [ ] PR レビュー承認 + CI 全パス + develop merge 完了 (main merge は別 PR)

---

## 7. リスクと対応

設計方針書 §11 リスク表を再掲し、Day 別の検出 / 緩和タイミングを明記。

| リスク | 影響度 | 検出 Day | 緩和策 |
|--------|--------|---------|--------|
| PHOTON eval/runtime checkpoint load 失敗 (S7-001) | 高 | Day 1 (smoke test) | smoke test 不通で実装修正、リリース blocker |
| Catastrophic forgetting (英語 -10pp 超) | 高 | Day 3-4 (1K step 毎 FastAPI eval) | 5pp 超で早期停止、最良 checkpoint dynamic 選定 |
| mulmoclaude checkpoint 紛失 | 中 | Day 1 (Task 0.3) | 紛失時 scratch 学習 + 工数 +2-3 日、別 Issue 起票 |
| `resume_from` LR 起点誤り | 高 | Day 2-3 (Task 5.2 / lr-finder) | `test_resume_from_continual_learning` CI、`min_lr=3e-6 ≠ 0` 必須 (DR2-005) |
| corpus 機密漏洩 | 高 | Day 1 (Task 0.1) | `.gitignore` 最初の commit、pre-commit hook、PR diff 確認 |
| 制度文書本文の外部 LLM 送信 (DR3-002) | 高 | Day 1-2 (Task 2.3) | 外部 provider 既定禁止、承認なしで使用しない |
| eval set リーク / near-duplicate | 高 | Day 2 (verify_corpus) | リーク検出時 corpus 再生成、5-gram Jaccard で near-duplicate も拒否 |
| GPU 占有 / #137 競合 (DR3-006) | 中 | Day 3 (Task 6.1) | #137 先行完了確認、lock file で serialize、#137 rerank A/B (CPU) は Day 4-5 並列可 |
| self-hosted runner workspace 汚染 (DR4-007) | 中 | Day 3-4 | repo local lockfile、issue 固有 worktree、pre/post `git status --short` |
| 65,536 tokens/optimizer-step による所要時間過小評価 | 中 | Day 3 (lr-finder) | 1K step lr-finder で sec/optimizer-step 実測、5 日超なら `grad_accum=4` / 10K early stop / 夜間実行 |
| Disk 容量 50GB 不足 | 中 | Day 1 (Task 0.2) / Day 2-4 (df 監視) | 中間 checkpoint 即時削除、最終 3 ポイントのみ保持 |
| CI で新 checkpoint 取得不可 | 中 | Day 5 (Task 8.1) | LFS or external fetch step、smoke test |
| Turn 5-6 NC < 6% 達成失敗 | 中 | Day 4-5 (Task 7.4) | 3-6% 帯は軽量 fine-tune 5K step、>6% は Phase 2 完了基準再定義 別 Issue (DR2-011 / S3-012) |
| `corpora_mix` silent normalize | 中 | Day 2 (Task 5.1) | `__post_init__` + `iterate_mixed_batches` で strict ValueError、warning fallback 禁止 (DR1-003) |
| `iterate_mixed_batches` peak memory 逼迫 | 低-中 | Day 3 (lr-finder) | 5,000 sessions × 4K tokens × bf16 ≈ 40 MB の試算。実測 3 GB 超なら Iterator 化 |
| Training data extraction (publish 時) | 高 | Day 5 (Task 8.7) | DR4-004 5 軸チェック、private/ACL storage 既定、必要なら follow-up Issue |
| 他 PR との merge conflict | 低 | Day 3-5 | Day 3-4 期間中は trainer.py / data.py / yaml 関連 PR 保留、Day 5 develop rebase |

---

## 8. 並列開発との調整

- **#137 institutional 多言語 embedding A/B**: **Day 3 学習開始前に #137 close を必須確認** (`gh issue view 137 --json state`)。学習 phase (Day 3-4) は GPU lock で直列。rerank A/B (CPU) は Day 4-5 eval phase と並列可。S7-005 / DR3-006。
- **#133 多言語 reranker A/B**: Phase A merge 済 (PR #136)。Phase B は本 Issue と並走可能 (CPU 系)。
- **develop merge**: 5 日間で他 PR が develop に merge された場合、Day 5 ロールアウト前に rebase + 全テスト再実行 (Task 8.8)。
- **commandmatedev / worktree status 監視**: 失敗または利用不能なら fail-closed とし、`git fetch` / `git status --short` / `gh issue view 137` / 関連 config diff 確認を手動で通すまで Day 3 学習を開始しない (DR3-006 / DR4-007)。

---

## 9. 次のアクション (作業計画承認後)

1. **ユーザーに作業計画の承認を求める** (特に Day 3-4 GPU 占有 17-33 GPU-hours の合意、#137 close 確認、`/tmp/photon-mlx-gpu.lock` 取得方針)。
2. Day 1 Phase 0 から実装開始 (TDD)。最初の commit は `.gitignore` 更新 (Task 0.1)。
3. `/progress-report` で日次進捗報告 (Day 1 / Day 2 / Day 3 終了時)。
4. Day 5 完了後 `/create-pr` で PR 作成 + follow-up Issue 3 件 (DR1-004 / DR1-011 / DR2-010) 同時起票。
