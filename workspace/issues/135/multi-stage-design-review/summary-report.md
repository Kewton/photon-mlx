# Issue #135 マルチステージ設計レビュー完了報告

**Issue**: feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合
**完了日**: 2026-04-26
**実行コマンド**: `/multi-stage-design-review 135`
**設計方針書**: `workspace/design/issue-135-photon-retrain-design-policy.md` (498 → 667 行、+169 行 / +33.9%)

## ステージ別結果

| Stage | レビュー種別 | レビュアー | Must / Should / Nice | 反映 | 設計方針書行数 |
|-------|------------|-----------|---------------------|------|--------------|
| 1 | 通常レビュー（設計原則） | Claude opus | 3 / 6 / 2 = 11 | 11/11 | 498 → 615 (+117) |
| 2 | 整合性レビュー | Claude opus | 3 / 6 / 2 = 11 | 11/11 | 615 → 627 (+12) |
| 3 | 影響分析レビュー | Codex (gpt-5.5 high) | 2 / 4 / 0 = 6 | 6/6 | 627 → 631 (+4) |
| 4 | セキュリティレビュー | Codex (gpt-5.5 high) | 4 / 4 / 0 = 8 | 8/8 | 631 → 667 (+36) |

**累計指摘**: Must Fix 12 / Should Fix 20 / Nice to Have 4 = **36 件**、すべて設計方針書に反映済。

## 主要改善点

### Stage 1 (設計原則 / Claude opus)

- **DR1-001 SOLID**: `generate_institutional_training_corpus.py` の責務 8 集約を `build_sessions` / `verify_corpus` / `_corpus_core` の 3 段分割に明記
- **DR1-002 PHOTON 規約**: `photon_mlx/checkpoint.py` 新設で trainer 直 import を回避 (Dependency Inversion)
- **DR1-003 Boundary**: `iterate_mixed_batches` と `TrainingConfig.__post_init__` の sum(weights) 検証を strict ValueError に統一
- **DR1-005 YAGNI**: `val_corpora_mix` dict を廃止し `val_split: float` (0.05) で train pool 内分割に簡素化
- **DR1-007 Observability**: JP/EN ratio を sequence-ratio (制御目標) と token-ratio (測定値) に分離

### Stage 2 (整合性 / Claude opus)

- **DR2-001/002 Issue ↔ 設計方針書 sync**: Issue 本文の `val_corpora_mix` および `photon_mlx.trainer.load_checkpoint` 直接 import 残存を `gh issue edit` で修正、双方向 sync 完了
- **DR2-003 テスト整合**: `photon_mlx/tests/test_training.py::TestData` 同居 (L23-110) と新規 `test_data_mixed.py` の並立を明確化
- **DR2-004 内部整合**: §11 リスク表の compute 計算を `effective batch=32` で再計算
- **DR2-009 Codex 反映補強**: Issue 本文 hyperparam 列挙に `min_lr=3e-6` を追記し設計方針書 §設計判断 #3 と完全一致

### Stage 3 (影響分析 / Codex gpt-5.5 high)

- **DR3-001 checkpoint.py 抽出時の境界**: 既存 tests (TestCheckpoint が TrainState 直接 assert) を踏まえた trainer 互換 wrapper の必要性、import 表面 (5 module) を明示
- **DR3-002 LLM provider 送信リスク**: 制度文書本文を OpenAI 等に送信する場合の規約リスクを §11 に追加、デフォルトを "qwen (mlx local)" に固定
- **DR3-003 旧表記残存**: val_corpora_mix の旧表記が Day 別タスクサマリ・影響範囲表に残存していた箇所を一掃
- **DR3-004 pytest 既知失敗**: `tests/test_generate_training_corpus.py` の 2 件 pre-existing failure を CI 比較基準とし、新規追加で 507 → 535 件 (+28) 想定を明記
- **DR3-005 .gitignore 確認**: 既存 `git ls-files data/training/` で tracked ファイル無確認の手順を Day 1 タスク化
- **DR3-006 commandmatedev fail-closed**: worktree status 監視失敗時の fail-closed 条件 (60 秒以内 status=running 確認できなければ即停止)

### Stage 4 (セキュリティ / Codex gpt-5.5 high) — Must Fix 4 件

- **DR4-001 generate script CLI 入力検証**: --corpus-dir のパストラバーサル検出 (絶対パス + symlink resolve)、--sessions 上限 (1,000,000)、partial output の atomic write、失敗時 cleanup
- **DR4-002 JSONL/corpus_paths 入力検証**: tokens 配列の int / 0 ≤ id < vocab_size / 長さ上限 (1M)、corpus_paths キーの allowlist (`./data/`、`workspace/` 配下のみ)
- **DR4-003 checkpoint 改竄検出**: state.json と weights.npz の SHA-256 hash を別ファイル (`integrity.json`) に保存、load 時に必ず照合、不一致時 ValueError
- **DR4-004 training data extraction / membership inference**: 公開 release の判断基準を 5 軸 (corpus 規模 / 一意性 / canary 検出 / k-anonymity / membership inference test) で表形式定義、§13 機密漏洩防止セクション新設
- **DR4-005 eval リーク検証強化**: session_id 重複に加え、tokens 配列の prefix 一致 (Jaccard > 0.7) も検出
- **DR4-006 誤 commit / secret 自動検出**: pre-commit hook (`detect-secrets`, `truffleHog`) を `.pre-commit-config.yaml` に必須化
- **DR4-007 GPU runner serialize**: self-hosted runner で training 実行時の他 PR 干渉防止 (worktree-level lock file `/tmp/photon-mlx-gpu.lock`)
- **DR4-008 dependency pinning**: `requirements.txt` / `pyproject.toml` の主要 ML dependency (mlx, mlx-lm, sentence-transformers, rank-bm25) を ==X.Y.Z で pin、CVE 確認可能化

## 残課題と次フェーズへの申し送り

1. **DR4-004 公開判断**: 採用 checkpoint を public release する場合、§13 機密漏洩防止の 5 軸チェックを Day 5 ロールアウト直前に必ず実施。差分プライバシー (DP-SGD) は本 Issue 範囲外、follow-up Issue で検討。
2. **mulmoclaude 600 step checkpoint の物理確認** (Phase 0.5 仮説検証 #5 Unverifiable): 設計方針書 §補足 Day 1 で smoke test 必須化済。実行時に紛失が判明した場合は scratch 学習 (工数 +N 日 risk) を発動。
3. **PHOTON eval/runtime checkpoint load 経路** (S7-001 / DR1-002 / DR3-001): 設計方針書 §11 リスクトップ + §10 影響範囲 + Day 1 smoke test の 3 重縛りで実装誤りを早期検出。

## 次のアクション

- [x] 設計方針書の最終確認 (667 行、§1-§13 + Day 別タスクサマリ)
- [ ] `/work-plan 135` で作業計画立案 (Phase 4)
- [ ] `/pm-auto-dev 135` または分割 Issue で TDD 実装 (Phase 5)
  - **重要**: Phase 5 は実 GPU 学習 (3-5 日) を含むため、実行前に必ずユーザー確認 + Day 1 smoke test 通過後に着手

## 出力ファイル

```
workspace/issues/135/multi-stage-design-review/
├── stage1-review-context.json / stage1-review-result.json / stage1-apply-result.json
├── stage2-review-context.json / stage2-review-result.json / stage2-apply-result.json
├── stage3-review-result.json / stage3-apply-result.json
├── stage4-review-result.json / stage4-apply-result.json
└── summary-report.md (本ファイル)
```

設計方針書: `workspace/design/issue-135-photon-retrain-design-policy.md` (667 行)
