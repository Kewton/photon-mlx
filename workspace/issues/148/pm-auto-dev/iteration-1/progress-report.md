# Issue #148 pm-auto-issue2dev 完了報告

**実施日**: 2026-04-27
**Issue**: #148 — `test(eval): re-establish true baseline — fixed PHOTON pipeline + new LLM upgrade (Qwen3.5-9B / Gemma4-26B)`
**PR #1 (本 worktree)**: https://github.com/Kewton/photon-mlx/pull/150

## 全体サマリー

| Phase | 内容 | ステータス | 主要成果物 |
|-------|------|----------|----------|
| 1 | マルチステージ Issue レビュー (8 stages) | ✅ 完了 | 34 findings 全反映 |
| 2 | 設計方針書作成 | ✅ 完了 | 842 行 (`workspace/design/issue-148-rebaseline-design-policy.md`) |
| 3 | マルチステージ設計レビュー (4 stages) | ✅ 完了 | 32 findings 全反映 |
| 4 | 作業計画立案 | ✅ 完了 | 501 行 (`workspace/issues/148/work-plan.md`), 3-PR 分割 |
| 5 | TDD 自動開発 (Phase A0 のみ) | ✅ 完了 | 22 新規テスト + 1 invariant test, 1124 既存テスト維持 |
| 6 | 完了報告 (本書) | ✅ 完了 | PR #150 作成 |

## Phase 1: マルチステージ Issue レビュー

| Stage | 担当 | Must Fix | Should Fix | Nice to Have | reviewer 検証 |
|-------|------|---------|-----------|------------|------------|
| 1 (通常) | claude-opus | 4 | 5 | 3 | - |
| 2 (反映) | claude-sonnet | - | - | - | - |
| 3 (影響範囲) | claude-opus | 4 | 5 | 4 | - |
| 4 (反映) | claude-sonnet | - | - | - | - |
| 5 (通常 2回目) | **codex** | 2 | 2 | 0 | ✅ `reviewer="codex"` |
| 6 (反映 2回目) | **codex** | - | - | - | - |
| 7 (影響範囲 2回目) | **codex** | 1 | 4 | 0 | ✅ `reviewer="codex"` |
| 8 (反映 2回目) | **codex** | - | - | - | - |

累計: **Must Fix 11 / Should Fix 16 / Nice to Have 7 = 34 findings、全反映**

### Codex で発見された silent bug (重要)
- **S5-001**: Phase A0 checkpoint load 実装場所を `photon_mlx/inference.py:137` (実は WARNING 文言のみ) に誤指定 → `photon_mlx.trainer.load_checkpoint` が正しい実装場所
- **S5-002**: Phase B (新 LLM × PHOTON eval) と Phase C/D (vocab_size Qwen2.5 維持) の方針矛盾 → baseline-only に限定
- **S7-001**: #135 unblock 条件 (Phase A0+A 完了 vs Issue 全体完了) の不整合 → Phase A0+A 完了に統一

## Phase 3: マルチステージ設計レビュー

| Stage | 担当 | Must Fix | Should Fix | Nice to Have | reviewer 検証 |
|-------|------|---------|-----------|------------|------------|
| 1 (設計原則) | claude-opus | 1 | 7 | 6 | - |
| 2 (整合性) | claude-opus | 1 | 4 | 3 | - |
| 3 (影響分析) | **codex** | 2 | 2 | 0 | ✅ `reviewer="codex"` |
| 4 (セキュリティ) | **codex** | 0 | 5 | 1 | ✅ `reviewer="codex"` |

累計: **Must Fix 4 / Should Fix 18 / Nice to Have 10 = 32 findings、全反映**

### Codex で発見された silent bug
- **DR3-001**: `model.checkpoint_path` を `.safetensors` 単体パスとして例示 → 実 API は `weights.npz` + `state.json` directory を要求 → directory 形式に修正
- **DR3-002**: §3 fail-fast 方針と §4 / §10 unit test 例 (WARNING 継続) の矛盾 → fail-fast に統一

## Phase 5: TDD 実装 (Phase A0 のみ)

### 実装ファイル

| ファイル | 種別 | 行数 | 内容 |
|---------|------|------|------|
| `baseline_reporag/photon_pipeline.py` | M | +187 | 3 helper functions + `_build_photon_deps` 拡張 |
| `baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py` | A | +536 | 22 unit tests |
| `tests/test_pipeline_factory_yaml_invariants.py` | M | +21 | 新規 invariant test |
| `configs/institutional_docs_photon.yaml` | M | (small) | checkpoint_path placeholder |
| `docs/deployment.md` | M | (small) | PHOTON_CHECKPOINT_ROOT / PHOTON_ALLOW_RANDOM_INIT 説明 |
| `docs/troubleshooting.md` | M | (small) | checkpoint load 失敗時の対処 |

### Codex Code Review 結果

**Iteration 1**: 4 findings (high 2, medium 2)
- CB-001 (high): PHOTON_CHECKPOINT_ROOT 相対 path 解決バグ → 修正
- CB-002 (medium): `.exists()` → `.is_file()` → 修正
- CB-003 (high): PHOTON_ALLOW_RANDOM_INIT を Phase A eval で許す案内 → 削除
- CB-004 (medium): `_validate_repo_id` の例外文に raw 値 → redaction

**Iteration 2**: 2 findings (high 1, low 1)
- CB-005 (high): MLX import abort in fresh test environment → **follow-up issue 化** (本 PR 範囲外、ローカル env では tests pass)
- CB-006 (low): `from pathlib import Path, os` typo → 修正

### 品質チェック結果

| チェック | コマンド | 結果 |
|---------|---------|------|
| ruff check | `ruff check .` | ✅ All checks passed! |
| ruff format | `ruff format --check .` | ✅ 150 files left unchanged |
| pytest (新規) | `pytest baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py tests/test_pipeline_factory_yaml_invariants.py -v` | ✅ 22+6 = 28 pass |
| pytest (全体) | `pytest baseline_reporag/tests/ tests/ photon_mlx/tests/ torch_ref/tests/ --timeout=60` | ✅ 1124 pass (既知 2 failure 未変) |

## PR 構成

3-PR 分割の PR #1 を本 worktree で作成:

| PR | 内容 | branch | 状態 |
|----|------|--------|------|
| **#1 (本 PR)** | Phase A0 (code) + Phase A (config + docs) | feature/issue-148-rebaseline | https://github.com/Kewton/photon-mlx/pull/150 |
| #2 | Phase B 新 LLM baseline-only eval | (未作成) | TBD |
| #3 | Phase C adoption + integration | (未作成) | TBD |

### Commit 構成

| commit | 内容 | files |
|--------|------|-------|
| `a38c8d3` | feat(photon_pipeline): fail-loud checkpoint loading + security guards | photon_pipeline.py + 2 tests |
| `e6a5171` | feat(institutional): set checkpoint_path + update deployment docs | yaml + 2 docs |
| `a39a777` | docs(issue-148): pm-auto-issue2dev completion artifacts | workspace/* |

## 残作業 (本 PR merge 後)

1. mulmoclaude 600-step ckpt の所在確認・配置 (担当: kewton)
2. Phase A eval 実行:
   - `python -m baseline_reporag.cli --config configs/institutional_docs_photon.yaml ...` smoke test
   - FastAPI MT eval × 2 runs
   - Institutional MT eval × 2 runs
3. レポート作成 (`reports/gate2_judgment_v5_post_s7001.md`, `reports/institutional_photon_mt_eval_v2.md`)
4. **#135 Phase 6-8 解禁** (本 PR merge + Phase A eval 完了が必要条件)

## Follow-up Issues 推奨

- **CB-005 follow-up**: `baseline_reporag/photon_pipeline.py` の MLX top-level import を遅延化、または test 側で sys.modules stubbing で fresh test env での abort を回避
- **mulmoclaude ckpt artifact**: リポ内 / HF hub / shared server のいずれに配置するかの方針確定

## 完了条件チェック

- [x] Phase 1 完了 (Issue 本文 34 findings 全反映)
- [x] Phase 2 完了 (設計方針書 842 行)
- [x] Phase 3 完了 (設計レビュー 32 findings 全反映)
- [x] Phase 4 完了 (作業計画 501 行)
- [x] Phase 5 完了 (Phase A0 TDD 実装、22 tests + 1 invariant、Codex review iter1+iter2 通過)
- [x] Phase 6 完了 (本書 + PR #150 作成)
- [x] reviewer 検証 (Stage 5/7 issue review + Stage 3/4 design review すべて `reviewer="codex"`)
- [x] ruff check / format pass
- [x] pytest 全パス (除既知 2 件)
- [x] 3-commit 分割で git push 完了
- [x] PR #150 を develop 向けに作成済 (https://github.com/Kewton/photon-mlx/pull/150)
