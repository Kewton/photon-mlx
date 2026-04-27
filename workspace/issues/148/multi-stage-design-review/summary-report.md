# マルチステージ設計レビュー完了報告 — Issue #148

**実施日**: 2026-04-27
**設計方針書**: `workspace/design/issue-148-rebaseline-design-policy.md`

## ステージ別結果

| Stage | レビュー種別 | レビュアー | Must Fix | Should Fix | Nice to Have | 反映数 | ステータス |
|-------|------------|----------|---------|-----------|------------|-------|----------|
| 1 | 通常 (設計原則 SOLID/KISS/YAGNI/DRY) | claude-opus | 1 | 7 | 6 | 14/14 | 完了 |
| 2 | 整合性 | claude-opus | 1 | 4 | 3 | 8/8 | 完了 |
| 3 | 影響分析 | **codex** | 2 | 2 | 0 | 4/4 | 完了 |
| 4 | セキュリティ | **codex** | 0 | 5 | 1 | 6/6 | 完了 |

**累計**: Must Fix 4 / Should Fix 18 / Nice to Have 10 = 32 findings、すべて反映 (skipped 0)

## reviewer フィールド検証 (Issue #140 / S7-001 follow-up)

| ファイル | reviewer | 判定 |
|---------|----------|------|
| stage3-review-result.json | `codex` | OK |
| stage4-review-result.json | `codex` | OK |

WARNING なし。Codex stage は必須化済 (skip なし)。

## 主要な発見と修正

### Stage 1 (設計原則) - Claude opus
- **DR1-001 (Must Fix)**: `checkpoint_path` 設定時の load 失敗を fail-soft (random-init 続行) としていた設計が S7-001 再発防止と矛盾 → fail-fast (`RuntimeError`) を default にし、`PHOTON_ALLOW_RANDOM_INIT=1` でのみ test/CI 限定で fail-soft 有効化
- DR1-002〜DR1-007: lazy import 整理、cfg fallback 削除、photon_rag backbone 差分対策の正当化、代替 slug 定量基準、smoke test peak RSS の全 LLM 共通化、PR commit 分離

### Stage 2 (整合性) - Claude opus
- **DR2-001 (Must Fix)**: photon_600m_paper.yaml を Qwen2.5 系と誤記 → 実コードは LLaMA-2 系 (vocab 32000) に訂正
- DR2-002〜DR2-005: API 表現統一、tokenizer_id mock 追加、bench/ スクリプト影響範囲明示、Drift metrics follow-up タイミング順序化

### Stage 3 (影響分析) - **Codex** (silent bug 発見)
- **DR3-001 (Must Fix)**: `model.checkpoint_path` を `.safetensors` 単体パスとして例示 → 実 API `photon_mlx.trainer.load_checkpoint` は directory (`weights.npz` + `state.json`) を要求 → 設計を directory 形式に修正
- **DR3-002 (Must Fix)**: §3 fail-fast 方針と §4 データフロー / §10 unit test 例 (WARNING 継続) の矛盾 → unit test を `test_build_photon_deps_raises_on_load_failure_by_default` に修正
- DR3-003: backbone 自動出力影響範囲を `bench/run_all.py` + `bench/tests/test_run_all.py` に具体化
- **DR3-004 (Should Fix)**: `ru_maxrss` の OS 別単位差 (macOS bytes / Linux KiB) → platform 分岐 helper 明記

### Stage 4 (セキュリティ) - **Codex**
- **DR4-001 (Should Fix, path traversal)**: checkpoint_path の root containment 検証追加 (`PHOTON_CHECKPOINT_ROOT` 配下限定、symlink escape 拒否)
- **DR4-002 (Should Fix, 入力検証)**: `model.model_id` / `tokenizer_id` の HF repo-id allowlist (URL/local path/traversal/control char 拒否)
- **DR4-003 (Should Fix, command injection)**: 自動化で `subprocess.run([...], shell=False)` argv list のみ許可
- **DR4-004 (Should Fix, secrets)**: private model_id / token / absolute path の redaction、secret scan 受入条件追加
- **DR4-005 (Should Fix, supply chain)**: HF artifact の revision 記録、`trust_remote_code=True` 範囲外化
- DR4-006 (Nice to Have): rollback config (`baseline_qwen25.yaml`) の secret-free 条件独立記載

## 設計方針書の変化

| 時点 | 行数 |
|-----|------|
| 初期作成 | 534 行 |
| Stage 1 反映後 | 630 行 (+95) |
| Stage 2 反映後 | (Stage 1 + 8 finding 反映) |
| Stage 3 反映後 | (Codex Must Fix 2 件 等で大幅修正) |
| Stage 4 反映後 (最終) | **842 行** (セキュリティ section 大幅追記) |

## 次のアクション

- [x] 設計方針書がレビュー反映済み
- [ ] Phase 4: `/work-plan 148` で作業計画立案
- [ ] Phase 5: `/pm-auto-dev 148` で TDD 開発
