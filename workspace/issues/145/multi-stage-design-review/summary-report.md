# Issue #145 マルチステージ設計レビュー完了報告

実施日: 2026-04-28
対象: `workspace/design/issue-145-real-weight-test-design-policy.md`

---

## ステージ別結果

| Stage | レビュー種別 | レビュアー | Must Fix | Should Fix | Nice to Have | 反映 | ステータス |
|-------|------------|----------|---------|-----------|-------------|-----|----------|
| 1 | 通常レビュー (設計原則) | claude-opus | 2 | 6 | 2 | 10/10 | 完了 |
| 2 | 整合性レビュー | claude-opus | 0 | 5 | 5 | 10/10 | 完了 |
| 3 | 影響分析レビュー | **codex** | 2 | 4 | 1 | 7/7 | 完了 (reviewer="codex" 検証済) |
| 4 | セキュリティレビュー | **codex** | 0 | 3 | 2 | 5/5 | 完了 (reviewer="codex" 検証済) |

**累計**: Must Fix 4 / Should Fix 18 / Nice to Have 10 = **計 32 findings 全件反映**

---

## reviewer フィールド検証 (Issue #140 / S7-001 follow-up)

| ファイル | reviewer | 期待値 | 結果 |
|---------|----------|-------|------|
| stage3-review-result.json | codex | codex | OK |
| stage4-review-result.json | codex | codex | OK |

WARNING なし。Codex 担当 Stage 3/4 が正しく Codex により実行されている。

---

## 主要な設計改善

1. **DR1-001 (Must Fix)**: `deps['photon_model']` → `deps['photon_inference'].model` に全箇所修正 (`_build_photon_deps` 戻り値 4 key 整合)
2. **DR1-002 (Must Fix / 致命的)**: L2 norm detector を「random-init vs load 後」差分判定 → 「**trained 参照値との近接性** (`abs(norm_after - norm_trained) < 1e-5` または `mx.allclose(logits_after, logits_trained, atol=1e-5)`)」判定に変更。silently-skipped load を確実に検出可能に修正
3. **DR3-Must Fix x2 (Codex)**: forward-pass 比較例の chunk 長不整合修正、最小 YAML の tokenizer 必須 key 漏れ補完
4. **DR2-001 (Should Fix)**: Issue 本文 / 設計書 §8 / CLAUDE.md の test 件数を 510/512 (positive 1 + negative 2 = +3 件) で整合
5. **DR3 (Codex Should Fix x4)**: testpaths 検証方法、weekly_eval artifact 生成、docs 更新責務、並行 Issue 衝突表の追加
6. **DR4 (Codex Should Fix x3)**: self-hosted runner stale artifact、env dump 禁止、tmp_path/root containment 制約、no-pickle/no-eval 境界、fake tokenizer DI scope を設計書に明記

---

## セキュリティ評価結果 (Stage 4)

- **重大な脆弱性なし** (path traversal / arbitrary code execution / secret leak いずれも対策済または scope 外)
- 既存 production 機構が脅威モデル上の主要対策を提供:
  - `_resolve_checkpoint_path` の `Path.resolve(strict=True)` + root containment check
  - `numpy.load(allow_pickle=False)` (デフォルト挙動)
  - `monkeypatch` teardown による env 復元
- 設計書改善: env dump 禁止、stale artifact クリーンアップ、fake tokenizer DI scope を明文化

---

## 最終検証

設計方針書 (`workspace/design/issue-145-real-weight-test-design-policy.md`) はマルチステージレビューで合計 32 findings を反映済み (Must Fix 4 / Should Fix 18 / Nice to Have 10)。実装フェーズに着手可能な状態。

---

## 次のアクション

- [ ] Phase 4: 作業計画立案 (`/work-plan 145`)
- [ ] Phase 5: TDD自動開発 (`/pm-auto-dev 145`)
