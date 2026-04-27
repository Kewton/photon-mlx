# Issue #148 マルチステージレビュー完了報告

**Issue**: #148 — `test(eval): re-establish true baseline — fixed PHOTON pipeline + new LLM upgrade (Qwen3.5-9B / Gemma4-26B)`
**実施日**: 2026-04-27
**Issue URL**: https://github.com/Kewton/photon-mlx/issues/148

## 仮説検証結果（Phase 0.5）

10 件の仮説をコードベースで照合:

| # | 仮説 | 判定 |
|---|------|------|
| H1 | S7-001 + #138 解消済 (#141, #146, #147 merged) | Confirmed |
| H2 | PhotonModel は random-init + StubTokenizer で動作していた | Confirmed |
| H3 | 現行 baseline LLM は Qwen2.5-Coder-14B-Instruct-4bit | Confirmed |
| H4 | configs/institutional_docs_photon.yaml に checkpoint_path 未明示 | Confirmed |
| H5 | mulmoclaude 600-step ckpt 存在 | Partially Confirmed |
| H6 | Latency は weight 非依存 | Confirmed |
| H7 | Baseline eval は photon_pipeline 非経由 | Confirmed |
| H8 | Gate 2 v4 数値 (NC baseline 21.7% / PHOTON 20.0%) | Confirmed |
| H9 | Qwen3.5-9B / Gemma4-26B が HF 上に存在 | **Unverifiable** |
| H10 | invariant test に LLM ハードコード | Confirmed (実は reranker model_id) |

## ステージ別結果

| Stage | レビュー種別 | レビュアー | Must Fix | Should Fix | Nice to Have | 反映数 | ステータス |
|-------|------------|----------|---------|-----------|------------|-------|----------|
| 1 | 通常レビュー（1回目） | claude-opus | 4 | 5 | 3 | - | 完了 |
| 2 | 指摘事項反映（1回目） | claude-sonnet | - | - | - | 12/12 | 完了 |
| 3 | 影響範囲レビュー（1回目） | claude-opus | 4 | 5 | 4 | - | 完了 |
| 4 | 指摘事項反映（1回目） | claude-sonnet | - | - | - | 13/13 | 完了 |
| 5 | 通常レビュー（2回目） | **codex** | 2 | 2 | 0 | - | 完了 |
| 6 | 指摘事項反映（2回目） | **codex** | - | - | - | 4/4 | 完了 |
| 7 | 影響範囲レビュー（2回目） | **codex** | 1 | 4 | 0 | - | 完了 |
| 8 | 指摘事項反映（2回目） | **codex** | - | - | - | 5/5 | 完了 |

**累計**: Must Fix 11 / Should Fix 16 / Nice to Have 7 = 34 findings、すべて反映 (skipped 0)

## reviewer フィールド検証 (Issue #140 / S7-001 follow-up)

| ファイル | reviewer | 判定 |
|---------|----------|------|
| stage5-review-result.json | `codex` | OK |
| stage7-review-result.json | `codex` | OK |

WARNING: なし (Codex が正規にレビュー実施、Claude による上書きなし)

## Codex クロスレビューで発見された silent bug (重要)

### S5-001 (Must Fix): Phase A0 checkpoint load 実装責務の誤指定
- **問題**: Stage 4 反映時、Phase A0 受入条件に `photon_mlx/inference.py:137 が load を実行することを確認` と記載されたが、実際の `inference.py:137` は WARNING 文言のみで load 実装はない
- **正しい実装場所**: `photon_mlx.trainer.load_checkpoint` / `_build_photon_deps` (baseline_reporag/photon_pipeline.py)
- **修正**: Phase A0 の対象を正しい実装場所に変更

### S5-002 (Must Fix): Phase B/C/D の方針矛盾
- **問題**: Phase B で「新 LLM × PHOTON eval 実施」、Phase C/D で「photon yaml は Qwen2.5 vocab_size: 152064 維持」が同時成立せず
- **修正**: Phase B を新 LLM 2 件の baseline-only eval に限定、新 LLM + PHOTON 本格 eval は Phase D/#135 へ延期

### S7-001 (Must Fix): #135 unblock 条件の不整合
- **問題**: Phase B baseline-only 化後、ブロック対象 (`本 Issue 解消が #135 GPU 着手前の必須前提`) と PR 運用方針 (Phase A0+A 完了で unblock) が矛盾
- **修正**: #135 GPU 着手の unblock 条件は Phase A0+A 完了であると明記

### S7-002 (Should Fix): YAML provider silent misconfiguration
- **問題**: Phase B 候補表の `provider` 列が YAML の `model.provider` と誤読され、`model.provider: "qwen3.5"` 等で書かれると意味論破綻
- **修正**: 表の列名を `family label` に変更、新規 yaml は `model.provider: "mlx_lm"` を維持

これらは opus 単独レビューが見落とした実装ギャップであり、Codex クロスレビューの価値が顕在化。

## Issue 本文の変化

| 時点 | サイズ |
|-----|-------|
| 初期 (original) | 9,589 bytes |
| Stage 2 反映後 | 14,080 bytes |
| Stage 4 反映後 | 28,677 bytes |
| Stage 6 反映後 | 29,747 bytes |
| Stage 8 反映後 (最終) | 33,973 bytes |

3.5 倍に拡張、特に Phase A0 (実装ギャップ対策)、Phase B (baseline-only 限定)、PR 運用方針、リスク節が大幅追記。

## 次のアクション

- [x] Issue 本文がレビュー反映済み
- [ ] Phase 2: `/design-policy 148` で設計方針書作成
- [ ] Phase 3: `/multi-stage-design-review 148`
- [ ] Phase 4: `/work-plan 148`
- [ ] Phase 5: `/pm-auto-dev 148`
