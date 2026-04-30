# Issue #179 — TDD実装 進捗レポート

## Issue #179: feat(ui): parallel baseline/PHOTON comparison mode in chat (G7)

---

## 実装フェーズ結果

| Phase | タスク | ステータス |
|-------|--------|-----------|
| 1 | `baseline_reporag/comparison.py` 新規作成 | 完了 |
| 2 | `scripts/compare_baseline_photon.py` リファクタリング | 完了 |
| 3 | `tests/test_comparison.py` 新規作成 | 完了 |
| 3 | `tests/test_compare_baseline_photon.py` 既存テスト確認 | 完了 |
| 4 | `app/photon_app.py` 比較モード UI 実装 | 完了 |

---

## 生成・変更ファイル

### 新規作成
- `baseline_reporag/comparison.py` — VariantResult / DeltaResult / ComparisonResult / run_variant_with_pipeline / compare / compute_delta / _build_and_run
- `tests/test_comparison.py` — 12 テスト

### 変更
- `scripts/compare_baseline_photon.py` — VariantResult re-export、build_pipeline をモジュール名前空間に公開、run_variant をモンキーパッチ互換に修正
- `app/photon_app.py` — 比較モード UI 追加（トグル / 2カラム / delta セクション / clear 拡張 / eviction フィルタ）

---

## テスト結果

### 新規テスト (12件)
- `TestVariantResult::test_dataclass_fields` — PASSED
- `TestRunVariantWithPipeline::test_maps_query_result_to_variant_result` — PASSED
- `TestRunVariantWithPipeline::test_no_citation_flag_preserved` — PASSED
- `TestRunVariantWithPipeline::test_config_path_defaults_to_empty` — PASSED
- `TestComputeDelta::test_latency_delta_and_pct` — PASSED
- `TestComputeDelta::test_cited_overlap_jaccard_full` — PASSED
- `TestComputeDelta::test_cited_overlap_jaccard_no_overlap` — PASSED
- `TestComputeDelta::test_cited_overlap_jaccard_partial` — PASSED
- `TestComputeDelta::test_cited_overlap_both_empty` — PASSED
- `TestComputeDelta::test_latency_delta_zero_baseline` — PASSED
- `TestCompare::test_returns_comparison_result` — PASSED
- `TestCompare::test_both_pipelines_queried` — PASSED

### 既存テスト (6件)
- `TestRunVariant::test_returns_variant_result_with_pipeline_data` — PASSED
- `TestRunVariant::test_falls_back_to_cfg_repo_id_when_repo_id_empty` — PASSED
- `TestPrintTextReport::test_emits_question_variants_and_summary` — PASSED
- `TestPrintTextReport::test_warns_on_no_citation` — PASSED
- `TestToJsonPayload::test_includes_question_and_variant_dicts` — PASSED
- `TestIndent::test_indents_each_line` — PASSED

### 全体テスト: 1601 passed, 3 failed (pre-existing)
- `tests/test_generate_training_corpus.py` の 2 件は CLAUDE.md 記載の既知の既存失敗
- `baseline_reporag/tests/test_photon_pipeline.py::TestBuildPhotonDepsRealTokenizer::test_vocab_size_mismatch_raises` は既存失敗

---

## 品質チェック結果

| チェック項目 | コマンド | 結果 |
|-------------|----------|------|
| テスト | `python -m pytest` | 1601 passed, 3 pre-existing failures |
| リント | `ruff check .` | 警告 0 件 |
| フォーマット | `ruff format --check . --exclude scripts/train_photon.py` | 差分なし（train_photon.py は既存の未フォーマット） |

---

## 設計判断メモ

1. **`run_variant_with_pipeline` を DRY の中心に置く**: UI（パイプラインキャッシュ済み）と CLI（設定からビルド）の両方がここに委譲。
2. **モンキーパッチ互換性**: `scripts/compare_baseline_photon.py` のモジュール名前空間に `build_pipeline` を保持することで既存テストが継続動作。
3. **eviction フィルタ**: `_is_comparison_pipeline_key()` で比較モード用キャッシュが通常モードの eviction から保護される。
4. **比較履歴は session_state のみ**: `AppState.chat_histories` への永続化なし（比較モードは揮発性）。
5. **プロバイダー検証**: `_check_comparison_eligible()` が両設定の `model.provider` を検査し、トグルの有効・無効を制御。
