# 作業計画: Issue #179 — 比較モード UI

## Issue 概要

**Issue番号**: #179  
**タイトル**: feat(ui): parallel baseline/PHOTON comparison mode in chat (G7)  
**サイズ**: M  
**優先度**: High  
**依存Issue**: なし（PR #168 で compare スクリプト実装済み）

---

## Phase 1: コアモジュール実装（baseline_reporag/comparison.py）

### Task 1.1: データモデル定義
- **成果物**: `baseline_reporag/comparison.py` — `VariantResult`, `ComparisonResult`, `DeltaResult`
- **依存**: なし
- **内容**:
  - `VariantResult`: variant_id, config_path, answer, cited_chunk_ids, no_citation, latency_total_ms, latency_retrieval_ms, latency_generation_ms, memory_peak_mb
  - `DeltaResult`: latency_delta_ms, latency_delta_pct, cited_overlap_jaccard（answer_similarity は MVP 外）
  - `ComparisonResult`: question, baseline, photon, delta

### Task 1.2: `run_variant_with_pipeline` 実装（低レベル API）
- **成果物**: `baseline_reporag/comparison.py`
- **依存**: Task 1.1
- **内容**:
  - `pipeline.query(question, session_id, repo_id)` → `VariantResult` への変換
  - `QueryResult.latency` (LatencyBreakdown) と `memory` (MemorySnapshot) のフィールドマッピング
  - 実行ロジックはここに一本化（DRY 原則）

### Task 1.3: `compute_delta` 実装
- **成果物**: `baseline_reporag/comparison.py`
- **依存**: Task 1.1
- **内容**:
  - `latency_delta_ms = photon.latency_total_ms - baseline.latency_total_ms`
  - `latency_delta_pct = (delta / baseline.latency_total_ms) * 100`
  - `cited_overlap_jaccard = |A∩B| / |A∪B|`（cited_chunk_ids の Jaccard 係数）

### Task 1.4: `compare` 実装（高レベル API）
- **成果物**: `baseline_reporag/comparison.py`
- **依存**: Task 1.2, 1.3
- **内容**:
  - `ThreadPoolExecutor(max_workers=2)` で baseline/PHOTON を並列実行
  - `run_variant_with_pipeline` を各スレッドで呼び出し、メインスレッドで結果を集約

### Task 1.5: `_build_and_run` 実装（private、CLI 用）
- **成果物**: `baseline_reporag/comparison.py`
- **依存**: Task 1.2
- **内容**:
  - `config_path` から `load_config` → `build_pipeline` → `run_variant_with_pipeline` に委譲
  - `scripts/compare_baseline_photon.py` の後方互換のため private として公開

---

## Phase 2: scripts 側リファクタリング

### Task 2.1: `scripts/compare_baseline_photon.py` 更新
- **成果物**: `scripts/compare_baseline_photon.py`
- **依存**: Phase 1 完了
- **内容**:
  - `VariantResult` を `baseline_reporag.comparison` から re-export
  - `build_pipeline`, `override_repo_for_pipeline` を scripts モジュールの名前空間にインポート（モンキーパッチ可能に）
  - `run_variant` を `_build_and_run` の薄いラッパーとして再定義
  - `main()` は変更なし

---

## Phase 3: テスト実装

### Task 3.1: `tests/test_comparison.py` 新規作成
- **成果物**: `tests/test_comparison.py`
- **依存**: Phase 1 完了
- **内容**:
  - `run_variant_with_pipeline`: pipeline をモックして `VariantResult` 変換を検証
  - `compute_delta`: 正常系（latency delta, Jaccard overlap）
  - `compare`: `run_variant_with_pipeline` をモックして並列実行・delta 計算を検証

### Task 3.2: `tests/test_compare_baseline_photon.py` 更新確認
- **成果物**: `tests/test_compare_baseline_photon.py`
- **依存**: Task 2.1
- **内容**:
  - `VariantResult` / `run_variant` が scripts モジュール経由で依然アクセス可能か確認
  - `monkeypatch.setattr(module, 'build_pipeline', ...)` が引き続き機能するか確認
  - 必要に応じてテストを `baseline_reporag.comparison` 直接テストに書き直し

---

## Phase 4: UI 実装（app/photon_app.py）

### Task 4.1: `_check_comparison_eligible` 実装
- **成果物**: `app/photon_app.py`
- **依存**: Phase 1 完了
- **内容**:
  - `proj.photon_config_path` が設定されているか
  - `proj.config_path != proj.photon_config_path`
  - baseline cfg の `model.provider != "photon"`
  - PHOTON cfg の `model.provider == "photon"`
  - config ロード失敗時は `False` を返す

### Task 4.2: pipeline cache eviction 拡張
- **成果物**: `app/photon_app.py`
- **依存**: なし（既存コード修正）
- **内容**:
  - `_COMPARISON_SUFFIXES = ("_comparison_baseline", "_comparison_photon")` 定数追加
  - `_is_comparison_pipeline_key(key)` ヘルパー追加
  - 既存 eviction ループに除外フィルタを追加:
    ```python
    keys_to_evict = [
        k for k in st.session_state
        if k.startswith(prefix) and k != pipeline_key
        and not _is_comparison_pipeline_key(k)
    ]
    ```

### Task 4.3: `_render_comparison_mode` 実装
- **成果物**: `app/photon_app.py`
- **依存**: Task 4.1, 4.2, Phase 1
- **内容**:
  - baseline / PHOTON pipeline を `pipeline_{proj.name}_comparison_baseline/photon` キーでキャッシュ
  - `compare(baseline_pipeline, photon_pipeline, question, proj.repo_id, ...)` 呼び出し
  - `st.columns(2)` で baseline 列 / PHOTON 列に結果表示
  - 各列: answer, latency, cited chunks, no_citation, drift panel（PHOTON 列のみ）
  - Delta セクション: latency_delta_pct, cited_overlap_jaccard
  - 比較モード専用フラグ `comparison_photon_unavailable_{project_name}` で PHOTON 利用不可を管理

### Task 4.4: `page_chat` にトグル追加
- **成果物**: `app/photon_app.py`
- **依存**: Task 4.1, 4.3
- **内容**:
  - `st.toggle("⚖ 比較モード", disabled=not comparison_eligible)` を追加
  - トグル ON: `_render_comparison_mode(proj, project_name, session_key)` を呼ぶ
  - トグル OFF: 既存の通常モードを呼ぶ（`_render_normal_mode` に切り出すか既存コードのまま）

### Task 4.5: クリアボタン拡張
- **成果物**: `app/photon_app.py`
- **依存**: Task 4.3
- **内容**:
  - クリア時に `_baseline` / `_photon` サフィックスの session_state キーも削除
  - 比較モード履歴は `st.session_state` のみ管理（`save()` は不要）

---

## Phase 5: 品質チェック

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| テスト | `python -m pytest` | 全テストパス（既存 + 新規） |
| リント | `ruff check .` | 警告 0 件 |
| フォーマット | `ruff format --check .` | 差分なし |

---

## 実装順序（推奨）

```
Task 1.1 → 1.2 → 1.3 → 1.4 → 1.5
→ Task 3.1（comparison テスト）
→ Task 2.1（scripts リファクタ）
→ Task 3.2（既存テスト確認）
→ Task 4.1 → 4.2 → 4.3 → 4.4 → 4.5
→ Phase 5（品質チェック）
```

## Definition of Done

- [ ] `baseline_reporag/comparison.py` 新規作成（公開 API: `run_variant_with_pipeline`, `compare`）
- [ ] `scripts/compare_baseline_photon.py` リファクタ（re-export + `run_variant` ラッパー）
- [ ] `tests/test_comparison.py` 新規作成（全パス）
- [ ] `tests/test_compare_baseline_photon.py` 既存テスト全パス
- [ ] `app/photon_app.py` 比較モードトグル + 2column 表示
- [ ] `python -m pytest` 全パス
- [ ] `ruff check .` 警告 0 件
- [ ] `ruff format --check .` 差分なし
