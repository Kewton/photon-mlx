# 設計方針書: Issue #179 — 比較モード UI (parallel baseline/PHOTON comparison)

## 概要

| 項目 | 内容 |
|------|------|
| Issue | #179 feat(ui): parallel baseline/PHOTON comparison mode in chat (G7) |
| 作成日 | 2026-04-30 |
| ステータス | Draft |

## 目的

Streamlit チャットページに「比較モード」トグルを追加し、1 つの質問入力で baseline と PHOTON の応答を side-by-side で並列表示する。ユーザーが PHOTON のメリット（レイテンシ削減・drift 検知・working memory）を直感的に体感できる手段を提供する。

---

## アーキテクチャ概要

```
┌──────────────────────────────────────────────────────┐
│               Streamlit UI (app/photon_app.py)        │
│  ┌─────────────────────────────────────────────────┐ │
│  │  page_chat()                                     │ │
│  │  ┌────────────┐  ⚖ 比較モード toggle            │ │
│  │  │ 通常モード  │──────────────────────────────── │ │
│  │  │ (既存)     │  比較モード ON                   │ │
│  │  └────────────┘  ┌──────────┬──────────────────┐│ │
│  │                  │Baseline  │ PHOTON            ││ │
│  │                  │ col (left)│ col (right)       ││ │
│  │                  └──────────┴──────────────────┘│ │
│  │                  Delta セクション                 │ │
│  └─────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────┤
│   baseline_reporag/comparison.py  (新規)              │
│   ┌────────────────────────────────────────────────┐ │
│   │ VariantResult  (移動元: scripts/compare_*.py)  │ │
│   │ ComparisonResult                               │ │
│   │ DeltaResult                                    │ │
│   │                                                │ │
│   │ [公開 API - 2つに絞る (DR1-002)]               │ │
│   │ run_variant_with_pipeline(pipeline, ...)       │ │
│   │   ← 低レベル実行（UI・テスト共用）               │ │
│   │ compare(baseline_pipeline, photon_pipeline,    │ │
│   │   question, ...) -> ComparisonResult           │ │
│   │   ← 並列実行 + delta 計算（高レベル）            │ │
│   │                                                │ │
│   │ [private]                                      │ │
│   │ _build_and_run(config_path, ...) -> VariantResult│ │
│   │   ← CLI/scripts 用ラッパー (DR1-001)            │ │
│   │ compute_delta(baseline, photon) -> DeltaResult │ │
│   └────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────┤
│   baseline_reporag/pipeline_factory.py (既存)         │
│   build_pipeline(cfg) / override_repo_for_pipeline()  │
├──────────────────────────────────────────────────────┤
│   scripts/compare_baseline_photon.py (既存 → 薄い re-export) │
└──────────────────────────────────────────────────────┘
```

---

## データモデル

### VariantResult（移動）

`scripts/compare_baseline_photon.py` から `baseline_reporag/comparison.py` へ移動。

```python
@dataclass
class VariantResult:
    variant_id: str          # "baseline" | "photon"
    config_path: str
    answer: str
    cited_chunk_ids: list[str]
    no_citation: bool
    latency_total_ms: float
    latency_retrieval_ms: float
    latency_generation_ms: float
    memory_peak_mb: float
```

### ComparisonResult（新規）

```python
@dataclass
class ComparisonResult:
    question: str
    baseline: VariantResult
    photon: VariantResult
    delta: DeltaResult
```

### DeltaResult（新規）

MVP スコープとして `answer_similarity` は除外（DR1-004 / YAGNI）。

```python
@dataclass
class DeltaResult:
    latency_delta_ms: float         # photon - baseline
    latency_delta_pct: float        # (delta / baseline) * 100
    cited_overlap_jaccard: float    # |A∩B| / |A∪B|
    # answer_similarity は MVP 外。将来の拡張として別 Issue で対応。
```

---

## 設計判断

### 判断 #1: comparison.py の API 設計（CLI vs UI 分離）

**選択肢**:
- A: `run_comparison(question, baseline_config_path, photon_config_path, ...)` のみ — 毎回 `build_pipeline` を呼ぶ
- B: `run_comparison_with_pipelines(question, baseline_pipeline, photon_pipeline, ...)` を追加 — UI は cached pipeline を渡す
- C: A のみで UI も対応（キャッシュは UI 層で管理、comparison.py は pure function）

**決定**: **API を2つに絞る（DR1-001/002/003 反映）**
- 公開 API: `run_variant_with_pipeline`（低レベル実行）+ `compare`（高レベル並列実行+delta）
- `run_variant`（config_path 版）は `_build_and_run` として private 化し、`scripts/compare_baseline_photon.py` の re-export が使う形にする
- クエリ実行と `VariantResult` への変換ロジックは `run_variant_with_pipeline` に一本化し DRY を維持

**理由**:
- `run_variant_with_pipeline` に実行ロジックを集約することで DRY 達成（DR1-003）
- 公開 API を2つに絞ることで KISS 達成（DR1-002）
- CLI 向けの pipeline 構築は private 関数 `_build_and_run` に隠蔽（DR1-001）
- `comparison.py` は `st.session_state` に依存しない純粋な計算モジュール

**トレードオフ**:
- メリット: API surface が小さく、利用者がどれを呼ぶか迷わない
- デメリット: `run_variant` が外部から呼べなくなるため、既存 CLI テストの修正が必要
- リスク: 低（`scripts/compare_baseline_photon.py` は `_build_and_run` を呼ぶか、scripts 内に同等実装を持つ）

---

### 判断 #2: 並列実行の実装方式

**選択肢**:
- A: `concurrent.futures.ThreadPoolExecutor` — GIL のため CPU バウンドは並列化されないが、IO バウンド（MLX 推論）には有効
- B: `asyncio` — Streamlit との相性が悪い（event loop 競合）
- C: 逐次実行 — 実装が最もシンプルだが体験が悪い

**決定**: **選択肢 A** — `ThreadPoolExecutor(max_workers=2)` で baseline/PHOTON を並列実行。

**理由**:
- MLX 推論はスレッドをブロックするが GIL を解放するため実質並列実行が可能
- 各スレッドは `VariantResult` を返すのみで `session_state` に触れない（スレッドセーフ）
- メインスレッドで `executor.map` / `as_completed` の結果を受け取り `st.rerun()` で更新

**トレードオフ**:
- メリット: シンプルな実装、Streamlit と相性が良い
- デメリット: Python GIL により完全な CPU 並列にはならない
- リスク: MLX が GIL を解放しない箇所がある場合は逐次になる可能性（許容）

---

### 判断 #3: pipeline cache eviction の拡張

**現状**: `_run_query` は `prefix = f"pipeline_{proj.name}_"` でマッチして古いキーを evict する。

**問題**: 比較モード用キー `pipeline_{proj.name}_comparison_baseline` も同 prefix にマッチして evict される。

**決定**: eviction ロジックを `_COMPARISON_KEY_SUFFIX = ("_comparison_baseline", "_comparison_photon")` 等のサフィックスで除外する。比較モード専用の eviction 関数 `_evict_comparison_pipelines(proj_name)` を別途設ける。

```python
# eviction 対象外のサフィックス
_COMPARISON_SUFFIXES = ("_comparison_baseline", "_comparison_photon")

def _is_comparison_pipeline_key(key: str) -> bool:
    return any(key.endswith(s) for s in _COMPARISON_SUFFIXES)
```

---

### 判断 #4: 比較モード履歴の永続化

**決定**: 比較モード用の会話履歴（`f"chat_{project_name}_baseline"` / `f"chat_{project_name}_photon"`）は `st.session_state` のみで管理し、`AppState.chat_histories` には保存しない。

**理由**:
- `AppState` は `.cache/photon_app_state.json` に永続化されるためキャッシュファイルが肥大化する
- 比較モードはセッション横断の履歴管理を必要としない（毎セッション独立）

---

### 判断 #5: provider 検証の実施タイミング

**決定**: トグル表示前に `proj.config_path` と `proj.photon_config_path` をロードし、`cfg.model.provider` を確認してトグルの有効/無効を決定する。

検証条件（全て満たす場合のみトグル有効化）:
1. `proj.photon_config_path` が設定されている（空でない）
2. `proj.config_path != proj.photon_config_path`（同一 config でない）
3. baseline cfg の `model.provider` が `"photon"` でない
4. PHOTON cfg の `model.provider` が `"photon"` である

---

## レイヤー構成と責務

| モジュール | 変更種別 | 責務 |
|-----------|---------|------|
| `baseline_reporag/comparison.py` | **新規** | `VariantResult`, `ComparisonResult`, `DeltaResult`, `run_variant_with_pipeline`（公開）, `compare`（公開）, `_build_and_run`（private）, `compute_delta` |
| `scripts/compare_baseline_photon.py` | **変更（薄い re-export）** | `VariantResult`, `run_variant` を `baseline_reporag.comparison` から re-export。`main()` は変更なし |
| `app/photon_app.py` | **変更** | `page_chat()`: 比較モードトグル + 2column + Delta 表示。`_pipeline_cache_key`/eviction: 比較モード対応。クリアボタン: 両履歴対応 |
| `tests/test_comparison.py` | **新規** | `VariantResult`, `ComparisonResult`, `compute_delta`, `run_comparison_with_pipelines` のユニットテスト（pipeline はモック） |
| `tests/test_compare_baseline_photon.py` | **変更** | `run_variant` / `VariantResult` の import 先を `baseline_reporag.comparison` に変更（または scripts の re-export 経由のまま） |

---

## セキュリティ設計

| 脅威 | 対策 |
|------|------|
| `session_state` スレッド安全性 | 各スレッドは結果 dataclass を返すのみ（`session_state` 書き込みはメインスレッドのみ） |
| config path の外部入力 | `proj.config_path` / `proj.photon_config_path` は既存の Project dataclass 経由で管理済み |
| MLX OOM | 比較モード有効化時に 16GB 以下の環境では警告表示（必須ではない）|

---

## 実装方針（主要変更点）

### 1. `baseline_reporag/comparison.py` 新規作成

DR1-001/002/003 反映: 公開 API を2つに絞り、実行ロジックを `run_variant_with_pipeline` に一本化。

```python
from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class VariantResult:
    variant_id: str
    config_path: str
    answer: str
    cited_chunk_ids: list[str]
    no_citation: bool
    latency_total_ms: float
    latency_retrieval_ms: float
    latency_generation_ms: float
    memory_peak_mb: float

@dataclass
class DeltaResult:
    latency_delta_ms: float
    latency_delta_pct: float
    cited_overlap_jaccard: float
    # answer_similarity: MVP 外 (DR1-004)

@dataclass
class ComparisonResult:
    question: str
    baseline: VariantResult
    photon: VariantResult
    delta: DeltaResult

# ===== 公開 API =====

def run_variant_with_pipeline(
    pipeline, question: str, session_id: str, repo_id: str, variant_id: str,
    config_path: str = "",
) -> VariantResult:
    """低レベル実行 API（UI・テスト共用）。実行ロジックはここに一本化。"""
    result = pipeline.query(question=question, session_id=session_id, repo_id=repo_id)
    return VariantResult(
        variant_id=variant_id,
        config_path=config_path,
        answer=result.answer,
        cited_chunk_ids=list(result.cited_chunk_ids),
        no_citation=bool(result.no_citation),
        latency_total_ms=float(result.latency.total_ms),
        latency_retrieval_ms=float(result.latency.retrieval_ms),
        latency_generation_ms=float(result.latency.generation_ms),
        memory_peak_mb=float(result.memory.peak_mb),
    )

def compare(
    baseline_pipeline, photon_pipeline,
    question: str, repo_id: str,
    baseline_session_id: str, photon_session_id: str,
) -> ComparisonResult:
    """高レベル API: ThreadPoolExecutor で並列実行 + delta 計算。"""
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_b = ex.submit(run_variant_with_pipeline, baseline_pipeline, question, baseline_session_id, repo_id, "baseline")
        f_p = ex.submit(run_variant_with_pipeline, photon_pipeline,   question, photon_session_id,   repo_id, "photon")
    b, p = f_b.result(), f_p.result()
    return ComparisonResult(question=question, baseline=b, photon=p, delta=compute_delta(b, p))

def compute_delta(baseline: VariantResult, photon: VariantResult) -> DeltaResult:
    """Jaccard overlap + latency delta 計算（pure function）。"""
    ...

# ===== private =====

def _build_and_run(
    variant_id: str, config_path: str, question: str, repo_id: str, session_id: str,
) -> VariantResult:
    """CLI / scripts 向け: config_path から pipeline を構築して run_variant_with_pipeline に委譲。"""
    from baseline_reporag.config import load_config
    from baseline_reporag.pipeline_factory import build_pipeline, override_repo_for_pipeline
    cfg = load_config(config_path)
    override_repo_for_pipeline(cfg, repo_id or cfg.repo.repo_id)
    pipeline = build_pipeline(cfg)
    return run_variant_with_pipeline(pipeline, question, session_id, repo_id or cfg.repo.repo_id, variant_id, config_path)
```

### 2. `scripts/compare_baseline_photon.py` — `_build_and_run` 経由の re-export

```python
# 後方互換: run_variant は _build_and_run の薄いラッパーとして維持
from baseline_reporag.comparison import _build_and_run, VariantResult  # noqa: F401

def run_variant(variant_id, config_path, question, repo_id, session_id) -> VariantResult:
    return _build_and_run(variant_id, config_path, question, repo_id, session_id)
```

### 3. `app/photon_app.py:page_chat()` の変更骨格

```python
def page_chat():
    ...
    # provider 検証
    comparison_eligible = _check_comparison_eligible(proj)

    comparison_mode = st.toggle(
        "⚖ 比較モード",
        disabled=not comparison_eligible,
        help="baseline と PHOTON を並列比較します" if comparison_eligible else "baseline/PHOTON config が未設定です",
    )

    if comparison_mode:
        _render_comparison_mode(proj, project_name)
    else:
        _render_normal_mode(proj, project_name)  # 既存ロジックをリファクタ
```

---

## 品質基準

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| テスト | `python -m pytest` | 全テストパス（既存 510 件 + 新規） |
| リント | `ruff check .` | 警告 0 件 |
| フォーマット | `ruff format --check .` | 差分なし |

---

## 影響ファイル一覧

| ファイル | 変更種別 | 優先度 |
|---------|---------|--------|
| `baseline_reporag/comparison.py` | 新規 | 高 |
| `app/photon_app.py` | 変更 | 高 |
| `scripts/compare_baseline_photon.py` | 変更（re-export） | 中 |
| `tests/test_comparison.py` | 新規 | 高 |
| `tests/test_compare_baseline_photon.py` | 変更 | 中 |

---

## 未解決事項

- [ ] 16GB 以下の環境でのメモリ警告の実装有無 → optional として Issue に記載済み
- [ ] `tests/test_compare_baseline_photon.py` は `run_variant` (scripts 内のラッパー) 経由で引き続きテスト可能か確認（re-export 戦略変更による影響）

## 変更履歴

| 日付 | 変更内容 |
|------|---------|
| 2026-04-30 | 初版作成 |
| 2026-04-30 | DR1-001/002/003 反映: 公開 API を `run_variant_with_pipeline` + `compare` の2つに絞り、`_build_and_run` を private 化 |
| 2026-04-30 | DR1-004 反映: `DeltaResult.answer_similarity` を MVP スコープ外として削除 |
