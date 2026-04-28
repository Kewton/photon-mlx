# Issue #144 PM Auto Dev 進捗報告 (iteration-1)

**Issue**: #144 (test(retrieval): ruri-small-v2 (V2) build failure follow-up — manual spiece.model download workaround)
**ブランチ**: feature/issue-144-ruri-v2-build
**完了日時**: 2026-04-28
**実装者**: Claude (TDD/refactor opus + Codex code review)

---

## 実装サマリー

本 PM-auto-dev iteration-1 では、Issue #144 の **コード変更が必要な範囲** (Task 1.1 + Task 1.2) を実装し、Codex によるクロスレビューを経て品質受入を完了した。build/eval 実行 (Task 2.x) と report 反映 (Task 3.x) は重い手動運用のため、後続の運用フェーズに委譲する。

### 実装対象 (本フェーズで完了)

| Task | 内容 | ステータス |
|------|------|----------|
| Task 1.1 | `requirements.txt` に `huggingface_hub` を追加 | ✅ Completed |
| Task 1.2 | `scripts/_setup_ruri_models.py` 新規作成 | ✅ Completed |
| Codex review CB-001 | `__main__` での sanitized error wrapping (DR4-002 担保) | ✅ Addressed |
| Codex review CB-002 | `RURI_MODELS`/`RURI_FILES` の空 tuple guard (fail-loud at import) | ✅ Addressed |
| Refactor | `def main()` 抽出 + 既存 scripts 命名規約整合 + 冗長コメント削減 | ✅ Completed |

### 実装範囲外 (運用フェーズへ)

| Task | 内容 | 理由 |
|------|------|------|
| Task 2.1 | `configs/_experiments/institutional_V2.yaml` 作成 | gitignored、手動運用 |
| Task 2.2 | variant config security validation gate 実行 | Task 2.1 後の手動 step |
| Task 2.3 | `scripts/ingest_repo.py` で V2 ingest 再実行 | ~5-10 分の手動運用 |
| Task 2.4 | `scripts/build_indexes.py` で V2 build | ~30-60 分の重い手動運用 (4228 docs re-embed) |
| Task 2.5 | `scripts/run_baseline_eval.py` で V2 eval | ~30 分の手動運用 |
| Task 3.1 | `reports/institutional_retrieval_ab.md` の V2 行更新 | Task 2.5 結果待ち |
| Task 3.2 | `configs/institutional_docs.yaml:84-87` コメント更新 | Task 2.5 結果待ち |
| Task 3.3 | (条件付き) V2 採用低確率分岐対応 | 発火時のみ (V2 が V4 を上回った場合) |

---

## 変更ファイル

### Commit 対象 (PR で push)

| ファイル | 変更種別 | 行数 |
|---------|---------|------|
| `requirements.txt` | modified | +3 行 (空行 + コメント + huggingface_hub) |
| `scripts/_setup_ruri_models.py` | added | 92 行 |

### Commit 対象外 (workspace/ 配下、設計成果物)

- `workspace/design/issue-144-ruri-v2-build-design-policy.md` (Phase 2/3 で作成・更新)
- `workspace/issues/144/issue-review/` (Phase 1 マルチステージレビュー成果)
- `workspace/issues/144/multi-stage-design-review/` (Phase 3 設計レビュー成果)
- `workspace/issues/144/work-plan.md` (Phase 4 作業計画)
- `workspace/issues/144/pm-auto-dev/iteration-1/` (Phase 5 TDD/受入/refactor 成果)

---

## 品質チェック結果

| チェック項目 | コマンド | 結果 |
|-------------|----------|------|
| Lint | `ruff check .` | ✅ PASS (All checks passed!) |
| Format (changed files) | `ruff format --check scripts/_setup_ruri_models.py requirements.txt` | ✅ PASS |
| Format (project-wide) | `ruff format --check .` | ⚠ 2 件 pre-existing (`baseline_reporag/photon_pipeline.py`, `scripts/train_photon.py`、本 task 由来でない) |
| Test (invariants) | `pytest tests/test_pipeline_factory_yaml_invariants.py -v` | ✅ PASS (7/7、bge-m3 pin 不変) |
| Test (no-scaffolding) | `pytest tests/test_no_scaffolding_in_prod.py -v` | ✅ PASS (1/1) |
| Test (full suite) | `pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | ✅ 1256/1259 PASS (3 件 fail はいずれも pre-existing、CLAUDE.md 既知) |
| Import smoke | `python -c "import importlib.util; ..."` | ✅ PASS (RURI_MODELS / RURI_FILES / 3 関数 callable 確認) |

---

## Codex クロスレビュー結果 (Phase 2.5)

| Finding | Severity | 内容 | 対応 |
|---------|---------|------|------|
| CB-001 | medium | `__main__` で sanitized error wrapping を実装で担保 (DR4-002) | ✅ 反映 (`__main__` block で try/except + `RURI_SETUP_DEBUG=1` opt-in traceback) |
| CB-002 | low | `RURI_MODELS`/`RURI_FILES` 空 tuple guard | ✅ 反映 (module load 時の fail-loud RuntimeError) |

両 finding 反映後、verdict: **needs_fix → pass** に upgrade 可能。

---

## 主要な実装ポイント

### 1. `scripts/_setup_ruri_models.py` の構造

```python
"""Module docstring (security/log hygiene policy 含む)"""

# Top-level imports + ImportError fallback
try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
except ImportError as e:
    raise ImportError("...") from e

# 定数 + fail-loud at import (CB-002)
RURI_MODELS: tuple[str, ...] = ("cl-nagoya/ruri-small-v2",)
RURI_FILES: tuple[str, ...] = ("spiece.model",)
if not RURI_MODELS: raise RuntimeError(...)
if not RURI_FILES: raise RuntimeError(...)

# 関数定義
def fetch_ruri_files() -> None: ...  # sanitized log (filename + size のみ)
def verify_load() -> None: ...        # SentenceTransformer 単独 load 検証

# def main() で entry point 抽出 (refactor で既存 scripts 命名規約と整合)
def main() -> None:
    try:
        fetch_ruri_files()
        verify_load()
    except Exception as exc:
        print(f"ERROR: setup failed: {type(exc).__name__}", file=sys.stderr)
        if os.environ.get("RURI_SETUP_DEBUG") == "1":
            traceback.print_exc(file=sys.stderr)
        raise SystemExit(1) from None

if __name__ == "__main__":
    main()
```

### 2. 設計方針書との整合 (DR4-002 sanitized log policy)

- 成功 log: `[OK] fetch: {model_id}:{filename} ({size_bytes} bytes)` (filename + size のみ、HF cache full path なし)
- 失敗時: `ERROR: setup failed: {type(exc).__name__}` (exception type のみ、raw traceback なし)
- raw traceback 出力: `RURI_SETUP_DEBUG=1` 環境変数が明示的にセットされた場合のみ
- `SystemExit(1) from None` で chained traceback も抑止
- HF token / PAT / full cache path / raw traceback を Issue コメントに貼らない方針を **コードで担保**

---

## 次のアクション

### 本 PR の取り扱い
- [ ] `git status` 確認 (commit 対象は requirements.txt + scripts/_setup_ruri_models.py のみ)
- [ ] commit + push
- [ ] `/create-pr` で develop 向け PR 作成

### 運用フェーズで実施 (PR レビュー前 or マージ後)
- [ ] Task 1 受入: `python scripts/_setup_ruri_models.py` で `verify_load()` 成功確認 + sanitized log を Issue コメント記録
- [ ] `python -m pip check` + `pip show huggingface_hub transformers sentence-transformers` resolved version を Issue コメント記録 (DR4-003)
- [ ] Task 2.1: `configs/_experiments/institutional_V2.yaml` 作成 (yq / python -c で 4 行 override)
- [ ] Task 2.2: variant config security validation gate 実行 (設計方針書 §3.4 Step 2.5)
- [ ] Task 2.3: V2 ingest (`scripts/ingest_repo.py --repo ... --commit 9e500539... --config configs/_experiments/institutional_V2.yaml --repo-id institutional_documents_V2`)
- [ ] Task 2.4: V2 build_indexes (`scripts/build_indexes.py ...`、~30-60 分)
- [ ] Task 2.5: V2 run_baseline_eval (`scripts/run_baseline_eval.py ...`、~30 分)
- [ ] Task 3.1: `reports/institutional_retrieval_ab.md` V2 行に NC rate / category breakdown / latency 反映
- [ ] Task 3.2: `configs/institutional_docs.yaml:84-87` コメント更新
- [ ] V4 採用判定不変の確認 (V2 < V4 の場合) → Issue close
- [ ] (条件付き) V2 採用低確率分岐: invariant test + deployment docs 更新 → 別 PR

### Option A 失敗時のフォールバック
- 設計方針書 §3.4 「失敗時のフォールバック分岐」表に従い Option B (専用 venv) または Option C (上流報告) を選択
- Option B 採用時のみ `.gitignore` に `.venv-ruri-*/` を追加 (DR3-003)

---

## メモリ・キャッシュ

- HF cache 推奨: `~/.cache/huggingface/hub/models--cl-nagoya--ruri-small-v2/` 配下に `spiece.model` (439 KB) + 既存 cache が揃うこと
- ruri-small-v2 モデル size: 272 MB (model.safetensors)
- V2 build 時のメモリ使用量想定: 4228 docs × max_input_chars=2048 × small embedding (67M params) で M3 Ultra で軽量

---

## 関連リンク

- Issue: https://github.com/Kewton/photon-mlx/issues/144
- 親 Issue (closed): https://github.com/Kewton/photon-mlx/issues/137
- 親 PR (merged): https://github.com/Kewton/photon-mlx/pull/142
- 設計方針書: `workspace/design/issue-144-ruri-v2-build-design-policy.md`
- Issue review summary: `workspace/issues/144/issue-review/summary-report.md`
- Design review summary: `workspace/issues/144/multi-stage-design-review/summary-report.md`
- Work plan: `workspace/issues/144/work-plan.md`
- TDD result: `workspace/issues/144/pm-auto-dev/iteration-1/tdd-result.json`
- TDD fix result: `workspace/issues/144/pm-auto-dev/iteration-1/tdd-fix-result.json`
- Codex code review: `workspace/issues/144/pm-auto-dev/iteration-1/codex-review-result.json`
- Acceptance result: `workspace/issues/144/pm-auto-dev/iteration-1/acceptance-result.json`
- Refactor result: `workspace/issues/144/pm-auto-dev/iteration-1/refactor-result.json`
