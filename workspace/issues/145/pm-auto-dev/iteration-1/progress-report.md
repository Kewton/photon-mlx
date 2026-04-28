# Issue #145 PM Auto Dev 進捗報告 (iteration-1)

実施日: 2026-04-28
対象: Issue #145 「test(photon): real-weight integration test (split from #139, depends on #135)」
ブランチ: feature/issue-145-real-weight-test

---

## Phase 別完了状況

| Phase | 内容 | ステータス | 備考 |
|-------|------|----------|------|
| P1 | tests/integration/ + pyproject.toml + discovery smoke | 完了 | testpaths = 6 dir (4 documented + bench/tests + evals/tests preserved) |
| P2 | configs/photon_test_minimal.yaml | 完了 | `_tiny_cfg` 同型、`tokenizer_id` 必須 key 含む |
| P3 | conftest.py + positive path test | 完了 | autouse: env isolation + MagicMock guard + MLX skip |
| P4 | negative path tests (without/with bypass) | 完了 | corrupt ckpt 経由で RuntimeError / WARNING 両方 pin |
| P5 | weekly_eval.yml integration step | 完了 | continue-on-error なし、env dump なし、stale log 防止 |
| P6 | CLAUDE.md + docs + Issue 本文 | 完了 | 510/512 で統一、private API 追従 checklist 追加 |
| P7 | quality gate | 完了 | 詳細は下記 |

---

## 品質ゲート結果

| チェック項目 | コマンド | 結果 |
|-------------|----------|------|
| Integration test 単独 | `python -m pytest tests/integration/test_photon_real_weights.py -v` | **3/3 pass、3.29 秒** (60 秒予算 18x マージン) |
| 全テスト | `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | **1260 passed + 3 (integration) = 1263 pass、3 pre-existing failed** |
| ruff check | `ruff check .` | **All checks passed** (新規ファイル含む全件 pass) |
| ruff format (新規ファイル) | `ruff format --check tests/integration/` | pass |
| pytest discovery | `python -m pytest --collect-only -q` | 1297 tests collected、エラーなし |

### Pre-existing failures (本 Issue scope 外)

1. `tests/test_generate_training_corpus.py::TestMain::test_main_cli_tokenizer_id` (CLAUDE.md 既知)
2. `tests/test_generate_training_corpus.py::TestMain::test_main_uses_tokenize_text` (CLAUDE.md 既知)
3. `baseline_reporag/tests/test_photon_pipeline.py::TestBuildPhotonDepsRealTokenizer::test_vocab_size_mismatch_raises` (`@pytest.mark.real_hf_loader` 経路、ネットワーク/HF allowlist 依存)

### Pre-existing ruff format violations (本 Issue scope 外)

- `baseline_reporag/photon_pipeline.py`
- `scripts/train_photon.py`

main 上で既に違反しており、本 Issue で touch していない。修正は別 Issue で扱うべき。

---

## 設計適合性

設計方針書に対する 32 findings (Must Fix 7 / Should Fix 18 / Nice to Have 7) はすべて実装に反映済み。主要な実装ピン:

| 設計 ID | 内容 | 実装箇所 |
|--------|------|---------|
| DR1-001 | `deps['photon_inference'].model` 経由でアクセス | test line 212 |
| DR1-002 | trained 参照値との近接性 (< 1e-5) で load 検証 | test line 218-221 |
| DR1-003 | autouse fixture を `_photon_env_isolation` の parameter dependency で順序 pin | conftest line 39-44 |
| DR2-003 | `_real_photon_model_guard` で MagicMock 残留を起動時 assert | conftest line 39-50 |
| DR2-004 | corrupt ckpt の `state.json` を `CheckpointState` schema 準拠化、`integrity.json` 不在 | test line 137-160 |
| DR3-001 | docs/code_review_checklist.md に private API 変更時の追従 checklist 追記 | checklist line 79 |
| DR4-001 | weekly_eval.yml で `continue-on-error: true` を使わず fail-fast | weekly_eval line 21-32 |
| DR4-003 | `set -x` / `printenv` / `env` 不使用、stale log 防止 | weekly_eval line 21-32 |
| DR4-005 | `monkeypatch` 経由で fake tokenizer を patch | conftest line 94-105 |

---

## 成果物一覧

### 新規ファイル

```
tests/integration/__init__.py
tests/integration/conftest.py                  (110 行 / fixtures)
tests/integration/test_photon_real_weights.py  (292 行 / 3 tests + helpers)
configs/photon_test_minimal.yaml               (47 行)
pyproject.toml                                 (20 行 / [tool.pytest.ini_options])
workspace/design/issue-145-real-weight-test-design-policy.md
workspace/issues/145/...                       (review/design-review/work-plan/pm-auto-dev)
```

### 変更ファイル

```
.github/workflows/weekly_eval.yml  (+15 行 / integration step + artifact paths)
CLAUDE.md                          (1 行 / test count 507/509 → 510/512)
docs/code_review_checklist.md      (+1 行 / private API 追従 checklist)
GitHub Issue #145 本文             (508/510 → 510/512 数字統一)
```

---

## 次のアクション

- [ ] commit & push (ユーザー判断)
- [ ] PR 作成 (`/create-pr 145`)
- [ ] CI 確認 (weekly_eval.yml の workflow_dispatch dry-run)

---

## Follow-up Issue 候補 (本 Issue scope 外)

1. **候補方針 A の段階導入**: 実 production checkpoint (`step_003000`) 利用 + 実 tokenizer + 実 retrieval full E2E
2. **PR-level blocking integration workflow** の新規追加
3. **`verify_integrity=True` plumbing** to `trainer.load_checkpoint`
4. **CLAUDE.md 全 test 数の正確化**: 現状 507/509 表記は実測 1260 pass と大きく乖離。本 Issue では指針通り 510/512 に更新したが、根本的な現状把握が必要
5. **`baseline_reporag/photon_pipeline.py` / `scripts/train_photon.py` のフォーマット修正** (ruff format pre-existing violations)
6. **`test_vocab_size_mismatch_raises` (real_hf_loader)** failure 原因究明
