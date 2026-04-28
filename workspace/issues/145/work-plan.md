# Issue #145 作業計画書

## Issue: test(photon): real-weight integration test (split from #139, depends on #135)

- **Issue 番号**: #145
- **サイズ**: M (新規 test 3 件 + config + ci wiring)
- **優先度**: Medium (S7-001 follow-up, regression detection 強化)
- **依存 Issue**: #135 (S7-001 fix / 解消済 main マージ済), #139 (Task 1+3 / 解消済)
- **採用方針**: B (セルフホスト型最小 e2e、repo 完結)
- **設計方針書**: `workspace/design/issue-145-real-weight-test-design-policy.md` (32 findings 反映済)

---

## 1. タスク分解

### Phase 1: 基盤準備 (P1)

- [ ] **Task 1.1**: `tests/integration/` ディレクトリ + `__init__.py` 作成
  - 成果物: `tests/integration/__init__.py` (空ファイル)
  - 依存: なし

- [ ] **Task 1.2**: `pyproject.toml` 新規作成 (`[tool.pytest.ini_options]` のみ最小追加)
  - 成果物: `pyproject.toml`
  - 内容:
    ```toml
    [tool.pytest.ini_options]
    testpaths = [
        "tests",
        "baseline_reporag/tests",
        "photon_mlx/tests",
        "torch_ref/tests",
    ]
    markers = [
        "integration: integration tests requiring real PhotonModel + checkpoint",
        "slow: tests taking >5 seconds",
    ]
    ```
  - 依存: なし
  - 検証: `python -m pytest --collect-only -q` の nodeid set を testpaths 設定前後で比較し既存 test の欠落がないこと (DR3-003)

- [ ] **Task 1.3**: discovery smoke test
  - 成果物: なし (検証のみ)
  - 検証コマンド: `python -m pytest --collect-only -q tests/integration/ 2>&1` でエラー 0 件
  - 検証コマンド: `python -m pytest --markers 2>&1 | grep -E "integration|slow"` で markers 認識
  - 依存: Task 1.1, 1.2

### Phase 2: 最小 config (P2)

- [ ] **Task 2.1**: `configs/photon_test_minimal.yaml` 新規作成
  - 成果物: `configs/photon_test_minimal.yaml`
  - 仕様 (`photon_mlx/tests/test_training.py::_tiny_cfg` 同型):
    - `base_embed_dim: 16`
    - `hidden_size: 64`
    - `intermediate_size: 128`
    - `num_heads: 4` (yaml 側) / `num_attention_heads` (Pydantic ModelConfig 側) の **キー名対応を脚注で確認**
    - `head_dim: 16`
    - `tokenizer.vocab_size: 256`
    - `hierarchy.chunk_sizes: [4, 4]`
    - `encoder_layers_per_level: [1, 1]`
    - `decoder_layers_per_level: [1, 1]`
  - 依存: Task 1.3
  - 検証: `from baseline_reporag.config import load_config` で load 可能、Pydantic validation 通過

### Phase 3: positive path test (P3)

- [ ] **Task 3.1**: `tests/integration/conftest.py` 新規作成 (fixture 共通化)
  - 成果物: `tests/integration/conftest.py`
  - fixture (DR1-009 反映の実行順保証 — 統合 fixture 化推奨):
    - `_integration_setup` (autouse=True, scope="function"):
      - env isolation: `PHOTON_CHECKPOINT_ROOT` / `PHOTON_ALLOW_RANDOM_INIT` の delenv
      - `_real_photon_model_guard`: `assert not isinstance(photon_mlx.model.PhotonModel, MagicMock)`
      - `mlx_available_or_skip`: MLX import 失敗で skip
    - `fake_tokenizer` fixture (vocab_size=256 + 必要 method 群)
  - 依存: Task 2.1
  - 検証: 単独 test 起動で fixture 競合なし、teardown で env 復元

- [ ] **Task 3.2**: `tests/integration/test_photon_real_weights.py` の positive path 実装
  - 成果物: `tests/integration/test_photon_real_weights.py` (新規)
  - 関数: `test_photon_pipeline_with_self_trained_minimal_ckpt`
  - 仕様 (設計方針書 §5 (b) detector 準拠):
    1. `mx.random.seed(42)` 明示 call
    2. 最小 config load → `PhotonModel(cfg.photon)` 構築 → 1 step 学習 (forward+backward+optimizer.step) → trained_model 保持
    3. `photon_mlx.trainer.save_checkpoint(model, state, tmp_path / "test_ckpt")`
    4. `monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))`
    5. `cfg.model.checkpoint_path = "test_ckpt"`
    6. `_build_photon_deps(cfg)` 直接呼出 → `deps['photon_inference'].model` 取得
    7. **検出器 (b)**: `mx.allclose(logits_after, logits_trained, atol=1e-5)` または `abs(_flat_l2_norm(model_after) - _flat_l2_norm(model_trained)) < 1e-5` で trained 参照値との **近接性** を assert
    8. smoke: `deps['photon_inference'].generate_answer(..., max_new_tokens=1)` でクラッシュなし
  - 依存: Task 3.1, 2.1
  - 検証: 単独実行 (`python -m pytest tests/integration/test_photon_real_weights.py::test_photon_pipeline_with_self_trained_minimal_ckpt -v`) で 60 秒以内に pass

### Phase 4: negative path tests (P4)

- [ ] **Task 4.1**: `_make_corrupt_checkpoint` ヘルパ実装
  - 成果物: 同 test ファイル内のヘルパ関数
  - 仕様:
    - tmp_path 配下に `corrupt_ckpt/` ディレクトリ作成
    - `state.json` を `CheckpointState` schema 準拠で記述 (DR2-004 反映)
    - `weights.npz` を不正バイナリ (`b"\x00" * 100`) で上書き
    - `integrity.json` は **作らない** (`verify_integrity=False` で WARNING + return される DR2-004 指摘箇所)
  - 依存: Task 3.2

- [ ] **Task 4.2**: `test_photon_load_failure_without_bypass` 実装
  - 成果物: 同 test ファイル
  - 仕様:
    - `_make_corrupt_checkpoint(tmp_path)` 呼出
    - `PHOTON_CHECKPOINT_ROOT` を tmp_path に設定 (env 1 つ)
    - `cfg.model.checkpoint_path = "corrupt_ckpt"`
    - `with pytest.raises(RuntimeError): _build_photon_deps(cfg)` を assert
  - 依存: Task 4.1
  - 検証: 60 秒以内に pass

- [ ] **Task 4.3**: `test_photon_load_failure_with_bypass` 実装
  - 成果物: 同 test ファイル
  - 仕様:
    - `_make_corrupt_checkpoint(tmp_path)` 呼出
    - `monkeypatch.setenv("PHOTON_ALLOW_RANDOM_INIT", "1")`
    - `monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))`
    - `cfg.model.checkpoint_path = "corrupt_ckpt"`
    - `deps = _build_photon_deps(cfg)` で deps が返る (= 続行) を assert
    - `caplog` で WARNING ('random-init' / 'PHOTON_ALLOW_RANDOM_INIT' substring 含む) を assert
  - 依存: Task 4.1
  - 検証: 60 秒以内に pass

### Phase 5: CI wiring (P5)

- [ ] **Task 5.1**: `.github/workflows/weekly_eval.yml` step 追加
  - 成果物: `.github/workflows/weekly_eval.yml`
  - 挿入位置: `Install dependencies` step の **後**
  - 仕様 (DR4-001/003 反映):
    - step 名: `Run real-weight integration test`
    - command: `set -o pipefail && LOG_BASENAME="logs/integration_test_${GITHUB_RUN_ID:-local}" && python -m pytest tests/integration/test_photon_real_weights.py -v 2>&1 | tee "${LOG_BASENAME}.log"` + `--junitxml="${LOG_BASENAME}.xml"`
    - `continue-on-error: true` は **使わない** (S7-001 / fail-fast)
    - `set -x` / `printenv` / `env` は使わない (env dump 禁止)
    - artifact 既存 `eval-results-${{ github.run_number }}` に `logs/integration_test_*.log` / `logs/integration_test_*.xml` を追加 (`if: always()`)
    - 前回 run の stale log を取り込まない (run-scoped LOG_BASENAME)
  - 依存: Task 4.3
  - 検証: yml syntax check (`yamllint` or `gh workflow view`)、`workflow_dispatch` で dry-run 実行成功

### Phase 6: docs / CLAUDE.md 整合 (P6)

- [ ] **Task 6.1**: `CLAUDE.md` 品質チェック行更新
  - 成果物: `CLAUDE.md`
  - 変更: 約 507/509 → **510/512** (positive 1 + negative 2 = +3 件)
  - 依存: Task 4.3 (test 件数確定後)

- [ ] **Task 6.2**: Issue #145 本文の受入条件数字確定
  - 成果物: GitHub Issue 本文 (gh issue edit)
  - 変更: 約 508/510 → **510/512**
  - 依存: Task 6.1

- [ ] **Task 6.3**: `docs/code_review_checklist.md` 更新
  - 成果物: `docs/code_review_checklist.md`
  - 内容: private API signature 変更時の integration test 追従確認を追記
  - 依存: Task 5.1

- [ ] **Task 6.4**: docs/deployment.md / docs/troubleshooting.md の `PHOTON_ALLOW_RANDOM_INIT` 適用範囲補足要否判定
  - 成果物: docs (必要なら更新、不要なら判定理由を本作業計画書に記録)
  - 内容: valid-shaped load failure のみ bypass 適用、checkpoint_path 未設定/不在/shape 不正は対象外であることを ops 視点で補足するか判断
  - 依存: Task 6.3

### Phase 7: 全品質ゲート通過 (P7)

- [ ] **Task 7.1**: 全 test 実行
  - コマンド: `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v`
  - 基準: 約 510/512 通過 (pre-existing failure 2 件除く)
  - 依存: Task 6.4

- [ ] **Task 7.2**: integration test 単独実行
  - コマンド: `python -m pytest tests/integration/test_photon_real_weights.py -v`
  - 基準: 全件 pass、60 秒以内
  - 依存: Task 7.1

- [ ] **Task 7.3**: ruff check / format
  - コマンド:
    - `ruff check .` (警告 0 件)
    - `ruff format --check .` (差分なし)
  - 依存: Task 7.2

- [ ] **Task 7.4**: pytest discovery 整合性確認
  - コマンド: `python -m pytest --collect-only -q | wc -l`
  - 基準: testpaths 設定前後で nodeid 欠落なし (Task 1.2 で取得した baseline と比較)
  - 依存: Task 7.3

---

## 2. 品質チェック項目

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| 全テスト | `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | 約 510/512 通過 (pre-existing failure 2 件除く) |
| Integration 単独 | `python -m pytest tests/integration/test_photon_real_weights.py -v` | 全件 pass、60 秒以内 |
| Lint | `ruff check .` | 警告 0 件 |
| Format | `ruff format --check .` | 差分なし |
| Discovery | `python -m pytest --collect-only` | エラーなし、testpaths 範囲のみ収集、既存 test 欠落なし |

---

## 3. Definition of Done

Issue 完了条件:

- [ ] tests/integration/__init__.py + test_photon_real_weights.py + conftest.py が新規追加されている
- [ ] pyproject.toml が新規追加され markers (integration, slow) + testpaths が認識される
- [ ] configs/photon_test_minimal.yaml が新規追加され Pydantic validation 通過
- [ ] positive path test (`_with_self_trained_minimal_ckpt`) が trained 参照値との近接性で load 検証
- [ ] negative path test 2 件 (without/with bypass) が RuntimeError / WARNING を pin
- [ ] weekly_eval.yml に integration step 追加 (continue-on-error なし、env dump なし、stale log 防止)
- [ ] CLAUDE.md test 件数 510/512 / Issue #145 本文 510/512 / docs/code_review_checklist.md 更新
- [ ] `python -m pytest` 全 510/512 件 pass (pre-existing failure 2 除く)
- [ ] integration test 単独 60 秒以内に完走
- [ ] `ruff check .` 警告 0 件
- [ ] `ruff format --check .` 差分なし

---

## 4. リスクと対応

| リスク | 確率 | 影響 | 対応 |
|-------|-----|-----|-----|
| MLX nondeterminism で flaky | 中 | 中 | seed 固定 + numpy 経由比較 + L2 close threshold |
| autouse fixture 干渉 | 低 | 高 | tests/integration/ 分離 + `_real_photon_model_guard` |
| pytest discovery 破壊 | 低 | 中 | testpaths 設定前後で nodeid set 比較 (DR3-003) |
| 60s budget 超過 | 低 | 低 | M3 Ultra 上で 5-15 秒、4x マージン |
| `_build_photon_deps` signature 変更 | 中 (将来) | 中 | private API 依存は受容、変更時は追従 |

---

## 5. 次のアクション

1. **/pm-auto-dev 145** で TDD 実装着手 (Phase 1-7 を順次実行)
2. 完了後 `/create-pr` で PR 作成
