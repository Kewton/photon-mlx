# Issue #145 設計方針書 — PHOTON real-weight integration test

> **対象 Issue**: [#145 test(photon): real-weight integration test (split from #139, depends on #135)](https://github.com/Kewton/photon-mlx/issues/145)
>
> **採用方針**: 候補方針 B (セルフホスト型最小 e2e) — repo 完結、外部 ckpt 取得なし
>
> **依存解消状況** (2026-04-28 時点):
> - 必須 #135 (S7-001 fix) — main マージ済 (commit 8c13517 / PR #157)
> - 推奨 #139 (Task 1 + Task 3) — main マージ済 (PR #147)
>
> **設計レビュー反映済**: 累計 32 findings (Stage 1: 10 / Stage 2: 10 / Stage 3: 7 / Stage 4: 5) — Stage 1-4 (claude-opus + codex、Stage 4 セキュリティレビュー DR4-001..005 反映済)

---

## 1. ゴール

実 `PhotonModel` + test 内で生成した実 checkpoint + production load path (`_build_photon_deps` → `_load_photon_checkpoint` → `photon_mlx.trainer.load_checkpoint`) を通る integration test を **1 件以上追加** し、「random-init weight が production で silently 動いていた」型 (Issue #135 / S7-001) の構造的事故を CI で検出可能にする。

scope:
- 採用: 方針 B (1 step 学習 → save → production load path → smoke)
- 外: 方針 A (実 production ckpt `step_003000` 利用、実 tokenizer / 実 retrieval full E2E) → follow-up Issue 化
- 外: PR-level blocking integration workflow の新規追加 → follow-up

---

## 2. 技術スコープ

| ファイル | 変更種別 | 影響度 |
|---------|---------|-------|
| `tests/integration/__init__.py` | 新規 | 低 |
| `tests/integration/test_photon_real_weights.py` | 新規 | **高** (本 Issue の主成果物) |
| `tests/integration/conftest.py` | 新規 (fixture 共通化必要に応じ) | 中 |
| `configs/photon_test_minimal.yaml` | 新規 (test 専用最小 PhotonConfig、`_tiny_cfg` 同型) | 中 |
| `pyproject.toml` | 新規 (`[tool.pytest.ini_options]` のみ。markers + testpaths) | 中 |
| `tests/__init__.py` | 新設しない (default) / fallback として保留 — 設計判断 #8 参照 [DR2-005 反映] | 低 |
| `.github/workflows/weekly_eval.yml` | step 追加 (`Run real-weight integration test`) | 中 |
| `CLAUDE.md` | 品質チェック行の test 件数更新 (約 507/509 → 約 510/512) | 低 |
| `docs/deployment.md` / `docs/troubleshooting.md` / `docs/code_review_checklist.md` | 整合確認 (PHOTON_ALLOW_RANDOM_INIT 適用範囲補足の要否判定) | 低 |

**変更しない (本 Issue scope 外で固定)**:
- `baseline_reporag/photon_pipeline.py` の `_build_photon_deps` / `_resolve_checkpoint_path` / `_load_photon_checkpoint` — 既存 production API signature を維持 [S7-004]
- `photon_mlx/trainer.py` の `load_checkpoint` / `save_checkpoint` — そのまま使用
- `photon_mlx/checkpoint.py` の `load_checkpoint(verify_integrity=...)` plumbing への trainer wrapper 拡張 — follow-up [S5-003]
- `photon_mlx/inference.py` の `_check_weight_initialization` WARNING ロジック — そのまま

---

## 3. アーキテクチャ判断

### 設計判断 #1: 採用方針 (B vs A)

**選択肢**:
- A: 実 production checkpoint (`step_003000` 等) を取得し full E2E (実 tokenizer + 実 retrieval + `pipeline.query`) を回す
- B: tmp_path 内で 1 step 学習 → save → production load path → PhotonInference smoke (実 tokenizer / 実 retrieval は scope 外)

**決定**: **B (本 Issue), A は follow-up Issue 化**

**理由**:
- A は (i) 実 ckpt 取得手段 (HF Hub upload / S3 mirror / self-hosted runner キャッシュ)、(ii) HF allowlist 整備 (`baseline_reporag/photon_pipeline.py:340-342` `_validate_repo_id`)、(iii) CI runtime budget (現在 weekly_eval は 180 分タイムアウト) の決定が必要であり、本 Issue の主目的「load path の構造的事故検出」には過剰
- B は外部リソース不要、M3 Ultra 上で 5-15 秒、self-hosted runner なら blocking 化も検討可能
- B でも `_build_photon_deps` → `_load_photon_checkpoint` → `photon_mlx.trainer.load_checkpoint` の **production load path 全体を実コード経由で通す** ため、S7-001 型 silent bug の検出力は十分

**トレードオフ**:
- メリット: 軽量・自己完結・flaky 制御容易
- デメリット: 「実用 size weight」 e2e ではない (production と最小 cfg の挙動差を検出できない)
- リスク: A 移行時に test 構造を作り直す必要がある — 共通化しない方針で OK

---

### 設計判断 #2: PhotonModel 実体化方法 (`_build_photon_deps` 直接 vs `pipeline_factory` 経由)

**選択肢**:
- (i) `_build_photon_deps(cfg)` を直接呼ぶ (private API 依存)
- (ii) `pipeline_factory.build_pipeline(cfg_yaml)` 経由で PhotonRAGPipeline を完成させる

**決定**: **(i) `_build_photon_deps` 直接呼出** (default)

**理由**:
- 本 Issue の主目的は **checkpoint load path integration** であり、retrieval/embedding まで通す必要なし
- (ii) は tmp_path 内に最小 corpus + index 準備が必要で test setup が大幅増 (60 秒予算超過リスク)
- private API 依存を受容する根拠 (DR1-007 反映):
  - (a) `_build_photon_deps` の signature 変更を本 Issue 単独でなく既存 boundary test 群 (`baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py`) も巻き込んで強制改修する力学が働き、boundary test と本 integration test の両方が同じ private API に依存する **「2 重構造的圧力」** として signature 安定化を担保する
  - (b) public API (pipeline_factory) 経由は最小 corpus + index 準備が必要で 60 秒予算違反
  - boundary test (= MagicMock 置換ベースの spec test) と integration test (= production load path 全体を実コード経由で通す) は**責務が異なる** (前者は code path の単位検証、後者は wire-up 検証) ため、単に「前例あり」ではなく上記 2 点で SRP / DIP に対する根拠を補強
- model 取り出し方法 (DR1-001 反映): `deps['photon_inference'].model` 経由で取得する (`PhotonInference.__init__` line 189 で `self.model = model` を保持)。`_build_photon_deps` 戻り値は `{photon_inference, safe_recgen, photon_cfg, tokenizer}` の 4 key のみで `photon_model` key は存在しないため
- `deps['photon_inference']` 経由で `generate_answer(..., max_new_tokens=1)` 程度の smoke を流せば load 結果を検証可能

**トレードオフ**:
- メリット: 軽量、test focus 明確
- デメリット: private API 依存 — `_build_photon_deps` signature 変更時に test 改修が必要
- リスク: signature 変更は `_build_photon_deps` 利用 test 全体に波及するが、本 Issue 単独でなく既存 test 群も影響を受けるため **既知のリスク** として受容 (signature 変更時は本 test も追従する責務を明記、§9 リスク表とも整合)

---

### 設計判断 #3: random-init detector 実装方式

**背景**: Issue 起票時の `pipeline.last_pruning_score_distribution.std()` は架空 API。実在 API ベースの (a)-(d) から選択。

**用語定義 [DR2-006 / DR2-007 反映]**:
- **silent load skip**: load 経路が exception や bypass を通らず、wire-up 不整合で `load_checkpoint` が呼ばれずに完了する事象 (本 Issue の検出対象、本文書では旧表記 'silently-skipped load' / 'load skip' をこの用語に統一)
- **load failure**: `load_checkpoint` が exception を投げる事象 (`PHOTON_ALLOW_RANDOM_INIT` bypass の対象)
- **load success**: `load_checkpoint` が exception なく完了し weight が反映される事象
- **weight closeness 検証**: trained 参照値と load 後の値が `abs(norm_after - norm_trained) < 1e-5` で近接していることの検証 (旧 'weight identity' / '同一性' を統一。bit-perfect equality は MLX nondeterminism の余地を残すため意図的に避ける — DR1-010 と整合)

**選択肢**:
- (a) checkpoint integrity hash 検証 (DR4-003 SHA-256)
- (b) **weight closeness 検証 (load 後 trained 一致)** (**trained_model の参照値 vs load 後 model の値の近接性**) [DR2-007 反映]
- (c) `PHOTON_ALLOW_RANDOM_INIT` bypass のネガティブテスト
- (d) `_score_prune_candidates` private API 直接呼出 + raw_score std 比較

**決定**: **(b) + (c) 併用**

**理由** (S3-010 推奨案を採用):
- (b) は **positive path** (学習データ load 経路 = load success) を pin
- (c) は **negative path** (load failure 時の bypass 経路) を pin
- 両者併用で random-init silently 動作型事故を **load success / load failure / silent load skip の 3 系統** から検出可能
- 実装コスト合計 15-20 行
- (a) は flaky 時のフォールバック (bit-perfect 保証あり)
- (d) は private API 依存度高すぎるため不採用

**(b) の検出器ロジック (DR1-002 / DR2-007 反映 / 重要)**:
- 当初案: `abs(norm_before - norm_after) > 1e-4` で「変化あり」を検出 → **silent load skip を検出できない**
- 失敗パターン: `norm_before` (seed=42 random-init#1) と `norm_after` (load success 時=trained / silent load skip 時=`PhotonModel(photon_cfg)` で生成された random-init#2) は、どちらの場合でも 1e-4 以上の差を返してしまい、silent load skip 時に検出器が見逃す
- **採用案 (weight closeness 検証)**: trained_model を save 直前に保持し、load 後の model と **近接性** (`abs(norm_after - norm_trained) < 1e-5` または `mx.allclose(logits_after, logits_trained, atol=1e-5)`) で判定
  - load success → trained と一致 (近接) → assertion pass
  - silent load skip → trained と乖離 (random-init#2 と trained の差) → assertion fail で検出
- forward pass logits の `mx.allclose` 比較は `photon_mlx/tests/test_training.py:220-260` の `test_load_model_forward_consistency` と同 pattern (より頑健)

**実装注意 (S3-005, DR1-010 反映)**:
1. `mx.random.seed(42)` を test 冒頭で明示 call (PhotonConfig seed に頼らない)
2. 比較は `mx.eval(param)` 後に `np.asarray(param)` 経由で numpy ベース
3. weight closeness 検証は trained_model の参照値と load 後 model の値の **近接性** で行う (`abs(norm_after - norm_trained) < 1e-5` または `mx.allclose(logits_after, logits_trained, atol=1e-5)`)。bit-perfect (`==`) は MLX nondeterminism の余地を残すため避ける

---

### 設計判断 #4: pyproject.toml の scope (新規 vs 既存活用)

**現状**: pyproject.toml / pytest.ini は repo に未存在。markers (`integration`, `slow`) は `pytest --markers` で warning が出る。

**選択肢**:
- (a) pyproject.toml を新規作成、`[tool.pytest.ini_options]` のみ追加
- (b) pytest.ini を新規作成
- (c) markers を使わず file path で integration test を区別

**決定**: **(a) — `pyproject.toml [tool.pytest.ini_options]` のみ最小追加**

**理由** (S3-003):
- markers (`integration`, `slow`) を pytest 標準で登録するため markers config は必要
- `[tool.ruff]` / `[build-system]` / `[project]` セクションは追加しない (既存 `ruff check .` 警告 0 件挙動を保持、ruff config は cwd ベース default に委ねる)
- `testpaths` も明記して discovery 範囲を pin (`tests`, `baseline_reporag/tests`, `photon_mlx/tests`, `torch_ref/tests` の 4 ディレクトリ) — S3-002 と整合

**新規 pyproject.toml 内容**:

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

---

### 設計判断 #5: 最小 config (`configs/photon_test_minimal.yaml`)

**選択肢**:
- (a) 既存 `configs/photon_tiny.yaml` を流用 (hidden=640, 80M params)
- (b) test 専用最小 yaml を新規 (`_tiny_cfg` 同型, base_embed_dim=16, hidden=64)

**決定**: **(b)**

**理由** (S5-004):
- (a) は学習に分単位かかるため 60 秒予算違反
- (b) は `photon_mlx/tests/test_training.py::_tiny_cfg` と同型: `base_embed_dim=16`, `hidden_size=64`, `intermediate_size=128`, `num_heads=4`, `head_dim=16`, `tokenizer.vocab_size=256`, `chunk_sizes=[4, 4]`, `encoder/decoder_layers_per_level=[1, 1]`
- 1 step (forward+backward+optimizer.step) で 1-3 秒、save_checkpoint で +1-2 秒
- **キー名注記 [DR2-002 反映]**: yaml 表記では `num_heads: 4` を採用する (baseline の `_build_photon_deps` が `cfg.model.get("num_heads", 4)` で読む key 名 — `baseline_reporag/photon_pipeline.py:284`、前例 `configs/photon_small.yaml` line 175)。一方 `photon_mlx/tests/test_training.py::_tiny_cfg` は `ModelConfig` を直接構築するため `num_attention_heads=4` / `num_key_value_heads=4` の field 名を使う。本 Issue の `configs/photon_test_minimal.yaml` は yaml 経由で baseline Config に load されるため `num_heads` を採用すること (yaml に `num_attention_heads:4` と書くと `cfg.model.get("num_heads", 4)` は default 4 を返し動作はするが意図と異なる)
- **必須 YAML key [DR3-002 反映]**: `_build_photon_deps(cfg)` は `cfg.tokenizer.tokenizer_id` を `_validate_tokenizer_id` に渡してから `_load_hf_tokenizer(tokenizer_id, expected_vocab_size)` を呼ぶため、fake tokenizer fixture を使う場合でも `tokenizer_id` は省略しない。test 専用値として allowlist 風の syntactically valid ID (例: `Kewton/fake-photon-test-tokenizer`) を置く。加えて `inference.safe_recgen_enabled: false` を明示し、load-path integration test に不要な SafeRecGen 構築を避ける。
- **CI runtime 内訳積算 [DR2-010 反映]**: 3 件合計 (positive 1 + negative 2) で 13-23s。positive: 構築 1-2s + L2 norm 2×0.5s + 1-step 1-3s + save 1-2s + load 1-2s + smoke 0.5-1s + L2 norm 2×0.5s ≈ 7-13s / negative 各 3-5s。§9 リスク表 「CI runtime 60s 超過」の根拠

**最小 YAML skeleton [DR3-002 反映]**:

```yaml
model:
  architecture: photon
  base_embed_dim: 16
  hidden_size: 64
  intermediate_size: 128
  num_heads: 4
  head_dim: 16
  encoder_layers_per_level: [1, 1]
  decoder_layers_per_level: [1, 1]
  max_position_embeddings: 128
  checkpoint_path: null

hierarchy:
  chunk_sizes: [4, 4]

tokenizer:
  tokenizer_id: Kewton/fake-photon-test-tokenizer
  vocab_size: 256

inference:
  safe_recgen_enabled: false
```

---

### 設計判断 #6: tokenizer fixture 方針

**選択肢**:
- (a) `_load_hf_tokenizer` を実 HF download で呼ぶ (CI で download 失敗リスク)
- (b) `_load_hf_tokenizer` を local fake fixture に置き換え

**決定**: **(b)** (S3-009 default 推奨)

**理由**:
- HF allowlist 整備の前提で実 tokenizer download は本 Issue scope 外 (方針 A 領域)
- 本 Issue は load path integration が主目的であり、実 tokenizer の挙動検証は不要
- `vocab_size` は `_load_hf_tokenizer(tokenizer_id, expected_vocab_size)` の `expected_vocab_size` (= 最小 cfg `tokenizer.vocab_size: 256`) と一致させ、#138 tokenizer mismatch invariant と矛盾しない

**実装方法**:
- `tests/integration/conftest.py` に `fake_tokenizer` fixture を定義 (SimpleNamespace + `vocab_size=256` + 必要 method の最小実装)
- `monkeypatch.setattr("baseline_reporag.photon_pipeline._load_hf_tokenizer", lambda *a, **k: fake_tokenizer)` で置換
- **DI 境界 [DR4-005 反映]**: fake tokenizer の `monkeypatch.setattr` は pytest default の function scope fixture 内に閉じる。module/session scope fixture や import-time patch は使わない。これにより production code への影響は test 関数 teardown で復元され、他 test へ伝播しない。

---

### 設計判断 #7: CI 実行戦略

**現状**:
- 現行 main に PR-level blocking workflow なし
- `.github/workflows/weekly_eval.yml` (cron: Monday 00:00 UTC, runs-on: self-hosted, timeout 180min) のみ
- MLX 依存テスト (PhotonModel 構築) は self-hosted Apple Silicon runner 必須

**選択肢**:
- (a) PR-level blocking workflow を新規作成
- (b) weekly_eval.yml に integration step を追加 (scheduled/manual)
- (c) 両方

**決定**: **(b) — weekly_eval.yml に追加。`continue-on-error: true` 禁止で fail-fast** (S7-001)

**理由**:
- (a) は新規 workflow 追加スコープが大きく本 Issue 範囲外
- (b) は既存 cron で integration test を週次実行、PR を block しない代わりに scheduled/manual を赤にすることで検出
- `continue-on-error: true` を使うと「CI で検出可能」というゴールを満たさない

**運用詳細** (S3-004):
- 挿入位置: `Install dependencies` step の **後** (長時間 eval の前に 60 秒 budget の load-path regression を fail-fast で検出)
- 失敗時挙動: integration test failure で workflow を fail させる
- artifact: 既存 `eval-results-${{ github.run_number }}` artifact に `logs/integration_test_*.log` を追加 (`if: always()`)
- log 生成 [DR3-004 / DR4-001 反映]: artifact に追加するだけでは空参照になるため、integration step 自体で `mkdir -p logs` 後に run-scoped basename (`LOG_BASENAME="logs/integration_test_${GITHUB_RUN_ID:-local}"`) を作り、`python -m pytest tests/integration/test_photon_real_weights.py -v --tb=short --junitxml="${LOG_BASENAME}.xml" | tee "${LOG_BASENAME}.log"` を実行する。GitHub Actions の shell は `bash` とし、pipe の失敗を隠さないよう `set -o pipefail` を使う。
- artifact path [DR3-004 / DR4-001 反映]: upload 対象に `logs/integration_test_*.log` と `logs/integration_test_*.xml` を含め、`if: always()` で失敗時も保存されることを dry-run / manual run で確認する。self-hosted runner の workspace 共有状態に備え、log file 名に `GITHUB_RUN_ID` を含めて前回 run の stale integration log を上書き・混入させない。
- secret leak policy [DR4-003 反映]: integration step では `set -x`, `printenv`, `env`, `echo $TOKEN` 相当の env dump を禁止する。`PHOTON_CHECKPOINT_ROOT` は `tmp_path`、`PHOTON_ALLOW_RANDOM_INIT` は literal flag のみで secret ではないが、artifact には pytest stdout/JUnit XML 以外を追加しない。
- 実行時間予算: integration test は **60 秒以内に完走** (実測 5-15 秒予測に対して 4x 安全マージン)

---

### 設計判断 #8: pytest discovery 安全策 (`tests/__init__.py` 要否)

**現状**: `tests/` 直下は `__init__.py` 不在のフラット構成

**選択肢**:
- (a) `tests/__init__.py` を新設 (rootdir 推定の安定化)
- (b) 新設せず、`pyproject.toml [tool.pytest.ini_options] testpaths` で discovery 範囲を pin

**決定**: **(b) を default、(a) は fallback** (S3-002)

**理由**:
- 現状で discovery は動作しており不必要な変更を避ける
- `testpaths` 明記により `tests/integration/` 配下の discovery が壊れる可能性は低減
- 設計フェーズで `pytest tests/integration/test_photon_real_weights.py -v` を空テストで先行実行し discovery が壊れないか確認
- 壊れる場合のみ (a) を採用

---

## 4. テスト構造詳細

### 4-1. テスト関数構成

```
tests/integration/test_photon_real_weights.py
├── test_photon_pipeline_with_self_trained_minimal_ckpt  (主 test, 方針 B)
│   ├── Step 1: 最小 config で PhotonModel(cfg) を構築 → seed=42, weight L2 norm 計測 (random-init#1, 参考値)
│   ├── Step 2: 1 step 学習 (forward+backward+optimizer.step) → trained_model を取得
│   ├── Step 3: save_checkpoint(trained_model, state, tmp_path / "test_ckpt") で実生成
│   │       ※ save 直前に norm_trained = _flat_l2_norm(trained_model) を保持
│   │         (load 後の比較対象 = "trained 参照値" として使用)
│   ├── Step 4: monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
│   ├── Step 5: cfg.model.checkpoint_path = "test_ckpt" を注入し _build_photon_deps(cfg) を呼ぶ
│   ├── Step 6: deps['photon_inference'] で smoke (max_new_tokens=1)
│   └── Step 7: detector (b) weight closeness 検証 [DR1-001/002 / DR2-006/007 反映]:
│           model_after = deps['photon_inference'].model  ← key は photon_model ではない
│           load 後 model の weight が trained 参照値と **近接** (`abs(norm_after - norm_trained) < 1e-5`)
│           であることを assert (silent load skip 時は random-init#2 と trained で乖離 → fail)
├── test_photon_load_failure_without_bypass  (detector (c) negative path 1)
│   ├── valid-shaped だが load 失敗の checkpoint を用意 (corrupt weights.npz)
│   ├── env 未設定で _build_photon_deps(cfg) を呼ぶ
│   └── RuntimeError が raise されることを assert
└── test_photon_load_failure_with_bypass  (detector (c) negative path 2)
    ├── 同 corrupt checkpoint
    ├── monkeypatch.setenv("PHOTON_ALLOW_RANDOM_INIT", "1")
    └── WARNING + 続行 (deps が返る) を assert
```

### 4-2. fixture 構成

`tests/integration/conftest.py`:
- `_photon_env_isolation` (autouse=True) [DR1-003 反映 — typo `_phsoton_` → `_photon_` 訂正]: `PHOTON_CHECKPOINT_ROOT` / `PHOTON_ALLOW_RANDOM_INIT` の env を test 開始時にクリア
- `fake_tokenizer`: SimpleNamespace + vocab_size=256 (DR1-008 反映: 詳細は §4-2-1 参照)
- `_real_photon_model_guard` (autouse=True): `assert not isinstance(photon_mlx.model.PhotonModel, MagicMock)` で MagicMock patch 残留を検出 [S3-006]
- `mlx_available_or_skip`: MLX import 失敗時に test を skip (CI runner 互換性ガード)

**autouse fixture 実行順保証 [DR2-003 反映]**:
autouse fixture 2 個 (`_photon_env_isolation` / `_real_photon_model_guard`) の実行順は、本 Issue では本質的に依存しない (env 汚染と MagicMock 残留は独立な検査) が、可読性と将来の事故予防のため **fixture parameter 依存で順序を pin** する:

```python
@pytest.fixture(autouse=True)
def _real_photon_model_guard(_photon_env_isolation):  # ← parameter 依存で順序 pin
    """env クリーン後に MagicMock 残留を検出 (順序保証)。"""
    import photon_mlx.model
    from unittest.mock import MagicMock
    assert not isinstance(photon_mlx.model.PhotonModel, MagicMock), (
        "PhotonModel is patched with MagicMock — autouse fixture leakage detected"
    )
    yield
```

または fixture を 1 つに統合 (`_photon_test_setup`) して順序を実装で固定する案も同等に許容。Issue #140 / S7-001 follow-up で強化された review checklist (CLAUDE.md '## Code Review Checklist') と同質の autouse 順依存事故を防ぐため、設計フェーズで実行順を明示しておく。

#### 4-2-1. fake_tokenizer の必要 method (DR1-008 反映)

既存 `baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py:91-96` の MagicMock pattern を再利用する方針。具体的には:
- `vocab_size=256` (cfg `tokenizer.vocab_size` と一致、#138 invariant 整合)
- `pad_token_id=0`
- `encode.return_value=[1, 2, 3]` (smoke 用最小)
- `decode.return_value=""` (max_new_tokens=1 の出力捨て)

**実装フェーズ前 (P1/P2) の確定作業**: `deps['photon_inference'].generate_answer` 経路で参照される method を実コードから列挙して conftest に最小実装を確定する (yak-shaving 防止)。

**注記 [DR2-009 反映]**: `PhotonInference.__init__` 内の `_check_weight_initialization` (`photon_mlx/inference.py:114-144`) は `model.token_embed` のみ参照するため tokenizer method の対象外。列挙対象は `deps['photon_inference'].generate_answer` (max_new_tokens=1) の call graph のみで十分。

### 4-3. autouse fixture 干渉対策 (DR1-003 反映)

既存 `baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py:75-123` の autouse fixture `_patch_heavy_deps` は `monkeypatch.setattr("photon_mlx.model.PhotonModel", MagicMock(...))` で module-level シンボルを patch する。`tests/integration/` 配下に分離することの根拠を **正しい pytest scope semantics** で示すと:

- `_patch_heavy_deps` は **autouse=True かつ default function scope** のため、各 test 関数の **teardown で monkeypatch が自動復元される**。これが分離の **真の保証**
- conftest.py は配置ディレクトリ単位で読み込まれるため、`baseline_reporag/tests/conftest.py` の autouse fixture は `tests/integration/` 配下の test に **そもそも適用されない**
- 念のため `_real_photon_model_guard` で各 integration test 開始時に `assert not isinstance(photon_mlx.model.PhotonModel, MagicMock)` を実施し、起動時 assertion で MagicMock 残留を検出

**訂正前の誤認**: 「testpaths 明記により別 session として discovery される」と説明していたが、`testpaths` は **discovery scope** であり session 分離とは無関係。実態は `monkeypatch teardown` + `conftest.py` のディレクトリ局所性が分離の保証。将来 fixture scope を session/module に変更する場合、この理解がないと分離が壊れた時の判断を誤る (Issue #140 / S7-001 と同質の構造的事故防止)。

### 4-4. plugin 依存 (S7-003)

- pytest-xdist / pytest-forked への依存は **追加しない**
- 単独・シリアル実行: `python -m pytest tests/integration/test_photon_real_weights.py -v`
- security note [DR4-005]: `PHOTON_CHECKPOINT_ROOT` / `PHOTON_ALLOW_RANDOM_INIT` は process-wide env であるため、本 integration test は xdist/forked 前提にしない。将来 parallel 実行へ移行する場合は env 操作を worker-local に閉じる設計を別途確認する。

---

## 5. random-init detector 実装方式 (詳細)

### (b) weight identity 検証 — positive path (DR1-001 / DR1-002 / DR1-005 / DR1-006 反映)

```python
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
from mlx.utils import tree_flatten
from photon_mlx.model import PhotonModel
from photon_mlx.loss import photon_loss
from photon_mlx.trainer import save_checkpoint, TrainState

def _flat_l2_norm(model):
    """全パラメータの L2 norm 合計を numpy 経由で計算 (lazy eval 罠回避)。

    DR1-005 反映: nn.Module.parameters() は nested dict / list を返すため
    .values() で top-level だけ走査するとサブモジュールの dict/list を
    np.asarray にかけて失敗する。photon_mlx/model.py:741-743 の
    count_parameters() と同じく tree_flatten を使う。
    """
    total = 0.0
    for _name, p in tree_flatten(model.parameters()):
        mx.eval(p)
        total += float(np.linalg.norm(np.asarray(p).ravel()))
    return total


def _run_one_step(model, cfg):
    """1 step 学習 (forward+backward+optimizer.step) — DR1-006 反映。

    既存 photon_mlx/tests/test_training.py:264-292 (test_tiny_overfit) と同 pattern。
    """
    batch = mx.random.randint(0, cfg.tokenizer.vocab_size, (2, 16))

    def loss_fn(m, b):
        logits, _ = m(b, labels=b)
        total, _ = photon_loss(logits, b, 0.0)
        return total

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = mlx.optimizers.Adam(learning_rate=1e-3)
    _loss, grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    return model, TrainState(step=1)


def test_photon_pipeline_with_self_trained_minimal_ckpt(tmp_path, monkeypatch, ...):
    mx.random.seed(42)

    # Step 1: random-init#1 (参考値、現行検出器では使わない)
    cfg = load_minimal_cfg()
    model_before = PhotonModel(cfg.photon)
    _norm_random_init = _flat_l2_norm(model_before)  # 参考用

    # Step 2-3: 1 step 学習 + save
    trained_model, state = _run_one_step(model_before, cfg)
    norm_trained = _flat_l2_norm(trained_model)  # ★ save 直前の参照値を保持
    save_checkpoint(trained_model, state, tmp_path / "test_ckpt")

    # Step 4-5: load via production path
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
    cfg.model.checkpoint_path = "test_ckpt"
    deps = _build_photon_deps(cfg)
    # DR1-001 反映: deps['photon_model'] は存在しない。
    # PhotonInference.__init__ (line 189) で self.model = model を保持しているため取得可能。
    model_after = deps["photon_inference"].model
    norm_after = _flat_l2_norm(model_after)

    # Detector (b): load 経路が trained 参照値と一致することを検証 (DR1-002 反映)
    # 旧案 (norm_before vs norm_after > 1e-4) では silently-skipped load を見逃すため、
    # trained_model の参照値との 近接性 で判定する。
    assert abs(norm_after - norm_trained) < 1e-5, (
        f"weights not loaded from ckpt: norm_trained={norm_trained}, norm_after={norm_after}"
    )

    # (より頑健な代替案) forward logits を mx.allclose で比較する形も可。
    # photon_mlx/tests/test_training.py:220-260 の test_load_model_forward_consistency 参照。
    # seq_len = 16  # math.prod(cfg.hierarchy.chunk_sizes)。[4, 4] のため PhotonModel の chunk 整列条件を満たす。
    # x = mx.random.randint(0, cfg.tokenizer.vocab_size, (1, seq_len))
    # logits_trained, _ = trained_model(x)
    # mx.eval(logits_trained)
    # logits_after, _ = model_after(x)
    # mx.eval(logits_after)
    # assert mx.allclose(logits_after, logits_trained, atol=1e-5).item()
```

### (c) PHOTON_ALLOW_RANDOM_INIT bypass — negative path (DR1-004 反映)

```python
def _make_corrupt_checkpoint(tmp_path):
    """valid shape の checkpoint dir を作り weights.npz を破損させる。

    DR1-004 反映: state.json は CheckpointState (photon_mlx/checkpoint.py:30-43)
    の field のみを含む valid 最小 schema にする。vocab_size 等の未知 key は
    `Ignoring unknown state.json keys` の WARNING を生み、(c) の caplog assertion を
    brittle にするため除外する。

    DR2-004 反映 — load path の流れ:
    本 fixture は意図的に integrity.json を作らない。
    photon_mlx/checkpoint.py:117-130 の _verify_integrity(strict=False) は
    integrity.json 不在時に WARNING + return で続行する (verify_integrity=False)。
    そのため production 経路
      _build_photon_deps
        → _load_photon_checkpoint
        → photon_mlx.trainer.load_checkpoint
        → photon_mlx.checkpoint.load_checkpoint(verify_integrity=False)
    は integrity check を skip し、load フェーズの mx.load(weights.npz) で初めて fail する。
    これにより (i) integrity check の特定挙動 (DR4-003) を本 Issue scope 外に保ちつつ、
    (ii) corrupt weights.npz による load failure → except 句 → RuntimeError or
    PHOTON_ALLOW_RANDOM_INIT bypass の経路を実コード経由で pin できる。
    verify_integrity=True の trainer wrapper plumbing は follow-up Issue (S5-003)。
    """
    ckpt_dir = tmp_path / "corrupt_ckpt"
    ckpt_dir.mkdir()
    state_payload = {
        "step": 1,
        "best_val_loss": float("inf"),
        "best_step": 0,
        "patience_counter": 0,
        "train_losses": [],
        "val_losses": [],
    }
    (ckpt_dir / "state.json").write_text(json.dumps(state_payload))
    (ckpt_dir / "weights.npz").write_bytes(b"\x00" * 100)  # 不正バイナリ
    return ckpt_dir

def test_photon_load_failure_without_bypass(tmp_path, monkeypatch):
    _make_corrupt_checkpoint(tmp_path)
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
    cfg = load_minimal_cfg()
    cfg.model.checkpoint_path = "corrupt_ckpt"

    with pytest.raises(RuntimeError):
        _build_photon_deps(cfg)

def test_photon_load_failure_with_bypass(tmp_path, monkeypatch, caplog):
    _make_corrupt_checkpoint(tmp_path)
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
    monkeypatch.setenv("PHOTON_ALLOW_RANDOM_INIT", "1")
    cfg = load_minimal_cfg()
    cfg.model.checkpoint_path = "corrupt_ckpt"

    deps = _build_photon_deps(cfg)
    assert deps is not None
    assert any("random-init" in r.message.lower() or "PHOTON_ALLOW_RANDOM_INIT" in r.message
               for r in caplog.records if r.levelname == "WARNING")
```

---

## 6. セキュリティ設計

| 脅威 | 対策 | 優先度 |
|------|------|--------|
| **path traversal via `checkpoint_path`** | 既存 `_resolve_checkpoint_path` (line 589) が `Path.resolve(strict=True)` + root containment check 済。`../outside/evil` や symlink escape は `baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py` の boundary test で既に pin されている。本 integration test は production 経路を通すことが目的なので、positive/negative とも `checkpoint_path` は `"test_ckpt"` / `"corrupt_ckpt"` のような tmp root 相対 basename に固定し、追加の traversal bypass を作らない [DR4-001] | 高 (既存対策 + 設計制約) |
| **arbitrary file read** | checkpoint fixture は必ず `tmp_path / <name>` 配下に作成し、`/tmp`, `~/.cache`, repo 外絶対パス、symlink を入力にしない。`PHOTON_CHECKPOINT_ROOT=str(tmp_path)` と root containment により test 外ディレクトリ読み取りを防ぐ [DR4-002] | 中 |
| **tmp_path leak** | `monkeypatch.setenv` で test scope のみ、teardown で env 復元。log に出る可能性がある path は pytest tmp_path のみで secret ではないが、CI step では env dump (`set -x`, `printenv`, `env`) を禁止する [DR4-003] | 中 |
| **arbitrary code execution via checkpoint** | `photon_mlx.checkpoint.load_checkpoint` は `mx.load(path / "weights.npz")` と JSON read の経路であり、設計上 `np.load(..., allow_pickle=True)`, pickle, object dtype `.npy/.npz`, eval/exec/subprocess は使わない。corrupt fixture は raw invalid bytes (`b"\x00" * 100`) のみを書き、外部提供 checkpoint を読まない [DR4-004] | 中 |
| **CI runner 上の state 汚染 / stale artifact** | checkpoint は `tmp_path`、logs は `logs/integration_test_${GITHUB_RUN_ID}.log/xml` の run-scoped 名に限定する。self-hosted runner の workspace が再利用されても前回 run の integration log が混入しないよう file 名を run-scoped にする [DR4-001] | 中 |
| **DI / fixture isolation** | fake tokenizer patch は function scope の `monkeypatch.setattr` に限定し、module/session scope や import-time patch を禁止。`_real_photon_model_guard` と `monkeypatch` teardown で他 test への伝播を検出・復元する [DR4-005] | 中 |
| **secret leak via log/artifact** | tmp_path / fake tokenizer を使うため API key/credentials の経路なし。artifact は `logs/integration_test_*.log/xml` のみに限定し、環境変数一覧・HF token・GitHub token・外部 model path を出力しない。`PHOTON_ALLOW_RANDOM_INIT` warning は literal flag であり secret ではない [DR4-003] | 低 |

---

## 7. 衝突確認 (S5-005)

| Issue | 衝突内容 | 対策 |
|-------|---------|------|
| #138 (tokenizer mismatch) | `_load_hf_tokenizer` invariant | fake tokenizer の `vocab_size` は cfg と一致、invariant を逸脱しない |
| #140 (random-init WARNING) | embedding variance WARNING | caplog で許容、最小 1-step ckpt は trained checkpoint ではないため WARNING を品質指標として扱わない |
| #143 (eval reproducibility) | quality eval を実施しない | scope 外 |
| #154 (eval bugs) | `.github/workflows/weekly_eval.yml` の artifact/log 整理で衝突可能 | integration log は `logs/integration_test_*` prefix に固定し、既存 eval log / repo-id isolation の命名と混在させない |
| #144 (ruri-v2 build) | `pyproject.toml`, `CLAUDE.md`, `.github/workflows/weekly_eval.yml` の並行編集で merge conflict 可能 | 本 Issue は pytest 設定・品質チェック行・weekly integration step のみに変更を限定し、dependency/build-system セクションを追加しない |
| #156 (refusal-aware) | citation grading を実施しない | scope 外 |

---

## 8. 品質基準

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| 全テスト | `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | 約 510/512 通過 (本 Issue で +3 件 = positive 1 + negative 2、pre-existing failure 2 件除く) [DR2-001 反映] |
| Integration test 単独 | `python -m pytest tests/integration/test_photon_real_weights.py -v` | 全件パス、60 秒以内 |
| Lint | `ruff check .` | 警告 0 件 |
| Format | `ruff format --check .` | 差分なし |
| pytest discovery | `python -m pytest --collect-only` | エラーなし、testpaths 範囲のみ収集 |

**注記 [DR2-001 反映]**: Issue #145 受入条件の「約 508/510」表記は Issue 起票時 (S3-008) に detector (b) のみ +1 件で計算されており、S3-010 / DR1-002 で追加された detector (c) negative-path 2 件 (`test_photon_load_failure_without_bypass` / `test_photon_load_failure_with_bypass`) を反映できていない数字となっている。本設計確定後の正しい数字は **510/512** であり、§10 P6 で Issue 本文と `CLAUDE.md` の整合更新を実装フェーズの責任として明記する。

---

## 9. リスクと回避策

| リスク | 確率 | 影響 | 回避策 |
|-------|-----|-----|-------|
| MLX nondeterminism で flaky | 中 | 中 | seed 固定 + L2 norm 閾値比較 + numpy 経由 |
| autouse fixture 干渉 | 低 | 高 | `_real_photon_model_guard` 起動時 assertion + `tests/integration/` 分離 |
| pytest discovery 破壊 | 低 | 中 | `testpaths` 明記 + 設計フェーズで空テスト先行確認。加えて `--collect-only -q` の nodeid 比較で implicit discovery されていた test の欠落を検出する [DR3-003] |
| corrupt ckpt の意図せざる成功 | 低 | 高 | shape check は `_build_photon_deps` の existing validation に委ねる、test 側で前提条件を明確に |
| CI runtime 60s 超過 | 低 | 低 | M3 Ultra 上で 5-15 秒、4x マージン |
| `_build_photon_deps` signature 変更 | 中 (将来) | 中 | private API 依存は受容、変更時は test も追従 |
| `_build_photon_deps` signature 変更 PR で integration test 改修漏れ [DR2-008 / DR3-005 反映] | 中 (将来) | 中 | weekly_eval.yml は PR-blocking でないため検知が翌週月曜まで遅延。緩和策: (i) follow-up Issue で PR-level workflow を新設 (本 Issue scope 外)、(ii) `docs/code_review_checklist.md` に「`_build_photon_deps` signature 変更時は `tests/integration/test_photon_real_weights.py` の追従を確認」を追記する。本 Issue の影響ファイルに docs が含まれているため P6 の scope 内で扱う |

---

## 10. 実装フェーズ

| Phase | 作業 | 完了条件 |
|-------|------|---------|
| **P1: 基盤準備** | `tests/integration/__init__.py`, `pyproject.toml` 新規作成、空 test で discovery 確認 | `pytest --collect-only` でエラーなし、markers 認識。**testpaths 設定前後で `python -m pytest --collect-only -q` の nodeid set を比較し、既存 test の欠落がないことを確認** (`wc -l` は warning/header 行の影響を受けるため補助情報扱い) [DR3-003] |
| **P2: 最小 config** | `configs/photon_test_minimal.yaml` 作成、PhotonConfig が load 可能、**fake_tokenizer の必要 method 列挙** (deps['photon_inference'].generate_answer 経路を実コードから読む) | `from baseline_reporag.config import load_config` で OK、必要 method リストが §4-2-1 と一致 |
| **P3: positive path test** | `test_photon_pipeline_with_self_trained_minimal_ckpt` 実装 + fixture | 1 step 学習 + save + load + L2 比較 pass |
| **P4: negative path tests** | `test_photon_load_failure_without_bypass` + `_with_bypass` 実装 | RuntimeError / WARNING 両方 pass |
| **P5: weekly_eval.yml** | step 追加、`continue-on-error: true` なし。`set -o pipefail` + run-scoped `LOG_BASENAME="logs/integration_test_${GITHUB_RUN_ID:-local}"` + `tee "${LOG_BASENAME}.log"` + `--junitxml="${LOG_BASENAME}.xml"` で artifact 実体を生成。`set -x` / `printenv` / `env` は使わない [DR4-001 / DR4-003] | yml syntax check + manual/dry-run。失敗時も `logs/integration_test_*.log` / `logs/integration_test_*.xml` が artifact に含まれ、前回 run の stale log が混入しない |
| **P6: docs/CLAUDE.md 整合** | test 件数更新、`PHOTON_ALLOW_RANDOM_INIT` 適用範囲補足要否判定。**[DR2-001 / DR3-005 反映] 実装フェーズで以下を整合更新する責任:** (i) `CLAUDE.md` 品質チェック行 (約 507/509 → **510/512**)、(ii) Issue #145 本文の受入条件文言 (約 508/510 → **510/512**)、(iii) `docs/code_review_checklist.md` に private API signature 変更時の integration test 追従確認を追記、(iv) `docs/deployment.md` / `docs/troubleshooting.md` に `PHOTON_ALLOW_RANDOM_INIT` の test/ops 適用範囲補足が必要か確認 — Stage 1 反映で test 構造が +3 件 (positive 1 + negative 2) に確定したため、508/510 (Issue 起票時の +1 件想定) ではなく 510/512 が正しい数字 | 整合 OK、CLAUDE.md と Issue #145 本文の数字が 510/512 で揃い、docs 更新要否の判断結果が残っている |
| **P7: 全 quality gate** | `python -m pytest`, `ruff check .`, `ruff format --check .` | 全 pass |

---

## 11. follow-up Issue 候補

- 候補方針 A: 実 production checkpoint (`step_003000`) を取得して E2E test を回す (HF Hub upload / S3 mirror / self-hosted runner キャッシュ + HF allowlist 整備)
- PR-level blocking integration workflow の新規追加
- `verify_integrity=True` の `trainer.load_checkpoint` への plumbing

---

## レビュー反映履歴

### Stage 1 通常レビュー (claude-opus, 2026-04-28)

入力: `workspace/issues/145/multi-stage-design-review/stage1-review-result.json` (Must Fix 2 / Should Fix 6 / Nice to Have 2 = 計 10 findings)

| ID | Severity | カテゴリ | 反映箇所 | 概要 |
|----|----------|----------|---------|------|
| DR1-001 | Must Fix | 論理性 | §3 設計判断 #2 / §4-1 Step 7 / §5 (b) | `deps['photon_model']` は存在しない。`deps['photon_inference'].model` (PhotonInference line 189) 経由に修正 |
| DR1-002 | Must Fix | 論理性 | §3 設計判断 #3 / §4-1 Step 3,7 / §5 (b) | L2 norm 検出器を「random-init#1 vs load 後」差分から「trained 参照値との **近接性** (`abs(norm_after - norm_trained) < 1e-5`)」に変更。silently-skipped load を検出可能に |
| DR1-003 | Should Fix | DRY | §4-2 / §4-3 | typo `_phsoton_env_isolation` → `_photon_env_isolation` 修正。autouse fixture 分離の根拠を「monkeypatch teardown + conftest.py のディレクトリ局所性」に訂正 (testpaths は session 分離ではない) |
| DR1-004 | Should Fix | 論理性 | §5 (c) | corrupt ckpt の state.json を `CheckpointState` 仕様準拠の最小 schema (`step`, `best_val_loss`, `best_step`, `patience_counter`, `train_losses`, `val_losses`) に修正。不要 `vocab_size` key 除外 |
| DR1-005 | Should Fix | 可読性 | §5 (b) | `_flat_l2_norm` を `mlx.utils.tree_flatten` ベースに書き換え (既存 `count_parameters()` と pattern 一致) |
| DR1-006 | Should Fix | テスタビリティ | §5 (b) | `_run_one_step` の中身 (loss=photon_loss, optimizer=Adam(lr=1e-3), batch=(2,16)) を inline で明示 |
| DR1-007 | Should Fix | SOLID | §3 設計判断 #2 | private API 依存を受容する根拠を「(a) 2 重構造的圧力で signature 安定化を担保、(b) public API 経由は 60 秒予算違反」に補強 |
| DR1-008 | Should Fix | KISS | §4-2 / §4-2-1 (新設) / §10 P2 | `fake_tokenizer` の必要 method (`vocab_size`, `pad_token_id`, `encode`, `decode`) を §4-2-1 で明示し、P2 完了条件に「method 列挙」を追加 |
| DR1-009 | Nice to Have | YAGNI | §10 P1 | P1 完了条件に「testpaths 設定前後で `--collect-only` 件数が一致」を追加 |
| DR1-010 | Nice to Have | 一貫性 | §3 設計判断 #3 実装注意 3 | 閾値文言を「changed > 1e-4」から「近接性 `< 1e-5` または `mx.allclose(atol=1e-5)`」に書き換え (DR1-002 と一貫) |

**反映件数**: 10 件 / 据置: 0 件

### Stage 2 整合性レビュー (claude-opus, 2026-04-28)

入力: `workspace/issues/145/multi-stage-design-review/stage2-review-result.json` (Must Fix 0 / Should Fix 5 / Nice to Have 5 = 計 10 findings)

| ID | Severity | カテゴリ | 反映箇所 | 概要 |
|----|----------|----------|---------|------|
| DR2-001 | Should Fix | 品質基準整合性 | §8 注記 / §10 P6 完了条件 | Issue 本文 (508/510) と §8 (510/512) の不整合を §10 P6 の整合更新責任として明記。+3 件 = positive 1 + negative 2 の内訳を §8 表に明示。CLAUDE.md (507/509 → 510/512) と Issue #145 本文の整合更新を P6 で担保 |
| DR2-002 | Should Fix | コードベース整合性 | §3 設計判断 #5 末尾注記 | 最小 cfg 寸法表記の yaml キー名 (`num_heads`) と `_tiny_cfg` の field 名 (`num_attention_heads` / `num_key_value_heads`) の混在を脚注で明示。本 Issue の `configs/photon_test_minimal.yaml` は yaml なので `num_heads` を採用 |
| DR2-003 | Should Fix | fixture整合性 | §4-2 末尾追加 | autouse fixture 2 個の実行順を fixture parameter 依存で pin (`def _real_photon_model_guard(_photon_env_isolation): ...`) する旨を明示。fixture 統合案も同等に許容 |
| DR2-004 | Should Fix | コードベース整合性 | §5 (c) `_make_corrupt_checkpoint` docstring | `integrity.json` 不在 → `_verify_integrity(strict=False)` で WARNING + return → `mx.load(weights.npz)` で fail という load path の流れを docstring に明示。verify_integrity=True plumbing は follow-up (S5-003) と再確認 |
| DR2-005 | Should Fix | Issue整合性 | §2 表 `tests/__init__.py` 行 | 「(要検証)」表記を「新設しない (default) / fallback として保留 — 設計判断 #8 参照」に書き換え (確定後表現に統一) |
| DR2-006 | Nice to Have | 用語統一 | §3 設計判断 #3 用語定義追加 / §4-1 Step 7 | 'silently-skipped load' / 'load skip' を **'silent load skip'** に統一。'load failure' / 'load success' の定義も §3 #3 冒頭に追加 |
| DR2-007 | Nice to Have | 用語統一 | §3 設計判断 #3 (b) タイトル / 用語定義 / §4-1 Step 7 タイトル | 'weight identity 検証' を **'weight closeness 検証 (load 後 trained 一致)'** に統一 (DR1-010 の bit-perfect 否定と整合) |
| DR2-008 | Nice to Have | 設計判断間整合性 | §9 リスク表に新規行追加 | private API signature 変更時の integration test 改修漏れリスクを §9 に追加。緩和策として code_review_checklist.md 1 行追記を提案 (本 Issue では docs を編集しない方針との整合確認 Note を併記し scope 判断を実装フェーズに委ねる) |
| DR2-009 | Nice to Have | コードベース整合性 | §4-2-1 末尾注記 | `_check_weight_initialization` (`inference.py:114-144`) は `model.token_embed` のみ参照し tokenizer method の対象外である旨を P2 列挙作業の補助情報として明示 |
| DR2-010 | Nice to Have | 設計判断間整合性 | §3 設計判断 #5 末尾 / §9 リスク表との整合 | 60s budget の内訳積算 (positive 7-13s / negative 各 3-5s / 3 件合計 13-23s) を §3 #5 に追加。§9 「CI runtime 60s 超過 確率: 低」の根拠として参照可能に |

**反映件数**: 10 件 / 据置: 0 件 (Should Fix 5 全件 + Nice to Have 5 全件適用)

### Stage 3 影響範囲レビュー (codex, 2026-04-28)

入力: `workspace/issues/145/multi-stage-design-review/stage3-review-result.json` (Must Fix 2 / Should Fix 4 / Nice to Have 1 = 計 7 findings)

| ID | Severity | カテゴリ | 反映箇所 | 概要 |
|----|----------|----------|---------|------|
| DR3-001 | Must Fix | forward-pass妥当性 | §5 (b) forward logits 代替案 | `chunk_sizes: [4, 4]` の product=16 に対して `(1, 8)` input は PhotonModel の chunk 整列条件を満たさないため、`seq_len=16` に修正。`photon_mlx/tests/test_training.py:220-260` と同じく `mx.eval` 後に `mx.allclose(...).item()` で判定する形へ補強 |
| DR3-002 | Must Fix | 設定ファイル波及 | §3 設計判断 #5 | `_build_photon_deps` が必須参照する `tokenizer.tokenizer_id` と、不要な SafeRecGen 構築を避ける `inference.safe_recgen_enabled: false` を最小 YAML skeleton に明記 |
| DR3-003 | Should Fix | pytest discovery | §9 / §10 P1 | `testpaths` 追加時の implicit discovery 欠落検出を `wc -l` ではなく `--collect-only -q` の nodeid set 比較に変更 |
| DR3-004 | Should Fix | CI/CD artifact | §3 設計判断 #7 / §10 P5 | weekly_eval の artifact path 追加だけではログが生成されないため、`tee logs/integration_test_photon_real_weights.log` と `--junitxml` による artifact 実体生成を明記 |
| DR3-005 | Should Fix | document波及 | §9 / §10 P6 | `docs/code_review_checklist.md` 更新責務を P6 scope 内に確定し、deployment/troubleshooting の `PHOTON_ALLOW_RANDOM_INIT` 補足要否確認も完了条件へ追加 |
| DR3-006 | Should Fix | 並行Issue衝突 | §7 | #154 / #144 の並行編集リスクを衝突表に追加し、`weekly_eval.yml`, `pyproject.toml`, `CLAUDE.md` の merge conflict 対策を明記 |
| DR3-007 | Nice to Have | メタデータ整合 | 冒頭 / §2 | 設計レビュー累計を Stage 1-3 の 27 件に更新し、`CLAUDE.md` の想定件数を §8/P6 と同じ 510/512 に統一 |

**反映件数**: 7 件 / 据置: 0 件 (Must Fix 2 全件 + Should Fix 4 全件 + Nice to Have 1 適用)

### Stage 4 セキュリティレビュー (codex, 2026-04-28)

入力: `workspace/issues/145/multi-stage-design-review/stage4-review-result.json` (Must Fix 0 / Should Fix 3 / Nice to Have 2 = 計 5 findings)

| ID | Severity | カテゴリ | 反映箇所 | 概要 |
|----|----------|----------|---------|------|
| DR4-001 | Should Fix | CI runner | §3 設計判断 #7 / §6 / §10 P5 | self-hosted runner の workspace 再利用で stale integration log が artifact に混入しないよう、`GITHUB_RUN_ID` を含む run-scoped log/xml 名に変更 |
| DR4-002 | Should Fix | path traversal / file read | §6 | `checkpoint_path` は tmp root 相対 basename (`"test_ckpt"` / `"corrupt_ckpt"`) に固定し、`../`, symlink, repo 外絶対パスは既存 boundary test の責務として分離する旨を明記 |
| DR4-003 | Should Fix | secret leak | §3 設計判断 #7 / §6 / §10 P5 | weekly_eval integration step で `set -x`, `printenv`, `env` 等の env dump を禁止し、artifact を pytest stdout/JUnit XML のみに限定 |
| DR4-004 | Nice to Have | code execution / eval/exec | §6 | corrupt checkpoint fixture は raw invalid bytes のみを書き、`np.load(..., allow_pickle=True)`, pickle, object dtype, Python `eval`/`exec`, subprocess を使わない方針を明記 |
| DR4-005 | Nice to Have | DI / fixture isolation | §3 設計判断 #6 / §4-4 / §6 | fake tokenizer patch を function scope `monkeypatch.setattr` に限定し、module/session scope や import-time patch を禁止。process-wide env を使うため xdist/forked 前提にしない旨も明記 |

**反映件数**: 5 件 / 据置: 0 件 (Should Fix 3 全件 + Nice to Have 2 全件適用)
