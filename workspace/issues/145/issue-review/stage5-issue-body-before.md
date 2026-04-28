## 背景

#139 (Stub/Mock pattern audit) のスコープ整理 (Stage 1 review S1-001 / S1-002 / S1-008) の結果、当初 Task 2 として含まれていた **real-weight integration test** を本 Issue に切り出した。Issue 起票時 (2026-04-26) は #135 が未マージであったため follow-up に分離したが、2026-04-28 時点で依存はすべて解消済みであり、**着手可能状態**である。

経緯:
- #139 Task 2 は #135 (S7-001 fix, commit `2dbf458` で導入された `model.checkpoint_path` 機構) を前提に設計されていた
- Issue 起票時 (2026-04-26) は #135 が `feature/issue-135-photon-retrain` 上のみ存在し main にマージされていなかった
- **2026-04-28 時点で #135 は main にマージ済** (commit 8c13517 / PR #157)。`baseline_reporag/photon_pipeline.py` line 256 (`_build_photon_deps`)、line 365-417 (checkpoint resolve 経路)、line 626-633 (`_load_photon_checkpoint`) で `cfg.model.checkpoint_path` を resolve し `photon_mlx.trainer.load_checkpoint` (trainer.py:80) で実 weight を読む経路が main 上に存在する
- **2026-04-28 時点で #139 (Task 1 + Task 3) も main にマージ済** (PR #147, 2026-04-27)

## 依存

- **必須 (解消済)**: #135 (本格再学習 / S7-001 fix の checkpoint loading 機構) が main にマージ済 — commit 8c13517 / PR #157 (2026-04-28)
- **推奨 (解消済)**: #139 (Task 1 + Task 3 / scaffolding audit + invariant test) が main にマージ済 — PR #147 (2026-04-27)。production path から `_StubTokenizer` 等が排除済みであり、本 test はクリーンに書ける状態

## ゴール

実 PhotonModel + 実 (or 実用最小) checkpoint + 実 tokenizer + 実 retrieval を通る end-to-end query test を 1 件以上追加し、「random-init weight が production で silently 動いていた」型の構造的事故を CI で検出可能にする。

## 変更内容

新規ファイル: `tests/integration/test_photon_real_weights.py`

### 候補方針 A: 既存最小 checkpoint を使用 (実 ckpt 利用) — **本 Issue scope 外 / follow-up Issue 化**

```python
@pytest.mark.integration
@pytest.mark.slow
def test_photon_pipeline_with_real_checkpoint(tmp_path):
    """E2E: 実 checkpoint + 実 tokenizer + 実 retrieval で query が走る"""
    ckpt = path_to_smallest_committed_or_downloadable_ckpt()  # configs/photon_tiny.yaml 等
    cfg_yaml = make_test_config(ckpt_path=ckpt)
    pipeline = build_pipeline(cfg_yaml)
    result = pipeline.query("What is FastAPI?")
    assert result.answer is not None
    assert len(result.evidence) > 0
    # random-init detector: 実在 API ベースの選択肢 (a)-(d) で実装する (受入条件 (3) 参照)
```

候補方針 A は本 Issue では follow-up に明示的に外す。理由: (a) `photon_institutional_retrain_20260428/step_003000` のような production ckpt を取得する手段 (HF Hub upload / self-hosted runner キャッシュ / S3 mirror) の決定、(b) CI runtime budget (現在 weekly_eval は 180 分タイムアウト) の評価、(c) HF allowlist 整備 (`baseline_reporag/photon_pipeline.py:340-342` の `_validate_repo_id`) が必要。本 Issue 完了後の follow-up Issue (タイトル例: `test(photon): real production checkpoint integration test`) に分離する。

### 候補方針 B: セルフホスト型最小 e2e (**本 Issue 採用**, repo 完結)

```python
@pytest.mark.integration
def test_photon_pipeline_with_self_trained_minimal_ckpt(tmp_path):
    """最小 PhotonConfig で 1 step 学習 → save → load → query が走る"""
    # 1. 最小 config (vocab=256, hidden=64, layers=2 等) で PhotonModel を構築
    # 2. ダミーデータで 1 step 学習 (seed=42 固定で deterministic 挙動を保証;
    #    1 step の forward+backward+optimizer.step は最小 cfg でおよそ 1-3 秒 / save_checkpoint で +1-2 秒 /
    #    学習前後で `model.parameters()` の sample tensor が確実に変化することを `mx.array_equal` の否定で確認可能)。
    #    tmp_path に checkpoint 保存する際は `photon_mlx.trainer.save_checkpoint(model, state, path)` を直接呼ぶ。
    # 3. 保存した checkpoint dir を `cfg.model.checkpoint_path` に注入し、
    #    以下のいずれかで PhotonModel を実体化する (Stage 3 影響範囲レビューで確定):
    #    (i) `_build_photon_deps(cfg)` を直接呼ぶ — 軽量で test 観点を満たす (default 推奨)
    #        (private API 依存だが、`baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py` で
    #         既に同関数を patch ベースで叩いている前例あり)
    #    (ii) `pipeline_factory.build_pipeline(cfg_yaml)` 経由で PhotonRAGPipeline を完成させる
    #        — 実 retrieval/embedding まで通すが、tmp_path に最小 corpus + index 準備が必要
    # 4. (i) の場合は `deps['inference']` 経由で 1 query 相当を流す。
    #    (ii) の場合は `pipeline.query(...)` で result.answer を assert
    # 5. random-init detector: 受入条件 (3) の (a)-(d) いずれかを採用 (default: (b) weight identity 検証 + (c) PHOTON_ALLOW_RANDOM_INIT bypass の併用、S3-010 参照)
```

利点: 外部リソース不要、CI で確実に再現、最小 cfg + 1 step + tmp_path I/O で M3 Ultra 上で 5-15 秒程度、self-hosted runner なら blocking PR への組み込みも検討可能。
欠点: 「実用 size の weight を通す」 e2e ではないため、Issue #135 想定の random-init bug 検出力は方針 A より弱い。

→ **採用**: 方針 B を本 Issue で実装、方針 A は別 follow-up Issue で段階導入する。

## 受入条件

- [ ] `tests/integration/test_photon_real_weights.py` に real-weight integration test を 1 件以上追加
- [ ] テスト構造の最低保証 (mock を使わない範囲を明示):
   - [ ] `PhotonModel(photon_cfg)` の実インスタンスが構築されること (PhotonModel を patch しない)
   - [ ] `photon_mlx.trainer.load_checkpoint` (実体は `photon_mlx.checkpoint.load_checkpoint`、`trainer.py:80` で lazy re-export) が実 npz weight ファイルから state を復元すること (`load_checkpoint` を patch しない / `baseline_reporag.photon_pipeline._load_photon_checkpoint` も patch しない — この 2 箇所が production の load 経路) [S3-001]
   - [ ] 既存テスト `baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py` は `_load_photon_checkpoint` を patch しているため、本 integration test は "`_load_photon_checkpoint` を patch しない" ことで前者と差別化される (検出力の境界を明示) [S3-001]
   - [ ] checkpoint dir は test 内で `photon_mlx.trainer.save_checkpoint` を呼んで実生成する (mock dir を使わない)
   - [ ] tokenizer は HF download を避けるため `_load_hf_tokenizer` を local fixture に置き換えてよい (test isolation のため許容)
   - [ ] integration test 内で `photon_mlx.model.PhotonModel` の **実体** が import されることを検証する fixture を用意する (e.g., `import photon_mlx.model; assert not isinstance(photon_mlx.model.PhotonModel, MagicMock)`)。これにより既存 `test_photon_pipeline_checkpoint_load.py` の autouse fixture `_patch_heavy_deps` (line 75-123) による MagicMock patch 残留を test 開始時に検出可能 [S3-006]
   - [ ] integration test は他 test と独立して走らせるため `tests/integration/` 配下に分離 (既に scope 内で確定済)。pytest-xdist 並列実行時は `--forked` か `-n 0` (integration test のみシリアル化) を README / CI コマンドに明記 [S3-006]
   - [ ] `monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))` で test scope のみ checkpoint root を tmp_path に向ける (既存 `baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py:142` の前例に従う) [S3-007]
   - [ ] tmp_path 配下に `save_checkpoint(model, state, tmp_path / "test_ckpt")` で生成した dir を `cfg.model.checkpoint_path` に注入 (相対パス `"test_ckpt"` で記述、`_resolve_checkpoint_path` が root 配下に解決)。`_resolve_checkpoint_path` は `Path.resolve(strict=True)` を呼ぶため tmp_path 経由でも実在性が要求される (空 dir では fail) [S3-007]
   - [ ] 既存の本番 `./checkpoints/photon_institutional_retrain_20260428/step_003000/` とは tmp_path scope 化により完全分離 (干渉なし) [S3-007]
- [ ] random-init detector を、実在する API に基づいた以下のいずれか (or 複数併用) で実装すること (Stage 3 影響範囲レビュー or 設計フェーズで確定):
   - 推奨組み合わせ (S3-010): **(b) weight identity 検証 (positive path) + (c) PHOTON_ALLOW_RANDOM_INIT bypass ネガティブテスト (negative path)** の 2 案併用 — 学習データ load 経路と bypass 経路の両方を pin することで random-init silently 動作型の事故を多角的に検出可能。実装コスト合計 15-20 行程度。
   - (a) **checkpoint integrity hash 検証** — `photon_mlx/checkpoint.py` の `integrity.json` (DR4-003) が存在し、test 内で `_verify_integrity` 経由で SHA-256 一致を確認できること (構造的に "checkpoint が読まれた" を保証)。`load_checkpoint(verify_integrity=True)` 引数追加で簡潔だが、SHA-256 比較は "checkpoint が物理的に存在し改竄されていない" を保証するのみで "weight が model に load された" の verify には (b) との組み合わせが必要 [S3-010]
   - (b) **weight identity 検証** (default 推奨) — random-init `PhotonModel(cfg)` のパラメータと、checkpoint load 後のパラメータの L2 norm / sample tensor が異なることを assert
     - 実装注意 (S3-005):
       1. `mx.random.seed(42)` を test 冒頭で明示 call (PhotonConfig の seed 設定に頼らず、test 自身で seed を固定)
       2. 比較は `mx.eval(param)` 後に `np.asarray(param)` 経由で numpy ベースで行う (`mx.array_equal` の lazy eval 罠回避)
       3. L2 norm 比較は閾値 (e.g., `abs(norm_before - norm_after) > 1e-4`) で実装し bit-perfect 比較は避ける
     - フォールバック: (b) で flaky が発生する場合は (a) checkpoint integrity hash 検証 (DR4-003 の SHA-256 ベース、bit-perfect 保証) に切替可能
   - (c) **PHOTON_ALLOW_RANDOM_INIT bypass のネガティブテスト** — checkpoint_path 未設定時に WARNING ログが出る + production-safe path では RuntimeError が出ることを assert (`baseline_reporag/photon_pipeline.py:415-419 / 408-413` の挙動を pin する)
   - (d) `_score_prune_candidates` を private API として直接呼び、raw_score の std を比較する (`photon_mlx/inference.py:536`、private API 依存のリスクあり)
   - 注: 元の Issue で記載されていた `pipeline.last_pruning_score_distribution.std()` は **実コードベース上に存在しない架空 API** であり採用不可
- [ ] CI 実行戦略を以下の事実認識のもとで確定する (Stage 3 影響範囲レビュー or 設計フェーズで決定):
   - 現行 main には blocking PR workflow が定義されておらず、`.github/workflows/weekly_eval.yml` (cron: Monday 00:00 UTC, runs-on: self-hosted, timeout 180min) のみ存在する
   - MLX 依存テスト (PhotonModel 構築) は self-hosted Apple Silicon runner 必須
   - pytest marker (`integration`, `slow`) は pyproject.toml/pytest.ini が repo に未存在のため、本 Issue で markers を新規登録する (`pyproject.toml [tool.pytest.ini_options] markers = [...]`)
   - 推奨: weekly_eval.yml に `Run real-weight integration test` step を追加 (blocking ではない nightly 相当) + 毎 PR では `pytest -m "not integration"` で除外する形を採用
   - 運用詳細 (S3-004):
     - 挿入位置: `Run multi-turn eval` step の **後** (eval 完走後に integration test を走らせ、eval 結果に影響を与えない)
     - 失敗時挙動: `continue-on-error: true` で job 全体を fail させない (nightly notification はログ確認で対応)
     - artifact: 既存 `eval-results-${{ github.run_number }}` artifact に `logs/integration_test_*.log` を追加 (`if: always()` で常時収集)
     - 実行時間予算: integration test は **60 秒以内に完走することを受入条件で定める** (実測 5-15 秒予測に対して 4x の安全マージン)
- [ ] (推奨 / S1-009) `PHOTON_ALLOW_RANDOM_INIT` の挙動を pin する: env 未設定 + checkpoint 不在 → RuntimeError、env=1 + checkpoint 不在 → WARNING + 続行、を両方 assert (`baseline_reporag/photon_pipeline.py:397-413` の挙動を回帰防止)
- [ ] 既存テスト + 新規テスト全パス (`python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` 全件パス、約 507/509、残り 2 件は CLAUDE.md 記載の pre-existing failure。本 Issue 完了後は約 508/510 に件数更新 [S3-008])
- [ ] `ruff check .` 警告 0 件

## 影響ファイル (scope 内)

- `tests/integration/__init__.py` (新規, 既存 tests/ 配下に integration/ サブディレクトリは未存在)
- `tests/integration/test_photon_real_weights.py` (新規)
- (要検証) `tests/__init__.py` を新設するか — 現状 `tests/` 直下は `__init__.py` 不在のフラット構成。pytest rootdir 推定が integration/ 配下でも壊れないか、設計フェーズで `pytest tests/integration/test_photon_real_weights.py -v` を空テストで先行実行して確認する。安全策: `tests/__init__.py` を新設しない方針を採る場合、`pyproject.toml [tool.pytest.ini_options]` で `testpaths = ["tests", "baseline_reporag/tests", "photon_mlx/tests", "torch_ref/tests"]` を明記して discovery 範囲を pin する [S3-002]
- `configs/photon_test_minimal.yaml` (新規, test 専用最小 PhotonConfig — 既存 `configs/photon_tiny.yaml` を流用しない理由: tiny も hidden=640 / 80M params で学習に分単位かかるため。candidate B の vocab=256 / hidden=64 / layers=2 想定)
- `pyproject.toml` (新規, 本 Issue scope は `[tool.pytest.ini_options] markers = ["integration", "slow"]` のみ追加。`[tool.ruff]` / `[build-system]` / `[project]` セクションは追加しない (既存の `ruff check .` 警告 0 件挙動を保持するため、ruff config は cwd ベースのデフォルト動作に委ねる)。`[tool.pytest.ini_options]` には併せて `testpaths` を明記し既存 4 ディレクトリ (`tests`, `baseline_reporag/tests`, `photon_mlx/tests`, `torch_ref/tests`) を pin する (S3-002 と整合) [S3-003]
- `.github/workflows/weekly_eval.yml` (nightly に integration test step を追加する場合、運用詳細は受入条件 CI 実行戦略の S3-004 項を参照)
- (推奨) `CLAUDE.md` の「品質チェック」コマンド整合確認: integration test を default で除外する場合は `pytest -m "not integration"` フラグ追加要否を判定。default で含める場合は test 件数 (約 507 → 約 508) 更新要否を判定 [S3-008]
  - 採用方針 (推奨): 既存コマンド `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` は integration test を含む (5-15 秒程度の追加なので blocking で良い)。CLAUDE.md 更新は test 件数を「約 508/510 通過」に更新する程度で十分。
  - 代替方針: integration test を nightly のみ走らせる場合 (CI 戦略 S1-004 で nightly 採用なら) 、開発者ローカルでは default 含む / CI weekly_eval.yml で `--ignore=tests/integration/` を追加し別 step で integration test を実行。CLAUDE.md には integration 専用コマンドを別行で追記。

## scope 外 (follow-up Issue 化)

- 候補方針 A (実 production ckpt `step_003000` を取得して E2E test を回す) — checkpoint 取得経路 (HF Hub / S3 / self-hosted runner キャッシュ) の決定が必要
- PR-level blocking integration workflow の新規追加

## 関連

- 切り出し元: #139 (Stub/Mock pattern audit) — Task 1 + Task 3 を先行
- 必須依存: #135 (本格再学習 / S7-001 fix) — main マージ済 (commit 8c13517 / PR #157)
- 推奨依存: #139 (Task 1 + Task 3) — main マージ済 (PR #147)
- 緊急修正: #138 (tokenizer mismatch) — マージ済み (#141)

## レビュー履歴

- 切り出し根拠: #139 Stage 1 通常レビュー S1-001 / S1-002 / S1-008
  - workspace/issues/139/issue-review/stage1-review-result.json
- 切り出し時刻: 2026-04-26
- Stage 1 通常レビュー (1回目) 反映 (2026-04-28): S1-001 (依存ステータス更新), S1-002 (架空 API 修正 / random-init detector を実在 API ベース (a)-(d) に書き換え), S1-003 (候補方針 A scope 外化), S1-004 (CI 戦略の判断材料追記), S1-005 (wire-up 経路明示), S1-006 (影響ファイル整合), S1-007 (mock 粒度明示), S1-008 (1 step 学習所要時間補足), S1-009 (PHOTON_ALLOW_RANDOM_INIT 負経路テスト追加)
  - workspace/issues/139/issue-review/stage1-review-result.json
  - workspace/issues/145/issue-review/stage1-review-result.json
- Stage 3 影響範囲レビュー (1回目) 反映 (2026-04-28): S3-001 (`_load_photon_checkpoint` 実体名固定 / patch 不可境界明示), S3-002 (`tests/__init__.py` 追加要否を影響ファイルに明示 + pyproject.toml `testpaths` 安全策), S3-003 (pyproject.toml scope 最小化、`[tool.ruff]` 追加しない方針), S3-004 (weekly_eval.yml への step 挿入位置・continue-on-error・artifact・60 秒予算の運用詳細), S3-005 (MLX nondeterminism 対策の `mx.random.seed(42)` 明示 call / `mx.eval` + numpy 経由比較 / L2 閾値 1e-4), S3-006 (autouse fixture 干渉対策の `MagicMock` 検出 fixture + `--forked`/`-n 0` 指針), S3-007 (`PHOTON_CHECKPOINT_ROOT` の `monkeypatch.setenv` 経由設定方針), S3-008 (CLAUDE.md 整合 / test 件数 508/510 更新方針), S3-010 (H5 代替案併用方針 (b)+(c) 推奨を受入条件に追記)。S3-009 (tokenizer 置換方針 default 推奨明示) は Nice to Have、実装者判断に委ねるため据置。
  - workspace/issues/145/issue-review/stage3-review-result.json

