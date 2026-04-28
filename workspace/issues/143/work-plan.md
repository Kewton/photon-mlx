# Issue #143 作業計画書

作成日: 2026-04-28
ブランチ: `feature/issue-143-eval-reproducibility` (作成済)
設計方針書: `workspace/design/issue-143-eval-reproducibility-design-policy.md`
レビュー反映: Issue Stage 1-8 (32 findings) + 設計 Stage 1-4 (31 findings) 全反映済

---

## Issue: institutional eval reproducibility — Qwen 14B nondeterminism causes ±1.7pt baseline drift

- **Issue 番号**: #143
- **サイズ**: **L** (~655 LOC + 10h manual run)
- **優先度**: High (Issue #135 PHOTON 採用判定の前提に直結)
- **依存 Issue**: #156 (Step 5-6 前にマージ完了必須)
- **関連 Issue**: #137 (CLOSED, 元), #135 (PHOTON 採用), #138 (CLOSED, 追加対応不要)

---

## サマリー

institutional eval (baseline + multi-turn) の Qwen 14B nondeterminism による NC rate ±1.7pt 揺れを seed 固定で抑え、A/B 判定 threshold (-2pt) をノイズより大きく確保する。`Generator.generate(*, seed)` / `Pipeline.query(*, seed)` を keyword-only で追加し、eval scripts のみが `cfg.run.seed` を明示的に伝播。`--runs N` (1 ≤ N ≤ 20) で multi-run 集計、Task 4 で 10-run noise floor を実測する。

**設計の要**:
- `seed=None` デフォルトで CLI/server/Streamlit interactive 経路を完全保護
- `cfg.run.seed` / `cfg.run.deterministic` を source of truth にして既存 dead key を蘇らせる
- `--repo-id` silent bug を同時修正
- merge order: **#156 → #143** に固定

---

## Phase 1: 実装タスク (Step 1-7)

### Step 1: helper layer (前提)

- [ ] **Task 1.1**: `baseline_reporag/eval/run_config.py` 新規作成
  - 成果物: `resolve_eval_seed(cfg) -> int | None` 公開関数 + `_validate_run_block(run_dict)` private 関数
  - 仕様:
    - `cfg.run` 欠落時 → default `seed=42, deterministic=True`
    - `type(seed) is int` 厳密判定 (bool/float/NaN/str は `TypeError`)
    - `0 <= seed < 2**32` range 検証 (範囲外は `ValueError`)
    - `deterministic` 非 bool は `TypeError`
    - `deterministic=False` のとき `seed=None` を返す
  - 依存: なし
  - 推定 LOC: +30
- [ ] **Task 1.2**: `baseline_reporag/tests/test_run_config.py` 新規作成 (TDD red→green)
  - 成果物: 10 test cases:
    1. `run` 欠落 → default 42 / True
    2. `run.deterministic=False` → seed=None
    3. `run.seed=42` → 42
    4. `run.deterministic=True` (bool) → 42
    5. `run.seed=true` (YAML bool) → `TypeError`
    6. `run.seed="42"` (str) → `TypeError`
    7. `run.seed=3.14` (float) → `TypeError`
    8. `run.seed=-1` → `ValueError`
    9. `run.seed=2**32` → `ValueError`
    10. `run.deterministic="false"` (str) → `TypeError`
  - 依存: Task 1.1
  - 推定 LOC: +50

### Step 2: API 拡張 (Generator + Pipeline + PhotonPipeline)

- [ ] **Task 2.1**: `baseline_reporag/generation/generator.py:Generator.generate(*, seed: int | None = None)` 追加
  - シグネチャ: `def generate(self, messages: list[dict], max_new_tokens: int | None = None, *, seed: int | None = None) -> str:`
  - 実装: `if seed is not None: import mlx.core as mx; mx.random.seed(seed)` を generate 呼び出し直前に挿入 (= QwenMLXAdapter と同パターン、ADR-11 で B 採択 = 層分離)
  - **重要**: `if seed:` ではなく `if seed is not None:` (seed=0 silent bug 防止 / DR3-002)
  - 依存: なし (既存呼び出し点 4 箇所は keyword-only のため後方互換)
  - 推定 LOC: +10
- [ ] **Task 2.2**: `baseline_reporag/pipeline.py:RepoRAGPipeline.query(*, seed: int | None = None)` 追加
  - シグネチャ: `def query(self, question: str, session_id: str = "", repo_id: str = "", *, seed: int | None = None) -> QueryResult:`
  - 実装: `seed is not None` のとき `self.generator.generate(messages, seed=seed)` を呼ぶ。`seed=None` のとき既存の `self.generator.generate(messages)` (引数 shape 維持で 17+ MagicMock 後方互換)
  - 依存: Task 2.1
  - 推定 LOC: +15
- [ ] **Task 2.3**: `baseline_reporag/photon_pipeline.py:PhotonRAGPipeline.query(*, seed)` 追加
  - シグネチャ: `def query(self, question: str, session_id: str = "", repo_id: str = "", *, seed: int | None = None) -> QueryResult:`
  - 実装: Qwen-only path / Qwen fallback (3 箇所: L1030, L1043, L1394) で `bl.generator.generate(messages, max_new_tokens=followup_tokens, seed=seed)` を伝播
  - 依存: Task 2.1
  - 推定 LOC: +20
- [ ] **Task 2.4**: `baseline_reporag/tests/test_pipeline_integration.py` に seed 伝播 unit test 追加
  - 確認内容:
    - `RepoRAGPipeline.query(seed=42)` → `mock_gen.generate.assert_called_with(messages, seed=42)`
    - `RepoRAGPipeline.query(seed=None)` → `mock_gen.generate.assert_called_with(messages)` (既存と同じ引数 shape、後方互換)
    - 既存の `mock_gen.generate.return_value = ...` 8 件のテストが TypeError なしで pass
  - 依存: Task 2.2
  - 推定 LOC: +10
- [ ] **Task 2.5**: `baseline_reporag/tests/test_photon_pipeline.py` に seed 伝播 unit test 追加
  - 確認内容: Qwen-only path / fallback 3 path 全てで seed が伝播される (mock 17+ 件後方互換)
  - 依存: Task 2.3
  - 推定 LOC: +5

### Step 3: eval scripts 4 件への seed 伝播 + --repo-id silent bug fix

- [ ] **Task 3.1**: `scripts/run_baseline_eval.py` 修正
  - 変更: (a) `from baseline_reporag.eval.run_config import resolve_eval_seed`、(b) `seed = resolve_eval_seed(cfg)` を pipeline 構築前に呼ぶ、(c) `pipeline.query(..., seed=seed)` で渡す、(d) **`--repo-id` silent bug fix**: `cfg.repo.repo_id = repo_id` を `build_pipeline(cfg)` 直前に明示反映 (現状: query 時のみ override されて index load との不一致)
  - 依存: Task 1.1, Task 2.2
  - 推定 LOC: +20
- [ ] **Task 3.2**: `scripts/run_multi_turn_eval.py` 修正
  - 変更: 同上の (a)(b)(c) のみ (`--repo-id` 反映は既存実装で正しい)
  - 依存: Task 1.1, Task 2.3
  - 推定 LOC: +10
- [ ] **Task 3.3**: `scripts/retrieval_grid_search.py` 修正 (DR3-001 反映)
  - 変更: `resolve_eval_seed(cfg)` 呼出 + `pipeline.query(..., seed=seed)` 伝播
  - 依存: Task 1.1, Task 2.2
  - 推定 LOC: +10
- [ ] **Task 3.4**: `scripts/run_stress_eval.py` 修正 (DR3-001 反映)
  - 変更: `resolve_eval_seed(cfg)` 呼出 + `pipeline.query(..., seed=seed)` 伝播
  - 依存: Task 1.1, Task 2.2
  - 推定 LOC: +10
- [ ] **Task 3.5**: `tests/test_retrieval_grid_search_smoke.py` / `tests/test_run_stress_eval.py` に seed 伝播 smoke test 追加 (新規 or 追記)
  - 依存: Task 3.3, Task 3.4
  - 推定 LOC: +40

### Step 4: determinism integration test (受入条件 1)

- [ ] **Task 4.1**: `evals/tests/test_eval_determinism.py` 新規作成
  - 成果物:
    - `@pytest.mark.skipif(not _HAS_MLX, reason="requires MLX")`
    - 同一 prompt × seed=42 で 2-run の `Generator.generate()` を呼び、出力 (cited_chunk_ids, no_citation) が完全一致することを assert
    - 1 prompt 検証で CI 速度維持 (LLM 起動を含むため self-hosted runner のみ実走)
  - 仮検証戦略 (リスク §11): もし mlx-lm 内部 nondeterminism で完全一致しない場合 → 2-run の token 編集距離 < 5 を soft assert に切替 (Issue 受入時に決定)
  - 依存: Task 2.1, Task 3.1
  - 推定 LOC: +50

### Step 5: --runs N + predictions schema 拡張 (#156 マージ後)

> **前提**: Step 5-6 着手前に Issue #156 (is_refusal 出力欠落) のマージ完了を確認する

- [ ] **Task 5.1**: `scripts/run_baseline_eval.py` に `--runs N` 引数追加
  - 仕様: argparse `type=int_in_range(1, 20)` (custom validator) で `1 <= N <= 20` を強制 (DR4-002 反映)。default `--runs 1` (=既存互換)
  - schema: predictions JSONL に `run_index: int`, `run_seed: int` の **2 fields** を必須出力 (DR1-006 / ADR-5: run_id は ADR-5 で計算復元方針)
  - 実装: 外側で `for run_index in range(args.runs):` ループ、各 run で `run_seed = resolve_eval_seed(cfg)` (固定 seed 戦略 / Task 4 noise floor 用) または `run_seed = base_seed + run_index` (seed sensitivity 戦略、未採用 / 別 mode)
  - 依存: Task 1.1, Task 3.1
  - 推定 LOC: +35
- [ ] **Task 5.2**: `scripts/run_multi_turn_eval.py` に `--runs N` 引数追加
  - 同様の bounded validator + per-run iteration + 2 fields 出力
  - 依存: Task 1.1, Task 3.2
  - 推定 LOC: +35
- [ ] **Task 5.3**: `tests/test_run_baseline_eval.py` / `tests/test_run_multi_turn_eval.py` に bounds test 追加 (新規 or 追記)
  - 確認: `--runs 0` / `--runs 21` / `--runs -1` / `--runs abc` で argparse error
  - 依存: Task 5.1, Task 5.2
  - 推定 LOC: +20

### Step 6: aggregator per-run 集計 (record_type=static/multi_turn)

- [ ] **Task 6.1**: `scripts/aggregate_institutional_baseline.py` の per-run 集計対応
  - 変更:
    - `compute_per_run_stats(records, group_field)` 共通 helper 切り出し (DR1-005 反映、KISS/YAGNI 受容)
    - `record_type=static`: `group_field='eval_id'` で per-run 集計 (既存)
    - `record_type=multi_turn`: `group_field=('session_id', 'turn_id')` で per-run 集計 (新規) — REQUIRED_FIELDS は別系統
    - 出力 schema 追加: `mean, std, min, max, n_runs, seeds`
    - **後方互換 (DR3-003)**: `run_index/run_seed` を REQUIRED_FIELDS に **入れない**。欠落時は単一 run として正規化 (= 旧 JSONL を読んでも壊れない)
  - 依存: Task 5.1
  - 推定 LOC: +90
- [ ] **Task 6.2**: `tests/test_aggregate_institutional.py` に test 追加 (既存ファイルへの追記)
  - test cases:
    - per-run NC rate / latency の `mean/std/min/max/n_runs/seeds` 算出
    - `record_type=multi_turn` での MT predictions 集計
    - 旧 JSONL (run_index/run_seed なし) を読んでも壊れず単一 run として扱う
  - 依存: Task 6.1
  - 推定 LOC: +50

### Step 7: compare_generators の seed 伝播

- [ ] **Task 7.1**: `scripts/compare_generators.py` に `resolve_eval_seed(cfg)` + `pipeline.query(seed=seed)` 追加
  - 依存: Task 1.1, Task 2.2
  - 推定 LOC: +10
- [ ] **Task 7.2**: `tests/test_compare_generators.py` に seed 伝播確認を追加
  - 依存: Task 7.1
  - 推定 LOC: +10

---

## Phase 2: ablation / noise floor / 文書化 (Step 8-10)

### Step 8: temperature=0 ablation (受入条件 Task 2)

- [ ] **Task 8.1**: institutional eval を seed=42 で `temperature=0.2` と `temperature=0.0` で各 1 run 実行 (約 2-3 時間/run)
  - 比較: NC rate / 出力品質 (degenerate output の有無)
  - 出力: `reports/institutional_temperature_ablation_v143.md` (新規)
- [ ] **Task 8.2**: 結果に応じて `configs/institutional_docs.yaml` の generation.temperature を 0.0 に変更 (採用) または 0.2 維持 + 文書化
  - 推定 LOC: +30 (報告書 + config 変更)

### Step 9: 10-run noise floor (受入条件 Task 4)

- [ ] **Task 9.1**: `python scripts/run_baseline_eval.py --config configs/institutional_docs.yaml --repo-id institutional_documents --runs 10 --output reports/noise_floor_run.jsonl` (約 10h, manual)
- [ ] **Task 9.2**: aggregator で `mean ± std` を抽出
- [ ] **Task 9.3**: `reports/institutional_eval_noise_floor.md` 新規作成
  - schema: `mean, std, min, max, n_runs, seeds, computed_at_commit, judgment_threshold (mean - 2*std), per_run table`
  - 期待値: std ≤ 0.5pt (合格)、std ≤ 1.0pt (現状改善ライン)
  - 推定 LOC: +50

### Step 10: 文書更新 (受入条件 Task 5)

- [ ] **Task 10.1**: `CLAUDE.md`「現在のメトリクス」を Task 4 完了後の seed=42 固定 mean ± std で更新
  - 推定 LOC: +10
- [ ] **Task 10.2**: `docs/deployment.md` に「seed 固定の有無」セクション追加 (eval = seed=42 固定、interactive = seed=None)
  - 推定 LOC: +15
- [ ] **Task 10.3**: `docs/troubleshooting.md` に FAQ 追加 (「回答が seed 固定後も揺れる場合」: mlx-lm 内部 nondeterminism 由来、本リポジトリ範囲外)
  - 推定 LOC: +15
- [ ] **Task 10.4**: `docs/code_review_checklist.md` に追記 (DR1-007 / ADR-11 関連: seed keyword-only / seed 注入時の両者同時更新)
  - 推定 LOC: +5

---

## Phase 3: PR 作成 (Step 11)

- [ ] **Task 11.1**: 全変更を commit (Step 1-10 を 7-10 個の論理 commit に分割: Step 1-2 / Step 3 / Step 4 / Step 5 / Step 6 / Step 7 / Step 8 / Step 9 / Step 10)
- [ ] **Task 11.2**: 品質チェック完全パス確認 (下記)
- [ ] **Task 11.3**: `gh issue edit 143` で受入条件 Task 3 の文言を **2 fields** に同期 (DR2-002): `run_index/run_seed/run_id` → `run_index/run_seed の 2 fields (run_id は計算復元)`
- [ ] **Task 11.4**: `/create-pr` で PR 作成 (タイトル: `fix(eval): institutional eval reproducibility — seed pinning + multi-run aggregation (#143)`)

---

## 品質チェック項目

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| ユニットテスト (Issue 範囲) | `python -m pytest baseline_reporag/tests/test_run_config.py baseline_reporag/tests/test_pipeline_integration.py baseline_reporag/tests/test_photon_pipeline.py tests/test_aggregate_institutional.py tests/test_compare_generators.py tests/test_run_baseline_eval.py tests/test_run_multi_turn_eval.py tests/test_retrieval_grid_search_smoke.py tests/test_run_stress_eval.py -v` | 全パス |
| 全テスト (CI 受入基準) | `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | 全パス (test_eval_determinism は MLX 環境のみ) |
| 整合性 integration test | `python -m pytest evals/tests/test_eval_determinism.py -v` (self-hosted MLX のみ) | 2-run 完全一致 |
| リント | `ruff check .` | 警告 0 件 |
| フォーマット | `ruff format --check .` | 差分なし |
| 既存 baseline 疎通 | `python -m baseline_reporag.cli --config configs/baseline.yaml --repo-id fastapi_fastapi --question "test"` | 応答あり (interactive seed=None デフォルト確認) |

---

## 内部依存関係グラフ

```
Step 1 (helper + test)
    ↓
Step 2 (Generator/Pipeline/Photon API + tests)
    ↓
Step 3 (4 eval scripts + repo-id fix + tests)
    ↓
Step 4 (determinism integration test)
    ↓
[必須前提] Issue #156 マージ完了確認
    ↓
Step 5 (--runs + predictions schema)
    ↓
Step 6 (aggregator per-run + record_type)
    ↓
Step 7 (compare_generators + test)
    ↓
Step 8 (temperature ablation, manual ~3h)
    ↓
Step 9 (10-run noise floor, manual ~10h)
    ↓
Step 10 (4 docs 更新)
    ↓
Step 11 (PR 作成 + Issue body 同期)
```

---

## Definition of Done

- [ ] Step 1-7 (実装) すべて完了、各 Step ごとに pytest pass
- [ ] Step 8 ablation 実施、結果を `reports/institutional_temperature_ablation_v143.md` に記録
- [ ] Step 9 noise floor 計測、結果を `reports/institutional_eval_noise_floor.md` に記録
- [ ] Step 10 4 docs 更新済
- [ ] `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/` 全パス
- [ ] `ruff check .` 警告 0 件
- [ ] `ruff format --check .` 差分なし
- [ ] CI weekly_eval.yml が `--runs 1` のままで pass (ADR-7)
- [ ] Issue #143 受入条件 Task 1-5 すべて check
- [ ] Issue #143 body の Task 3 受入条件 (2 fields) 同期済 (DR2-002)
- [ ] PR レビュー承認

---

## リスク管理

| リスク | 緩和策 (実装段階) |
|------|--------|
| seed 固定で 2-run が完全一致しない | Task 4.1 で先に試し、`reports/institutional_eval_noise_floor.md` の std で実測。完全一致しない場合 token 編集距離 soft assert に切替 |
| Issue #156 と REQUIRED_FIELDS 衝突 | Step 5-6 着手前に `gh issue view 156 --json state` で CLOSED 確認、aggregator の REQUIRED_FIELDS を rebase |
| 17+ MagicMock test の TypeError | デフォルト `seed=None` 維持。`mock_gen.generate.return_value = ...` は引数受け流しで動く |
| `--runs` 誤指定 DoS | argparse `type=int_in_range(1, 20)` validator |
| YAML `run.seed: true` silent | `type(seed) is int` 厳密判定 + bool TypeError |
| Task 9 (~10h) 中断 | predictions JSONL は run ごと逐次 append (resume 可能) |
| temperature=0 で degenerate | Task 8.1 で 1 prompt 比較し採否判定 |

---

## 推定工数

| Phase | 内容 | 工数 |
|-------|------|------|
| Phase 1 (Step 1-7) | 実装 + unit/integration test | ~12-16h (人手) |
| Phase 2 Step 8 | temperature ablation manual run | ~2-3h |
| Phase 2 Step 9 | 10-run noise floor manual run | ~10-12h |
| Phase 2 Step 10 | 文書更新 | ~1h |
| Phase 3 Step 11 | PR 作成 + multi-stage-review 反映 | ~1-2h |
| **合計** | | **~26-34h** |

---

## 次のアクション

- [x] 作業計画書承認
- [ ] **/pm-auto-dev 143** で TDD 自動開発開始 ← 次のフェーズ (Step 1-7 実装)
- [ ] Step 8-9 (manual run) は別途実機実行
- [ ] /create-pr で PR 自動作成
