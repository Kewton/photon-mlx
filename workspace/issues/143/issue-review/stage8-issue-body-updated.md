## 背景

#137 A/B 実験 (5-variant retrieval test、PR #142 マージ済 / Issue #137 CLOSED) で、V0 baseline の NC rate が以下のように乖離した:

| 計測 | 計測時期 | V0 NC rate | コミット |
|---|---|---|---|
| prior measurement | 2026-04 (post-#125 + #126) | **11.21%** | `reports/institutional_baseline_static.md` |
| #137 同セッション再計測 | 2026-04-26 | **12.93%** | `reports/institutional_retrieval_ab.md` |
| 差分 | — | **+1.72pt** | — |

#137 worker は「同セッション/同コミットで全 variant を再実行した V0=12.93% を比較基準」として採用判定を成立させた (相対比較の論理は妥当)。しかし **+1.72pt の不確かさ** は Issue #137 受入基準「-2pt 以上改善」の thresh の 86% に相当する大きさで、看過できない。

## 原因

Qwen2.5-Coder-14B-Instruct-4bit の生成 nondeterminism:

- 設定: `temperature=0.2`, `top_p=0.9`, `do_sample=false`。ただし `do_sample` フラグは生成経路 (`baseline_reporag/generation/generator.py`) に伝達されておらず、実装上は `make_sampler(temp=0.2, top_p=0.9)` 経由のサンプリング経路となっている (mlx_lm.sample_utils.make_sampler は `temp=0.0` のときのみ argmax greedy)。
- **現状観測される事実**: seed を固定しない場合、同一 prompt × 同一 config で 2-run の出力 (citation 形式・引用 ID・NC 判定) が異なる。実機エビデンスは `reports/institutional_retrieval_ab.md` の V0 baseline +1.72pt drift。
- **推定原因 (mlx-lm 内部に依存し本リポジトリのコードでは確定不能)**: 4-bit 量子化 kernel の matmul order、GPU/MPS reduction 順序、prompt の token-level cache 等、mlx-lm の内部サンプリング/KV cache 実装に起因する nondeterminism。コード追跡では確定できないため、本 Issue では「seed を毎回固定して再現性を確保する」ことに焦点を絞る。
- これらの累積により異なる run で異なる token が選ばれ、citation grader (post-processor) の citation 不在判定が変わり、**NC rate が ±1-2pt 揺れる**。

なお `mlx_lm.generate()` は `seed` 名前付き引数を受け付けない (`generate(model, tokenizer, prompt, verbose=False, **kwargs)`、`stream_generate(... max_tokens=256, draft_model=None, **kwargs)` どちらにも seed パラメータ無し) ため、**generate 直前に `mx.random.seed(seed)` を呼ぶ実装方式が唯一の手段**である。

## 影響

- A/B 実験の判定 threshold (-2pt) が、**baseline 自身のノイズ (~1.7pt) と同じオーダー**
- このため将来の retrieval 改善 (1-2pt 改善) は noise 内で見えない
- 「同 variant の 2-run 平均」を取らないと、改善判定が不安定
- PR #142 (V4) 採用後の再計測は、Task 1-4 完了後に別途 1 回実施推奨 (本 Issue スコープ外)

**現在のメトリクス再計測リスク**: CLAUDE.md「現在のメトリクス」(Gate 2 v6) で採用済の PHOTON `step_003000` の Turn 5-6 NC 0.00% / follow-up p50 12,092ms は未固定 seed 計測。Task 1 適用後、PHOTON 採用判定 (#135 Phase 8) の前提が崩れていないか seed=42 で再計測 (multi-turn + static、各 1 run)、結果を `reports/institutional_photon_mt_eval_v2_3k.md` に追記する。Turn 5-6 NC ≥ 6% (MVP 基準超え) を確認した場合は Issue #135 採用判定の再評価が別 Issue として必要。

## ゴール

institutional eval (および類似の MT/Static eval) を **再現性ある計測** にするための施策を導入し、judgment threshold を ノイズより大きく確保する。

## 変更内容 (案)

### Task 1: 厳密な決定性化 (best effort)

`mlx_lm.generate()` には seed 引数が無いため、(a) process 起動時のグローバル seed と (b) `pipeline.query()` (= generate) 呼び出しごとの `mx.random.seed()` 再注入の **2 段構え** で実装する。

**(a) process 起動時のグローバル seed (script entry point で 1 回だけ):**

```python
# scripts/run_baseline_eval.py / scripts/run_multi_turn_eval.py
import os, random
import numpy as np

seed = int(getattr(cfg.run, "seed", 42))
deterministic = bool(getattr(cfg.run, "deterministic", True))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
if deterministic:
    random.seed(seed)
    np.random.seed(seed)
```

**config 型検証 (S7-001)**: `cfg.run.seed` / `cfg.run.deterministic` は eval seed の source of truth とするが、`bool(getattr(...))` のような truthy cast は使わない。`resolve_eval_seed(cfg) -> int | None` helper を scripts 共通に置き、`run` ブロック欠落時は default seed=42 / deterministic=True、`run.deterministic` は bool 以外を TypeError、`run.seed` は int かつ `0 <= seed < 2**32` に限定する。これにより YAML の `deterministic: "false"` が truthy 扱いで固定 seed 有効化される silent bug を防ぐ。

**注意 (S5-003)**: `PYTHONHASHSEED` は Python interpreter 起動時に決まるため、script 内で `os.environ['PYTHONHASHSEED'] = ...` を設定しても現在の process の hash seed は固定されない。hash seed まで固定する運用が必要な場合は `PYTHONHASHSEED=42 python scripts/run_baseline_eval.py ...` のように起動前 env として指定し、docs/troubleshooting.md に記載する。

**(b) query (generate) ごとの MLX RNG 再注入:**

実装は既存 `Generator.generate(self, messages: list[dict], max_new_tokens: int | None = None)` シグネチャに **keyword-only で seed 引数を追加** する形を取る (第一引数名 `messages` を維持し、4 箇所の既存呼び出し点と後方互換):

```python
# baseline_reporag/generation/generator.py
def generate(
    self,
    messages: list[dict],
    max_new_tokens: int | None = None,
    *,
    seed: int | None = None,
) -> str:
    self._load()
    if seed is not None:
        import mlx.core as mx
        mx.random.seed(seed)
    prompt = self._tokenizer.apply_chat_template(messages, ...)
    return mlx_lm.generate(self._model, self._tokenizer, prompt=prompt, max_tokens=..., sampler=self._sampler)
```

**統一方針 (S1-005 / S5-001 / S5-002)**: 既存 `baseline_reporag/eval/institutional/llm_client.py:QwenMLXAdapter.generate()` (Issue #135 Day 3) の seed 注入パターン (`mx.random.seed(seed)` → `make_sampler(temp)` → `mlx_lm.generate(...)`) に揃え、`baseline_reporag/generation/generator.py:Generator` 側にも `seed: int | None = None` 引数を追加する。MLX RNG state は generate ごとに進むため起動時 1 回の seed 設定だけでは N 番目の query の再現性が崩れる点に注意。

**eval/interactive の seed 伝播 API (S5-001)**:
- `RepoRAGPipeline.query(..., seed: int | None = None)` と `PhotonRAGPipeline.query(..., seed: int | None = None)` を追加する。
- eval scripts は `seed = cfg.run.seed if cfg.run.deterministic else None` を `pipeline.query(..., seed=seed)` に明示的に渡す。`cfg.run.seed` / `cfg.run.deterministic` は既存 config の source of truth とし、hard-coded `SEED = 42` と二重管理しない。
- CLI / FastAPI / demo など interactive 経路は query のデフォルト `seed=None` のままとし、現状の自然なゆらぎを維持する。
- PHOTON 経路で Qwen fallback を呼ぶ場合も同じ seed を `bl.generator.generate(..., seed=seed)` に伝播する。
- `Generator(seed=...)` という constructor 引数は現コードに存在しないため、採用する場合は `Generator.__init__` の API 追加を明記する。採用しない場合は `pipeline.query(..., seed=...)` に統一する。

**既存テスト後方互換 (S3-003)**:
- `pipeline.py` / `photon_pipeline.py` の `self.generator.generate(...)` 呼び出しは、`query(seed=None)` のとき追加の seed 引数を渡さないことで既存 MagicMock テスト (`test_pipeline_integration.py`, `test_photon_pipeline.py` の合計 17+ 件) との後方互換を維持する。`query(seed=42)` の eval 経路だけ明示的に `seed` を渡す。
- `evals/tests/test_eval_determinism.py` は実 `Generator` を使う統合テストとし、`@pytest.mark.skipif(not _HAS_MLX)` で MLX 未インストール環境では skip する (CI workflow の self-hosted MLX 環境でのみ走る)。

**CLI / server (interactive) 経路の扱い (S3-004)**: eval scripts (`run_baseline_eval.py`, `run_multi_turn_eval.py`) のみ明示的に `pipeline.query(..., seed=cfg.run.seed)` で固定する。これにより interactive CLI / FastAPI 経路は現状の自然なゆらぎを維持。`docs/deployment.md` / `docs/troubleshooting.md` で挙動を案内 (Task 5 文書更新)。

**`--repo-id` override の silent bug 修正 (S5-004)**: `scripts/run_baseline_eval.py` は `scripts/run_multi_turn_eval.py` と同様に、`repo_id = args.repo_id or cfg.repo.repo_id` を決めた直後、`build_pipeline(cfg)` より前に `cfg.repo.repo_id = repo_id` を反映する。これにより index load と query filter の repo_id 不一致を防ぐ。

**compare_generators への波及 (S7-004)**: `scripts/compare_generators.py` は Qwen vs PHOTON の評価系 script で `build_pipeline(cfg)` → `pipeline.query(...)` を使うため、seed 固定対象に含めるか明示的に対象外とする。含める場合は `resolve_eval_seed(cfg)` を使って `pipeline.query(..., seed=seed)` を渡し、`tests/test_compare_generators.py` を更新する。

### Task 2: temperature=0 への切替 (greedy 厳密化)

```yaml
# configs/institutional_docs.yaml の generation: ブロック (L191-196) を対象とする
generation:
  temperature: 0.0  # was 0.2
  top_p: 1.0        # neutralize since greedy
  do_sample: false
```

**スコープ**: `configs/institutional_docs.yaml` の `generation:` ブロック (L191-196) のみを対象とする。`inference:` ブロック (L256-269) は PHOTON 専用経路 (`photon_generation_enabled=true` 時のみ参照) のため Issue #143 のスコープ外。

trade-off: temperature=0 は repetitive/degenerate 出力を起こしやすい。本 corpus (institutional 制度文書) では事実回答が中心で OK の可能性大。Task 1 (seed 固定) 完了後に institutional eval で 1 回 ablation し、NC rate / 出力品質を比較した上で採用判定を行う。

### Task 3: 2-run 平均化を eval pipeline に組込

```bash
python scripts/run_baseline_eval.py --config X --runs 2  # 新引数
```

報告書テーブルに `mean ± std` 列を追加。

**出力 schema / 集計仕様 (S5-005)**:
- `--runs N` は predictions JSONL に `run_index`, `run_seed`, `run_id` を必須出力する。単一 JSONL に append する場合も、run 識別子で group 可能にする。
- 固定 seed 決定性検証では全 run で `run_seed=cfg.run.seed` を使う。seed sensitivity を測る場合は `cfg.run.seed..cfg.run.seed+N-1` など別 mode として明示する。
- aggregator は `run_index` ごとに `eval_id` set の完全一致を検証してから per-run NC rate / latency を計算し、`mean`, `std`, `min`, `max`, `n_runs`, `seeds` を report に出す。全 records を単純に pooled 集計して `mean ± std` の代替にしない。

**multi-turn 集計方針 (S5-006)**: `scripts/aggregate_institutional_baseline.py` は現状 static predictions (`eval_id`, `category`) 前提。`run_multi_turn_eval.py --runs` は、(a) `.summary.json` に `run_index/run_seed` と per-run NC/follow-up latency を追加する、または (b) aggregator を `record_type=static|multi_turn` 対応に拡張し MT 用 REQUIRED_FIELDS を別定義する、のどちらかを実装前に選ぶ。

**CI 採用方針 (S7-002)**: `.github/workflows/weekly_eval.yml` は、(a) 当面 `--runs 1` の scalar smoke として維持し、Task 4 noise floor は手動実行に限定する、または (b) `--runs 2` を採用して timeout を引き上げ、`scripts/ci_eval_check.py` が per-run `mean/std` と `mean - 2*std` threshold を読む、のどちらかを実装前に決める。現状の `ci_eval_check.py` は JSONL 全体を pooled 集計するだけなので、multi-run std の保護にはならない。

**Streamlit eval runner の扱い (S7-003)**: `app/components/eval_panel.py` / `app/photon_app.py` の async eval runner は `--output <job>.json` の scalar summary (`done_q`, `total_q`, `p50_latency_ms`, `nc_rate`) を読む。UI は単一 run のままなら `--runs` は CLI/weekly/manual 専用と文書化する。UI から multi-run を扱うなら `build_eval_job_cmd(runs=...)`、`EvalJob` の mean/std fields、`_sync_eval_job()`、`tests/test_photon_app_components.py` / `tests/test_photon_app_eval_jobs.py` を更新する。

### Task 4: ノイズ量定量化レポート

10-run × V0 baseline を実行 (約 10h 計算)、NC rate の std を確定。
将来の judgment threshold は `mean - 2*std` を下回る改善のみ採用。

- **seed 戦略**: 固定 `cfg.run.seed` (default 42) で 10 回反復 (Task 1 完了後の決定性が真であれば全 run 一致 → std=0 が期待値)。完全決定性が達成不能の場合は `cfg.run.seed..cfg.run.seed+9` で各 1 回として `mean ± std` を計測。
- **期待値**: std ≤ 0.5pt (合格)、std ≤ 1.0pt (現状改善ライン)。
- **出力 schema**: `## Summary` セクションに `mean, std, min, max, n_runs, seeds, computed_at_commit` を含める。
- **実行主体**: Issue #143 worker が手動実行 (~10h)。

## 受入条件

- [ ] Task 1: `Generator.generate(seed=42)` 引数追加 + `RepoRAGPipeline.query(seed=...)` / `PhotonRAGPipeline.query(seed=...)` 追加 + eval scripts から `cfg.run.seed` を伝播。`evals/tests/test_eval_determinism.py` 新規作成。同一 prompt × 同一 seed の 2-run が `cited_chunk_ids` および `no_citation` で完全一致することを assert (1 prompt 検証で CI 速度維持)
- [ ] Task 1: `resolve_eval_seed(cfg)` helper を追加し、`run` 欠落時 default、`run.deterministic` bool validation、`run.seed` int/range validation を unit test で固定する
- [ ] Task 1: `RepoRAGPipeline.query(seed=42)` が `generator.generate(..., seed=42)` を呼ぶこと、`query(seed=None)` が既存引数 shape を維持すること、`PhotonRAGPipeline.query(seed=42)` が Qwen-only path / PHOTON fallback path の両方へ seed を伝播することを unit test で固定する
- [ ] Task 1: `scripts/run_baseline_eval.py --repo-id` は `build_pipeline(cfg)` 前に `cfg.repo.repo_id` へ反映し、`run_multi_turn_eval.py` と同じ index load / query filter 挙動にする
- [ ] Task 2: temperature=0 採用または「temperature=0.2 のままで decision に影響しない理由」を文書化
- [ ] Task 3: --runs N 引数追加、predictions に `run_index/run_seed/run_id` を出力、aggregator が pooled ではなく per-run `mean/std/min/max/n_runs/seeds` 集計対応
- [ ] Task 3: weekly CI は `--runs 1` 維持か `--runs 2` 採用かを明記し、採用する場合は `weekly_eval.yml` timeout と `ci_eval_check.py` の per-run mean/std 対応を更新
- [ ] Task 3: Streamlit eval runner は single-run のまま維持するか multi-run UI を追加するかを明記し、必要なら `app/components/eval_panel.py`, `app/photon_app.py`, 関連 tests を更新
- [ ] Task 3: `scripts/compare_generators.py` は seed 固定対象に含めるか対象外にするかを明記し、含める場合は `cfg.run.seed` を伝播する
- [ ] Task 4: 10-run noise floor 計測 + reports/institutional_eval_noise_floor.md 出力
- [ ] 既存 eval scripts test 全パス
- [ ] Task 5 (文書更新):
  - CLAUDE.md「現在のメトリクス」を Task 4 完了後の seed=42 固定 mean ± std で更新
  - docs/deployment.md に「seed 固定の有無」セクションを追加 (eval は seed=42 固定、interactive は seed=None)
  - docs/troubleshooting.md に「回答が seed 固定後も揺れる場合」FAQ を追加 (mlx-lm 内部の nondeterminism 由来、本リポジトリ範囲外)

## 影響ファイル

- scripts/run_baseline_eval.py (seed 固定 + --runs)
- scripts/run_multi_turn_eval.py (seed 固定 + --runs、Issue #143 範囲)
- scripts/aggregate_institutional_baseline.py (multi-run aggregation)
- baseline_reporag/pipeline.py (`query(seed=...)` 追加)
- baseline_reporag/photon_pipeline.py (`query(seed=...)` 追加 + PHOTON fallback への seed 伝播)
- baseline_reporag/pipeline_factory.py (`cfg.run.seed` / `cfg.run.deterministic` を source of truth とする場合の wiring 確認)
- configs/baseline.yaml (`run.seed` / `run.deterministic` は既存 key。dead key にしない)
- configs/institutional_docs.yaml (temperature 検討)
- scripts/compare_generators.py (評価系 script として seed 固定対象に含めるか判断)
- app/components/eval_panel.py (`--runs` を UI から渡す場合)
- app/photon_app.py (multi-run summary を UI に表示する場合)
- tests/test_compare_generators.py
- tests/test_photon_app_components.py / tests/test_photon_app_eval_jobs.py (`--runs` UI 対応時)
- baseline_reporag/tests/test_pipeline_integration.py / baseline_reporag/tests/test_photon_pipeline.py (`query(seed=...)` 伝播・後方互換)
- tests/test_ci_eval_check.py (`ci_eval_check.py` を per-run 対応する場合)
- reports/institutional_eval_noise_floor.md (新規)
- baseline_reporag/generation/generator.py (sampler temp 制御 + generate 直前 seed 固定)
- demo/run_demo.py (Generator() コンストラクタ呼び出し点 — seed 既定値で動作確認のみ)
- .github/workflows/weekly_eval.yml (--runs N 採用時 timeout-minutes 引き上げ要否を確認)
- scripts/ci_eval_check.py (Task 1-4 完了後、`STATIC_NC_MAX=0.30 / MT_NC_MAX=0.35` が seed 固定後の実測 NC と整合しているか確認、必要なら mean - 2*std で Task 4 noise floor を反映)
- evals/tests/test_eval_determinism.py は実 MLX を要求するため `@pytest.mark.skipif(not _HAS_MLX)` で MLX 未インストール環境では skip。CI 経路は self-hosted runner (weekly_eval.yml) のみで実走

## 内部依存関係

- Task 1 (seed 固定) — 先行必須
- Task 2 (temperature 検証) — Task 1 完了後 institutional eval で 1 回 ablation
- Task 3 (--runs N 集計) — Task 1 完了後 (Task 2 と並列可)
- Task 4 (10-run noise floor) — Task 1-3 完了後 (Task 1 が決定性を達成しているかの最終検証)

## 並列性

#137 / #135 / #115 / #139 / #140 と独立、いつでも並列実施可。

## 関連

- 元: #137 (5-variant A/B、本問題が顕在化)
- 関連: #138 (CLOSED, tokenizer mismatch fix 反映済 — 本 Issue では追加対応不要)
- 関連: #135 (本格再学習 eval、本問題の再発を防ぐため Phase 6-8 前に解消推奨)
- 関連: #156 (run_multi_turn_eval.py の is_refusal 出力欠落 bug、OPEN) — Task 3 (aggregator multi-run 対応) は #156 修正後または同時に実施。両方とも `aggregate_institutional_baseline.py` を変更するため、merge order は **#156 → #143** を推奨。並行マージ時は `REQUIRED_FIELDS` (現状 10 件、#156 の `is_refusal`、#143 の `run_index/run_seed/run_id`) と --runs 平均化ロジックが互いに rebase conflict を起こす。
