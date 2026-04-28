# Issue #143 設計方針書 — institutional eval reproducibility

作成日: 2026-04-28
ブランチ: `feature/issue-143-eval-reproducibility`
対象 Issue: [#143](https://github.com/Kewton/photon-mlx/issues/143)
レビュー反映: マルチステージ Issue レビュー Stage 1-8 (Claude opus + Codex 計 32 findings) 反映済 (`workspace/issues/143/issue-review/summary-report.md`)

---

## 1. 目的

institutional eval (baseline + multi-turn) を **再現性ある計測** にする。具体的には、Qwen2.5-Coder-14B-Instruct-4bit の生成 nondeterminism による NC rate の ±1.7pt 揺れを seed 固定で抑え、A/B 実験の judgment threshold (-2pt) をノイズより大きく確保する。同時に、決定性を活かした multi-run 集計と noise floor 計測を導入し、将来の retrieval 改善 (1-2pt) を「検出可能」にする。

**スコープ外** (今 Issue では扱わない):
- mlx-lm 内部 (4-bit kernel / KV cache) の決定性化 — 外部ライブラリ範囲。本 Issue は「seed 固定で best-effort 再現性を確保」までを担う。
- PR #142 (V4) の再計測 — Task 1-4 完了後に別途 1 回実施。
- Issue #135 採用判定 (`step_003000` Turn 5-6 NC 0.00%) の再評価 — 再計測で NC ≥ 6% を確認した場合のみ別 Issue で対応。

---

## 2. アーキテクチャ位置

PHOTON-RepoRAG の eval 経路と production 経路に対する変更スコープ。

```
                    [eval scripts (seed=cfg.run.seed)]
                    │
  ┌────────────────┼────────────────────────┐
  │                │                        │
scripts/run_baseline_eval.py   scripts/run_multi_turn_eval.py
scripts/compare_generators.py  scripts/retrieval_grid_search.py
scripts/run_stress_eval.py
                    │
                    ▼
              build_pipeline(cfg)
                    │
                    ▼
   ┌────────────────────────────────────────┐
   │  RepoRAGPipeline / PhotonRAGPipeline   │  <-- query(..., seed=None) 追加 (keyword-only)
   │      .query(question, ..., seed)       │
   └────────────────────┬───────────────────┘
                        │
                        ▼
        baseline_reporag/generation/generator.py
                Generator.generate(messages,
                                   max_new_tokens=None,
                                   *, seed=None)   <-- seed 追加 (keyword-only)
                        │
              if seed is not None: mx.random.seed(seed)
                        │
                        ▼
                 mlx_lm.generate(...)             <-- mlx-lm は seed を受けない

   [interactive 経路 (seed=None でデフォルト)]
   ─ baseline_reporag/cli.py
   ─ baseline_reporag/server.py
   ─ app/photon_app.py
   ─ demo/run_demo.py
```

**設計の要**:
- `seed=None` をデフォルトとし、既存 MagicMock テスト (17+ 件) と interactive 経路を**触らない**。
- eval scripts のみ `seed = resolve_eval_seed(cfg)` を `pipeline.query(seed=seed)` に明示的に渡す。
- `cfg.run.seed` / `cfg.run.deterministic` が source of truth (現状 yaml に存在するが未使用 — silent dead key 化していたものを蘇らせる)。

---

## 3. レイヤー別変更

| レイヤー | モジュール | 変更内容 | 影響度 |
|---------|-----------|---------|------|
| **Config** | `baseline_reporag/config.py` | 変更なし (TypedConfig は permissive に維持し、`cfg.run.seed`/`run.deterministic` 型検証は helper 側へ集約) | - |
| **Helpers (新規)** | `baseline_reporag/eval/run_config.py` (新規) | `resolve_eval_seed(cfg) -> int \| None` helper。config 型/範囲 validation の source of truth | - |
| **Generation** | `baseline_reporag/generation/generator.py` | `Generator.generate(*, seed)` keyword-only 引数追加。generate 直前で `mx.random.seed(seed)` | 中 |
| **Pipeline** | `baseline_reporag/pipeline.py` | `RepoRAGPipeline.query(question, session_id='', repo_id='', *, seed=None)` (既存 positional 互換 + seed のみ keyword-only) 追加、`generator.generate(messages, seed=seed)` 伝播 | 中 |
| **Photon Pipeline** | `baseline_reporag/photon_pipeline.py` | `PhotonRAGPipeline.query(question, session_id='', repo_id='', *, seed=None)` 追加。Qwen-only path / Qwen fallback (3 箇所: `photon_pipeline.py` L1030, L1043, L1394) で seed 伝播 | 中 |
| **EvalJob (UI)** | `app/photon_app.py:EvalJob` | 変更なし (ADR-8 で UI single-run 維持) | - |
| **Eval scripts** | `scripts/run_baseline_eval.py` | (a) `--runs N` 引数 (`1 <= N <= 20`)、(b) `cfg.run.seed` 伝播、(c) **`--repo-id` silent bug 修正** (`build_pipeline` 前に `cfg.repo.repo_id = repo_id`)、(d) predictions JSONL に `run_index/run_seed` 出力 (`run_id` は ADR-5 により非保持) | 高 |
| **Eval scripts** | `scripts/run_multi_turn_eval.py` | `--runs N` 引数 (`1 <= N <= 20`)、`cfg.run.seed` 伝播、`.summary.json` に per-run summary 追加 | 高 |
| **Compare** | `scripts/compare_generators.py` | `cfg.run.seed` 伝播 (eval 系 script のためスコープ内) | 低 |
| **Eval scripts** | `scripts/retrieval_grid_search.py` | `run_eval_inproc()` から `pipeline.query(..., seed=seed)` へ伝播。grid search は `--runs` 対象外だが、生成依存 metric の再現性を固定 | 中 |
| **Eval scripts** | `scripts/run_stress_eval.py` | `run_real()` の sequential multi-turn stress run で `cfg.run.seed` を `pipeline.query(..., seed=seed)` へ伝播。`--runs` 対象外 | 中 |
| **Aggregator** | `scripts/aggregate_institutional_baseline.py` | `record_type=static` (既存) + `record_type=multi_turn` (新規) per-run 集計、`mean/std/min/max/n_runs/seeds` 出力 | 中 |
| **CI** | `scripts/ci_eval_check.py` | weekly が `--runs 1` 維持の場合は変更なし、`--runs 2` 採用時は per-run mean/std 読込 (今 Issue では `--runs 1` 維持決定 = 変更なし) | 低 |
| **Workflow** | `.github/workflows/weekly_eval.yml` | timeout 確認のみ (今 Issue は `--runs 1` 維持決定) | 低 |
| **Tests (新規)** | `evals/tests/test_eval_determinism.py` | 同一 prompt × 同一 seed の 2-run 完全一致 assert (`@pytest.mark.skipif(not _HAS_MLX)`) | - |
| **Tests (修正)** | `baseline_reporag/tests/test_pipeline_integration.py` | `query(seed=...)` 伝播 unit test 追加 (`generator.generate.assert_called_with(..., seed=42)`) | 中 |
| **Tests (修正)** | `baseline_reporag/tests/test_photon_pipeline.py` | Qwen-only path / fallback path の seed 伝播 unit test | 中 |
| **Tests (修正)** | `tests/test_compare_generators.py` | `cfg.run.seed` 伝播確認 | 低 |
| **Tests (修正)** | `tests/test_retrieval_grid_search_smoke.py` | `run_eval_inproc()` が `seed` を query kwargs に渡すことを確認 | 低 |
| **Tests (新規)** | `tests/test_run_stress_eval.py` | `run_real()` が `cfg.run.seed` を query kwargs に渡すことを MagicMock で確認 | 低 |
| **Tests (修正)** | `tests/test_aggregate_institutional.py` | per-run 集計 (`mean/std/...`) 追加検証 (既存ファイル `tests/test_aggregate_institutional.py` への追記。新規ファイル作成しない) | 中 |
| **Tests (変更なし)** | `tests/test_ci_eval_check.py` | 変更なし (ADR-7 で `--runs 1` 維持のため) | - |
| **Demo (変更なし)** | `demo/run_demo.py` | 変更なし (`Generator()` の seed 既定値で動作確認のみ、interactive 経路は seed=None) | - |
| **Pipeline factory (変更なし)** | `baseline_reporag/pipeline_factory.py` | 変更なし (`resolve_eval_seed` は eval scripts 側で呼ぶため pipeline_factory 内には wiring 不要) | - |
| **Docs** | `CLAUDE.md` | 「現在のメトリクス」を Task 4 完了後の seed=42 固定 mean ± std で更新 | - |
| **Docs** | `docs/deployment.md` | 「seed 固定の有無」セクション追加 | - |
| **Docs** | `docs/troubleshooting.md` | 「回答が seed 固定後も揺れる場合」FAQ 追加 | - |
| **Docs** | `docs/code_review_checklist.md` | seed keyword-only ルール (DR1-007) + seed 注入時の両者同時更新 (ADR-11) を追記 | - |
| **Reports** | `reports/institutional_eval_noise_floor.md` | 新規。10-run × seed=42 の noise floor 計測結果 | - |

### §3.1 最終 API シグネチャ (DR1-007 反映)

設計方針書全体で `*` 区切りの keyword-only seed を採用する。最終 signature は以下の通り。

```python
# baseline_reporag/pipeline.py
class RepoRAGPipeline:
    def query(
        self,
        question: str,
        session_id: str = "",
        repo_id: str = "",
        *,
        seed: int | None = None,
    ) -> QueryResult: ...

# baseline_reporag/photon_pipeline.py
class PhotonRAGPipeline:
    def query(
        self,
        question: str,
        session_id: str = "",
        repo_id: str = "",
        *,
        seed: int | None = None,
    ) -> QueryResult: ...

# baseline_reporag/generation/generator.py
class Generator:
    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        *,
        seed: int | None = None,
    ) -> str: ...
```

`*` で区切ることで既存 positional 呼出 (`pipeline.query('q', 'sid', 'rid')`、`generator.generate(messages, max_new_tokens=128)`) は破壊しない。`seed` は keyword-only であるため、新規呼出は必ず `seed=42` のように明示する。Code Review Checklist (`docs/code_review_checklist.md`) に「query/generate の seed は keyword-only でなければならない (positional 4 個目以降を増やすと既存呼出を破壊するため)」を追加する (Task 5 文書更新タスクに含む)。

### §3.2 `cfg.run.*` の命名規則 (DR1-009 反映)

既存 keys (`run.name_prefix: str`, `run.seed: int`, `run.deterministic: bool`) を踏襲する。`run` ブロックは「eval ジョブ識別 (`name_prefix`) + 決定性 policy (`seed` / `deterministic`)」の責務を持つ複合 policy ハブである。

新 key を追加する場合は以下のいずれかに従う:
- `run.<noun>` (eval ジョブ属性: `name_prefix`, `seed` 等)
- `run.<adjective>` (boolean policy: `deterministic` 等)

これは既存 `is_symbol_graph_enabled` (`indexing.symbol_graph.enabled`) の `<scope>.<feature>.enabled` 接尾辞パターンとは差がある。差は意図的: `run` ブロックは複合 policy ハブとして 1 ブロック内で複数 type (str/int/bool) を許容する。一方 `<scope>.<feature>.enabled` は単一 boolean feature flag 専用。

---

## 4. 設計判断 (Architecture Decision Records)

### ADR-1: seed 引数の API シグネチャ

**選択肢**:
- A: `Generator.__init__(seed=42)` でコンストラクタに持たせる
- B: `Generator.generate(*, seed: int | None = None)` で keyword-only に持たせる
- C: `Pipeline.query(*, seed)` から透過的に渡す (Generator は変更しない)

**決定**: **B + C 両方** (`Generator.generate(*, seed)` と `Pipeline.query(*, seed)` の両方に keyword-only 引数を追加)

**最終 signature** (DR1-007 反映):
- `RepoRAGPipeline.query(self, question: str, session_id: str = '', repo_id: str = '', *, seed: int | None = None) -> QueryResult`
- `PhotonRAGPipeline.query(self, question: str, session_id: str = '', repo_id: str = '', *, seed: int | None = None) -> QueryResult`
- `Generator.generate(self, messages: list[dict], max_new_tokens: int | None = None, *, seed: int | None = None) -> str`

**選択肢 比較表 (KISS 観点) — DR1-004 反映**:

| 軸 | C 単独 (Pipeline.query のみ) | B+C 両方 (本設計の決定) |
|------|------|------|
| 実装コスト | Pipeline 1 箇所 (約 5 行) | Pipeline + Generator 2 箇所 (約 10 行、+5 行差) |
| API 表面積 | 1 箇所 | 2 箇所 |
| Unit test 容易性 | mock 経由で `assert_called_with(seed=42)` 可 (mock を Generator に注入すれば書ける) | mock 経由で同 shape の assert 可 |
| 直接呼出の柔軟性 | Pipeline 経由必須 (test_eval_determinism.py は Pipeline 経由でも書ける) | Generator を直接 instantiate して seed 検証可 |
| 将来性 | LLMClient Protocol 統合 (#110 系) 時に Protocol へ seed 追加 → Generator も追従が必要 | Protocol へ seed が入るならば Generator はすでに対応済 |
| 後方互換 | 既存 4 件呼出は無修正 (Pipeline 側のみ) | 既存 4 件呼出は無修正 (`*` 区切りで keyword-only 追加) |

**理由**:
- 実装コストの差は 5 行で等価、test 容易性も等価。決定的差分は (a) 直接呼出柔軟性、(b) 将来性。
- Generator は LLMClient 相当の低レベル primitive。将来 LLMClient Protocol に統合される見込み (#110 系) があり、その時点で `seed` 引数が Protocol 必須になる蓋然性が高い。先行投資として B+C を採用する。
- Pipeline は eval scripts が触る高レベル API なので、scripts → Pipeline → Generator と自然に伝播するルートを用意。
- Constructor (A) にすると Generator instance 一つで複数 seed を扱えなくなり、Task 4 (seed=42..51) が冗長。
- 既存呼び出し点 (4 箇所、すべて positional) は keyword-only を `*` 区切りで追加するため**シグネチャ破壊なし**。
- ただし C 単独でも本 Issue の受入条件 (Task 1 同一 prompt × 同一 seed の 2-run 完全一致) は達成可能であり、将来 #110 系が別方向に進んだ場合は C 単独へ refactor 可能 (Pipeline 内で Generator を mock しなくなる程度の影響)。

**トレードオフ**:
- メリット: 後方互換 (既存 17+ MagicMock テスト無修正)、eval/interactive 双方の責務分離 (eval scripts のみ seed 指定)、unit test の自然な assert (`generator.generate.assert_called_with(..., seed=42)`)、将来 Protocol 統合への先行投資
- デメリット: API 表面積が 2 箇所増える (KISS 観点で +5 行のコスト)
- リスク: Pipeline → Generator の seed 伝播実装漏れ → unit test で防御 (受入条件 Task 1)

**multi-call 内 seed 戦略 (DR1-003 反映)**:

PHOTON path 内で複数回 `Generator.generate` を呼ぶ場合 (`photon_pipeline.py` L1030, L1043, L1394 の Qwen fallback 3 箇所)、各 generate の直前で `mx.random.seed(seed)` を呼ぶため、**全呼出が seed=42 起点の同一 RNG state から始まる** (= 同じ token sequence を生成する可能性がある)。これは `Generator.generate` の決定性保証 (= 同 prompt × 同 seed → 同 output) を multi-call にも素直に拡張した形であり、本 Issue では intentionally accept する。

理由:
- 受入条件 Task 1 (2-run 完全一致) を multi-call path でも壊さない最も単純な設計。
- 「N 回目の呼出で異なる token を望む (=各回で diversity を出したい)」要件は本 Issue では発生しない (eval は決定性が目的)。
- multi-call 間で異なる seed を望む場合は `seed + i` のように shift する設計を別 Issue で導入する選択肢を残す (本 Issue では不要)。

実装注意:
- `photon_pipeline.py` の Qwen fallback 3 箇所 (L1030, L1043, L1394) には `# seed propagated to Qwen-only fallback path (each call resets MLX RNG; intentional for determinism)` のコードコメントを残す。
- 各 fallback path への seed 伝播は `test_photon_pipeline.py` の Qwen-only / fallback path unit test で `mock_gen.generate.assert_called_with(..., seed=42)` を 3 箇所別個に assert する。

### ADR-2: seed 設定の管理

**選択肢**:
- A: hard-coded `SEED = 42` を eval scripts に書く
- B: `cfg.run.seed` / `cfg.run.deterministic` を source of truth (既存 YAML キー) にして `resolve_eval_seed(cfg)` helper を共通化
- C: 環境変数 `BASELINE_SEED` で外部から注入

**決定**: **B** (config 駆動 + helper)

**理由**:
- 既存 `configs/baseline.yaml` / `configs/institutional_docs.yaml` に `run.seed: 42 / run.deterministic: true` が**既に存在**するが、コード側で参照されていない silent dead key になっている (Codex S5-002 で発見)。これを蘇らせる方が DRY
- helper (`resolve_eval_seed(cfg) -> int | None`) で型検証 (S7-001) を集中化: `run` 欠落時 default `42`、`run.deterministic` は bool 以外を `TypeError`、`run.seed` は `type(seed) is int` かつ `0 <= seed < 2**32` に限定。`bool(getattr(...))` の truthy cast は使わない (`deterministic: "false"` が truthy になる silent bug 防止)
- 環境変数 (C) は eval scripts の起動時に余計な設定を強いるため棄却

**トレードオフ**:
- メリット: config 一元化、テスト容易性、validation 集中
- デメリット: 新規 helper module 追加 (`baseline_reporag/eval/run_config.py`)
- リスク: 既存 yaml の `run.seed` が int 以外で書かれている場合に init 時エラー → 既存値はすべて int=42 で確認済 (`grep -n 'run:' configs/*.yaml`)

**helper の SRP 緩和方針 (DR1-002 反映)**:

`resolve_eval_seed(cfg)` は `(1) run ブロック欠落 default 補完`、`(2) run.deterministic bool 型検証`、`(3) deterministic=False 時の seed=None 返却`、`(4) run.seed int 型検証`、`(5) seed の range (0 <= seed < 2**32) 検証` の 5 責務を担う。これは既存 `is_symbol_graph_enabled` (単一 bool key 検証) より粒度が大きいが、以下の方針で SRP を緩和する:

- **採用案**: 単一 helper `resolve_eval_seed(cfg) -> int | None` を維持し、内部で `_validate_run_block(run_dict) -> dict` private 関数として型検証 (上記 2, 4, 5) を分離する。public API は 1 個に保ちつつ、helper 内部で「validation」と「determinism gating policy」を関数分割。
- 棄却案: `validate_run_block(cfg) -> RunConfig` namedtuple を public で返し eval scripts 側で `RunConfig.seed if RunConfig.deterministic else None` を組み立てる案 (= 呼び側の boilerplate が増えるため不採用)。

これにより public API 表面積は 1 個に保ったまま、内部 SRP は 2 関数に分かれる。unit test は public helper のレベルで網羅 (default / int / deterministic=False / 不正型 TypeError / range 外 ValueError / `run` 欠落) を独立 test case で書く (§7 に追加)。

**例外設計 (DR1-008 反映)**:

`resolve_eval_seed(cfg)` (および内部 `_validate_run_block`) の例外方針は以下の通り:

| エラー種別 | 例 | 例外 |
|---------|---|------|
| 型不一致 | `run.deterministic: "false"` (str)、`run.seed: "42"` (str)、`run.seed: true` (bool)、`run.seed: .nan` (float NaN) | `TypeError` |
| 値の範囲外 | `run.seed: -1`、`run.seed: 2**33`、`run.seed: 0.5` (int でない実数) | `ValueError` |

これは既存 `is_symbol_graph_enabled` の `TypeError`-only パターンよりも詳細だが、`seed` は range 制約 (`0 <= seed < 2**32`) を持つため値由来エラー (ValueError) を区別する。Pythonic に「型不一致 = TypeError、値の不正 = ValueError」という標準を踏襲。

**セキュリティ補足 (DR4-001 反映)**:

Python では `bool` が `int` の subclass なので、`isinstance(seed, int)` だけでは `run.seed: true` が `seed=1` として通る。実装は `type(seed) is int` で厳密判定し、`bool` / `float` / `NaN` / `str` をすべて `TypeError` とする。YAML 読込は既存通り `yaml.safe_load` を使い、`eval()` / `exec()` / 文字列式評価は使わない。

eval scripts 側 (`run_baseline_eval.py` 等) は両者を catch せず **fail-fast で起動時 abort** する。これは config を直す動機を与える (silent fallback すると seed が無効化されたまま eval が走ってしまうリスクを排除)。

**scripts 間 boilerplate の取扱い (DR1-010 反映)**:

`resolve_eval_seed` を 5 scripts (`run_baseline_eval.py` / `run_multi_turn_eval.py` / `compare_generators.py` / `retrieval_grid_search.py` / `run_stress_eval.py`) で再利用するが、各 script は `cfg を load → seed = resolve_eval_seed(cfg) → pipeline.query(seed=seed)` という同じ pattern を繰り返す。この boilerplate は受容する設計判断を取る。

理由:
- 各 script の意図 (= cfg からどう seed を取るか) が呼出箇所で明示的に読みやすい (helper を隠蔽するより教育的)。
- helper を `with_eval_seed(cfg, fn)` のような higher-order 関数にすると、scripts 側は `with_eval_seed(cfg, lambda q: pipeline.query(q))` のような lambda + 高階呼出になり、scripts の素直な手続き型読みやすさを損なう (overengineering リスク)。

棄却した代替案:
- `with_eval_seed(cfg, fn: Callable[[str], QueryResult]) -> Callable[[str], QueryResult]` という higher-order helper を提案。これは scripts ごとの 2-3 行の boilerplate を削減できるが、(a) scripts ごとに pipeline.query の signature が微妙に違う場合に高階関数の generics が膨らむ、(b) seed 渡し忘れリスクは unit test で既に防御済み、の 2 理由で却下。

代替策として「`resolve_eval_seed` の使い方を 1 サンプル (3 行) で `CLAUDE.md` および `docs/code_review_checklist.md` に明記し、新規 eval script 作成時の reference 実装とする」を Task 5 文書更新に含める。

### ADR-3: interactive 経路 (CLI / server / Streamlit) の seed デフォルト

**選択肢**:
- A: デフォルト `seed=42` (eval = interactive 同じ)
- B: デフォルト `seed=None` (eval scripts のみ明示指定)
- C: 環境変数 `BASELINE_DETERMINISTIC=true` で切替

**決定**: **B** (デフォルト `seed=None`)

**理由**:
- production 用途 (FastAPI / Streamlit / CLI) で「同じ質問が毎回完全に同じ回答」になると UX 退化
- eval scripts は明示的に `seed=resolve_eval_seed(cfg)` を渡すため切替が明示的
- 環境変数 (C) は実装増分の割に効果が薄い

**トレードオフ**:
- メリット: interactive UX 維持、既存 MagicMock テスト (`test_pipeline_integration.py` 等) と完全後方互換
- デメリット: eval scripts は明示的に seed を渡す責任を持つ (= 忘れると Issue #143 の意義が消失) → unit test で `assert_called_with(seed=42)` を fix
- リスク: 開発者が新規 eval script を書いた時に seed を渡し忘れる → CLAUDE.md 品質チェック節と code review checklist に「eval は cfg.run.seed を query に渡すこと」を追加

### ADR-4: temperature=0 への切替

**選択肢**:
- A: 即時 `temperature: 0.0` に切替
- B: `temperature: 0.2` 維持 + Task 1 後 institutional ablation で判定
- C: institutional は `0.0`、baseline は `0.2` (config 別)

**決定**: **B** (Task 1 完了後 ablation で判定 → 別 PR で確定)

**理由**:
- temperature=0 は repetitive/degenerate 出力リスクあり (Issue 本文 Task 2 trade-off 参照)
- 制度文書 (institutional) は事実回答中心で OK な見込みだが、定量的根拠 (Task 1 完了後の同 prompt 比較) を持って判定したい
- Task 1 で seed 固定が機能すれば temperature=0.2 でも決定性が出るため、必ずしも temperature=0 が必須ではない

**トレードオフ**:
- メリット: データ駆動の判定、退化リスクを 1 回 ablation で確認
- デメリット: 本 Issue で完全結論が出ない (Task 2 = 検証ステップ)
- リスク: ablation を実施しないまま Task 4 へ進むと「temperature=0 の方が効果的だった」と後で判明する → 受入条件 Task 2 で「採用または検証文書化」を必須化

### ADR-5: --runs N の predictions schema

**選択肢**:
- A: 単一 JSONL に append、`run_index` で識別
- B: run ごとに別 JSONL ファイル (`predictions_run0.jsonl`, `predictions_run1.jsonl`)
- C: `run_index` を含めず pooled 集計のみ

**決定**: **A** (単一 JSONL + `run_index/run_seed` の 2 fields)

**`--runs` 入力制限 (DR4-002 反映)**:

`--runs` は `int` かつ `1 <= runs <= 20` に制限する。default は `1`。Task 4 の noise floor は 10-run を想定しているため上限 20 で十分な余裕があり、誤操作や attacker-controlled CLI/UI 経由の `--runs 1000000` による self-hosted runner / developer machine の DoS を防ぐ。上限を超える長期計測が必要な場合は別 Issue で runner capacity と timeout を再設計する。

**追加 fields の絞り込み (DR1-006 反映)**:

予測 JSONL の追加 fields を **2 種** (`run_index: int`, `run_seed: int`) に絞る。`run_id` は `f"{run_index}_{run_seed}"` で計算可能なため別 field にしない。これにより:
- schema 拡張面積を最小化 (3 fields → 2 fields)
- #156 (`is_refusal`) との merge conflict 面積が減る
- downstream (aggregator / report) が必要なら `run_id` を `f"{run_index}_{run_seed}"` で計算して使える

**理由**:
- 既存 evaluation 経路 (`scripts/aggregate_institutional_baseline.py`) を最小改造で per-run 集計に拡張可能
- aggregator は `run_index` を group key として直接使えば `run_id` 文字列を保持する必要がない
- C (pooled) は Codex S5-005 で指摘された通り `mean ± std` を出せないため棄却
- B (別ファイル) はファイル数増加でファイル管理が煩雑

**トレードオフ**:
- メリット: 既存 aggregator の改造範囲が小さい、欠損 run の検出が容易、schema 拡張面積が最小 (2 fields)
- デメリット: REQUIRED_FIELDS 互換性 (#156 と衝突予定) は残るが、3 fields → 2 fields で衝突面積を縮小
- リスク: #156 (refusal-aware) との merge order を **#156 → #143** に固定する (受入時に明示) → Issue 「関連」セクションに記載済

### ADR-6: multi-turn の集計

**選択肢**:
- A: aggregator (`scripts/aggregate_institutional_baseline.py`) を `record_type=static|multi_turn` 対応に拡張
- B: `run_multi_turn_eval.py` の `.summary.json` 自身に per-run mean/std を埋め込み、aggregator は static 専用のまま

**決定**: **A** (aggregator を static + multi_turn 両対応に拡張)

**理由**:
- DRY (per-run 集計ロジックを 1 箇所に集約)
- 既存 `scripts/aggregate_institutional_baseline.py` は CLI から `--predictions multi_turn_predictions.jsonl --record-type multi_turn` で呼べるように拡張するのが自然
- B は per-run 集計を 2 箇所 (run_multi_turn_eval + aggregator) に重複実装するため非推奨

**トレードオフ**:
- メリット: 集計ロジック一元化、テスト追加容易
- デメリット: REQUIRED_FIELDS が 2 系統 (static = `eval_id, category, ...`, multi_turn = `session_id, turn_id, ...`) になり実装増分大
- リスク: REQUIRED_FIELDS 拡張が #156 と衝突 → ADR-5 と同じ merge order (#156 → #143)

**aggregator の拡張性 (DR1-005 反映)**:

現時点で `record_type` は `static` / `multi_turn` の 2 種のみで、追加の見込みは限定的なため **if-else 分岐で実装する** (KISS 優先 / YAGNI 受容)。将来 3 種以上に増えた段階で Strategy パターン (`RecordSchema` Protocol) へ refactor する。

ただし、内部実装では DRY 維持のため共通 helper `compute_per_run_stats(records, group_field)` を切り出す:

- `static` の場合: `group_field='eval_id'` (各 record は eval_id ごとにユニーク)
- `multi_turn` の場合: `group_field=('session_id', 'turn_id')` (tuple key で group)

per-run 集計ロジック (`mean / std / min / max / per_run`) は `compute_per_run_stats` 内に集中させ、`record_type` 分岐は (a) JSONL 読込、(b) REQUIRED_FIELDS 検証、(c) `group_field` 選択の 3 箇所のみで if-else を持つ。これにより「per-run 集計の数値ロジック」は 1 箇所に保ち、「record_type 分岐」は最小限に抑える。

**後方互換 validation (DR3-002 反映)**:

`run_index` / `run_seed` は Issue #143 以降の eval runner が出力する新 fields だが、aggregator の `REQUIRED_FIELDS` には含めない。historical JSONL を集計する場合は loader 内で `run_index=0` として単一 run に正規化し、`run_seed` が欠落している場合は `seeds` から除外する。これにより:
- Step 5 完了時点で旧 aggregator が新 fields を ignore できる
- Step 6 完了後も旧 predictions JSONL を silent に壊さない
- `tests/test_aggregate_institutional.py` に「run_index/run_seed 欠落の旧 JSONL を単一 run として扱う」case を追加できる

将来 OCP refactor 時は `RecordSchema` Protocol (`required_fields: list[str]`, `group_key: Callable[[dict], Hashable]`) を定義し、`record_type` ごとに実装クラスを差替える形へ移行できる構造を残す。

### ADR-7: weekly_eval.yml の `--runs` 採用方針

**選択肢**:
- A: weekly も `--runs 2` に切替、`ci_eval_check.py` を per-run mean/std 対応
- B: weekly は `--runs 1` 維持、Task 4 noise floor は manual run

**決定**: **B** (weekly は `--runs 1` 維持 = 現状互換)

**理由**:
- weekly CI の timeout は現在 180 min。`--runs 2` で約 360 min となり self-hosted runner の負荷が増す
- Task 4 noise floor (10-run × ~1h = ~10h) は元から手動運用前提
- 将来 `--runs 2` に上げる判断は別 Issue に切り出す

**トレードオフ**:
- メリット: 現 weekly CI workflow 変更なし、`ci_eval_check.py` 変更なし、merge risk 最小
- デメリット: weekly CI は scalar NC のみで multi-run noise を保護しない
- リスク: weekly が単一 run のままだと、もし Task 1 の決定性化が一部破綻した場合 weekly では検出できない → `evals/tests/test_eval_determinism.py` (1 prompt 統合テスト) でカバー

### ADR-8: Streamlit eval runner の扱い

**選択肢**:
- A: UI は単一 run のまま、`--runs` は CLI/weekly/manual 専用
- B: UI から `--runs N` を渡せるように拡張、`EvalJob` schema に mean/std 追加

**決定**: **A** (UI は単一 run 維持)

**理由**:
- Streamlit は人間が試行錯誤する用途で multi-run 集計の必要性が薄い
- B を採ると `app/components/eval_panel.py`, `app/photon_app.py`, `tests/test_photon_app_*.py` の改造範囲が広がり Issue 範囲が肥大化
- UI で multi-run が欲しい場合は別 Issue で対応

**トレードオフ**:
- メリット: UI コード変更なし、`EvalJob` schema 維持、関連 tests 変更なし
- デメリット: UI から noise floor を見られない
- リスク: なし (manual run + report で代替可)

`EvalJob.mean / std` 等の追加は本 Issue 範囲外 (YAGNI)。将来 UI に multi-run 集計を出したい要件が発生した時に別 Issue で対応 (DR1-012 反映)。

### ADR-9: `compare_generators.py` の seed 伝播

**選択肢**:
- A: 含める (`pipeline.query(seed=...)` を渡す)
- B: 別 Issue で対応

**決定**: **A** (含める)

**理由**:
- compare_generators は Qwen vs PHOTON の評価系 script であり、Issue #143 の「再現性ある eval 計測」目的に合致
- 1 ファイル + 1 test 修正で済む
- 含めないと「side-by-side 比較だけ未固定 seed」という違和感が残る

**トレードオフ**:
- メリット: 評価系 script 全体で再現性ポリシー統一
- デメリット: スコープが微増 (`tests/test_compare_generators.py` 追加修正)
- リスク: なし

**YAGNI 観点 (DR1-011 反映)**:

`compare_generators` は本 Issue の主目的 (institutional eval reproducibility = Task 4 noise floor) と直接依存しない側方比較 script である。受入条件 Task 1-5 は `compare_generators` を含めなくても達成可能。

それでも本 Issue に含める理由は以下の 2 点:
1. **修正コストが小さい**: 1 ファイル + 1 test の小さな修正 (推定 +20 LOC)。別 Issue 化のオーバーヘッド (issue 起票 / review / PR / merge) が修正コストを上回る。
2. **認知負荷の回避**: 含めないと「eval 系 3 script のうち 2 つだけ seed 固定、1 つだけ未固定」という非対称性が生まれ、新規開発者が「なぜ compare_generators だけ違うのか」を都度確認する負荷が残る。

YAGNI トレードオフ (= 「現時点で必要か？」が原則) を認識した上で、認知負荷削減の観点から本 Issue に含めることを明記する。

### ADR-10: `--repo-id` silent bug の同時修正

**選択肢**:
- A: Issue #143 で同時修正 (`scripts/run_baseline_eval.py` を `run_multi_turn_eval.py` と同じパターンに揃える)
- B: 別 Issue を切る

**決定**: **A** (同時修正)

**理由**:
- Codex S5-004 で発見した silent bug。`scripts/run_baseline_eval.py` の `--repo-id` override が `build_pipeline(cfg)` 前に `cfg.repo.repo_id` へ反映されないため、index load と query filter で repo_id が乖離 → 空 retrieval リスク
- 同 script を seed 固定で触るため、同時修正が最小コスト
- 別 Issue 化すると seed 固定の eval が誤った baseline を含む可能性

**トレードオフ**:
- メリット: 1 PR で論理的に閉じる、再 review 不要
- デメリット: スコープ若干拡大 (regression test 1 件追加)
- リスク: なし (同型修正は `run_multi_turn_eval.py` で既に検証済み)

### ADR-11: seed 注入の重複ロジック (DR1-001 反映)

**背景**:

既存 `baseline_reporag/eval/institutional/llm_client.py:QwenMLXAdapter.generate` (Issue #135 Day 3 で導入) は seed 注入として以下を持つ:

```python
if seed is not None:
    import mlx.core as mx
    mx.random.seed(seed)
```

本 Issue で追加する `Generator.generate(*, seed)` (ADR-1) も同じパターン (`if seed is not None: mx.random.seed(seed)`) を取る。両者は (a) `mx.random.seed` 呼出、(b) `make_sampler(temp=...)` 経由、(c) `mlx_lm.generate(...)` 呼出 という同じ shape を持ち、本 Issue の seed 注入は QwenMLXAdapter のロジックの単純コピーになる。

> **注記 (DR2-006 反映)**: `QwenMLXAdapter.generate` は default `seed=42` (institutional grader は再現性必須)、本 Issue で追加する `Generator.generate` は default `seed=None` (interactive 経路維持) という non-symmetric default を持つ。helper 化する場合は default policy を呼び出し側で決める設計が必要。本 Issue では層分離 (B 採択) のためこの差分は扱わない。さらに `QwenMLXAdapter` は `try/except Exception: pass` で fail-soft している (実装: `baseline_reporag/eval/institutional/llm_client.py` line 105-111)。

**選択肢**:
- A: `seed_mlx_rng(seed: int | None) -> None` helper を `baseline_reporag/eval/run_config.py` に置き、`Generator.generate` と `QwenMLXAdapter.generate` の両方から呼ぶ (DRY 厳守)
- B: `Generator.generate` 側のみ追加し、`QwenMLXAdapter` は institutional grader 専用なので別系統と割り切る (層分離)
- C: `QwenMLXAdapter` を `Generator` の薄いラッパへ refactor (構造的統合)

**決定**: **B** (層分離 — 重複は受容する)

**理由**:
- seed 注入ロジックは 3-5 行で trivial。helper 化のオーバーヘッド (新規 helper の import / 名前空間 / unit test 追加) が DRY のメリットを上回る。
- `Generator` は production / eval 両方に使われる generic な LLM 生成 primitive、`QwenMLXAdapter` は institutional grader (citation grader / answer evaluator) 専用の adapter。両者は責務階層が異なる (production 線 vs institutional grader 線)。同じ pattern を持つことは「重複」というより「同じ MLX API への対応として収斂した独立実装」と解釈する。
- C (refactor) は QwenMLXAdapter を Generator に統合するため影響範囲が大きく、本 Issue のスコープ (re-eval reproducibility) を超える。
- 将来 `LLMClient` Protocol 統合 (#110 系) で両者が同じ Protocol を実装する段階で helper 化または基底クラス化を検討する。

**トレードオフ**:
- メリット: 実装 / test の追加コスト最小、institutional grader と production 線の責務分離維持、本 Issue スコープを最小化
- デメリット: `mx.random.seed(seed)` の 3 行が 2 箇所に存在 (DRY 緩和)
- リスク: 将来 mlx-lm が seed 注入 API を変更した場合、2 箇所修正が必要 → grep で検出可能 (`grep -rn 'mx.random.seed' baseline_reporag/`)、Code Review Checklist に「seed 注入を増やす場合は両者同時更新」を追記して対処 (Task 5 文書更新)

**棄却した代替案 A の詳細**:

`seed_mlx_rng(seed: int | None) -> None` を `baseline_reporag/eval/run_config.py` に配置し、両者から呼ぶ案。技術的には可能だが:
- `QwenMLXAdapter` は `baseline_reporag/eval/institutional/` 配下、`Generator` は `baseline_reporag/generation/` 配下にあり、両者から `baseline_reporag/eval/run_config.py` を import するのは依存方向として奇妙 (generation が eval を import することになる)。
- helper 配置を `baseline_reporag/_seed_utils.py` のような中立な場所にすると、新規 module の正当化が必要 (3 行 helper のために 1 module は重い)。

**棄却した代替案 C の詳細**:

`QwenMLXAdapter` を `Generator` の薄いラッパに refactor すると、institutional grader (citation grader 系) も `Generator` 経由で MLX を呼ぶ形になる。これは構造として綺麗だが:
- Issue #135 で institutional grader 系は独立 adapter として設計済 (citation grader 専用の prompt / sampling 戦略を持つ)。
- 本 Issue (eval reproducibility) のスコープを大きく超える refactor になる。
- 別 Issue (例: 「LLM 呼出統合」) で扱うべき。

---

## 5. データモデル変更

### predictions JSONL (static / multi-turn)

**現状フィールド (static)**: `eval_id, category, question, answer, cited_chunk_ids, no_citation, latency_ms, ...`

**追加フィールド (Issue #143)** — DR1-006 反映で 2 fields に絞る:
- `run_index: int` (0-indexed)
- `run_seed: int` (= cfg.run.seed for that run)

`run_id` は別 field として保持せず、必要時に downstream で `f"{run_index}_{run_seed}"` を計算して使う (schema 拡張面積を最小化)。

**現状フィールド (multi-turn)**: `session_id, turn_id, question, answer, cited_chunk_ids, no_citation, ...`

**追加フィールド**: 同上 (`run_index, run_seed`)

**旧 predictions JSONL の扱い**:

`run_index` / `run_seed` は新 runner の出力 schema では必須だが、aggregator の読込 validation では optional とする。欠落時は `run_index=0` として単一 run 扱いに正規化し、`run_seed` は `seeds` 集計から除外する。これにより過去の `*.predictions.jsonl` と Step 5/6 分離時の CI を保護する。

### `aggregate_institutional_baseline.py` 出力 (新規)

per-run summary section に以下を追加:
- `n_runs: int`
- `seeds: list[int]`
- `nc_rate: { mean: float, std: float, min: float, max: float, per_run: list[float] }`
- `latency_p50_ms: { mean, std, min, max, per_run }`

`mean - 2*std` を Task 4 で別途計算 (報告書 `reports/institutional_eval_noise_floor.md`)。

### `reports/institutional_eval_noise_floor.md` (新規)

```markdown
# Institutional Eval Noise Floor (Task 4)

## Summary
- mean: X.XX%
- std: X.XX
- min: X.XX%
- max: X.XX%
- n_runs: 10
- seeds: [42, 42, 42, ..., 42] # 完全決定性が真なら全て同 seed
- computed_at_commit: <commit-sha>
- judgment_threshold (mean - 2*std): X.XX%

## per_run
| run | seed | nc_rate | p50_latency_ms |
| 0 | 42 | ... | ... |
| ... |
```

---

## 6. セキュリティ設計

| 脅威 | 対策 |
|------|------|
| seed 値による入力情報漏洩 | seed は `type(seed) is int` のみ、log / JSONL には reproduction 用の `run_seed` と `run_index` のみ出力。`run_id` は保持しない (ADR-5) |
| YAML injection / 型混同経由の `cfg.run.seed` | 既存 `load_config()` は `yaml.safe_load` を使用。`resolve_eval_seed(cfg)` は `type(seed) is int` と `0 <= seed < 2**32` を検証し、`bool` / `float` / `NaN` / `str` を `TypeError`、範囲外 int を `ValueError`。`eval()` / `exec()` / 文字列式評価は使わない |
| `--runs N` による計算リソース DoS | `--runs` は default `1`、`1 <= runs <= 20` に argparse/custom validator で制限。weekly CI は `--runs 1` 固定、Streamlit は single-run 維持 |
| `--repo-id` から path traversal | 既存 `validate_repo_id()` / UI `_safe_id` と同じ `[A-Za-z0-9_-]+` allowlist を維持。今回 silent bug 修正のみ、新規 attack surface なし |
| `--output` / `--marker-file` / aggregator `--predictions` の path traversal | CLI は trusted operator tool とし、明示 path を任意に許す既存互換を維持する。一方、Streamlit から subprocess に渡す path は既存 `eval_panel.make_eval_paths()` で `reports/eval_runs/` と `logs/eval/` 配下に confined されるため、Issue #143 で `--runs` を UI に追加しない (ADR-8) |
| command injection | subprocess 経路は既存 `build_eval_job_cmd()` / `start_eval_job()` の argv list + `shell=False` を維持。`cfg.run.*` は subprocess argv に渡さず、script 内で `resolve_eval_seed(cfg)` が読むだけ |
| `mx.random.seed()` 依存性 | seed は bounded int のみ。`mx.random.seed(seed)` は MLX RNG state を設定するだけで外部 I/O や code loading を増やさない。mlx-lm の未知脆弱性は dependency update / pinning の一般リスクとして扱い、本 Issue で新規 attacker-controlled code path は作らない |
| Streamlit `--runs` 経路 | UI 単一 run 維持 (ADR-8) のため変更なし |
| document leakage | `docs/troubleshooting.md` / `docs/code_review_checklist.md` には seed wiring rule と troubleshooting のみ記載し、eval questions / model answers / filesystem absolute paths / tokens は転載しない。詳細ログは `logs/` / `reports/eval_runs/` (gitignore 済み) に留める |

---

## 7. テスト戦略

### Unit tests (mockable, fast)

- `baseline_reporag/tests/test_pipeline_integration.py`:
  - `RepoRAGPipeline.query(seed=42)` → `mock_gen.generate.assert_called_with(messages, max_new_tokens=..., seed=42)`
  - `RepoRAGPipeline.query(seed=None)` → `mock_gen.generate` は `seed` を渡さない or `seed=None` で呼ばれる (既存 17+ MagicMock テスト後方互換)
- `baseline_reporag/tests/test_photon_pipeline.py`:
  - PHOTON disabled path (Qwen-only) で seed 伝播
  - PHOTON enabled + Qwen fallback (3 箇所) で seed 伝播
- `baseline_reporag/tests/test_run_config.py` (新規) — DR1-002 / DR1-008 反映で独立 test case を以下の粒度で記述:
  - `test_resolve_eval_seed_default_returns_42()`: `cfg` に `run` ブロックがない → default で `seed=42` を返す
  - `test_resolve_eval_seed_int_seed()`: `run.seed=123, run.deterministic=True` → `seed=123` を返す
  - `test_resolve_eval_seed_deterministic_false_returns_none()`: `run.deterministic=False` → `seed=None` を返す (seed 値があっても無視)
  - `test_resolve_eval_seed_run_block_missing()`: `cfg` に `run` キー自体がない場合の挙動 (default で `seed=42`)
  - `test_resolve_eval_seed_invalid_deterministic_type_raises_typeerror()`: `run.deterministic="false"` (str) → `TypeError`
  - `test_resolve_eval_seed_invalid_seed_type_raises_typeerror()`: `run.seed="42"` (str) → `TypeError`
  - `test_resolve_eval_seed_bool_seed_raises_typeerror()`: `run.seed=True` (YAML `true`) → `TypeError` (`bool` は `int` subclass のため明示 guard)
  - `test_resolve_eval_seed_nan_seed_raises_typeerror()`: `run.seed=float("nan")` (YAML `.nan`) → `TypeError`
  - `test_resolve_eval_seed_negative_seed_raises_valueerror()`: `run.seed=-1` → `ValueError`
  - `test_resolve_eval_seed_overflow_seed_raises_valueerror()`: `run.seed=2**33` → `ValueError`
- `tests/test_run_baseline_eval.py` / `tests/test_run_multi_turn_eval.py` (既存がなければ新規):
  - `--runs 0` / `--runs 21` / `--runs 1000000` を argparse validation で reject
  - `--runs 1` / `--runs 10` / `--runs 20` を accept
- `tests/test_aggregate_institutional.py`:
  - per-run NC rate / latency の `mean/std/min/max/n_runs/seeds` 算出
  - `record_type=multi_turn` で MT predictions の集計
  - (test 追加対象 (per-run 集計 / record_type=multi_turn) は既存ファイル `tests/test_aggregate_institutional.py` への追記。新規ファイル `test_aggregate_institutional_baseline.py` は作成しない)
- `tests/test_compare_generators.py`:
  - `compare_generators` が `cfg.run.seed` を `pipeline.query(seed=...)` に伝播

### Integration tests (real MLX)

- `evals/tests/test_eval_determinism.py` (新規):
  - `@pytest.mark.skipif(not _HAS_MLX, reason="requires MLX")`
  - 同一 prompt × 同一 seed=42 を 2 回 generate し、`cited_chunk_ids` と `no_citation` で完全一致 assert
  - 1 prompt 検証で CI 速度維持 (LLM 起動を含むため重い → self-hosted runner のみ)

### Manual / Off-CI

- Task 2 ablation: institutional eval を seed=42 で `temperature=0.2` と `0.0` で各 1 run、結果比較
- Task 4 noise floor: institutional baseline V0 を seed=42 で 10 回反復 (~10h)、std 計測

---

## 8. 品質基準

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| ユニットテスト | `python -m pytest baseline_reporag/tests/ tests/ evals/tests/` | 全パス (`test_eval_determinism` は MLX なし環境で skip) |
| リント | `ruff check .` | 警告 0 件 |
| フォーマット | `ruff format --check .` | 差分なし |
| 既存 baseline 疎通 | `python -m baseline_reporag.cli --config configs/baseline.yaml --repo-id fastapi_fastapi --question "test"` | 応答あり (interactive 経路で seed=None デフォルト確認) |
| determinism integration | `pytest evals/tests/test_eval_determinism.py -v` (self-hosted MLX 環境) | 2-run 完全一致 |

**補注 (DR2-010 反映)**: 本表は本 Issue の影響範囲に絞った最小スコープ。CI / 受入時は CLAUDE.md 品質チェックの全 test scope (`torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/`) を通過させる。`evals/tests/` は本 Issue で新規追加するため明示。

---

## 9. 内部依存関係 / 実装順序

```
[Step 1] resolve_eval_seed(cfg) helper + unit test (run_config.py)
   ↓
[Step 2] Generator.generate(*, seed) + Pipeline.query(*, seed) + PhotonRAGPipeline.query(*, seed)
   ↓ (unit test で seed 伝播 fix)
[Step 3] eval scripts (`run_baseline_eval.py` / `run_multi_turn_eval.py` / `retrieval_grid_search.py` / `run_stress_eval.py`) に cfg.run.seed 伝播 + --repo-id silent bug fix
   ↓
[Step 4] evals/tests/test_eval_determinism.py 新規 (skipif MLX)
   ↓ (Task 1 完了 = Issue 受入条件 1-4 達成)
   ↓
   ↓ [前提 (DR2-008 反映)] Step 5-6 着手前に Issue #156 (is_refusal 出力欠落) のマージを完了させる。
   ↓   理由: 両者とも `run_multi_turn_eval.py` の predictions dict、predictions JSONL schema、
   ↓   REQUIRED_FIELDS / aggregator REQUIRED_FIELDS を変更するため、
   ↓   merge order #156 → #143 を ADR-5/6 で固定 (リスク §11)。
   ↓
[Step 5] --runs N 引数 (1 <= N <= 20) + predictions schema 拡張 (run_index/run_seed の 2 fields, ADR-5)
   ↓
[Step 6] aggregator per-run 集計 (record_type=static/multi_turn)
   ↓
[Step 7] compare_generators の seed 伝播 + test
   ↓ (Task 3 完了)
[Step 8] Task 2 ablation (institutional eval × temperature=0.0/0.2 各 1 run, 文書化)
   ↓ (Task 2 完了)
[Step 9] Task 4 noise floor 計測 (~10h, manual)
   ↓
[Step 10] Task 5 文書更新 (CLAUDE.md / docs/deployment.md / docs/troubleshooting.md)
   ↓
[Step 11] PR 作成 + multi-stage-design-review 反映 + Issue body の Task 3 受入条件文言を 2 fields に同期 (DR2-002)
```

CI 採用方針 (ADR-7) と Streamlit (ADR-8) は変更なしのため Step に含まない。

**Step 5-6 の境界に関する注釈 (DR1-013 反映)**:

Step 5 (`--runs N` 引数 + predictions schema 拡張) で predictions JSONL に新 fields (`run_index`, `run_seed`) を出力する際、aggregator (現行 = Step 6 で改造予定) は新 fields を **ignore する後方互換動作** で動作する。これは:
- 現行 aggregator (`scripts/aggregate_institutional_baseline.py`) が REQUIRED_FIELDS 以外の追加 fields をエラーにせず読み飛ばす実装を維持
- = Step 5 完了時点で、新 schema の predictions JSONL を旧 aggregator で読んでも `mean ± std` は出ないが「壊れない」(scalar NC を出すのみ)
- Step 6 で per-run 集計 (`mean / std / min / max / n_runs / seeds`) を有効化

代替案として **Step 5 と Step 6 を統合** し、predictions schema 拡張 + aggregator 拡張を 1 つの PR 単位にまとめる選択肢もあり。これは:
- メリット: PR 単位で「per-run 集計が機能する」状態を 1 つの変更で確認可能 (review 容易)
- デメリット: PR 1 件あたりの変更量が増える (LOC +60 + +120 = +180)

本設計では **Step 5-6 を分離 (現状の §9 通り) + 後方互換動作** を採用する (PR 粒度を細かく保つ)。実装時に PR レビューの粒度や進捗都合で統合を選択するのは PM 判断 (work-plan 段階で再検討可)。

**Step 10 / Step 11 境界の取扱い (DR2-009 反映)**:

Step 10 (Task 5 文書更新) の成果物を Step 11 PR の一部として出す。multi-stage-design-review の追加指摘で文書更新が必要になった場合は Step 11 内で再修正する (= 文書更新は PR 作成サイクルに内包)。

---

## 10. 影響範囲 (実装コミット粒度)

| Step | 影響ファイル | 推定 LOC |
|------|------------|---------|
| 1 | `baseline_reporag/eval/run_config.py` (新規)、`baseline_reporag/tests/test_run_config.py` (新規, 10 test cases: default / int / `deterministic=False` / `run` 欠落 / TypeError × 4 / ValueError × 2) | +80 |
| 2 | `generation/generator.py`, `pipeline.py`, `photon_pipeline.py`, `tests/test_pipeline_integration.py`, `tests/test_photon_pipeline.py` | +60 |
| 3 | `scripts/run_baseline_eval.py`, `scripts/run_multi_turn_eval.py`, `scripts/retrieval_grid_search.py`, `scripts/run_stress_eval.py`, `tests/test_retrieval_grid_search_smoke.py`, `tests/test_run_stress_eval.py` | +90 |
| 4 | `evals/tests/test_eval_determinism.py` (新規) | +50 |
| 5 | `scripts/run_baseline_eval.py`, `scripts/run_multi_turn_eval.py` (--runs bounded validator + schema), `tests/test_run_baseline_eval.py` / `tests/test_run_multi_turn_eval.py` (`--runs` bounds) | +90 |
| 6 | `scripts/aggregate_institutional_baseline.py`, `tests/test_aggregate_institutional.py` (test 追加対象 (per-run 集計 / record_type=multi_turn / 旧 JSONL 互換) は既存ファイル `tests/test_aggregate_institutional.py` への追記) | +140 |
| 7 | `scripts/compare_generators.py`, `tests/test_compare_generators.py` | +20 |
| 8 | `configs/institutional_docs.yaml` (Task 2 採用判定により), `reports/institutional_temperature_ablation_v143.md` (新規) | +30 |
| 9 | `reports/institutional_eval_noise_floor.md` (新規) | +50 |
| 10 | `CLAUDE.md`, `docs/deployment.md`, `docs/troubleshooting.md`, `docs/code_review_checklist.md` | +45 |

**推定合計**: ~655 LOC (本体 + tests + docs)

**LOC 内訳補注 (DR2-007 / DR3 / DR4 反映)**: Step 1 (+80 LOC) = helper +30 / tests +50 (bool/NaN seed TypeError test を含む)。test 1 case 平均 6-8 LOC。Step 3 (+90 LOC) = 4 scripts seed wiring + retrieval/stress tests。Step 5 (+90 LOC) = `--runs` bounded validator + schema + bounds tests。Step 6 (+140 LOC) = per-run / record_type / 旧 JSONL 互換 test。Step 10 (+45 LOC) = `CLAUDE.md` +10 / `docs/deployment.md` +15 / `docs/troubleshooting.md` +15 / `docs/code_review_checklist.md` +5。

---

## 11. リスクと緩和策

| リスク | 緩和策 |
|------|--------|
| Task 1 で seed 固定しても mlx-lm 内部の nondeterminism で 2-run が完全一致しない | `evals/tests/test_eval_determinism.py` を 1 prompt で先に試し、もし完全一致しない場合は 2-run の citation/answer **token 編集距離が小さい** ことの soft assert に切替 (本 Issue 受入時に決定)。Task 4 noise floor で std を実測し閾値を提案 |
| `--runs N` schema 変更で Issue #156 (refusal-aware) と merge conflict | merge order を **#156 → #143** に固定。特に `run_multi_turn_eval.py` の `pred` dict、predictions JSONL schema、aggregator REQUIRED_FIELDS が衝突点になるため、Step 5-6 着手前に #156 後の schema を再確認する。Issue 「関連」セクションに記載済 |
| 既存 17+ MagicMock テストが query(seed=...) で TypeError | デフォルト `seed=None` 維持 + `seed=None` の場合 generator.generate に seed を渡さないことで回避 |
| `--runs` 誤指定による self-hosted runner / developer machine の長時間占有 | argparse で `1 <= runs <= 20` を強制。weekly は `--runs 1`、Streamlit は single-run 維持。20 超の長期計測は別 Issue で扱う |
| YAML `run.seed: true` が Python の `bool`/`int` 継承で `seed=1` として通る | `type(seed) is int` を使い、`bool` / `float` / `NaN` / `str` を `TypeError` にする unit test を追加 |
| Task 4 (~10h) の中断 | log を JSONL に逐次 append、再開可能な resume option を `scripts/run_baseline_eval.py --runs` 設計時に検討 (今 Issue では simple loop でも可) |
| temperature=0 で institutional 出力が degenerate | Task 2 ablation で 1 prompt 比較し採否判定 (ADR-4) |

---

## 12. 受入条件 (Issue 本文より転記、設計判断と整合)

- [ ] Task 1: `Generator.generate(seed=42)` 引数追加 + `RepoRAGPipeline.query(seed=...)` / `PhotonRAGPipeline.query(seed=...)` 追加 + eval scripts から `cfg.run.seed` を伝播。`evals/tests/test_eval_determinism.py` 新規作成。同一 prompt × 同一 seed の 2-run が `cited_chunk_ids` および `no_citation` で完全一致することを assert
- [ ] Task 1: `resolve_eval_seed(cfg)` helper + `run` 欠落時 default、`run.deterministic` bool validation、`run.seed` int/range validation の unit test
- [ ] Task 1: `RepoRAGPipeline.query(seed=42)` が `generator.generate(..., seed=42)` を呼ぶ unit test、`query(seed=None)` の引数 shape 維持、`PhotonRAGPipeline.query(seed=42)` が Qwen-only / fallback 両 path で seed 伝播
- [ ] Task 1: `scripts/run_baseline_eval.py --repo-id` を `build_pipeline` 前に `cfg.repo.repo_id` へ反映 (silent bug fix)
- [ ] Task 2: temperature=0 採用または「temperature=0.2 のままで decision に影響しない理由」を文書化
- [ ] Task 3: `--runs N` 引数追加 (`1 <= N <= 20`)、predictions JSONL に `run_index/run_seed` の 2 fields を出力 (`run_id` は ADR-5 / DR1-006 で計算復元方針に変更)、aggregator が pooled ではなく per-run `mean/std/min/max/n_runs/seeds` 集計対応
- [ ] Task 3: weekly CI は `--runs 1` 維持 (ADR-7 で確定)
- [ ] Task 3: Streamlit eval runner は single-run 維持 (ADR-8 で確定)
- [ ] Task 3: `compare_generators.py` は seed 固定対象に含める (ADR-9)
- [ ] Task 4: 10-run noise floor 計測 + `reports/institutional_eval_noise_floor.md` 出力
- [ ] 既存 eval scripts test 全パス
- [ ] Task 5: CLAUDE.md「現在のメトリクス」更新、`docs/deployment.md` / `docs/troubleshooting.md` / `docs/code_review_checklist.md` 更新

> **注釈 (DR2-002 反映)**: Issue 本文の Task 3 受入条件文言は元 3 fields (`run_index/run_seed/run_id`) で記載されているが、設計判断 ADR-5 (DR1-006) で 2 fields (`run_index/run_seed`) に絞った。`run_id` は downstream で `f"{run_index}_{run_seed}"` の計算で復元する方針。Issue body の更新は §9 Step 11 PR 作成時に同時更新する。

---

## 13. 関連

- 本 Issue: [#143](https://github.com/Kewton/photon-mlx/issues/143)
- 元 Issue: [#137](https://github.com/Kewton/photon-mlx/issues/137) (5-variant A/B、CLOSED)
- 関連 Issue: [#138](https://github.com/Kewton/photon-mlx/issues/138) (CLOSED, tokenizer mismatch — 追加対応不要)、[#135](https://github.com/Kewton/photon-mlx/issues/135) (PHOTON 再学習採用)、[#156](https://github.com/Kewton/photon-mlx/issues/156) (OPEN, is_refusal 出力欠落 — merge order #156 → #143)
- レビュー結果: `workspace/issues/143/issue-review/summary-report.md` (32 findings 反映)
- 仮説検証: `workspace/issues/143/issue-review/hypothesis-verification.md`

---

## 14. 次フェーズ

- `/multi-stage-design-review 143` で 4 段階レビュー (通常 / 整合性 / 影響分析 / セキュリティ)
- `/work-plan 143` でタスク分解
- `/pm-auto-dev 143` で TDD 実装
