# Issue #135 設計方針書 — PHOTON 本格再学習 (制度文書ドメイン対応)

## 1. メタデータ

| 項目 | 値 |
|------|-----|
| Issue 番号 | #135 |
| Issue タイトル | feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合 (#117 conditional, #113 follow-up) |
| リポジトリ | https://github.com/Kewton/photon-mlx |
| ブランチ | `feature/issue-135-photon-retrain` |
| ベースブランチ | `develop` (PR は `develop` → `main`) |
| 作成日 | 2026-04-26 |
| 想定工数 | **3-5 日** (内訳: Day 1-2 データ準備 / Day 3-4 学習 / Day 4-5 eval & rollout) |
| 関連 Issue | **#113** (PR #134 merged, 仮説 C 確定の実測根拠) / **#117** (Epic Phase 2 conditional) / **#134** (#113 の merge PR) / **#137** (institutional 多言語 embedding/reranker A/B、本 Issue より先行完了が前提、S7-005 反映) |
| 設計 stage 累計指摘 | Must Fix 11 / Should Fix 22 / Nice to Have 5 = 38 件 (全件本書に反映済) + Stage 1 設計原則レビュー Must 3 / Should 6 / Nice 2 = 11 件 (DR1-001〜DR1-011、全件反映済) |
| 仮説検証ステータス | 12 件中 10 件 Confirmed、1 件 Unverifiable (mulmoclaude 600 step checkpoint 物理確認 = Day 1 必須タスク)、本 Issue 範囲で 1 件補足確認済 (institutional_documents 4228 .md は `/Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents/` に実在) |

---

## 2. ゴールと成功条件

### 2.1 ゴール (Issue 本文要約)

#113 の実測 (`reports/institutional_photon_mt_eval.md`) で確定した「**現行 PHOTON は制度文書ドメインで baseline より NC が 4.44pp / Turn 5-6 で 5.83pp 悪化**」という設計 §9 仮説 C を解消するため、`mulmoclaude (英語コード) 600 step checkpoint` を起点に **JP 制度文書 50%+ を混合した continual learning (10K-20K 累計 step)** を実施し、PHOTON checkpoint を制度文書ドメインへ適合させる。

### 2.2 成功条件 (測定可能指標)

| 指標 | 起点 (現行 PHOTON) | 目標 (最低) | 目標 (理想) | 計測方法 |
|------|-------------------|------------|------------|---------|
| **Turn 5-6 NC** | 10.83% | **< 6.0%** (仮説 B 達成 = MVP minimum) | **< 3.0%** (仮説 A 復元) | `scripts/run_multi_turn_eval.py` × 2-run (境界帯 5-7% は +2 run = 4-run 平均) |
| **NC overall** | 11.39% | < 6.94% (baseline 同等以下) | - | 同上 (補助指標) |
| **follow-up p50 latency** | 10,707 ms | **≤ 13,600 ms** (= baseline 19,426 ms × 0.7、-30% 以上の優位を維持) | 現行 -44.9% を可能な限り維持 | 同上 (eval が記録) |
| **FastAPI 系 MT NC regression** | gate2 v4 PHOTON+SR 6.7% | **≤ 11.7%** (= 6.7% + 5pp 以内) | 6.7% 同等 | `scripts/run_multi_turn_eval.py` + `configs/baseline.yaml` (provider photon) |
| **JP sequence 比 (制御目標、DR1-007)** | 0% | **= 50% (target、ε≤2pp)** | 50% exact | `iterate_mixed_batches` の sequence-level weighted sampling で制御。metadata.json `jp_sequence_ratio` で記録 |
| **JP token 比 (実測値、DR1-007)** | 0% | **≥ 50% ± 5pp** (catastrophic forgetting 抑制) | 50-60% | tokenizer encode 後の id 比 (per-corpus avg sequence token len から事前推定 + 実 yield された batch token 列で測定)。metadata.json `jp_token_ratio` で記録 |
| **学習 corpus sessions 数** | - | **≥ 2,000** | 5,000 | generate script の `--sessions` (上限 5,000) |
| **cross_reference / drill_down シナリオ比率** | - | **≥ 30%** (#113 で最も悪化したシナリオ重視) | 40-50% | generate script の category metadata 集計 |
| **既存 168 tests + photon_mlx tests pass 状態** | 全 pass | 同 (regression なし) | - | `python -m pytest --ignore=data/training/` |

最低条件 (Turn 5-6 NC < 6% AND follow-up p50 ≤ 13.6s) を**全て満たす checkpoint のうち、Turn 5-6 NC が最小のもの**を採用する単一指標最適化を採る (設計判断 #6 参照)。

---

## 3. アーキテクチャ全体像

### 3.1 本 Issue の関与範囲

CLAUDE.md のモジュール構成のうち、本 Issue で扱うのは **学習データ生成 → 学習 → checkpoint → runtime checkpoint load → eval → ドキュメント更新** の 1 経路のみ。検索 (BM25 / embedding / reranker)・chunking・FastAPI server は触らない。

```
                       Issue #135 のスコープ
   ┌────────────────────────────────────────────────────────────────┐
   │                                                                │
   │  /Users/.../myWebData/markdowndb/institutional_documents/      │
   │  (4228 .md, 機密) ─── 既存                                     │
   │              │                                                 │
   │              │ scripts/generate_institutional_training_corpus  │
   │              │   .py  (新規, S5-003 反映 CLI)                  │
   │              │   ├─ scripts/_corpus_core.py (新設, S3-004)     │
   │              │   └─ eval リーク検証内蔵 (S3-009)               │
   │              ▼                                                 │
   │  data/training/institutional/                                  │
   │   ├─ train_jp.jsonl   (1 session = 1 doc, S5-006)             │
   │   ├─ val_jp.jsonl                                              │
   │   └─ metadata.json    (jp_token_ratio, n_sessions 等)          │
   │  (.gitignore で除外、S3-003)                                   │
   │              │                                                 │
   │              │ data/processed/train_multi.jsonl (mulmoclaude   │
   │              │   既存 EN コード corpus, 50% mix)               │
   │              ▼                                                 │
   │  photon_mlx/                                                   │
   │   ├─ data.py     iterate_mixed_batches()  ★ 新関数のみ追加     │
   │   │              (既存 iterate_batches signature 不変、S3-001) │
   │   ├─ trainer.py  ★ 改修最小: TrainingConfig.train_corpora_mix  │
   │   │              が dict なら iterate_mixed_batches を呼ぶ分岐 │
   │   │              のみ。schedule/checkpoint/resume_from は不変  │
   │   ├─ model.py    ★ 改修なし (既存 PhotonModel)                  │
   │   └─ inference.py★ 改修なし                                    │
   │              │                                                 │
   │              │ resume_from = checkpoints/photon_mulmoclaude/   │
   │              │   step_600/  (既存、Day 1 物理確認必須)          │
   │              ▼                                                 │
   │  checkpoints/photon_institutional_retrain_<yyyymmdd>/          │
   │   ├─ step_010000/  (累計 step、S5-002, S7-002)                  │
   │   ├─ step_015000/                                              │
   │   ├─ step_020000/  各 leaf に weights.npz / state.json          │
   │   └─ final/                                                    │
   │              │ (採用 checkpoint のみ昇格コピー可)               │
   │              ▼                                                 │
   │  configs/institutional_docs_photon_retrain.yaml  (新規)         │
   │   model.checkpoint_path: ...step_NNNNNN/  ← leaf path           │
   │   training.train_corpora_mix: {jp.jsonl: 0.5, en.jsonl: 0.5}    │
   │   training.val_split: 0.05  (DR1-005: train pool 内 split)       │
   │              │                                                 │
   │              ▼                                                 │
   │  baseline_reporag/photon_pipeline.py::_build_photon_deps()     │
   │   ★ S7-001 + DR1-002: model.checkpoint_path を読み              │
   │     photon_mlx.checkpoint.load_checkpoint(model, path) を呼ぶ   │
   │     (trainer.py への直接 import は禁止 / inference 専用 module 群 │
   │      = model / inference / session / safe_recgen / checkpoint   │
   │      のみを baseline_reporag からは触る)                         │
   │     missing/corrupt は fail (silent fallback 禁止)              │
   │              │                                                 │
   │              ▼                                                 │
   │  scripts/run_multi_turn_eval.py + build_pipeline(cfg)          │
   │   (既存、改修なし)                                             │
   │              │                                                 │
   │              ▼                                                 │
   │  reports/institutional_photon_mt_eval_v2.md   (新規)            │
   │  reports/gate2_post_retrain_eval.md           (新規)            │
   │  reports/gate2_judgment_v5_post_retrain.md    (新規, S3-005)    │
   │  CLAUDE.md L156-160 / workspace/mvp/roadmap.md / metrics.md     │
   │                                                                │
   └────────────────────────────────────────────────────────────────┘
```

### 3.2 各レイヤーの責務一覧

| レイヤー | パス | 責務 | 改修種別 |
|---------|------|------|---------|
| 機密保護 | `.gitignore` | `data/training/` 除外 (本 Issue 最初の commit) | 拡張 |
| 学習データ生成 | `scripts/generate_institutional_training_corpus.py` | institutional_documents → 2,000+ session JSONL | 新規 |
| 共通 corpus core | `scripts/_corpus_core.py` | tokenize / pack / val split / `validate_tokenizer_id` 共通化 | 新規 |
| データロード | `photon_mlx/data.py` | `iterate_mixed_batches` 追加 (既存 API 不変) | 拡張 |
| 学習設定 | `torch_ref/config.py` | `TrainingConfig.train_corpora_mix` + `val_split: float` 追加 + strict validation (DR1-003 / DR1-005) | 拡張 |
| 学習 | `photon_mlx/trainer.py` | `corpora_mix` が dict なら mixed batch を使う分岐のみ | 拡張 (最小) |
| Config | `configs/institutional_docs_photon_retrain.yaml` | 学習用 hyperparam + checkpoint_path leaf | 新規 |
| Checkpoint I/O 抽出 | `photon_mlx/checkpoint.py` (新規軽量 module) | `load_checkpoint(model, path)` / `save_checkpoint(model, path, state)` を専用 module に切り出し。trainer.py は `from .checkpoint import load_checkpoint` 経由で再 export (DR1-002) | 新規 |
| Runtime checkpoint load | `baseline_reporag/photon_pipeline.py` | `model.checkpoint_path` を `photon_mlx.checkpoint.load_checkpoint()` で読む (S7-001 / DR1-002)。`photon_mlx.trainer` への直接 import は禁止 | 拡張 |
| Pipeline factory | `baseline_reporag/pipeline_factory.py` | photon provider 経路 (改修なし、smoke test 追加) | 確認のみ |
| Eval | `scripts/run_multi_turn_eval.py` | 既存利用 | - |
| Report | `reports/institutional_photon_mt_eval_v2.md` 等 | 旧/新/baseline 比較 | 新規 |
| Documentation | `CLAUDE.md` / `workspace/mvp/roadmap.md` / `workspace/mvp/metrics.md` | 採用 checkpoint 確定後にメトリクス更新 | 拡張 |
| CI | `.github/workflows/tests.yml` | `pytest --ignore=data/training/` 必須化 (CI にしない場合はローカル実行ログ添付) | 新規/任意 |

---

## 4. レイヤー構成と責務 (本 Issue 範囲)

| レイヤー | モジュール | 本 Issue での責務 | 改修種別 | Codex 指摘反映 |
|---------|-----------|-----------------|---------|--------------|
| データ生成 (entry/合成) | `scripts/generate_institutional_training_corpus.py` (新規) | **CLI entry point + 合成のみ** (DR1-001): `main()` は (a) `build_sessions(...)` (LLM 駆動の純粋生成関数) と (b) `verify_corpus(...)` (純粋検証関数) と (c) `_corpus_core.*` (tokenize/pack/val split) を順次呼び合成するだけ。CLI: `--corpus-dir <institutional_documents_dir> --provider auto --sessions 2000 --output data/training/institutional --val-ratio 0.1 --seed 42 [--resume] [--dry-run]`。**失敗率 5% 超で exit code 1 (DR1-009)**: 失敗カテゴリ = `{api_error, json_parse_error, scenario_misclassified, token_overflow}` のうち **`api_error + json_parse_error`** の和のみで 5% 閾値を計算 (= LLM 側の不可避な不安定性による失敗のみ)。`scenario_misclassified` は scenario 別 retry で補正対象、`token_overflow` は事前 truncate で対処し閾値計算には含めない。失敗率 ≤ 5% なら達成数 (例: 1,985 / 2,000 要求) で続行可、metadata に `n_sessions_requested / n_sessions_succeeded / failure_breakdown` の 3 項目で内訳記録 | 新規 | S5-003, S3-004, S3-009, **DR1-001**, **DR1-009** |
| データ生成 (LLM 駆動) | `scripts/generate_institutional_training_corpus.py::build_sessions` (純粋関数) | **責務 (b) only** (DR1-001): institutional_documents 走査 + LLM session 合成 (cross_reference / drill_down 重視)。`Iterator[Session]` を yield し、I/O・eval リーク検証・metadata 生成は持たない。`tests/test_generate_institutional_training_corpus.py` で LLM クライアントを fixture で差し替えて単体テスト可能 | 新規 (関数粒度) | **DR1-001** |
| データ生成 (検証) | `scripts/generate_institutional_training_corpus.py::verify_corpus` (純粋関数) | **責務 (e/f) only** (DR1-001): `verify_corpus(sessions: Iterable[Session], eval_path: Path) -> CorpusReport` は eval リーク検証 (session_id 集合の積) と JP 比率 / scenario 分布集計を行う純粋関数。LLM API は呼ばず metadata は dataclass `CorpusReport` で返す。`main()` がこれを受け取って metadata.json として書き出す | 新規 (関数粒度) | **DR1-001**, S3-009 |
| 共通 corpus core | `scripts/_corpus_core.py` (新規, private module) | **責務 (c/d) + (e の判定ロジック) を抽出** (DR1-001): tokenize / `pack_sequences` 互換 / val split / `validate_tokenizer_id` / `resolve_tokenizer_id` / `validate_eval_overlap(session_ids: set[str], eval_path: Path) -> int` を持つ。本 Issue では新 script のみが core を経由し、既存 `generate_training_corpus.py` は **触らず legacy 並存**。follow-up Issue (Day 5 起票) で書き換え + mulmoclaude corpus 再生成 diff 0 回帰テスト追加 (DR1-004) | 新規 | S3-004, **DR1-001**, **DR1-004** |
| データロード | `photon_mlx/data.py` | `iterate_mixed_batches(corpus_paths: dict[str, float], ...)` 追加。既存 `iterate_batches` の signature/挙動は完全に不変 | 拡張 (新関数追加のみ) | S3-001, S5-005 |
| 学習設定 | `torch_ref/config.py::TrainingConfig` | `train_corpora_mix: dict[str, float] \| None` + `val_split: float = 0.05` (DR1-005: val_corpora_mix dict 化は廃止し train pool 内 sequence-level split に簡素化) を追加。空 dict / 負 weight / 全 weight 0 / 非数値 weight / sum != 1.0 (±1e-6) を reject する strict `__post_init__` validation (DR1-003) | 拡張 | S7-003, **DR1-003**, **DR1-005** |
| 学習 | `photon_mlx/trainer.py` | `t_cfg.train_corpora_mix is not None` のとき `iterate_mixed_batches` を呼ぶ分岐追加。`_build_lr_schedule` / `load_checkpoint` / `train` ループの主要ロジックは不変 (resume_from + cosine schedule の既存実装をそのまま利用) | 拡張 (最小) | S5-001, S5-002, S5-005, S3-004 |
| Config | `configs/institutional_docs_photon_retrain.yaml` (新規) | `model.provider: photon` + `model.checkpoint_path: ./checkpoints/photon_institutional_retrain_<yyyymmdd>/step_NNNNNN` + 学習 hyperparam (`lr=3e-5`, `warmup_ratio=0.0`, `min_lr=3e-6`, `max_steps=10000-20000`, `train_corpora_mix` (sum=1.0 strict), `val_split=0.05`) | 新規 | S3-007, S5-005, S7-001, S7-002, **DR1-005** |
| Checkpoint I/O 抽出 | `photon_mlx/checkpoint.py` (新規軽量 module) | `load_checkpoint(model, path)` / `save_checkpoint(model, state, path)` を trainer.py から本 module に物理移管し、`CheckpointState` (または同等の軽量 state DTO) を `checkpoint.py` 側で定義する。`photon_mlx/trainer.py` 側は既存 `TrainState` API を維持する薄い wrapper / re-export とし、既存 `from photon_mlx.trainer import load_checkpoint, save_checkpoint, TrainState` callsite を破壊しない。trainer のみが必要とする `mlx.optimizers / cosine_decay / loss` 等の training 依存を import せずに weights.npz / state.json をロードできるようにする (DR1-002, DR3-001) | 新規 | **DR1-002** (Must Fix), **DR3-001** |
| Runtime checkpoint load | `baseline_reporag/photon_pipeline.py::_build_photon_deps` | `model.checkpoint_path` 設定時、`PhotonModel` 構築直後に `photon_mlx.checkpoint.load_checkpoint(model, checkpoint_path)` を呼ぶ。missing/corrupt は明示的に raise (silent fallback 禁止)。`photon_mlx.trainer` への直接 import は禁止し、baseline_reporag から触る photon_mlx 表面は inference 専用 module 群 (model / inference / session / safe_recgen / checkpoint) のみに保つ | 拡張 | **S7-001** (Must Fix) / **DR1-002** (Must Fix) |
| Pipeline factory | `baseline_reporag/pipeline_factory.py` | photon provider 分岐は不変。確認用 smoke test を `baseline_reporag/tests/` に追加 | 確認のみ + テスト追加 | S3-006, S7-001 |
| Eval | `scripts/run_multi_turn_eval.py` | 既存利用 (改修不要) | - | - |
| Report | `reports/institutional_photon_mt_eval_v2.md` (新規) | 旧 PHOTON / 新 PHOTON / baseline 比較表。**checkpoint metadata 列として `resume_base_step` / `additional_updates` / `cumulative_step` の 3 列を必ず含める (DR1-008、強制)**。`checkpoint_path` も併記 | 新規 | S7-002, **DR1-008** |
| FastAPI 系 regression report | `reports/gate2_post_retrain_eval.md` (新規) | `configs/baseline.yaml` (provider=photon に切替) で fastapi_fastapi corpus を 1 run、MT NC を測定 | 新規 | C-3 受入条件 |
| 採用判定 | `reports/gate2_judgment_v5_post_retrain.md` (新規) | v4 → v5 cross-link、新値で Gate 2 判定更新 | 新規 | S3-005 |
| 機密保護 | `.gitignore` | `data/training/` 除外 (本 Issue 最初の commit) | 拡張 | S3-003 |
| Documentation | `CLAUDE.md` / `workspace/mvp/roadmap.md` / `workspace/mvp/metrics.md` | 採用 checkpoint 確定後のメトリクス書換 + roadmap の「>6% 分岐実行済」マーキング | 拡張 | S3-005 |
| CI | `.github/workflows/tests.yml` | `pytest --ignore=data/training/` を CI 必須化 (任意、CI にしない場合はローカル実行ログを report に添付) | 新規 (任意) | S3-006, S7-004 |

---

## 5. 設計判断とトレードオフ

### 設計判断 #1: `iterate_mixed_batches` の signature と混合単位

- **選択肢 A (採用)**: 新関数 `iterate_mixed_batches(corpus_paths: dict[str, float], *, context_length, batch_size, seed, val_split=0.0)` を `photon_mlx/data.py` に追加 (DR2-006: `epochs=1` 引数は削除、§6.2 API 契約と表記統一)。混合は **corpus ごとに `load_jsonl → pack_sequences` で sequence pool を作り、micro-batch 構成時に pool 単位で weighted sampling**。epochs 概念は持たず、trainer.py の `while state.step < max_steps` ループが `batch_idx % len(train_batches)` で round-robin する (DR1-006 議論で list 戻り型を選んだため、epochs 自前管理は trainer 側に残す)。
- **選択肢 B**: 既存 `iterate_batches` の signature に `corpus_paths: dict | None` を追加して兼用。
- **決定**: **A** (新関数を追加し既存 API を完全凍結)。
- **理由**:
  1. `iterate_batches` には既存 callsite が **4 箇所** (`photon_mlx/trainer.py:297-303` の 2 箇所、`scripts/train_photon_quick.py:48-49` の 2 箇所) と **27+ tests** が依存しており、signature 変更は破壊リスクが高い (S3-001)。
  2. S5-005 指摘の「単純 concat 後 pack だと doc 長差で比率が崩れる」を回避するには「corpus ごとに pack して sequence pool 単位で weighted sampling」する独立した実装が必要で、引数追加では分岐が複雑化する。
  3. `pack_sequences` は corpus 間を跨がない (JP session ↔ EN コード sequence の連結禁止)。session 境界は delimiter `<|turn_sep|>` (通常文字列) で必ず明示する。
- **トレードオフ**: 関数が 2 種類になり API 表面が増えるが、後方互換と単一責務 (mix vs single) の分離が優先。validation は **`val_split=0.05` (DR1-005) が指定されていれば train pool 内から sequence-level split で val を切り出し (train_corpora_mix と同じ比率を val でも維持)、`val_split=0.0` (未指定) なら従来 `val_corpus` 単一 corpus に fallback する明示的分岐**とする (DR2-001 反映: `val_corpora_mix` dict は廃止)。

### 設計判断 #2: `resume_from` の `max_steps` 解釈と checkpoint 命名

- **選択肢 A (採用)**: `max_steps` は **累計 step (= `TrainState.step` の通算値)** として扱う。`resume_from=step_600` から `max_steps=15000` で実行 → 追加 14,400 step → checkpoint 名は `step_015000`。
- **選択肢 B**: `max_steps` を「追加 step 数」として再定義し trainer.py を改修。
- **決定**: **A**。`photon_mlx/trainer.py:339` の `while state.step < max_steps:` は累計 step を見ているため、**既存実装をそのまま利用するのが最も安全**。
- **理由**:
  1. trainer.py の主要ロジック (resume_from / cosine schedule / checkpoint 保存) を改修不要にすることで、既存 168 tests を破綻させない (S3-004, S5-002)。
  2. 累計 step 化により、save layout (`checkpoint_dir / step_NNNNNN`) が trainer の既存実装と完全一致 (S7-002)。
  3. config / report / runtime path の全てで `step_NNNNNN` という単一表記に統一可能。
- **トレードオフ**: 「10K/15K/20K = 累計 step」という暗黙のルールが必要。本書・config・report 全てで「`step_010000` は累計 10K (= resume base 600 + 追加 9,400 step)」と明記し誤解を防ぐ。
- **誤読防止 (DR1-008、強制ルール)**: `reports/institutional_photon_mt_eval_v2.md` の checkpoint metadata 列に **`resume_base_step` / `additional_updates` / `cumulative_step` の 3 列を必ず含める** ことを §10 / §4 で強制する。例: `step_010000` 行は `resume_base_step=600 / additional_updates=9400 / cumulative_step=10000` と記載。これにより leaf 名から累計と追加分の両方が一目で読み取れる。
- **例外発動条件 (DR1-008)**: 追加 step 数を厳密に leaf 名に出したい場合のみ `max_steps=10600/15600/20600` に変更し checkpoint 名 `step_010600` 等で表す。発動条件は (i) report レビュー段階で resume_base が誤読を招く事案が発生、または (ii) 別 Issue で「追加 step を主指標にする要件」が確定、のいずれか。本 Issue では発動しない。

### 設計判断 #3: LR schedule の起点と warmup_ratio

- **選択肢 A (採用)**: `warmup_ratio=0.0` のまま既存 `_build_lr_schedule` (`photon_mlx/trainer.py:154-183`) の挙動に委ねる。`warmup_steps=0 ∧ min_lr=3e-6 > 0` のケースでは `mlx.optimizers.cosine_decay(init=lr=3e-5, end=min_lr=3e-6)` が返るため、**初回 update から `lr=3e-5` 起点で `min_lr=3e-6` へ cosine decay**。
- **選択肢 B**: `warmup_ratio>0` を設定して低 LR 起点から立ち上げる、または新 schedule を実装。
- **決定**: **A**。S5-001 指摘で「min_lr→max_lr 起点」表現は実装と矛盾するため修正済。
- **理由**:
  1. trainer.py 改修不要を維持 (設計判断 #2 と整合)。
  2. continual learning では既存 600 step 重みが既に有意義な収束点なので、`lr=3e-5` (mulmoclaude `1.5e-4` の 1/5) という抑制された LR を初手から使うことで重み破壊を抑制する設計意図と一致。
  3. Day 3 着手時に **lr-finder (1K step small run、lr=1e-5 / 3e-5 / 1e-4 の 3 候補)** で val_loss を比較し、必要なら `lr` を再選定。
- **トレードオフ**: 初手から full LR が走るため、極端に大きな勾配で 600 step 重みを破壊する risk は残る。緩和策として **`test_resume_from_continual_learning` を学習開始前に CI に追加** し、resume + cosine reset の挙動を恒常確認する (リスク表参照)。

### 設計判断 #4: Turn separator の token 種別

- **選択肢 A (採用)**: `<|turn_sep|>` は **既存 tokenizer (Qwen2.5-Coder vocab) の通常文字列として encode する delimiter**。新 special token 追加なし、vocab/embedding resize なし。
- **選択肢 B**: tokenizer に special token を追加し vocab 拡張・embedding resize。
- **決定**: **A**。S5-006 指摘の通り、現行 `TokenizerConfig.vocab_size=152064` と `PhotonModel` embedding は固定であり、新 special token を導入すると vocab 外 ID で学習が壊れる。
- **理由**:
  1. tokenizer/embedding resize は本 Issue の射程を超えるため明確に範囲外。
  2. 通常文字列の delimiter でも turn 境界の認識は十分 (`<|turn_sep|>` は Qwen の BPE で安定して特定 token 列に encode される)。
- **トレードオフ**: turn 境界の attention bias が special token 並みに強くならない可能性があるが、本 Issue の目的は「制度文書ドメインへの語彙適合」であり turn boundary modeling 自体の改良ではないため許容。将来的に必要になれば別 Issue で vocab resize を扱う。

### 設計判断 #5: 学習用 corpus の出力 schema (1 session = 1 doc vs turn 別 doc)

- **選択肢 A (採用)**: **1 session = 1 doc**。`{"tokens": [...], "session_id": "session_NNNN", "scenario": "cross_reference|drill_down|...", "lang": "ja", "n_turns": 6}` 形式の JSONL。turn 境界は `<|turn_sep|>` で区切る。
- **選択肢 B**: turn ごとに別 doc として出力 (`session_id` だけ共通)。
- **決定**: **A**。
- **理由**:
  1. 本 Issue のゴールは **Turn 5-6 NC 改善**。直前 turn を「真に参照する」連続文脈を 1 sequence 内に保持する必要がある。
  2. `pack_sequences` は doc を連結して固定長 sequence にするが、**corpus 間を跨がない** 制約 (設計判断 #1) により JP session 同士は連結可能。1 session = 1 doc で投入すれば session 内の 6 turn が同 sequence に収まりやすく (4K token ≪ 2048 context だが session ごとの境界は維持)、Cross-turn 依存の学習信号が直接得られる。
  3. metadata で scenario / lang / n_turns を保持すれば、後続の比率検証 (cross_reference 30%+) と JP token 比 50%+ の集計が容易。
- **トレードオフ**: 1 session 4K token と context_length 2048 のミスマッチで一部 session が 2 sequence に分割されるが、`pack_sequences` の greedy bin-packing は session 内では順序保持されるため許容。

### 設計判断 #6: Checkpoint 採用ロジック (単一指標最適化 vs Pareto)

- **選択肢 A (採用)**: 最低条件 (Turn 5-6 NC < 6% AND follow-up p50 ≤ 13.6s AND FastAPI MT NC ≤ 11.7%) を満たす checkpoint のうち **Turn 5-6 NC が最小** のものを採用。境界帯 (5-7%) は **+2 run = 4-run 平均で再判定**。
- **選択肢 B**: NC / latency / FastAPI regression の 3 軸 Pareto 最適から人手判断。
- **決定**: **A**。
- **理由**:
  1. eval set は 30 sessions × Turn 5-6 = 60 turns で、1 turn ≈ ±1.67pp の noise。境界帯は統計的決定不能なため 4-run 必須。
  2. 単一指標最適化は採用判断を機械化でき、ロールアウト工程 (Day 5) を短縮。
  3. Pareto 最適は判断者依存性が高く再現性に劣る。
- **トレードオフ**: 「latency が悪いが Turn 5-6 NC が際立って良い」checkpoint が最低条件 (latency ≤ 13.6s) で除外される可能性がある。最低条件を緩める判断は別 Issue (`gate2_judgment_v5` で議論) とする。

### 設計判断 #7: Config 分離 (既存 yaml 編集 vs 新規 yaml)

- **選択肢 A (採用)**: 新規 `configs/institutional_docs_photon_retrain.yaml` を作成し、既存 `configs/institutional_docs_photon.yaml` (#112 / #126 baseline 再現用) は**直接編集せず維持**。
- **選択肢 B**: 既存 yaml を直接編集し commit history で diff を追う。
- **決定**: **A**。S3-007 / S7-001 反映。
- **理由**:
  1. #112 / #126 の baseline 再現性 (Static NC 11.21% 等) を**永続的に保護**するため、既存 yaml の path 指向は変えない。
  2. checkpoint_path / corpora_mix 等の新フィールドは新 yaml にのみ記述し、影響範囲を物理的に分離。
  3. PR diff も「新規追加」のみで、既存 eval pipeline の意図せざる変更を防ぐ。
- **トレードオフ**: yaml が 2 個になり保守コストは増えるが、再現性 > 重複削減。共通項を別 yaml に切り出す include 機構は本 Issue 範囲外 (現状 `load_photon_config` には include がない)。
- **follow-up (DR1-011, Nice to Have)**: 2 yaml の `model.{name, generation, photon}` セクション同等性を保証する `tests/test_institutional_yaml_diff.py` を **follow-up Issue として Day 5 中に起票** する。本 Issue 内では実装しない (再現性最優先、追加テストは別 Issue で扱う)。

---

## 6. データモデル

### 6.1 学習 corpus JSONL schema

```jsonc
// data/training/institutional/train_jp.jsonl の各行 (1 session = 1 doc)
{
  "tokens": [int, ...],                  // <|turn_sep|> を含む turn 連結 token 列
  "session_id": "session_0001",          // generate script 内で発番、eval set とは重複禁止
  "scenario": "cross_reference",         // cross_reference | drill_down | define | quantity | comparison | conclusion
  "lang": "ja",                          // ja | en (英語コードは mulmoclaude を再利用)
  "n_turns": 6,                          // session 内 turn 数 (4-6)
  "source_md": "<filename>.md"           // 主参照元 markdown (リーク監査用)
}
```

```jsonc
// data/training/institutional/metadata.json
{
  "n_sessions_requested": 2000,           // DR1-009: 要求数
  "n_sessions_succeeded": 1985,           // DR1-009: 達成数
  "failure_breakdown": {                   // DR1-009: 失敗カテゴリ別内訳
    "api_error": 8,
    "json_parse_error": 5,
    "scenario_misclassified": 2,           // 補正対象 (5% 閾値計算からは除外)
    "token_overflow": 0
  },
  "n_sessions": 1985,                      // 後方互換 (= n_sessions_succeeded)
  "jp_sequence_ratio": 0.50,               // DR1-007: 制御目標 (sequence-level、target = 0.5 exact)
  "jp_token_ratio": 0.52,                  // DR1-007: 実測値 (token-level、≥ 0.5 必須)
  "scenario_distribution": {
    "cross_reference": 0.20,
    "drill_down": 0.15,
    "define": 0.20,
    "quantity": 0.15,
    "comparison": 0.15,
    "conclusion": 0.15
  },
  "eval_overlap": 0,                     // eval set との session_id 重複数 (= 0 必須)
  "generated_at": "2026-04-27T...",
  "tokenizer_id": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
}
```

### 6.2 `iterate_mixed_batches` API 契約

```python
# photon_mlx/data.py に新規追加
def iterate_mixed_batches(
    corpus_paths: dict[str, float],          # 例: {"data/training/institutional/train_jp.jsonl": 0.5,
                                              #      "data/processed/train_multi.jsonl": 0.5}
    *,
    context_length: int,
    batch_size: int,
    seed: int = 42,
    shuffle: bool = True,
    val_split: float = 0.0,                  # DR1-005 / DR2-008: val_split > 0 なら (train, val) tuple を返す
) -> list[mx.array] | tuple[list[mx.array], list[mx.array]]:
    """End-to-end mixed-corpus loader.

    Implementation contract:
    - **Return type (DR1-006 / DR2-008)**:
        - `val_split == 0.0` (default): list[mx.array] (sequence-of-batches, not Iterator)
          を返し、既存 `iterate_batches` の signature 互換を保つ (S3-001 後方互換)。
        - `val_split > 0.0`: tuple[list[mx.array], list[mx.array]] = (train_batches, val_batches)
          を返す。trainer.py §6.4 分岐は `train_batches, val_batches = iterate_mixed_batches(..., val_split=t_cfg.val_split)`
          で tuple unpack する。
      Reason: align with iterate_batches signature (S3-001 / 既存 callsite 4 / 27 tests
      の signature 不変原則を最優先) so trainer.py の `for b in batches` ループに透過
      置換できる (val_split=0 のケース)。memory: 5,000 sessions × ~4K tokens × bf16 ≈ 40 MB の sequence pool が
      対象なので、context_length=2048 で pack 後も全体 < 1 GB に収まる想定。Day 3
      lr-finder で peak RAM を実測し、超過 (~3 GB 以上) なら Iterator[mx.array] に
      切り替える (リスク表参照)。
    - For each (path, weight) in corpus_paths, run load_jsonl → pack_sequences
      to build an INDEPENDENT sequence pool. pack_sequences DOES NOT cross
      corpus boundaries (S5-005).
    - Build the final batch list by weighted sampling at the SEQUENCE level
      from the pools (not at the doc level). The control target is sequence-level
      ratio (e.g. 0.5/0.5). The token-level ratio is a measured value, recorded
      to metadata.json for audit (DR1-007).
    - Returns a list of (batch_size, context_length) mx.array (compatible
      with iterate_batches return type so trainer.py can swap in transparently)
      when `val_split == 0.0`. When `val_split > 0.0`, returns a tuple
      `(train_batches, val_batches)` where both are the same list[mx.array]
      type; train_batches preserves the train_corpora_mix ratio, val_batches
      is sampled from the same pool with the held-out fraction (DR1-005, DR2-008).
    - **Strict validation (DR1-003)**: All boundary conditions raise ValueError
      (no silent normalize, no warning fallback):
        - corpus_paths is empty → ValueError
        - any weight is non-finite, <= 0, or non-numeric → ValueError
        - sum(weights) is not within 1e-6 of 1.0 → ValueError
      The same checks live in TrainingConfig.__post_init__ (§6.3) so config
      load fails BEFORE trainer is constructed; iterate_mixed_batches itself
      runs these checks defensively for direct callers (tests, scripts).
    - **Path resolution (DR1-003)**: corpus_paths keys are absolute paths.
      Normalization to absolute is performed at YAML load time
      (`load_photon_config` resolves relative paths against the YAML file's
      directory, not cwd). iterate_mixed_batches itself does NOT re-resolve
      paths; passing a relative path here is a programming error.
    - **Security validation (DR4-002)**:
        - Each corpus path MUST be `Path(path).resolve(strict=True)` and must be a
          regular file under an approved root (`data/training/` or `data/processed/`).
          symlink / hardlink escape outside approved roots is rejected before
          `load_jsonl`.
        - JSONL `tokens` MUST be a non-empty list[int] with length <=
          `context_length * 4`, every token `0 <= token < tokenizer.vocab_size`,
          and bool / float / negative / oversized int values rejected. Per-line
          byte size is capped (default 2 MiB) to avoid memory DoS.
        - Rejection is fail-fast `ValueError`; invalid lines are never skipped
          silently because ratio and eval-leak metrics depend on full corpus integrity.
    """
```

### 6.3 `TrainingConfig` 新フィールド

```python
# torch_ref/config.py: TrainingConfig 拡張
@dataclass
class TrainingConfig:
    # ... 既存フィールド ...
    train_corpus: str = ""                 # 既存 (互換性維持)
    val_corpus: str = ""                   # 既存
    train_corpora_mix: dict[str, float] | None = None   # 新規 (sum=1.0 strict, DR1-003)
    val_split: float = 0.05                              # 新規 (DR1-005, YAGNI)
                                                          # train pool 内から sequence-level で
                                                          # val を切り出す ratio。0.0 なら
                                                          # 既存 val_corpus にフォールバック。
                                                          # train と val で異なる比率を扱う
                                                          # ユースケースは本 Issue の範囲外

    def __post_init__(self) -> None:
        # 既存 validation ...
        # 新規 validation (S7-003 + DR1-003)
        # 全ての境界条件は strict ValueError (silent normalize / warning fallback 禁止)。
        # 同じ規則を photon_mlx/data.py::iterate_mixed_batches でも defensively に再実行する。
        for fname in ("train_corpora_mix",):  # val_corpora_mix は廃止 (DR1-005)
            mix = getattr(self, fname)
            if mix is None:
                continue
            if not isinstance(mix, dict) or len(mix) == 0:
                raise ValueError(f"{fname} must be a non-empty dict or None")
            for path, weight in mix.items():
                if not isinstance(weight, (int, float)) or not math.isfinite(weight):
                    raise ValueError(f"{fname} weight for {path} must be a finite number")
                if weight <= 0:
                    raise ValueError(f"{fname} weight for {path} must be > 0")
            total = sum(mix.values())
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError(
                    f"{fname} weights must sum to 1.0 (±1e-6). got sum={total}. "
                    "Normalize the dict explicitly in YAML; silent normalization is disabled (DR1-003)."
                )
```

### 6.4 trainer.py 分岐 (最小改修, DR1-005 反映)

```python
# photon_mlx/trainer.py: train() 内のデータロード部分のみ
# 既存 (line 297-303) の iterate_batches 呼び出しを以下で wrap
#
# 分岐は train=mix/single の 2 通りのみ (DR1-005: val_corpora_mix 廃止により
# train=mix/single × val=mix/single の 4 分岐から 2 分岐に簡素化)。
# train=mix の場合は train pool 内から val_split で sequence-level に切り出す。

if t_cfg.train_corpora_mix is not None:
    train_batches, val_batches = iterate_mixed_batches(
        t_cfg.train_corpora_mix,
        context_length=context_length,
        batch_size=batch_size,
        val_split=t_cfg.val_split,          # DR1-005: train pool 内から sequence-level に切り出す
    )
else:
    train_batches = iterate_batches(t_cfg.train_corpus, context_length, batch_size)
    val_batches = iterate_batches(t_cfg.val_corpus, context_length, batch_size, shuffle=False)
```

`val_split=0.05` の場合、train pool sequences のうち 5% を val として確保し、残り 95% を train として yield する。比率は train_corpora_mix と同一 (= JP/EN 50/50 を val でも維持)。これにより「val が train と異なる比率を持つ」ユースケースが必要になるまで 4 分岐は導入しない (YAGNI)。

### 6.5 `_build_photon_deps` の checkpoint load 分岐 (S7-001 / DR1-002)

```python
# baseline_reporag/photon_pipeline.py::_build_photon_deps
# PhotonModel 構築直後 (現行 line 308 の直後) に追加

model = PhotonModel(photon_cfg)

# S7-001 / DR1-002: model.checkpoint_path が設定されていれば load_checkpoint を呼ぶ。
# import は photon_mlx.checkpoint (training 不要、inference 専用 module) のみ。
# photon_mlx.trainer への直接 import は禁止 (Dependency Inversion 違反 / lazy MLX
# import 規約弱体化のため)。trainer.py 側は `from .checkpoint import load_checkpoint`
# として再 export し、本ファイルは新 module だけを参照する。
# missing/corrupt なら明示的に raise (silent fallback 禁止)。
# checkpoint_path は resolve(strict=True) し、approved root
# (`checkpoints/` or ACL 制限 external mount) 配下の regular directory のみ許可。
# symlink escape / `..` traversal / required file 以外の任意読み込みは禁止。
# manifest.json が存在する場合は weights.npz / state.json の sha256 を検証し、
# 不一致なら load しない (DR4-003)。
# 戻り値の CheckpointState は runtime では利用しない。trainer.py だけが
# CheckpointState <-> TrainState を変換し、既存 trainer import API を維持する。
checkpoint_path = getattr(cfg.model, "checkpoint_path", None)
if checkpoint_path:
    from photon_mlx.checkpoint import load_checkpoint as _load_ckpt
    ckpt_dir = Path(checkpoint_path)
    if not (ckpt_dir / "weights.npz").exists() or not (ckpt_dir / "state.json").exists():
        raise FileNotFoundError(
            f"PHOTON checkpoint missing required files at {ckpt_dir} "
            f"(weights.npz / state.json). Aborting eval to prevent silent fallback."
        )
	    _state = _load_ckpt(model, ckpt_dir)
	    mx.eval(model.parameters())
	```

**境界規約 (DR1-002 / DR3-001)**: `baseline_reporag/photon_pipeline.py` および `baseline_reporag/pipeline_factory.py` から `photon_mlx` への import 表面は、inference 系 module = `{model, inference, session, safe_recgen, checkpoint}` の 5 つのみとする。`photon_mlx.trainer` (TrainState / optimizer / cosine_decay / loss を top-level で参照) は baseline 起動時に引き込んではならない。`checkpoint.py` は `TrainState` を import せず、`state.json` schema を `CheckpointState` として保持する。trainer 互換 wrapper は既存 tests (`photon_mlx/tests/test_training.py::TestCheckpoint`) が `TrainState` を直接 assert しているため必須。

### 6.6 generate script の責務 3 分割 (DR1-001)

`scripts/generate_institutional_training_corpus.py` の 8 責務を以下 3 関数に分離し、`main()` は合成のみを担う。これにより §8 unit tests は LLM API を fixture でモックせずに verify / pack / leak detection を直接テストできる。

```python
# scripts/generate_institutional_training_corpus.py

@dataclass
class Session:
    session_id: str
    scenario: str           # cross_reference | drill_down | define | quantity | comparison | conclusion
    lang: str               # ja | en
    n_turns: int
    turns: list[str]        # 平文 (tokenize は _corpus_core.py 側で行う)
    source_md: str

@dataclass
class CorpusReport:
    n_sessions_requested: int
    n_sessions_succeeded: int
    failure_breakdown: dict[str, int]      # {"api_error": N, "json_parse_error": N, ...}
    eval_overlap: int                       # session_id 重複数 (= 0 必須)
    jp_sequence_ratio: float                # 制御目標 (sequence-level)
    jp_token_ratio: float                   # 実測値 (token-level)
    scenario_distribution: dict[str, float]

def build_sessions(
    corpus_dir: Path,
    n: int,
    scenarios: dict[str, float],
    *,
    llm_client: LLMClient,                  # 注入可能 (test では Mock)
    seed: int = 42,
) -> Iterator[Session]:
    """LLM 駆動の純粋生成関数。I/O・eval 検証・metadata 出力は持たない。"""
    ...

def verify_corpus(
    sessions: Iterable[Session],
    eval_path: Path,
    *,
    tokenizer: Any,                         # token-level 比率測定用
) -> CorpusReport:
    """純粋検証関数。LLM API は呼ばず CorpusReport を返す。
    
    `validate_eval_overlap(session_ids, eval_path)` は `_corpus_core.py` に定義。
    """
    ...

def main() -> int:
    """合成のみ: build_sessions → verify_corpus → _corpus_core.{tokenize,pack,val_split}
    → metadata.json 出力 → 失敗率 5% 超で exit 1。"""
    ...
```

**テスト容易性**: `verify_corpus` / `_corpus_core.validate_eval_overlap` / `_corpus_core.pack_sequences` 等は LLM 非依存で単体テスト可能。`build_sessions` のみ `llm_client` を Mock 注入する 1-2 件のテストに集約できる。

**CLI / I/O security contract (DR4-001)**:

- `--corpus-dir` は `resolve(strict=True)` 後に approved institutional root (`/Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents/`) 配下であること、かつ directory symlink でないことを確認する。`..` / symlink escape / hidden temp directory は拒否。
- `--output` は `data/training/institutional/` 配下のみ許可し、`*.tmp` に `0600` permission で書いて `verify_corpus` 成功後に atomic rename する。失敗時は partial output を cleanup し、公開可能性のある原文 / token 列を logs に出さない。
- `--sessions` は `1 <= sessions <= 5000`、`--seed` は `0 <= seed <= 2**32-1` の int、`--val-ratio` は `0.0 < val_ratio < 0.5` を strict validation し、argparse の型変換だけに依存しない。
- `build_sessions` は eval set path / eval payload を受け取らない。eval set は `verify_corpus` のみが読み、LLM provider prompt に eval examples を渡さない。

---

## 7. セキュリティ設計

| 脅威 | 対策 | 優先度 | 検出手段 |
|------|------|--------|---------|
| **JP 制度文書の機密漏洩 (corpus 平文 token 列のパブリックリポジトリ流出)** | (a) `.gitignore` に `data/training/` 追加を**本 Issue 最初の commit** で実施。(b) 派生学習物 (corpus / checkpoint) は public 配布対象外、private repo / ACL 制限 external storage のみ。(c) checkpoint 公開直前にサンプル decode を人手で確認 | **高** | pre-commit hook で `data/training/` 配下が staging に入っていないか確認 (任意)、PR diff の手動確認 |
| **eval set リーク (training corpus に eval session が混入)** | generate script 内で eval set (`data/eval_sets/institutional_multi_turn_eval.jsonl`) を読み込み、`session_id` 集合の積が空であることを **生成 step 内で assert**。非空なら exit code 1 で abort。`tests/test_generate_institutional_training_corpus.py` で pytest 必須化 | **高** | generate script 内 assert + pytest CI |
| **API キー漏洩 (LLM provider 利用時)** | `OPENAI_API_KEY` 等は環境変数のみで参照、generate script のログ・metadata は API キーをマスキング。`.env*` は既存 `.gitignore` で除外済み | 中 | 生成ログ確認、`gh secret` 経由で CI に渡す |
| **institutional_documents の外部 LLM provider 送信 (DR3-002)** | `--provider openai` 等の外部 provider は制度文書本文を第三者 API に送るため、データ分類と送信許可が明示されるまで使用禁止。既定は local / private 実行の provider (`qwen` 等) とし、外部 provider を使う場合は (a) 承認記録、(b) 最小 excerpt / redaction、(c) 送信対象 document_id / provider / timestamp の metadata 記録、(d) 生成ログに原文を出さない、を満たす | **高** | dry-run metadata review、provider 設定 review、生成ログ spot-check |
| **PHOTON+SR 出力からの training data extraction / membership inference (DR4-004)** | 採用 checkpoint は public release 禁止を既定とし、公開する場合は別 security review で (a) verbatim extraction red-team prompt、(b) canary string 非再現確認、(c) source_md ごとの k-anonymity / low-frequency 文書除外、(d) 必要なら DP/追加 redaction の適用可否を判定する。本 Issue では private / ACL 制限 storage のみ | **高** | red-team eval report、canary decode check、公開前承認 |
| **checkpoint / state.json 改竄 (DR4-003)** | checkpoint leaf に `manifest.json` を置き、`weights.npz` / `state.json` / config / corpus metadata の sha256、tokenizer_id、resume_base_step、cumulative_step を記録。load 時に hash 不一致 / symlink escape / approved root 外 path を reject。外部 storage / LFS から取得する場合は manifest hash を report に併記 | **高** | load_checkpoint smoke test、manifest verification test、LFS fetch 後 hash check |
| **corpus / checkpoint path traversal と symlink escape (DR4-001 / DR4-002 / DR4-003)** | `--corpus-dir` / `--output` / `corpus_paths` / `checkpoint_path` は全て `resolve(strict=True)` + approved root `relative_to` + regular file/dir check を通す。symlink は原則拒否し、必要な external mount は explicit allowlist でのみ許可 | **高** | path traversal / symlink pytest、PR review |
| **JSONL token payload DoS / malformed token (DR4-002)** | `tokens` は list[int]、長さ上限、token id 範囲、line byte size 上限を検証。bool / float / negative / vocab 範囲外 / 巨大配列は fail-fast。invalid line skip は禁止 | **高** | malformed JSONL pytest、peak RSS monitor |
| **dependency CVE / unpinned dependency drift (DR4-008)** | `requirements*.txt` は現状 unpinned のため、Day 1 で `pip freeze` から lock / SBOM を保存し、`pip-audit` または `osv-scanner` を実行する。High/Critical は upgrade / pin / risk acceptance を記録するまで training / release に進まない | 中 | audit log 添付、lockfile diff review |
| **mulmoclaude 600 step checkpoint 紛失** | Day 1 着手時に物理的所在を最優先で確認 (`ls checkpoints/photon_mulmoclaude/step_600/weights.npz`)。紛失時は scratch 学習 + 工数 +2-3 日 risk を受容し別 Issue 起票 | 中 | Day 1 smoke test (`load_checkpoint` で実 load 検証) |
| **Checkpoint 中間 step (1K 単位) による M3 Ultra SSD 逼迫** | (1) Day 1 に `df -h` で残量 50GB+ 確認、(2) 中間 checkpoint のうち `state.step % 1000 == 0 ∧ state.step ∉ {10000, 15000, 20000}` のものは eval 完了後 5K step 経過で `shutil.rmtree`、(3) 学習開始前に古い不要 checkpoint (mulmoclaude 以外) を archive へ退避 | 中 | 学習中の `df -h` 監視、リスク表項目 |
| **CI/self-hosted runner で新 checkpoint がフェッチ不可** | Day 5 着手前に `.github/workflows/*.yml` を確認し、`actions/checkout@v4` の `lfs: true` 有無 / external secret 有無を点検。LFS 採用なら `.gitattributes` 追加、external storage 採用なら fetch step 追加。`pytest` を CI 必須化する場合は `--ignore=data/training/` を確認 | 中 | CI ジョブ動作確認、`scripts/run_multi_turn_eval.py` smoke test |

---

## 8. テスト戦略

| 種別 | テスト対象 | 配置 | 件数目安 | Codex 反映 |
|------|----------|------|---------|-----------|
| Unit | `iterate_mixed_batches`: 50/50 sequence sampling 比率正確性 (sequence ratio 0.5±0.02 / token ratio metadata 記録の両方、DR1-007) / シード再現性 / 空 corpus → ValueError / 不正 weight reject / sum != 1.0 (±1e-6) → ValueError (DR1-003) / val_split による train/val 分割 (DR1-005) | **新規 `photon_mlx/tests/test_data.py` を作成し、`iterate_mixed_batches` 関連 unit test を集約** (DR2-003: 既存 `photon_mlx/tests/test_training.py::TestData` クラス L23-110 の `test_pack_sequences` / `test_create_batches` / `test_load_jsonl` は移動せず in-place 維持し、import 経路 `from photon_mlx.data import ...` は不変。新関数の責務分離と test 配置を一致させる SOLID 観点) | **5-6** | S3-001, S5-005, **DR1-003**, **DR1-005**, **DR1-007**, **DR2-003** |
| Unit | `iterate_batches` の signature/挙動が完全に不変であること (既存 27+ tests pass + signature snapshot test) | `photon_mlx/tests/test_data.py` (新規、snapshot 1 件追加。既存 `test_training.py::TestData` の 3 件は in-place 維持、DR2-003) | **1** | S3-001, **DR2-003** |
| Unit | `test_resume_from_continual_learning`: **前提条件として `min_lr=3e-6` (≠ 0) が yaml に記述されていること** (DR2-005、`_build_lr_schedule` の `cosine_decay(init=lr, end=min_lr)` 経路を有効化するため) / `warmup_ratio=0.0` で `lr=3e-5` 起点 / cosine decay → `min_lr=3e-6` / `state.step` が resume base + 追加 step で進む / optimizer state は新規 / 既存 600 step 重みが恒等でない (= 学習が走った) | `photon_mlx/tests/test_training.py` (新規) | **2-3** | S3-002, S5-001, S5-002, **DR2-005** |
| Unit | checkpoint 抽出互換性 (DR3-001): `photon_mlx.checkpoint.load_checkpoint` は `CheckpointState` を返し `TrainState` を import しない / `photon_mlx.trainer.load_checkpoint` は既存通り `TrainState` を返す / `save_checkpoint` は trainer 既存 callsite (`save_checkpoint(model, TrainState, path)`) と新 module callsite の両方を維持 / corrupt state.json の raise | `photon_mlx/tests/test_checkpoint.py` 新規 + 既存 `photon_mlx/tests/test_training.py::TestCheckpoint` pass (既存 checkpoint tests の移設/拡張を含め、純増は 1-2 件に抑える) | **2-3** | **DR3-001** |
| Unit | `TrainingConfig.train_corpora_mix` + `val_split` の defaults / YAML load 保持 / 不正 ratio reject (sum != 1.0 を含む、DR1-003) / 既存 yaml 後方互換 (val_corpora_mix 廃止により dict 化テストは削減、DR1-005) | `torch_ref/tests/test_model.py` (拡張) | **3-4** | S7-003, **DR1-003**, **DR1-005** |
| Unit | `generate_institutional_training_corpus.py` の 3 分割関数 (DR1-001): (a) `verify_corpus` の eval リーク検出 (= 0 / > 0 で exit code 1) / JP 比率 (sequence-level / token-level 両方) / scenario 分布計算、(b) `_corpus_core.validate_eval_overlap` 単体、(c) `main()` の dry-run / resume / 失敗率 5% 超 exit 1、(d) `build_sessions` は `llm_client` Mock 注入で 1-2 件のみ。LLM API 実呼び出しを伴うテストは 0 件 (CI 安定性確保) | `tests/test_generate_institutional_training_corpus.py` (新規) | **5-7** | S3-009, S5-003, **DR1-001** |
| Unit | Security validation (DR4-001 / DR4-002 / DR4-005): `--corpus-dir` / `--output` traversal reject、symlink escape reject、`--sessions` / `--seed` / `--val-ratio` 境界値、malformed JSONL token (負数 / bool / float / vocab 範囲外 / 巨大 line)、eval content near-duplicate/hash overlap reject | `tests/test_generate_institutional_training_corpus.py` + `photon_mlx/tests/test_data.py` | **4-6** | **DR4-001**, **DR4-002**, **DR4-005** |
| Unit | `_build_photon_deps`: `model.checkpoint_path` 設定時の `load_checkpoint` 呼び出し / missing checkpoint で `FileNotFoundError` / corrupt state.json で raise / `checkpoint_path` 未設定時の従来挙動 | `baseline_reporag/tests/test_photon_pipeline.py` (新規 or 拡張、モック checkpoint) | **3-4** | **S7-001** |
| Unit | Checkpoint security (DR4-003): `checkpoint_path` traversal / symlink escape reject、approved root 外 reject、`manifest.json` sha256 不一致 reject、state.json schema validation、LFS/external fetch 後 hash check | `photon_mlx/tests/test_checkpoint.py` + `baseline_reporag/tests/test_photon_pipeline.py` | **3-5** | **DR4-003** |
| Unit | `pipeline_factory.create_query_pipeline` photon 経路 smoke (モック checkpoint で起動) | `baseline_reporag/tests/test_pipeline_factory_lazy_mlx.py` 等 (拡張) | **1-2** | S3-006, S7-001 |
| Integration | 小規模 corpus (10 sessions) + 100 step での dry-run 学習 (回帰防止)。`corpora_mix` 経路を含む | `photon_mlx/tests/test_training_integration.py` (新規 or 既存拡張) | **1-2** | - |
| Eval | `run_multi_turn_eval` で旧 PHOTON / 新 checkpoint × 3 (10K/15K/20K) の NC / latency 比較。境界帯時 +2 run | `reports/institutional_photon_mt_eval_v2.md` (実測記録) | **6+ runs** (4-8h) | C-1, C-2 |
| Eval | FastAPI 系 retrieval regression: `configs/baseline.yaml` (provider=photon) で fastapi_fastapi 1 run | `reports/gate2_post_retrain_eval.md` (実測記録) | **1 run** (40min) | C-3 |

**CI 必須化方針 (S3-006 / S7-004 / DR3-004)**: `python -m pytest --ignore=data/training/ photon_mlx/tests/ baseline_reporag/tests/ tests/ torch_ref/tests/` を `.github/workflows/tests.yml` で実行する案を採る (PR レビューで CI / ローカル添付ログのいずれかに最終確定)。CI は `data/training/` や数 GB corpus を取得せず、すべての corpus/generate tests は `tmp_path` 上の synthetic fixture と mocked LLM client のみで実行する。既知の pre-existing failure (`tests/test_generate_training_corpus.py` 2 件) が残る場合は、修正 / `xfail(strict=True)` / 明示 allowlist のいずれかを PR 本文と実行ログに記録し、本 Issue 起因の新規 failure が 0 件であることを pass 条件にする。

**Security scan 方針 (DR4-006 / DR4-008)**: PR 前に `git diff --cached --name-only` で `data/training/` / `checkpoints/` / `.env` / provider logs が staging されていないことを確認し、可能なら `gitleaks detect --no-git --source .` 相当の secret scan を添付する。依存は unpinned のため、Day 1 に lock / SBOM (`pip freeze`) を保存し、`pip-audit` または `osv-scanner` の結果を report に添付する。

**pytest 設定の暗黙化 (DR2-010、Nice to Have)**: root 配下に `pyproject.toml` / `pytest.ini` / `setup.cfg` のいずれも存在しないため、pytest 設定は CI yaml の command line 引数 (`--ignore=data/training/ photon_mlx/tests/ baseline_reporag/tests/ tests/ torch_ref/tests/`) のみで担保される。テスト discovery 順序の不安定性 (新規 `tests/test_generate_institutional_training_corpus.py` と既存 `tests/test_generate_training_corpus.py` の collection 順 / fixture 衝突) を避けるため、`pyproject.toml` を新設して `[tool.pytest.ini_options] testpaths` を一元化する案を **follow-up Issue として起票** する。本 Issue では pyproject.toml 新設は範囲外 (DR1-011 の yaml diff test 同様、再現性最優先)。

---

## 9. 受入条件 (Issue 反映済) と達成方法

| # | 受入条件 | 達成方法 |
|---|---------|---------|
| 1 | `.gitignore` に `data/training/` 登録 (S3-003 / DR3-005) | 本 Issue 最初の commit で 1 行追加し、同時に `git ls-files data/training` が空であることを確認。既に tracked file がある場合は `git rm --cached` の対象を明示してから除外し、PR diff で corpus 平文が出ないことを確認 |
| 2 | JP 50%+ 学習 corpus 構築 (sessions ≥ 2,000、JP sequence ratio = 50% target / JP token ratio ≥ 50%±5pp 実測、cross_reference/drill_down ≥ 30%、eval 重複 0、人手 spot-check 80%、DR1-007) | `generate_institutional_training_corpus.py` の CLI と内蔵検証 + `metadata.json` (`jp_sequence_ratio` / `jp_token_ratio` 両方記録) で集計 |
| 3 | generate script 単体実行で eval リーク検出 → exit code 非ゼロ pytest (S3-009) | `tests/test_generate_institutional_training_corpus.py` で擬似リーク fixture から exit code 1 確認 |
| 4 | 新 script 関数 / CLI 引数が既存 `generate_training_corpus.py` と top-level 衝突しない (S3-004) | top-level 関数名 prefix を `_institutional_*` に統一、共通項は `_corpus_core.py` へ |
| 5 | turn delimiter `<|turn_sep|>` が special token でなく通常文字列 encode (S5-006) | generate script 内の tokenizer 動作を unit test で確認 (vocab ID が既存範囲) |
| 6 | `test_resume_from_continual_learning` 追加 (S3-002 / S5-001 / S5-002) | `photon_mlx/tests/test_training.py` に LR / step / 重み不変性 assertion |
| 7 | 既存 `iterate_batches` signature 不変 (S3-001) | snapshot test + 既存 4 callsite + 27 tests pass |
| 8 | `iterate_mixed_batches` の 50/50 sampling 検証 (S5-005) | unit test で sequence count 比率を assert |
| 9 | `TrainingConfig` mix config regression (S7-003) | `torch_ref/tests/test_model.py` 4 件追加 |
| 10 | 累計 step 10K/15K/20K 再学習完了 (S5-002) | trainer.py を unchanged で実行、checkpoint leaf path を eval/config に伝播 |
| 11 | 再学習後 PHOTON で **Turn 5-6 NC < 6%** (最低条件) | `run_multi_turn_eval` 2-run 平均、境界帯は 4-run |
| 12 | (理想) Turn 5-6 NC < 3% | 同上 |
| 13 | latency 優位 -30% 以上 (≤ 13.6s) 維持 | 同上、eval が `follow_up_p50_ms` を記録 |
| 14 | FastAPI 系 MT NC ≤ 11.7% | `configs/baseline.yaml` (provider=photon) で fastapi_fastapi 1 run |
| 15 | `reports/institutional_photon_mt_eval_v2.md` で旧/新/baseline 比較 + checkpoint metadata (S7-002) | report 生成 step に列追加 |
| 16 | 採用 checkpoint が指定保管先 (LFS / external) に存在し CI/CD で取得可能 | Day 5 で `.gitattributes` または external fetch step 追加、smoke test で確認 |
| 17 | `configs/institutional_docs_photon_retrain.yaml` 新規作成 (S3-007) | 既存 yaml は不変 |
| 18 | PHOTON runtime が採用 checkpoint を実際に load (S7-001) | `_build_photon_deps` 改修 + `baseline_reporag/tests/` smoke test |
| 19 | `pytest` 全 tests pass (約 507/509、`--ignore=data/training/`) | CI 必須化 or ローカル M3 Ultra 実行ログ添付 (S3-006 / S7-004 / DR3-004)。既知の `tests/test_generate_training_corpus.py` 2 failure を残す場合は allowlist / xfail を明記し、それ以外の regression 0 件を確認 |
| 20 | photon provider smoke test 新規 (`pipeline_factory.create_query_pipeline` 経路) | `baseline_reporag/tests/test_photon_pipeline.py` 新規 |
| 21 | docs 同時更新 (CLAUDE.md / gate2_v5 / roadmap.md / metrics.md) (S3-005) | Day 5 ロールアウト工程で commit |
| 22 | (Nice to Have) `docs/deployment.md` / `docs/troubleshooting.md` 更新 (S3-011) | 本 Issue 範囲で困難なら follow-up Issue 起票 |
| 23 | Security validation pass (DR4-001 / DR4-002 / DR4-003 / DR4-005) | traversal / symlink / malformed token / checkpoint manifest / eval near-duplicate の pytest 追加と pass |
| 24 | Secret / dependency audit pass (DR4-006 / DR4-008) | `data/training` / `checkpoints` / `.env` / provider logs の staged diff 0、secret scan、`pip-audit` or `osv-scanner` 結果添付 |
| 25 | checkpoint 公開禁止または公開前 security review 完了 (DR4-004) | 本 Issue は private / ACL 制限 storage のみ。public release が必要なら別 security review で extraction / membership inference / canary check を完了 |

---

## 10. 影響範囲

| ファイルパス | 改修種別 | 責務 | テスト追加 | 回帰リスク |
|------------|---------|------|----------|----------|
| `.gitignore` | 拡張 | `data/training/` 除外 | - | 極低 |
| `photon_mlx/data.py` | 拡張 (新関数のみ) | `iterate_mixed_batches` 追加、既存 `iterate_batches` 不変 | **新規 `photon_mlx/tests/test_data.py` 作成 (5-7 件、`iterate_mixed_batches` 関連 + signature snapshot)。既存 `photon_mlx/tests/test_training.py::TestData` クラス L23-110 (`test_pack_sequences` / `test_create_batches` / `test_load_jsonl`) は移動せず in-place 維持 (DR2-003)** | 低 (既存 callsite 4 + tests 27 を守る snapshot test) |
| `photon_mlx/trainer.py` | 拡張 (最小、分岐追加のみ) | `train_corpora_mix` が dict なら `iterate_mixed_batches` を使う分岐。schedule/checkpoint/resume_from は不変 | `photon_mlx/tests/test_training.py` 拡張 (`test_resume_from_continual_learning` 等 2-3 件) | 中 (既存 168 tests を守る、誤改修ガード必須) |
| `torch_ref/config.py` | 拡張 | `TrainingConfig.train_corpora_mix` + `val_split` 追加 + validation (DR1-005 / DR3-003: `val_corpora_mix` は追加しない) | `torch_ref/tests/test_model.py` 拡張 (4 件) | 低 |
| `baseline_reporag/photon_pipeline.py` | 拡張 (S7-001 / DR1-002) | `_build_photon_deps` で `model.checkpoint_path` を `photon_mlx.checkpoint.load_checkpoint()` で読む。`photon_mlx.trainer` への直接 import は禁止 | `baseline_reporag/tests/test_photon_pipeline.py` 新規/拡張 (3-4 件、import 表面の snapshot test 1 件追加) | **高** (PHOTON eval 全経路の起点、smoke test 必須) |
| `photon_mlx/checkpoint.py` | 新規 (DR1-002 / DR3-001) | `load_checkpoint(model, path)` / `save_checkpoint(model, state, path)` の物理実装を trainer.py から本 module へ移管。`CheckpointState` を定義し `TrainState` は import しない。baseline_reporag は本 module のみを参照し、runtime は戻り state を利用しない | `photon_mlx/tests/test_checkpoint.py` 新規 (2-3 件、既存 checkpoint tests の移設/拡張 + `CheckpointState` / trainer wrapper 等価性 / corrupt state) | 中 |
| `photon_mlx/trainer.py` (再 export / wrapper) | 拡張 (DR1-002 / DR3-001) | 既存 `load_checkpoint` / `save_checkpoint` API は後方互換維持。`checkpoint.py` の `CheckpointState` と既存 `TrainState` を変換する薄い wrapper を置き、既存 `photon_mlx/tests/test_training.py::TestCheckpoint` と外部 callsite を破壊しない | 既存 trainer tests pass + checkpoint wrapper test | 中 |
| `baseline_reporag/pipeline_factory.py` | 確認のみ (改修なし) | photon provider 分岐 | `baseline_reporag/tests/test_pipeline_factory_lazy_mlx.py` 拡張 (1-2 件) | 極低 |
| `scripts/generate_institutional_training_corpus.py` | 新規 | corpus 生成 CLI + eval リーク検証内蔵 | `tests/test_generate_institutional_training_corpus.py` 新規 (5-7 件) | 低 |
| `scripts/_corpus_core.py` | 新規 (private module) | tokenize / pack / val split 共通化 | 上記 test 経由でカバー | 低 |
| `scripts/generate_training_corpus.py` | **本 Issue では触らない** (DR1-004) | follow-up Issue で `_corpus_core.py` 経由に書き換え。現状は新 script のみが core を利用し、既存 script は legacy のまま並存。Day 5 中に follow-up Issue を **必ず起票** し本 Issue close と同時に link する (放置禁止) | follow-up Issue で対応 (回帰テスト = mulmoclaude corpus 再生成 diff 0) | 0 (本 Issue 範囲) |
| `configs/institutional_docs_photon_retrain.yaml` | 新規 | 学習用 hyperparam + leaf checkpoint_path | YAML load test (#9) | 低 |
| `configs/institutional_docs_photon.yaml` | **不変** (S3-007) | #112 / #126 baseline 再現用に保護 | - | 0 |
| `data/training/institutional/` | 新規 (機密、`.gitignore` 除外) | corpus 実体 | - | 0 (commit 対象外) |
| `checkpoints/photon_institutional_retrain_<yyyymmdd>/` | 新規 (LFS or external) | 学習成果物。`manifest.json` に sha256 / tokenizer_id / config hash / corpus metadata hash / step metadata を記録し、public release は別 security review まで禁止 (DR4-003 / DR4-004) | manifest verification test | 中 (CI fetch 経路、改竄検出必須) |
| `reports/institutional_photon_mt_eval_v2.md` | 新規 | 比較表 + checkpoint metadata 列として `resume_base_step` / `additional_updates` / `cumulative_step` の 3 列を強制 (S7-002 / DR1-008)。**旧 `reports/institutional_photon_mt_eval.md` (#113) は履歴として保持、_v2.md は冒頭で旧 report への cross-link を持ち、両 report は並列保持 (DR2-007、`reports/gate2_judgment_v4_final.md` → v5 と同一パターン)** | - | 0 |
| `reports/gate2_post_retrain_eval.md` | 新規 | FastAPI 系 retrieval 再 eval | - | 0 |
| `reports/gate2_judgment_v5_post_retrain.md` | 新規 | v4→v5 cross-link (S3-005) | - | 0 |
| `CLAUDE.md` (L156-160) | 拡張 | 現在のメトリクス書換 | - | 0 |
| `workspace/mvp/roadmap.md` | 拡張 | 「>6% 分岐実行済」マーキング | - | 0 |
| `workspace/mvp/metrics.md` | 拡張 | 再学習成果反映 | - | 0 |
| `.github/workflows/tests.yml` | 新規 (任意、S7-004) | `pytest --ignore=data/training/` CI 必須化 | - | 低 |
| `.github/workflows/weekly_eval.yml` | 確認のみ (S3-006) | LFS 取得設定確認 | - | 低 |
| `docs/deployment.md` / `docs/troubleshooting.md` | 拡張 (Nice to Have、S3-011) | checkpoint fetch 手順 / 不在時対処 | - | 0 |

---

## 11. リスクと緩和策

| リスク | 影響度 | 緩和策 | 検出 |
|--------|--------|--------|------|
| **PHOTON eval/runtime が新 checkpoint をロードしない (S7-001)** | **高** (Issue 全工程が無意味化) | Day 1 に `_build_photon_deps` 改修 + `baseline_reporag/tests/test_photon_pipeline.py` smoke test を**学習開始前**に CI に乗せる。新 yaml 経由 load を必須テスト化 | smoke test 失敗時は実装修正、リリース blocker |
| **Catastrophic forgetting (英語 mulmoclaude 性能低下 -10pp 超)** | 高 | mix ratio 50/50 厳守、英語 loss を別 metric として追跡。1K step 毎に FastAPI MT eval 実施し最良 checkpoint を dynamic に選定 | 1K step 毎の英語 loss curve、FastAPI MT NC の 5pp 超え検出で early stop |
| **mulmoclaude 600 step checkpoint 紛失** | 中 | Day 1 着手時に物理パス (`checkpoints/photon_mulmoclaude/step_600/`) を最優先で確認、欠損時は別 Issue 起票し scratch 再学習 (工数 +2-3 日 risk 受容) | Day 1 の `load_checkpoint` smoke test |
| **`resume_from` が想定外 LR 起点を取り 600 step 重みを破壊 (S3-002 / S5-001 / S5-002 / DR2-005)** | 高 | `test_resume_from_continual_learning` を**学習開始前**に CI に追加、**`warmup_ratio=0.0` ∧ `min_lr=3e-6` (≠ 0) のとき、`_build_lr_schedule` が `cosine_decay(init=3e-5, end=3e-6)` パス (trainer.py:178-181) を取り、初回 update から `lr=3e-5` 起点で `min_lr=3e-6` へ cosine decay すること**、`state.step` が resume 元 + 追加 step で進むことを CI で恒常確認。**`min_lr=0` だと L182-183 のフォールバック (`else: return lr`) で定数 LR になり cosine decay が effectively 無効化される事実を併記** (DR2-005) | unit test、初回 1K step の lr-finder で val_loss spike 検出 |
| **学習 corpus institutional_documents の機密漏洩 (S3-003 / DR3-005)** | 高 | `.gitignore` に `data/training/` を**最初の commit で**追加 + `git ls-files data/training` が空であることを確認。checkpoint / corpus は public 配布対象外 (private repo / ACL 制限) | pre-commit hook 確認、`git status --short` / `git diff --cached --name-only` / PR diff レビュー |
| **制度文書本文が外部 LLM provider に送信される (DR3-002)** | 高 | 外部 provider (`openai` 等) はデータ分類と送信許可が明示されるまで使用禁止。既定は local / private provider とし、外部送信時は redaction / 最小 excerpt / metadata audit を必須化 | generate script dry-run metadata、provider flag review、生成ログ spot-check |
| **eval set リーク / content near-duplicate bypass (DR4-005)** | 高 | generate script 内蔵検証は `session_id` 重複だけでなく、eval prompt / answer / source_md の normalized hash と 5-gram Jaccard などの near-duplicate 判定を含める。`build_sessions` は eval set を読まず、LLM provider prompt に eval examples を渡さない | リーク検出時は corpus 再生成、near-duplicate pytest |
| **GPU 占有 3-5 日 + #137 競合 (S7-005 / DR3-006)** | 中 | #137 を**先行完了**し採用/非採用を `configs/institutional_docs.yaml` と reports に反映してから #135 Day 3 学習開始。両者の実機 eval/training は同時実行しない。#133 (PR #136 merged) の rerank A/B (CPU) は Day 4-5 eval phase と並列可。`commandmatedev` / worktree status 監視が失敗または利用不能なら fail-closed とし、`git fetch`、`git status --short`、`gh issue view 137 --repo Kewton/photon-mlx --json state`、関連 config diff 確認を手動で通すまで Day 3 学習を開始しない | commandmatedev 監視 + fallback command のログ、Day 3 着手前に #137 close 確認 |
| **self-hosted runner / commandmatedev 同時アクセスによる workspace 汚染 (DR4-007)** | 中 | training/eval は repo local の lockfile (`.locks/issue-135-train.lock`) または `flock` で serialize し、runner workspace は issue 固有 worktree / run id 配下に分離する。開始時に `git status --short` が expected files のみであること、終了時に `data/training` / `checkpoints` / logs の permission と cleanup を確認 | lock acquisition log、run id workspace、pre/post git status |
| **65,536 tokens/step による GPU 占有時間過小評価 (S7-006 / DR1-010 / DR2-004)** | 中 | **原理計算 (DR1-010、order-of-magnitude check)**: M3 Ultra で PHOTON 階層 (≈ 14B 4-bit + bottom-up/top-down) の forward+backward が **~3-6 sec/optimizer step** (= 16 micro × forward+backward = ~190-380 ms/micro / `micro_batch_size=2` (effective batch_size = 2 × 16 = 32) / context_length=2048 / 1 optimizer step あたり累計 65,536 tokens) と仮定すると、累計 20K **optimizer step** = 60,000-120,000 sec = **17-33 GPU-hours ≈ 0.7-1.4 日** (DR2-004: `batch_size=4` 表記を `micro_batch_size=2 (effective batch=32)` に修正、'sec/step' は **optimizer step (= 16 micro-step を集約した 1 optimizer update)** あたりの時間)。10K optimizer step なら 8.5-17h ≈ 0.4-0.7 日。設計工数 3-5 日のうち学習所要は 1-2 日、残りはデータ準備 / eval / report に割当。Day 3 の **1K step lr-finder で sec/optimizer-step を実測** (集計単位は optimizer step、micro-step ではない) し原理計算と乖離 (≥ 2x) があれば再見積り。5 日超なら `grad_accum=4` (16K tokens/step に抑制) / 10K early stop / 夜間実行へ切替 | lr-finder の elapsed_s ログ (optimizer step 単位)、原理計算との乖離率を記録 |
| **Checkpoint 容量逼迫 (1K step × 10-20 = 7.5-15GB)** | 中 | 1K step 中間 checkpoint は eval 完了後 5K step 経過で `shutil.rmtree`、最終 3 ポイント (10K/15K/20K) のみ保持。Day 1 に `df -h` で残量 50GB+ 確認 | df 監視、リスク表項目 |
| **CI で新 checkpoint 取得不可 (S3-006 / S7-004)** | 中 | Day 5 着手前に `.github/workflows/*.yml` 確認、LFS 採用 (`.gitattributes` `checkpoints/photon_institutional_*/**`) または external fetch step 追加。CI で `pytest --ignore=data/training/` を必須化 | CI ジョブ動作確認 |
| **再学習で Turn 5-6 NC < 6% 達成失敗** | 中 (MVP Phase 2 失敗) | (1) 3-6% 帯の場合、本 Issue 内で**追加軽量 fine-tune 5K step** (compute 余裕あれば、`workspace/mvp/roadmap.md` L88-94 conditional 表 3-6% 行 (軽量 fine-tune 5K step) と > 6% 行 (本格再学習 10-20K step) の両方を参照。本 Issue 採用結果に応じて該当行に「実行済」マーキング、DR2-011)。(2) それでも < 6% に至らない場合、Phase 2 完了基準を「Turn 5-6 NC < Y%」に再定義する別 Issue を起票し本 Issue は close (S3-012) | C-2 採用判定で確定、最低条件未達で発動 |
| **学習データ品質不足 (前 turn 参照成立率 < 80%)** | 中 | 人手検証 20 サンプル spot-check、低品質 session 除外 + generate prompt 修正 | spot-check 結果、metadata の scenario 分布 |
| **generate script の失敗率閾値設計 (DR1-009)** | 中 | 失敗カテゴリを 4 種に分解 (`api_error / json_parse_error / scenario_misclassified / token_overflow`) し、**5% 閾値は `api_error + json_parse_error` の和のみで計算** (LLM 側不可避な不安定性のみ)。`scenario_misclassified` は scenario 別 retry で補正、`token_overflow` は事前 truncate で対処。閾値超過時は exit 1 で abort、それ以下は達成数 (例: 1,985 / 2,000 要求) で続行 (≥ 2,000 要件は再実行で対処)。metadata.json に `n_sessions_requested / n_sessions_succeeded / failure_breakdown` の 3 項目で内訳記録 | metadata.json `failure_breakdown` をレビュー時に必ず確認 |
| **`corpora_mix` の silent normalize による比率ズレ (DR1-003)** | 中 | `TrainingConfig.__post_init__` と `iterate_mixed_batches` で sum=1.0 (±1e-6) を strict 要求し、不一致は早期 `ValueError`。warning fallback / silent normalize は明示的に禁止。`load_photon_config` は relative path を YAML ファイルのディレクトリ基準で絶対化する規約 | unit test (sum=1.5 で ValueError、sum=0.999999 で OK)、yaml load 後 path が絶対であることを assert |
| **`iterate_mixed_batches` の list 返却で sequence pool 全件 mx.array 化が peak memory を逼迫 (DR1-006)** | 低-中 | 想定 sequence pool: 5,000 sessions × ~4K tokens × bf16 ≈ 40 MB、context_length=2048 で pack 後 < 1 GB に収まる試算。Day 3 lr-finder で peak RSS を実測し、3 GB 超なら `Iterator[mx.array]` に切替 (新関数につき signature 縛りなし)。trainer の `for b in batches` ループは list / iterator どちらも透過 | lr-finder の peak RSS メトリクス (`psutil.Process().memory_info().rss`) をログ |
| **他 PR との merge conflict (S3-008)** | 低 | Day 3-4 学習期間中は `photon_mlx/trainer.py` / `photon_mlx/data.py` / `configs/institutional_docs_photon*.yaml` への外部 PR マージを保留。Day 5 ロールアウト前に `develop` rebase + 全テスト再実行 | git status 監視 |

---

## 補足: Day 別タスクサマリ (実装計画フェーズ #4 への申し送り)

| Day | 主要タスク | 完了 gate |
|-----|----------|---------|
| Day 1 | (1) `.gitignore` 更新 + 最初の commit、(2) `mulmoclaude/step_600` 物理確認、(3) `_build_photon_deps` の checkpoint load 改修 + smoke test (**S7-001 を最優先**)、(4) `df -h` 残量確認、(5) `_corpus_core.py` 抽出、(6) dependency audit / secret scan / lockfile 方針確定 (DR4-006 / DR4-008) | smoke test pass、checkpoint load 失敗 path も test 済、audit log 添付、security validation 方針確定 |
| Day 2 | (6) `generate_institutional_training_corpus.py` 実装 + eval リーク検証内蔵、(7) corpus 生成 (2,000 sessions) + JP 比率 / scenario 分布 確認、(8) `iterate_mixed_batches` + `TrainingConfig.train_corpora_mix` + `val_split` + unit tests (DR3-003: `val_corpora_mix` は追加しない) | corpus metadata.json で全条件達成、unit tests pass |
| Day 3 | (9) `configs/institutional_docs_photon_retrain.yaml` 作成、(10) self-hosted runner / commandmatedev lock 取得 (DR4-007)、(11) **lr-finder 1K step small run** (lr=1e-5/3e-5/1e-4) で sec/step 実測 + ETA 再計算、(12) ETA OK なら本走行開始 (累計 step ~10K まで) | lr 確定、ETA ≤ 5 日、lock 取得ログあり |
| Day 4 | (12) 累計 step 15K → 20K 学習完了、(13) 各 checkpoint で `run_multi_turn_eval` 2-run、(14) 境界帯時 +2 run、(15) FastAPI 系 regression eval | 採用候補確定 |
| Day 5 | (16) `_build_photon_deps` 経由で採用 checkpoint load smoke test、(17) reports v2 / gate2_post_retrain / gate2_v5 作成、(18) docs 更新 (CLAUDE.md / roadmap.md / metrics.md)、(19) develop rebase + 全テスト再実行、(20) PR 作成 | 全受入条件達成 |
