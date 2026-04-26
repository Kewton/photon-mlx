## 背景

#113 (PR #134 merged, `e281660`) で **現行 PHOTON が制度文書ドメインで baseline より精度低い** ことが実測で確定:

| 指標 | baseline avg | PHOTON avg | Δ |
|------|-------------|-----------|---|
| NC overall | 6.94% | 11.39% | +4.44pp 悪化 |
| **NC Turn 5-6** | **5.00%** | **10.83%** | **+5.83pp 悪化** |
| follow-up latency p50 | 19,426ms | 10,707ms | **-44.9% 改善** ✅ |

設計 §9 仮説 C (Turn 5-6 NC > 6%) 該当 → **本格再学習 (10-20K step、JP 50%+、5 日工数)** を実施。

#117 Epic Phase 2 conditional 表の最重い分岐に該当。本 Issue で対応。

### 既存 mulmoclaude checkpoint の取扱
- **想定パス**: `checkpoints/photon_mulmoclaude/step_600/` (val_loss=0.4525)
- **取得方法**: ローカル M3 Ultra 上に配置済み想定。物理的所在の最終確認は実装担当が Day 1 着手時に行い、欠損時は別途 Issue 起票して scratch 再学習へ pivot。
- **resume 方針**: `photon_mlx/trainer.py` の `resume_from` パラメータに上記パスを渡す。**optimizer state は新規生成 / LR scheduler は新規 cosine** を割り当てる (continual learning で既存 600 step の重みは保持しつつ、schedule は本 Issue 用に reset)。
- **checkpoint 紛失時の risk**: scratch から再学習となり工数 +2-3 日 (リスク表に追記)。

## ゴール

PHOTON checkpoint を制度文書ドメインに適合させ、以下を達成する:
- **Turn 5-6 NC < 6%**（最低基準）
- **Turn 5-6 NC < 3%**（理想）
- **現行 PHOTON の baseline に対する latency 優位 -44.9% を維持** (具体的には follow-up p50 ≤ baseline×0.7 = **13.6s 以下** = -30% 以上 vs baseline)

## アプローチ

### A. データ準備（Day 1-2）

#### A-0. 機密保護とリポジトリ衛生 (本 Issue 最初の commit)
- **`.gitignore` に `data/training/` を追加** (本 Issue の最初の commit に含めることを必須とする)。これにより JP 制度文書 corpus (decode 可能な token 列を含む、機密含み得る) の誤 commit / リポジトリ肥大化を防止する。
- **機密前提**: 本 corpus は institutional_documents (JP 制度文書) を tokenize したものであり、平文 token 列のままパブリックリポジトリへ push 不可。後述 D 節の checkpoint 配布時も**派生学習物 (corpus / checkpoint) は public 配布対象外**とし、private repo / ACL 制限のある external storage のみで取扱う。

#### A-1. 学習 corpus 構築
- **既存 mulmoclaude (英語コード) 50%** を保持（catastrophic forgetting 回避）
- **制度文書 JP 50%+ 追加**: institutional_documents 4228 md から学習用 turn 系列を生成
  - **source**: `institutional_documents` raw corpus directory (既存 eval set と同じ元文書集合)。標準入力は `--corpus-dir <institutional_documents_dir>` または `INSTITUTIONAL_CORPUS_DIR` とし、`data/eval_sets/institutional_multi_turn_eval.jsonl` は **eval リーク検証入力としてのみ使用** する
  - **標準 CLI (S5-003 反映)**: `python scripts/generate_institutional_training_corpus.py --corpus-dir <institutional_documents_dir> --provider auto --sessions 2000 --output data/training/institutional --val-ratio 0.1 --seed 42`
    - `--provider`: `auto|openai|qwen` (既存 `generate_institutional_eval_set.py` と同じ provider 選択方針)
    - `--sessions`: 初期値 2000、上限目安 5000
    - `--resume` / `--dry-run`: eval set generator と同等の挙動を持たせる
    - 失敗率が 5% を超える場合は exit code 1 とし、生成品質不足として再実行または prompt 修正
  - **目標 sessions 数**: **≥ 2,000** (初期値 / 目標下限。JP token 比 50% を満たす最小値で算出。上限は 5,000 を目安にし、生成コストとデータ品質に応じて調整)
  - **eval set との重複禁止**: `data/eval_sets/institutional_multi_turn_eval.jsonl` (30 sessions × 6 turns) と `session_id` 重複ゼロを検証スクリプトで強制
  - **eval リーク検証の実装責務 (S3-009 反映)**: 新 `scripts/generate_institutional_training_corpus.py` の **出力 step 内で eval set を読み込み、session_id 集合の積が空であることを assert し、非空なら exit code 1 で abort する** (検証は generate script 自身に内蔵し、別 script への分離はしない)。
  - **generation script (S3-004 反映)**: 新規 `scripts/generate_institutional_training_corpus.py` を追加。既存 `scripts/generate_training_corpus.py` (FastAPI 系 chunk → token corpus 生成) と **共通ロジック (tokenize / pack_sequences / val split / `validate_tokenizer_id` / `resolve_tokenizer_id`) は `scripts/_corpus_core.py` (新設、private module) に抽出** し、新 script は institutional 固有の (a) session 構築 (turn_sep 挿入) (b) eval set との session_id 重複チェック を上乗せする形に分離する。既存 `generate_training_corpus.py` / `generate_institutional_eval_set.py` の top-level 関数名と衝突しないこと。重複実装を選ぶ場合は follow-up Issue で共通化する旨を明記。
  - **出力 schema (S5-006 反映)**: 1 session = 1 doc。`{"tokens": [...], "session_id": "...", "scenario": "cross_reference|drill_down|...", "lang": "ja"}` 形式の JSONL。turn 境界は **既存 tokenizer で通常文字列として encode する delimiter** `<|turn_sep|>` で区切る (新 special token は追加しない。tokenizer vocab / embedding resize は本 Issue 範囲外)。turn 単位の文脈を 1 session 内に保持する。
  - **photon_mlx/data.py 拡張 (S3-001 / S5-005 反映: 後方互換固定)**: **新関数 `iterate_mixed_batches(corpus_paths: dict[str, float], context_length, batch_size, ...)` を追加し、既存 `iterate_batches()` の signature は変更しない**。これにより既存 4 callsite (`photon_mlx/trainer.py:297-303` の 2 箇所、`scripts/train_photon_quick.py:48-49` の 2 箇所) と `photon_mlx/tests/test_training.py:23` 系既存 unit test (`test_pack_sequences` / `test_create_batches` / `test_load_jsonl` 等 27+ 件) は無改修で動作する。`torch_ref/config.py` の `TrainingConfig` に新フィールド `train_corpora_mix: dict[str, float] | None` と `val_corpora_mix: dict[str, float] | None` を追加し、`photon_mlx/trainer.py` 側は当該値が `None` のときは従来通り `iterate_batches` を呼び、`dict` のとき `iterate_mixed_batches` を呼ぶ分岐とする。
    - 実装契約: corpus ごとに `load_jsonl -> pack_sequences` で sequence pool を作り、micro-batch 生成時に pool 単位で weighted sampling する。JP/英語コード比率は **yield された sequence 数** を基準に 50/50 とし、実測 token 比も report に記録する。
    - `pack_sequences` は corpus 間を跨がない。JP session 同士の連結を許可する場合も、少なくとも delimiter で session 境界を明示する。
    - validation は `val_corpora_mix` が指定された場合に同じ混合比で作成し、未指定なら従来 `val_corpus` の単一 corpus を使う。
  - **規模 / 比率の目安**: 数千 sessions × 4-6 turns、1 session あたり ~4K token、総 token ~数千万 token (mulmoclaude と合算で 50/50)

#### A-2. データ品質
- 各 turn が前 turn を「真に参照する」依存関係を持つ session を優先
- cross_reference / drill_down シナリオ重視（#113 で最も悪化した 2 つ）
- session 内多様性: define → quantity → comparison → conclusion 等

### B. 学習実行（Day 3-4）

#### B-1. ハイパーパラメータ

**初期値 (small run で再調整可)**:
- `lr=3e-5` (初期値、mulmoclaude の 1.5e-4 を 1/5 にして既存重み破壊を抑制)
- `micro_batch=2`
- `grad_accum=16`
- `warmup_ratio=0.0` (continual learning のため warmup を切る)
- `min_lr=3e-6`
- `cosine schedule は continual で reset` (新規 cosine を本 Issue の max_steps 用に張り直し)。**既存 `_build_lr_schedule()` の挙動に合わせ、warmup_ratio=0.0 では初回更新が `lr=3e-5`、以後 `min_lr=3e-6` へ cosine decay する** (min_lr→max_lr の warmup は行わない)

**確定手順**:
- (a) Day 3 着手時に **lr-finder (1K step の small run、lr=1e-5 / 3e-5 / 1e-4 の 3 候補)** で val_loss curve を比較し、最良の lr を採用
- (b) micro_batch / grad_accum は M3 Ultra の OOM 限界で max を取る (初期値 2×16 = effective batch 32)
- (c) **Step 数 = 10K-20K (S5-002 反映)**: 本文中の 10K/15K/20K は `TrainState.step` の **累計 step** を指す。`resume_from=step_600` から実行するため、追加 optimizer updates はそれぞれ約 9.4K / 14.4K / 19.4K。追加 step 数として 10K/15K/20K を厳密に扱う必要が出た場合は `max_steps=10600/15600/20600` に変更し、checkpoint 名にも累計 step を明記する。
- (d) **Token 数見積もり (S5-004 反映)**: 初期値では 1 optimizer step あたり `micro_batch_size * gradient_accumulation_steps * context_length = 2 * 16 * 2048 = 65,536 tokens`。累計 20K step まで実行した場合の総処理量は約 1.31B tokens (resume 前 600 step 分を含む概算)。16K token/step に抑える必要が出た場合は `grad_accum=4` 相当に変更する。
- (e) **resume_from**: `checkpoints/photon_mulmoclaude/step_600/` (背景セクション参照)。optimizer state は新規、LR scheduler は新規 cosine

**trainer.py の改修方針 (S3-004 反映)**:
- `photon_mlx/trainer.py` の **既存 `resume_from` / cosine schedule ロジックをそのまま利用** する。本 Issue では trainer.py 本体の schedule / checkpoint ロジック改修は不要 (新フィールド `train_corpora_mix` / `val_corpora_mix` への分岐追加のみで、`load_checkpoint` / `_build_lr_schedule` / `train` の既存 path は変更しない)。意図せざる変更で既存 168 tests が破綻しないようガードする。

#### B-2. checkpoint 保存
- 1K step 単位で intermediate checkpoint (debug 用、容量管理のため未採用 checkpoint は早期削除可)
- **本番採用候補 (eval 対象)**: **累計 step 10K, 15K, 20K の 3 ポイントのみ** で eval 実行 (`step_010000` / `step_015000` / `step_020000`)

### C. eval 実行（Day 4-5）

#### C-1. 各 checkpoint で MT eval
- #113 と同じ pipeline (`scripts/run_multi_turn_eval.py`)
- 同じ eval set (`data/eval_sets/institutional_multi_turn_eval.jsonl`)
- **2-run × 各 checkpoint = 6+ runs** (#113 踏襲)

#### C-2. 比較指標と採用基準
- vs baseline (NC 6.94% / Turn 5-6 5.00% / latency 19.4s)
- vs 学習前 PHOTON (NC 11.39% / Turn 5-6 10.83% / latency 10.7s)
- **採用基準（単一指標最適化）**:
  - 最低条件: **Turn 5-6 NC < 6% AND follow-up p50 ≤ baseline×0.7 = 13.6s** を満たす checkpoint のうち、**Turn 5-6 NC が最小のもの** を採用
  - 最低条件を満たす checkpoint が無い場合は **Phase 2 pivot** (リスク表項目「再学習で Turn 5-6 NC < 6% 達成失敗」) を発動
- **統計的決定ルール**: Turn 5-6 NC が **境界帯 (5-7%)** の場合、追加で **+2 run** 実施し **4-run 平均で再判定** (eval set 30 sessions × Turn 5-6 = 60 turns で 1 turn = ±1.67pp の noise を考慮)

#### C-3. FastAPI 系 retrieval 性能の regression 確認
- **対象 corpus**: `fastapi_fastapi`
- **eval script**: `scripts/run_multi_turn_eval.py` + `configs/baseline.yaml` (photon provider 切替)
- **測定指標**: **MT no-citation rate** (gate2 v4 PHOTON+SR ベンチマーク 6.7% を base)
- **判定**: **+5pp 以内 (= 11.7% 以下)** で維持
- 併せて static NC・latency も同条件で記録 (補助指標)

### D. 採用判定とロールアウト（Day 5）

- **checkpoint ディレクトリ命名規則**: `checkpoints/photon_institutional_<step>_<yyyymmdd>/` (例: `checkpoints/photon_institutional_15000_20260428/`)
- **保管先判定**:
  - checkpoint 1 個 = ~750MB (small 377M, fp16)、採用 1 個のみ残す場合 → **git LFS 採用** (.gitattributes に `checkpoints/photon_institutional_*/**` 追加、既存 LFS 設定との衝突無し)
  - 採用候補 (10K/15K/20K) を全て保管する場合 = ~2.3GB → **external storage** (local NAS / S3 のいずれか、README に取得手順を記載)
  - **判定基準**: 採用後に残す checkpoint 数 × ~750MB が **1.5GB を超える場合は external storage**、それ以下なら git LFS
- **config 更新 (S3-007 / S5-005 反映: 後方互換)**: 既存 `configs/institutional_docs_photon.yaml` (#112 / #126 baseline 再現用) は **直接編集せず維持** し、本 Issue 用には **新規 `configs/institutional_docs_photon_retrain.yaml` を作成** する。新 yaml で `model.provider` (photon) と `paths.checkpoint_root` 配下の新パス (例: `./checkpoints/photon_institutional_15000_20260428/`)、および新 hyperparam (`lr=3e-5` / `warmup_ratio=0.0` / `max_steps=10000-20000` / `train_corpora_mix` / `val_corpora_mix`) を持つ。これにより #112 / #126 既存 eval の再現性を保つ。
- **CI/CD 事前確認 (S3-006 反映)**: **Day 5 着手前に GitHub Actions のジョブ定義を確認** (`.github/workflows/*.yml` で `actions/checkout@v4` の `lfs: true` 有無 / external secret 有無)。CI が新 checkpoint を取得不可なら**本 Issue 範囲内で CI 設定変更も併せて行う** (LFS 有効化 or external storage fetch step 追加)。`pytest` 全 tests 実行時に `data/training/` 以下が ignore される (CI で `--ignore=data/training/` 確認) こと。
- **pipeline_factory.py 整合**: photon provider 経路が新 checkpoint を pickup することを smoke test で確認。`baseline_reporag/tests/` に **モック checkpoint で `pipeline_factory.create_query_pipeline` の photon provider 経路 (pipeline_factory.py:44-67 / `_build_photon_deps`) を起動できる unit test** を新規追加する。
- **比較表出力**: `reports/institutional_photon_mt_eval_v2.md` に **旧 PHOTON 値・新 PHOTON 値・baseline 値** を 1 表で並べる
- **FastAPI 系 retrieval 再 eval**: `configs/baseline.yaml` + photon provider で同 checkpoint を eval し、`reports/gate2_post_retrain_eval.md` (新規) に記録
- **採用 checkpoint 確定後のドキュメント更新 (S3-005 反映)**:
  - `CLAUDE.md` L156-160「現在のメトリクス (Gate 2 v4)」を新 PHOTON 値で更新 (Static NC / MT NC / latency)
  - `reports/gate2_judgment_v4_final.md` は履歴として保持し、**新規 `reports/gate2_judgment_v5_post_retrain.md` を作成** (冒頭で v4 → v5 の関係を cross-link)
  - `workspace/mvp/roadmap.md` Phase 2 conditional 表の「**>6% 分岐実行済**」マーキングを追加
  - `workspace/mvp/metrics.md` に再学習成果反映
- **chunker / retrieval は変更しない** (本 Issue は LM checkpoint のみ差し替え)

## 影響ファイル

- `.gitignore` (新エントリ `data/training/` 追加、本 Issue 最初の commit、S3-003 反映)
- `photon_mlx/trainer.py` (**既存 `resume_from` / `_build_lr_schedule` / `train` の schedule / checkpoint ロジックは改修不要**。新フィールド `train_corpora_mix` / `val_corpora_mix` への分岐追加のみ。S3-004 / S5-001 / S5-002 / S5-005 反映で誤改修ガード)
- `photon_mlx/data.py` (**新関数 `iterate_mixed_batches(corpus_paths: dict[str, float], ...)` を追加し、既存 `iterate_batches()` の signature は変更しない**。corpus ごとの sequence pool から weighted sampling し、単純 concat 後 pack はしない。S3-001 / S5-005 反映で後方互換固定)
- `photon_mlx/tests/test_training.py` (**新規 test 追加**: (a) `test_resume_from_continual_learning` - resume_from + warmup_ratio=0.0 + cosine reset で step counter が初期値+1 / 既存重みが恒等でないこと / optimizer state が新規 / LR が既存実装通り `lr` 起点で cosine decay することを assert (S3-002 / S5-001 / S5-002 反映)、(b) `iterate_mixed_batches` 単体テスト: 50/50 sequence sampling、既存 `iterate_batches()` signature 不変、val mix 分岐)
- `tests/test_generate_institutional_training_corpus.py` (新規、eval リーク検出時に exit code 非ゼロになる pytest を含む、S3-009 反映)
- `torch_ref/config.py` (`TrainingConfig` に `train_corpora_mix: dict[str, float] | None` / `val_corpora_mix: dict[str, float] | None` 追加)
- `scripts/generate_institutional_training_corpus.py` (新規、訓練 corpus 生成、標準 CLI / provider 選択 / train-val split / eval リーク検証内蔵、S5-003 反映)
- `scripts/_corpus_core.py` (新設、private module、tokenize / pack / val split / `validate_tokenizer_id` / `resolve_tokenizer_id` を抽出、S3-004 反映)
- `data/training/institutional/` (新規 corpus、`.gitignore` で除外済み、機密保護対象)
- `checkpoints/photon_institutional_<step>_<yyyymmdd>/` (新規、git LFS or external storage、機密保護対象)
- `configs/institutional_docs_photon_retrain.yaml` (**新規**、本 Issue 用 hyperparam / checkpoint パス。既存 `configs/institutional_docs_photon.yaml` は #112 / #126 再現用に直接編集せず保持、S3-007 反映)
- `baseline_reporag/tests/` (photon provider smoke test 新規追加、S3-006 反映)
- `.github/workflows/*.yml` (CI 設定確認 / 必要時 LFS 有効化、S3-006 反映)
- `reports/institutional_photon_mt_eval_v2.md` (新規、旧 PHOTON / 新 PHOTON / baseline 比較表)
- `reports/gate2_post_retrain_eval.md` (新規、FastAPI 系 retrieval 再 eval)
- `reports/gate2_judgment_v5_post_retrain.md` (新規、v4 → v5 cross-link、S3-005 反映)
- `CLAUDE.md` L156-160 (現在のメトリクスを新値で更新、S3-005 反映)
- `workspace/mvp/metrics.md` (再学習成果反映)
- `workspace/mvp/roadmap.md` (Phase 2 conditional 表に「>6% 分岐実行済」マーキング、S3-005 反映)
- `docs/deployment.md` (新 checkpoint パスの初回 fetch 手順を追記、S3-011 反映、follow-up Issue で対応も可)
- `docs/troubleshooting.md` (checkpoint 不在時 (= LFS pull 未実行) の症状と対処を追記、S3-011 反映、follow-up Issue で対応も可)

## 受入条件

- [ ] **`.gitignore` に `data/training/` 登録済み** (本 Issue 最初の commit、S3-003 反映)
- [ ] **JP 50%+ 学習 corpus 構築** (以下のチェックリストを全て満たす):
  - [ ] `scripts/generate_institutional_training_corpus.py --corpus-dir <institutional_documents_dir> --provider auto --sessions 2000 --output data/training/institutional --val-ratio 0.1 --seed 42` で再現可能
  - [ ] sessions 数 ≥ 2,000
  - [ ] JP token 比 ≥ 50% (測定方法 = tokenizer encode 後の id 比)
  - [ ] cross_reference / drill_down シナリオが 30% 以上
  - [ ] eval set との session_id 重複ゼロ (検証スクリプトで確認)
  - [ ] **generate script 単体実行時、eval リーク検出で exit code 非ゼロになる pytest を新規追加** (S3-009 反映)
  - [ ] 無作為 20 サンプルの人手 spot-check で「前 turn 参照」成立率 ≥ 80%
  - [ ] **新 script の関数 / CLI 引数が既存 `generate_training_corpus.py` と top-level 関数名で衝突しない** (S3-004 反映)
  - [ ] turn delimiter `<|turn_sep|>` は新 special token 追加ではなく、既存 tokenizer で通常文字列として encode される (vocab resize 不要、S5-006 反映)
- [ ] **`photon_mlx/tests/test_training.py` に resume_from + cosine schedule reset の unit test を追加** (`test_resume_from_continual_learning`: warmup から始まらず既存実装通り `lr` 起点で cosine decay すること / step counter は resume 元 + 追加 step で進むこと / optimizer state 新規 / 既存重み非恒等 を検証、S3-002 / S5-001 / S5-002 反映)
- [ ] **既存 `photon_mlx/data.py` の `iterate_batches()` signature が無改修である** (`photon_mlx/trainer.py:297-303` / `scripts/train_photon_quick.py:48-49` / `photon_mlx/tests/test_training.py:23` 系の既存 4 callsite + 27+ tests が無改修で pass、S3-001 反映)
- [ ] **`iterate_mixed_batches` が corpus ごとの sequence pool から 50/50 weighted sampling し、単純 concat 後 pack で比率が崩れないことを unit test で確認** (S5-005 反映)
- [ ] 10-20K step 再学習完了 (累計 step 10K/15K/20K の 3 checkpoint を eval。追加 step 扱いに変更する場合は `max_steps=10600/15600/20600` を明記、S5-002 反映)
- [ ] 再学習後 PHOTON で **Turn 5-6 NC < 6%** （仮説 B 達成 = MVP minimum）
- [ ] **理想**: Turn 5-6 NC < 3% （仮説 A 復元）
- [ ] latency 優位（-30% 以上 vs baseline = follow-up p50 ≤ 13.6s）維持
- [ ] **FastAPI 系 retrieval 性能 regression**: 対象 corpus = `fastapi_fastapi`、eval script = `scripts/run_multi_turn_eval.py`、測定指標 = MT no-citation rate (gate2 v4 PHOTON+SR ベンチマーク 6.7% を base)、**判定 = +5pp 以内 (= 11.7% 以下)**
- [ ] `reports/institutional_photon_mt_eval_v2.md` で再測定値を記録 (旧 PHOTON / 新 PHOTON / baseline 比較表)
- [ ] **採用 checkpoint が指定保管先 (git LFS or external storage) に存在し、CI/CD で取得可能**
- [ ] **`configs/institutional_docs_photon_retrain.yaml` 新規作成** (既存 `configs/institutional_docs_photon.yaml` は #112 / #126 再現用に直接編集せず保持、S3-007 反映)
- [ ] **`pytest` 全 tests (約 507/509) が新 corpus / 新 checkpoint 取得手順下で破綻しないこと** (CI で `--ignore=data/training/` を確認、`baseline_reporag/tests/` の photon provider smoke test を含む、S3-006 反映)
- [ ] **`baseline_reporag/tests/` に photon provider smoke test 新規追加** (`pipeline_factory.create_query_pipeline` photon 経路をモック checkpoint で起動、S3-006 反映)
- [ ] **採用 checkpoint 確定後の docs 同時更新** (`CLAUDE.md` L156-160 / `reports/gate2_judgment_v5_post_retrain.md` 新規作成 / `workspace/mvp/roadmap.md` Phase 2 conditional 表の「>6% 分岐実行済」マーキング、S3-005 反映)
- [ ] **(Nice to Have) `docs/deployment.md` に新 checkpoint 初回 fetch 手順、`docs/troubleshooting.md` に checkpoint 不在時の対処を追記** (本 Issue で困難なら follow-up Issue 起票でも可、S3-011 反映)

## 想定 compute コスト

- **学習**: 10-20K step × M3 Ultra (推定 1-3 日)
- **eval**: 6+ runs × ~40 min = ~4 時間 (Turn 5-6 NC 境界帯時 +2 run = ~+1.3 時間)
- **合計**: **3-5 日**（学習が支配項）
- **token throughput 前提 (S5-004 反映)**: 初期値では 1 optimizer step = `2 * 16 * 2048 = 65,536 tokens`。累計 20K step までの処理量は約 1.31B tokens。16K token/step 前提で compute を抑える場合は `grad_accum=4` 相当に再設定する
- **checkpoint disk**: 1 個 ~750MB (small 377M, fp16) × 1K step 単位保存で 10-20 個 = **~7.5-15GB ピーク**。eval 対象外 (累計 10K/15K/20K 以外) は順次削除し、最終採用 1 個 + 候補 2 個 = ~2.3GB を保管
- **SSD 残量前提 (S3-010 反映)**: Day 1 着手時に `df -h` で **残量 50GB 以上** を確認 (checkpoint ピーク ~15GB + corpus ~数十 MB-GB + 既存 `checkpoints/` の他 commit 分との合算で 20GB+ 占有を見込む)

## リスクと緩和策

| リスク | 影響 | 緩和策 |
|-------|------|-------|
| catastrophic forgetting **発生抑制** (英語コード性能劣化) | -10pp 以上 | mix ratio 50/50 維持、英語コード loss を別 metric として追跡 |
| catastrophic forgetting **検出選別** | 採用 checkpoint 誤選定 | 1K step 毎に英語コード eval (FastAPI MT) 実施 + dynamic に最良 checkpoint 選定 |
| 再学習で Turn 5-6 NC < 6% 達成失敗 | MVP Phase 2 失敗 | **Phase 2 pivot の具体化 (S3-012 反映)**: (1) 本 Issue で 3-6% 帯にしか改善しなかった場合、**追加で軽量 fine-tune 5K step を本 Issue 内で実施** (compute 余裕あれば、roadmap.md L88-94 conditional 表 3-6% 分岐に該当)、(2) それでも < 6% に至らない場合、Phase 2 完了基準を「Turn 5-6 NC < Y%」に再定義する別 Issue を起票し本 Issue は close |
| 学習データ品質不足 | NC 改善せず | 人手検証 20% サンプル、低品質 session 除外 |
| compute 時間超過 (>5 日) | スケジュール影響 | 10K step を early-stop 候補として中間 eval |
| **mulmoclaude 600 step checkpoint 紛失** | scratch 再学習で工数 +2-3 日 | Day 1 着手時に物理的所在を最優先で確認、欠損時は別 Issue 起票 |
| **`resume_from` が想定外の LR 起点を取り 600 step 重みを破壊** (S3-002 / S5-001 / S5-002 反映) | 既存 mulmoclaude 重み喪失 = scratch 同等に劣化 | `photon_mlx/tests/test_training.py` に `test_resume_from_continual_learning` を**学習開始前**に追加し、warmup_ratio=0.0 時に既存実装通り `lr=3e-5` 起点で cosine decay すること、かつ `state.step` が resume 元 step から進むことを CI で恒常確認 |
| **学習 corpus institutional_documents の機密漏洩** (S3-003 反映) | JP 制度文書平文 token 列のパブリック流出 | `.gitignore` に `data/training/` 追加 (本 Issue 最初の commit) + checkpoint / corpus は public 配布対象外 (private repo / ACL 制限) |
| **checkpoint / corpus による M3 Ultra SSD 逼迫** (S3-010 反映) | 学習中断・データ喪失 (1K step 進行分喪失) | (1) Day 1 着手時に `df -h` で残量 50GB 以上を確認、(2) 1K step 単位 checkpoint のうち eval 対象外 (10K/15K/20K 以外) は **当該 step の eval 完了時点で即削除** するロジック (例: state.step % 1000 == 0 かつ state.step not in {10000, 15000, 20000} なら 5K step 経過後に shutil.rmtree)、(3) 学習開始前に古い不要 checkpoint (mulmoclaude 以外) を archive へ移動 |
| **他 PR との merge conflict** (S3-008 反映) | rebase コスト増 / 5 日工数超過 | Day 3-4 学習期間中は本 Issue 関連ファイルへの外部 PR マージを保留 (並列開発セクション参照) |

## 関連

- 元 Issue: #113 (PR #134 merged) — 本 Issue を発動した実測根拠
- Epic: #117 Phase 2 conditional 「本格再学習」シナリオ
- baseline: #112 / handicap fixes: #125 / #126
- 学習 corpus 既存: mulmoclaude (英語コード、600 step)

## 並列開発

#133 (多言語 reranker A/B、PR #136 = `aeb2233` merged 済) との並列可否は phase 別に判断:
- **Day 4-5 の eval phase**: #133 の rerank A/B は CPU 系 cross-encoder で実施可能なため**同時進行可**
- **Day 3-4 の学習 phase**: M3 Ultra GPU を 1-3 日連続占有するため、#133 の **学習系作業 (rerank モデル学習等) は後回し** (直列推奨)

### ファイル単位の編集衝突回避 (S3-008 反映)
- **Day 3-4 の学習期間中** は以下のファイルへの外部 PR マージを保留する (同ファイルへの並行編集 PR を持つ Issue があれば本 Issue 完了まで待つ):
  - `photon_mlx/trainer.py`
  - `photon_mlx/data.py`
  - `configs/institutional_docs_photon.yaml` (既存、#112 / #126 再現用、本 Issue では直接編集しないが他 PR 編集を避ける)
  - `configs/institutional_docs_photon_retrain.yaml` (本 Issue で新規)
- **Day 5 ロールアウト前** に `develop` の最新 commit に rebase し全テストを再実行する。
