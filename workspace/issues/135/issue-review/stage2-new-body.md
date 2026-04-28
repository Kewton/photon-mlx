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

#### A-1. 学習 corpus 構築
- **既存 mulmoclaude (英語コード) 50%** を保持（catastrophic forgetting 回避）
- **制度文書 JP 50%+ 追加**: institutional_documents 4228 md から学習用 turn 系列を生成
  - source: `data/eval_sets/institutional_multi_turn_eval.jsonl` の同領域 corpus
  - **目標 sessions 数**: **≥ 2,000** (初期値 / 目標下限。JP token 比 50% を満たす最小値で算出。上限は 5,000 を目安にし、生成コストとデータ品質に応じて調整)
  - **eval set との重複禁止**: `data/eval_sets/institutional_multi_turn_eval.jsonl` (30 sessions × 6 turns) と `session_id` 重複ゼロを検証スクリプトで強制
  - **generation script**: 新規 `scripts/generate_institutional_training_corpus.py` を追加 (既存 `generate_institutional_eval_set.py` は eval 用 grader 込みのため流用しない / 共通ロジックは関数抽出で共有)
  - **出力 schema**: 1 session = 1 doc。`{"tokens": [...], "session_id": "...", "scenario": "cross_reference|drill_down|...", "lang": "ja"}` 形式の JSONL。turn 境界は special token (`<|turn_sep|>`) で区切り、turn 単位の文脈を 1 シーケンスとして pack
  - **photon_mlx/data.py 拡張**: `iterate_batches()` を複数 corpus_path 受領可能に拡張するか、新関数 `iterate_mixed_batches(corpus_paths: dict[str, float])` を追加し、JP/英語コード = 50/50 サンプリングで yield
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
- `cosine schedule は continual で reset` (新規 cosine を本 Issue の max_steps 用に張り直し)

**確定手順**:
- (a) Day 3 着手時に **lr-finder (1K step の small run、lr=1e-5 / 3e-5 / 1e-4 の 3 候補)** で val_loss curve を比較し、最良の lr を採用
- (b) micro_batch / grad_accum は M3 Ultra の OOM 限界で max を取る (初期値 2×16 = effective batch 32)
- (c) **Step 数 = 10K-20K**: 根拠 = JP 50% mix で 1 step あたり ~16K token、20K step = 320M token = mulmoclaude 600 step の ~30 倍。377M small model の forgetting curve / Chinchilla scaling から十分な学習量と判断
- (d) **resume_from**: `checkpoints/photon_mulmoclaude/step_600/` (背景セクション参照)。optimizer state は新規、LR scheduler は新規 cosine

#### B-2. checkpoint 保存
- 1K step 単位で intermediate checkpoint (debug 用、容量管理のため未採用 checkpoint は早期削除可)
- **本番採用候補 (eval 対象)**: **10K, 15K, 20K の 3 ポイントのみ** で eval 実行

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
- **config 更新**: `configs/institutional_docs_photon.yaml` の `model.provider` (photon) と `paths.checkpoint_root` 配下の新パス (例: `./checkpoints/photon_institutional_15000_20260428/`) を更新
- **pipeline_factory.py 整合**: photon provider 経路が新 checkpoint を pickup することを smoke test で確認
- **比較表出力**: `reports/institutional_photon_mt_eval_v2.md` に **旧 PHOTON 値・新 PHOTON 値・baseline 値** を 1 表で並べる
- **FastAPI 系 retrieval 再 eval**: `configs/baseline.yaml` + photon provider で同 checkpoint を eval し、`reports/gate2_post_retrain_eval.md` (新規) に記録
- **chunker / retrieval は変更しない** (本 Issue は LM checkpoint のみ差し替え)

## 影響ファイル

- `photon_mlx/trainer.py`（必要なら学習設定拡張、resume_from 動作確認）
- `photon_mlx/data.py`（複数 corpus mix 対応で `iterate_batches()` 拡張、または新関数 `iterate_mixed_batches(corpus_paths: dict[str, float])` 追加）
- `scripts/generate_institutional_training_corpus.py`（新規、訓練 corpus 生成）
- `data/training/institutional/`（新規 corpus、規模次第で gitignore）
- `checkpoints/photon_institutional_<step>_<yyyymmdd>/`（新規、git LFS or external storage）
- `configs/institutional_docs_photon.yaml`（checkpoint パス更新）
- `reports/institutional_photon_mt_eval_v2.md`（新規、旧 PHOTON / 新 PHOTON / baseline 比較表）
- `reports/gate2_post_retrain_eval.md`（新規、FastAPI 系 retrieval 再 eval）
- `workspace/mvp/metrics.md`（再学習成果反映）

## 受入条件

- [ ] **JP 50%+ 学習 corpus 構築** (以下のチェックリストを全て満たす):
  - [ ] sessions 数 ≥ 2,000
  - [ ] JP token 比 ≥ 50% (測定方法 = tokenizer encode 後の id 比)
  - [ ] cross_reference / drill_down シナリオが 30% 以上
  - [ ] eval set との session_id 重複ゼロ (検証スクリプトで確認)
  - [ ] 無作為 20 サンプルの人手 spot-check で「前 turn 参照」成立率 ≥ 80%
- [ ] 10-20K step 再学習完了 (10K/15K/20K の 3 checkpoint を eval)
- [ ] 再学習後 PHOTON で **Turn 5-6 NC < 6%** （仮説 B 達成 = MVP minimum）
- [ ] **理想**: Turn 5-6 NC < 3% （仮説 A 復元）
- [ ] latency 優位（-30% 以上 vs baseline = follow-up p50 ≤ 13.6s）維持
- [ ] **FastAPI 系 retrieval 性能 regression**: 対象 corpus = `fastapi_fastapi`、eval script = `scripts/run_multi_turn_eval.py`、測定指標 = MT no-citation rate (gate2 v4 PHOTON+SR ベンチマーク 6.7% を base)、**判定 = +5pp 以内 (= 11.7% 以下)**
- [ ] `reports/institutional_photon_mt_eval_v2.md` で再測定値を記録 (旧 PHOTON / 新 PHOTON / baseline 比較表)
- [ ] **採用 checkpoint が指定保管先 (git LFS or external storage) に存在し、CI/CD で取得可能**

## 想定 compute コスト

- **学習**: 10-20K step × M3 Ultra (推定 1-3 日)
- **eval**: 6+ runs × ~40 min = ~4 時間 (Turn 5-6 NC 境界帯時 +2 run = ~+1.3 時間)
- **合計**: **3-5 日**（学習が支配項）
- **checkpoint disk**: 1 個 ~750MB (small 377M, fp16) × 1K step 単位保存で 10-20 個 = **~7.5-15GB ピーク**。eval 対象外 (10K/15K/20K 以外) は順次削除し、最終採用 1 個 + 候補 2 個 = ~2.3GB を保管

## リスクと緩和策

| リスク | 影響 | 緩和策 |
|-------|------|-------|
| catastrophic forgetting **発生抑制** (英語コード性能劣化) | -10pp 以上 | mix ratio 50/50 維持、英語コード loss を別 metric として追跡 |
| catastrophic forgetting **検出選別** | 採用 checkpoint 誤選定 | 1K step 毎に英語コード eval (FastAPI MT) 実施 + dynamic に最良 checkpoint 選定 |
| 再学習で Turn 5-6 NC < 6% 達成失敗 | MVP Phase 2 失敗 | 軽量 fine-tune (5K step) 派生路線、または Phase 2 pivot |
| 学習データ品質不足 | NC 改善せず | 人手検証 20% サンプル、低品質 session 除外 |
| compute 時間超過 (>5 日) | スケジュール影響 | 10K step を early-stop 候補として中間 eval |
| **mulmoclaude 600 step checkpoint 紛失** | scratch 再学習で工数 +2-3 日 | Day 1 着手時に物理的所在を最優先で確認、欠損時は別 Issue 起票 |

## 関連

- 元 Issue: #113 (PR #134 merged) — 本 Issue を発動した実測根拠
- Epic: #117 Phase 2 conditional 「本格再学習」シナリオ
- baseline: #112 / handicap fixes: #125 / #126
- 学習 corpus 既存: mulmoclaude (英語コード、600 step)

## 並列開発

#133 (多言語 reranker A/B) との並列可否は phase 別に判断:
- **Day 4-5 の eval phase**: #133 の rerank A/B は CPU 系 cross-encoder で実施可能なため**同時進行可**
- **Day 3-4 の学習 phase**: M3 Ultra GPU を 1-3 日連続占有するため、#133 の **学習系作業 (rerank モデル学習等) は後回し** (直列推奨)
