# TASKS

- Last Updated: 2026-04-12
- Owner: <your_name>
- Horizon: 24 weeks
- Primary Track: RepoRAG
- Control: Baseline RepoRAG
- Experimental Track: PHOTON-RAG + Safe RecGen

---

## 使い方

- `[ ]` 未着手
- `[~]` 進行中
- `[x]` 完了
- `[!]` 要判断
- `BLOCKED:` ブロッカーあり

このファイルは「思いつきメモ」ではなく、**毎週更新する実行計画**として使う。

---

## 最優先事項

- [ ] 対象ユースケースを RepoRAG に固定する
- [ ] baseline RepoRAG を動かす
- [ ] benchmark を freeze する
- [ ] citation 付き回答を安定化する
- [ ] PHOTON 実装前に control 系を完成させる

---

# Phase 0: Project Setup

## 0.1 仕様と意思決定
- [x] `spec.md` を作成する
- [x] `README.md` を作成する
- [x] `tasks.md` を作成する
- [ ] primary use case を 1 つに固定する
- [ ] non-goals をチーム内で合意する
- [ ] 成功条件を `spec.md` に対して最終確認する

## 0.2 リポジトリ初期化
- [x] ディレクトリ構成を作る
- [x] `configs/` の初期 YAML を置く
- [x] `scripts/` に雛形を置く
- [x] `reports/` と `demo/` を作る
- [x] `.env.example` を作る
- [x] `requirements.txt` または依存管理を確定する

## 0.3 開発ルール
- [x] run_id 命名規則を決める（`spec.md` §16.1 参照）
- [x] repo snapshot の固定ルールを決める（`spec.md` §14.4 参照）
- [x] benchmark freeze のタイミングを Week 2 に固定する（`spec.md` §18 参照）
- [x] grading rule を明文化する（`spec.md` §15.4 参照）
- [x] failure case の記録テンプレートを作る（`reports/failure_case_template.md`）

### Exit Criteria
- [x] 開発ルールが README / spec / tasks に反映されている
- [x] リポジトリ初期構成が作成済み
- [x] baseline 開発に入れる状態になっている

---

# Phase 1: Baseline RepoRAG

## 1.1 Repo ingestion
- [x] `scripts/ingest_repo.py` を作る
- [x] repo を commit hash 付きで取り込める
- [x] コード、README、docs、tests、configs を抽出する
- [x] chunking 戦略を定義する（Python: AST boundary、その他: plain）
- [x] chunk metadata を保存する（SQLite）
- [x] file path / symbol / section 情報を保持する

## 1.2 Indexing
- [x] lexical index を構築する（BM25Okapi）
- [x] embedding index を構築する（sentence-transformers + numpy）
- [x] symbol graph を構築する（AST call graph）
- [ ] dependency graph の最小版を作る（symbol graph で代替中、必要なら拡張）
- [x] index build の再実行を idempotent にする（INSERT OR REPLACE）

## 1.3 Retrieval
- [x] hybrid retrieval を実装する
- [x] lexical + embedding の重み付けを決める（config で制御）
- [x] symbol / graph expansion を追加する
- [x] chunk 近傍展開を入れる
- [x] cited chunk 優先ロジックを入れる
- [x] retrieval debug 出力を残す（JSONL ログ）

## 1.4 Baseline answer generation
- [x] baseline instruct model を決める（Qwen2.5-Coder-14B-Instruct-4bit）
- [x] answer prompt を定義する（`generation/prompt.py`）
- [x] citation 付き回答フォーマットを決める（`[C:N]` 形式）
- [ ] answer-time local refresh を入れる（`evidence_pack.py` に hook 追加予定）
- [x] no-citation 断定を抑制する（system prompt で強制）
- [x] simple session memory を入れる（`memory/session.py`）

## 1.5 Interface
- [x] CLI を作る（`baseline_reporag/cli.py`）
- [x] ローカル API を作る（`baseline_reporag/server.py`、FastAPI）
- [x] session_id 指定を実装する
- [x] repo_id 指定を実装する
- [x] ログ保存を実装する（JSONL per run）

### Exit Criteria
- [x] baseline RepoRAG が 1 コマンドで起動する（CLI 疎通確認済み）
- [x] citation 付き回答が出る（`[C:9]` で正しい chunk を引用）
- [x] 同一 session で follow-up できる（2ターン疎通確認済み）
- [x] retrieval / answer / citation がログに残る（JSONL 確認済み）

---

# Phase 2: Benchmark and Eval Freeze

## 2.1 Static Eval
- [ ] onboarding 問題を 30 問作る（ingest 完了後に生成）
- [ ] impact analysis 問題を 30 問作る
- [ ] bug localization 問題を 30 問作る
- [ ] change planning 問題を 30 問作る
- [x] 採点ルーブリックを付与する（`evals/static_eval_schema.md`）

## 2.2 Multi-turn Session Eval
- [ ] 6 ターン × 30 セッションを作る（ingest 完了後に生成）
- [x] 途中で話題が狭まるケースを入れる（schema: `topic_narrowing`）
- [x] 途中で話題が切り替わるケースを入れる（schema: `topic_shift`）
- [x] exact quote 要求ケースを入れる（schema: `exact_quote`）
- [x] diff / patch 要求ケースを入れる（schema: `diff_or_patch`）

## 2.3 Stress Eval
- [ ] 8 同時セッション用ケースを作る（ingest 完了後に生成）
- [x] session ごとの active memory 計測を入れる（eval.yaml に定義済み）
- [x] fallback 発火の記録を入れる（logger.py で記録済み）

## 2.4 Scoring
- [x] grader テンプレートを作る（`evals/grader_template.py`、judge: qwen3.5:27b）
- [x] correctness / grounding / usefulness を採点可能にする
- [x] wrong citation 判定ルールを作る（`citation.py`）
- [ ] stale memory 判定ルールを作る（Phase 7 で実装）

## 2.5 Freeze
- [ ] Week 2 末で benchmark を freeze する（`bench/freeze_benchmark.py` 準備済み）
- [x] freeze 後の変更ルールを定義する（`spec.md` §14.4）
- [ ] baseline スコアを保存する（ingest → serve → run_all 後に実行）

### Exit Criteria
- [ ] benchmark が再現可能（問題セット生成待ち）
- [ ] baseline スコアが 1 回分保存されている
- [ ] 以後の比較が benchmark 上で可能

---

# Phase 3: Logging, Profiling, and Reporting

## 3.1 必須ログ
- [x] run_id を保存する
- [x] session_id / turn_id を保存する
- [x] repo_id / repo_commit を保存する
- [x] retrieval chunk IDs を保存する
- [x] evidence pack IDs を保存する
- [x] cited chunk IDs を保存する
- [x] fallback flag / reason を保存する
- [x] latency breakdown を保存する（`profiler.py` → `pipeline.py` → JSONL）
- [x] memory peak を保存する（`tracemalloc` ベース）
- [x] answer text と grader score を保存する（grader score は採点実行後に追加）

## 3.2 Profiling
- [x] retrieval time を測る（`TurnProfiler.phase("retrieval")`）
- [ ] prefill time を測る（mlx_lm 内部分離は Phase 9 で対応）
- [ ] decode time を測る（同上）
- [x] end-to-end latency を測る
- [x] peak memory を測る
- [ ] session あたりメモリを推定する（stress eval 時に計測）

## 3.3 Reporting
- [x] `scripts/export_report.py` を作る
- [x] benchmark summary を出せるようにする（P50/P90/mean）
- [ ] failure case を抽出できるようにする（grader 実行後に対応）
- [ ] 比較グラフのひな形を作る

### Exit Criteria
- [x] benchmark run の結果が自動集計される
- [ ] failure analysis の入口がある（grader 実行後に対応）
- [x] baseline のレポートが出せる

---

# Phase 4: torch_ref

## 4.1 Minimal LM
- [x] `torch_ref/` を作る
- [x] decoder-only minimal LM を作る（LLaMA-style: RoPE, SwiGLU, GQA, RMSNorm）
- [x] teacher forcing を実装する
- [x] greedy decode を実装する
- [ ] cache 付き decode を実装する（Phase 9 で必要に応じて追加）

## 4.2 Correctness Tests
- [x] mask テストを書く（causal mask upper triangular, prefix invariance）
- [x] shape テストを書く（logits, labels, greedy decode）
- [x] 1 batch overfit テストを書く（200 step で loss < 10% of initial）
- [x] seed 固定の再現テストを書く
- [x] logits の健全性確認をする（finite, non-constant）

## 4.3 Reference Harness
- [ ] forward 比較用の入力セットを作る
- [x] PHOTON 実装の control に使えるようにする（common PhotonConfig）
- [x] common config 読み込みを実装する（`torch_ref/config.py`）

### Exit Criteria
- [x] minimal LM が安定して動く（11/11 tests passed）
- [x] basic correctness tests が通る
- [x] PHOTON 実装の比較対象として使える

---

# Phase 5: PHOTON Forward (MLX Main Track)

## 5.1 Core Modules
- [x] `photon_mlx/` を作る
- [x] chunker を実装する（ConcatChunker + LinearChunker）
- [x] hierarchical encoder を実装する（LLaMA-style blocks with RoPE）
- [x] converter を実装する（linear + pos_embed + RMSNorm）
- [x] local decoder を実装する（chunk-local causal attention）
- [x] output head を実装する（RMSNorm + Linear）

## 5.2 Forward Graph
- [x] bottom-up pass を実装する
- [x] top-down pass を実装する（_decode_level: prefix + chunk processing）
- [x] teacher-forced forward を実装する（cross-entropy loss）
- [ ] hidden state を必要に応じて保存する（Phase 7 で対応）
- [x] config から level 数を切り替えられるようにする（PhotonConfig.hierarchy.levels）

## 5.3 Testing
- [x] shape テストを書く（logits, labels, batch_size=1）
- [x] chunk boundary テストを書く（min T=16, longer T=128）
- [ ] local attention 範囲テストを書く（Phase 7 で drift 含めて追加）
- [x] tiny config で smoke test を通す（logits finite, loss positive, loss decreases）
- [ ] torch_ref と概念整合を確認する（forward 比較入力セット未作成）

## 5.4 Configs
- [x] `configs/photon_tiny.yaml` を作る
- [x] `configs/photon_small.yaml` を作る
- [x] loss / batch / context の最小設定を決める

### Exit Criteria
- [x] PHOTON の forward が tiny で安定動作する（9/9 tests passed）
- [x] teacher-forced eval の入口がある（forward + loss 動作確認済み）
- [x] module 単位のテストが通る

---

# Phase 6: Training Path

## 6.1 Data Pipeline
- [x] training 用コーパスを整える（`data.py`: JSONL → pack → batch）
- [ ] train / val split を固定する（コーパス生成スクリプト未実装）
- [x] tokenizer / packing を決める（greedy bin-packing）
- [x] context 長を tiny / small で分ける（configs で制御）

## 6.2 Loss
- [x] next-token loss を入れる（`loss.py`）
- [x] hierarchical consistency loss を入れる（`photon_loss` に hook、v1 weight=0.0）
- [x] loss 重みを config 化する（`recursive_loss_weight`）
- [x] loss の分解ログを残す（breakdown dict）

## 6.3 Training Loop
- [x] train loop を作る（`trainer.py`）
- [x] eval loop を作る（`evaluate()`）
- [x] checkpoint 保存を実装する（`save_checkpoint` / npz + state.json）
- [x] resume を実装する（`load_checkpoint`）
- [ ] gradient / memory 監視を入れる（Phase 9 で対応）

## 6.4 Early Experiments
- [x] tiny で overfit を確認する（テスト通過: 50 steps で loss < 50% initial）
- [ ] tiny で val loss 低下を確認する（コーパス生成後に実行）
- [ ] small で 1 本実験を通す
- [ ] loss 崩壊パターンを failure_cases に記録する

### Exit Criteria
- [x] tiny / small で学習が回る（overfit テスト通過）
- [x] loss が下がる
- [x] checkpoint と eval が再現可能

---

# Phase 7: Session Inference

## 7.1 Session Memory
- [ ] session pack を PHOTON 入力へ変換する
- [ ] multi-turn state を保持する
- [ ] cited chunk を memory に反映する
- [ ] topic shift 検知のための特徴量を残す

## 7.2 Inference Path
- [ ] hierarchical prefill を実装する
- [ ] session update path を実装する
- [ ] answer-time local refresh を実装する
- [ ] reply 生成の grounding を強制する

## 7.3 Drift Metrics
- [ ] latent drift を計測する
- [ ] token agreement を計測する
- [ ] logit KL を計測する
- [ ] stale memory 兆候を記録する

## 7.4 Comparison
- [ ] Baseline vs PHOTON-RAG を同じ harness で比較する
- [ ] static eval を回す
- [ ] multi-turn session eval を回す
- [ ] 初回レポートを作る

### Exit Criteria
- [ ] PHOTON-RAG が end-to-end で動く
- [ ] multi-turn bench が回る
- [ ] drift が計測できる

---

# Phase 8: Safe RecGen

## 8.1 Trigger Design
- [ ] drift threshold を定義する
- [ ] topic shift threshold を定義する
- [ ] exact quote trigger を入れる
- [ ] diff / patch trigger を入れる
- [ ] high-risk query trigger を入れる
- [ ] confidence-based trigger を入れる

## 8.2 Fallback Actions
- [ ] local evidence refresh 強化を実装する
- [ ] re-retrieval を実装する
- [ ] hierarchical prefill refresh を実装する
- [ ] baseline path 退避を実装する

## 8.3 Logging
- [ ] fallback reason を保存する
- [ ] fallback 前後の latency を保存する
- [ ] fallback 前後の quality 差分を保存する

## 8.4 Validation
- [ ] multi-turn eval で missed fallback を数える
- [ ] wrong citation 改善を確認する
- [ ] stale memory 改善を確認する
- [ ] latency 悪化の許容範囲を確認する

### Exit Criteria
- [ ] Safe RecGen が発火する
- [ ] 誤答抑制の効果が確認できる
- [ ] レイテンシ改善を極端に壊さない

---

# Phase 9: Mac Optimization

## 9.1 Runtime Tuning
- [ ] fixed shape decode step を導入する
- [ ] padding 方針を決める
- [ ] compile 対象関数を切り出す
- [ ] warmup 手順を決める
- [ ] memory 測定を安定化する

## 9.2 Performance Experiments
- [ ] single session latency を測る
- [ ] 8 同時セッションを測る
- [ ] follow-up ターンの改善を測る
- [ ] memory / session を baseline と比較する
- [ ] prefill / decode 比率を分析する

## 9.3 Stability
- [ ] 長時間 run でリークがないか確認する
- [ ] repeated benchmark の分散を測る
- [ ] config ごとのベンチ結果を保存する

### Exit Criteria
- [ ] 最適化前後の差が数値で示せる
- [ ] benchmark の再現性がある
- [ ] Mac 単機で主要実験が回る

---

# Phase 10: Demo, Report, Release Decision

## 10.1 Demo
- [ ] CLI デモを整える
- [ ] 対象 repo での代表質問を 5 本用意する（シナリオ確定済み、下記参照）
- [ ] follow-up が速いことを見せるデモを作る
- [ ] citation と fallback を見せるデモを作る

### Demo シナリオ（fastapi/fastapi 対象）

| # | 軸 | 1ターン目 | follow-up | 見せたいもの |
|---|---|---|---|---|
| 1 | オンボーディング | repo の全体構成と主要モジュールを教えて | 依存性注入の仕組みをコードで説明して → どこから読み始めればいい？ | working memory の引き継ぎ、session consistency |
| 2 | 影響範囲分析 | `Depends()` の実装を変えると波及先はどこか | その中で壊れやすい箇所はどこ？ | graph expansion + citation precision |
| 3 | 障害解析 | 認証ミドルウェアで 401 が返る原因候補を 3 つ出して | 一番怪しい箇所のコードを示して | multi-turn で仮説を絞る流れ |
| 4 | Safe RecGen 発火 | `security.py` の `get_current_user` を exact quote で出して | その関数の引数の型を確認して | exact quote で Safe RecGen が発火し局所再読する様子 |
| 5 | 変更計画の比較 | 認可ロジックを middleware に移す案と decorator に残す案を比較して | 最小変更で済む方法はどれ？ | drift 検知 + fallback、citation 付き比較回答 |

## 10.2 Reports
- [ ] `reports/benchmark_report.md` を作る
- [ ] `reports/failure_cases.md` を作る
- [ ] baseline 比較表を作る
- [ ] Gate 判断用メモを作る

## 10.3 Decision
- [ ] baseline 統合可否を判断する
- [ ] 研究枝として公開するか判断する
- [ ] 限定ベータ投入可否を判断する
- [ ] 次フェーズ候補を 3 つに絞る

### Exit Criteria
- [ ] 第三者に見せられる report がある
- [ ] failure cases が整理されている
- [ ] 次アクションが明確になっている

---

# Ongoing Tasks

## 継続的にやること
- [ ] 毎週 benchmark を 1 回回す
- [ ] failure case を追加する
- [ ] retrieval エラーと generation エラーを切り分ける
- [ ] stale memory ケースを記録する
- [ ] benchmark 逸脱を検知したらメモする
- [ ] `reports/` を更新する

## 品質のための確認
- [ ] exact quote は必ず局所再読しているか
- [ ] patch / diff は grounding しているか
- [ ] wrong citation が増えていないか
- [ ] no-citation assertion が増えていないか

---

# Week-by-Week Plan

## Week 1
- [ ] repo skeleton を作る
- [ ] spec / README / tasks を確定する
- [ ] 対象 repo を 1 つ選ぶ
- [ ] ingest 方針を決める

## Week 2
- [ ] baseline retrieval を動かす
- [ ] citation 付き回答を出す
- [ ] benchmark を freeze する
- [ ] Gate 1 判定材料を集める

## Week 3
- [ ] static eval を埋める
- [ ] multi-turn eval を埋める
- [ ] logging / profiling を入れる
- [ ] baseline レポート 1 本目を出す

## Week 4
- [ ] baseline を安定化する
- [ ] Gate 1 を判定する
- [ ] PHOTON 実装開始判断をする

## Week 5
- [ ] torch_ref minimal LM を作る
- [ ] basic tests を通す
- [ ] photon_mlx skeleton を作る

## Week 6
- [ ] chunker / encoder を実装する
- [ ] converter / local decoder を実装する
- [ ] tiny forward smoke test を通す

## Week 7
- [ ] teacher-forced path を作る
- [ ] PHOTON module tests を増やす
- [ ] tiny config を安定化する

## Week 8
- [ ] initial PHOTON eval を回す
- [ ] drift metrics の計測を始める
- [ ] failure cases を残す

## Week 9
- [ ] training loop を作る
- [ ] tiny overfit を確認する
- [ ] checkpoint 保存を入れる

## Week 10
- [ ] tiny val を通す
- [ ] small config を 1 本回す
- [ ] loss breakdown を確認する

## Week 11
- [ ] session inference の入口を作る
- [ ] session memory の初版を入れる
- [ ] answer-time local refresh を合わせる

## Week 12
- [ ] PHOTON-RAG を end-to-end で動かす
- [ ] Gate 2 を判定する
- [ ] 継続方針を決める

## Week 13
- [ ] multi-turn eval を PHOTON で回す
- [ ] baseline 差分を確認する
- [ ] stale memory パターンを分類する

## Week 14
- [ ] token agreement / logit KL を入れる
- [ ] topic shift 検知を入れる
- [ ] wrong citation ケースを分析する

## Week 15
- [ ] Safe RecGen の trigger を実装する
- [ ] fallback action を実装する
- [ ] fallback logging を入れる

## Week 16
- [ ] exact quote / patch ケースで検証する
- [ ] fallback recall を計測する
- [ ] quality 改善を確認する

## Week 17
- [ ] fixed shape decode step を導入する
- [ ] compile 対象を決める
- [ ] warmup と計測を安定化する

## Week 18
- [ ] 最適化前後を比較する
- [ ] 8 同時セッションで測る
- [ ] memory / session を比較する

## Week 19
- [ ] benchmark report の下書きを作る
- [ ] failure case report を整理する
- [ ] demo シナリオを作る

## Week 20
- [ ] Gate 3 を判定する
- [ ] プロダクト投入可否の方向性を決める

## Week 21
- [ ] レポートを清書する
- [ ] 代表例デモを整える
- [ ] baseline 統合案を書く

## Week 22
- [ ] internal review を行う
- [ ] 次フェーズ候補を整理する
- [ ] 必要な追加実験を 1 回だけ実施する

## Week 23
- [ ] 最終ベンチを再実行する
- [ ] report / demo / failure cases を固定する

## Week 24
- [ ] Gate 4 を判定する
- [ ] 続行 / 公開 / 統合の意思決定を行う
- [ ] 次フェーズ用の新 spec を起草する

---

# Gate Checklist

## Gate 1
- [ ] baseline RepoRAG が安定稼働
- [ ] benchmark が再現可能
- [ ] citation 付き回答が出る
- [ ] logging がある

## Gate 2
- [ ] PHOTON forward が安定
- [ ] tiny / small で学習が回る
- [ ] drift 指標が取得できる
- [ ] follow-up 改善の兆候がある

## Gate 3
- [ ] Safe RecGen が有効
- [ ] 誤答率が改善
- [ ] レイテンシ改善が残る
- [ ] stale memory が抑制される

## Gate 4
- [ ] benchmark report 完成
- [ ] failure cases 完成
- [ ] demo 完成
- [ ] 次フェーズ意思決定完了

---

# Resolved Decisions

- [x] primary target repo は `fastapi/fastapi`
- [x] secondary holdout repo は `ml-explore/mlx`
- [x] product baseline model は `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`
- [x] research control baseline は matched-size の LLaMA-style decoder-only
- [x] Safe RecGen の fallback threshold は v1 では fixed、v2 で learned calibrator を検討
- [x] PHOTON config は `tiny` / `small` / `paper600m` の 3 本で管理（`configs/` 参照）
- [x] PHOTON default chunk size は `[4, 4]`
- [x] 初期 `recursive_loss_weight` は `0.0`
- [x] LLM judge model は `qwen3.5:27b`
- [x] demo シナリオは 5 本（オンボーディング・影響範囲・障害解析・Safe RecGen 発火・変更計画比較）

# Open Questions

- [ ] retrieval の control condition をどこまで固定するか
- [ ] demo で見せるユースケースをどれに絞るか

---

# Notes

- PHOTON 実装の前に、必ず baseline と benchmark を固定する
- exact quote / diff / patch は必ず局所再読
- 「速いが危ない」をそのまま採用しない
- control を最後まで維持する
- Done の基準は「比較できる」「再現できる」「説明できる」