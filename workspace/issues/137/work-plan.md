# Work Plan — Issue #137 institutional 多言語 embedding/reranker 5-variant 実機 A/B

## Issue: feat(retrieval): institutional 多言語 embedding/reranker 5-variant 実機 A/B (#133 Phase B)
- **Issue 番号**: #137
- **サイズ**: M (実機 4-5h + 採用反映 1h、計 ~半日)
- **優先度**: High (#135 直列ブロッカー)
- **依存 Issue**: #133 (Phase A 完了済 / PR #136 / `96e7b45`)
- **ブランチ**: `feature/issue-137-institutional-ab` (作成済)
- **設計方針書**: `workspace/design/issue-137-institutional-multilingual-ab-design-policy.md`

---

## 全体構造

本 Issue は **(A) Phase B 実機 A/B 評価 → (B) 採否判定 → (C) 採用結果に応じた最小コード変更** の 3 段。Phase B 実機作業 (A) はコード変更ゼロ (gitignored variant config + 実機 build/eval)、PR に含む code change は (C) のみ。

```
[Phase 1: 準備] → [Phase 2: 実機 A/B] → [Phase 3: 採否判定 + レポート] → [Phase 4: コード反映 (採用ありの場合)] → [Phase 5: PR + マージ]
```

---

## Phase 1: 準備 (Pre-flight checks, ~30 min)

### Task 1.1: 環境確認
- [ ] `pip show sentence-transformers` で version 確認 (要 >= 2.3.0、bge-m3/bge-reranker-v2-m3 サポート)
- [ ] 利用可能ディスク空き容量確認 (`df -h`) — 追加 ~7GB (HF cache 5.8GB + index 0.9-1.2GB)
- [ ] `~/.cache/huggingface/` の現状容量を `du -sh` で記録 (Phase B 後 cleanup 比較用)
- [ ] `data/indexes/institutional_documents/` の現状を `ls -la` で確認 (V0/V3 共有用 baseline)

### Task 1.2: base config の HEAD commit 記録 (DRY 違反許容ルール)
- [ ] `git log -1 --format="%H %s" -- configs/institutional_docs.yaml` で HEAD を記録
- [ ] PR description に「Phase B 期間中は `configs/institutional_docs.yaml` の更新を保留」と明記する旨をメモ
- [ ] (もし他 PR で更新進行中なら) #135 等の関連作業者と共有

### Task 1.3: V0 baseline 再確認
- [ ] `reports/institutional_baseline_static.md` の NC rate (11.21%) と eval set (`data/eval_sets/institutional_static_eval.jsonl`、116Q) を確認
- [ ] `data/indexes/institutional_documents/embedding/model_id.txt` が `intfloat/multilingual-e5-small` になっているか確認

---

## Phase 2: 実機 A/B 評価 (~4-5h、worker scope 外、人手 or 主セッション側)

### Task 2.1: 5 variant config 作成
- [ ] `mkdir -p configs/_experiments/`
- [ ] `configs/institutional_docs.yaml` を 5 回コピーし `institutional_V[0-4].yaml` を作成
  - V0: `model_id="intfloat/multilingual-e5-small"`, reranker=`cross-encoder/ms-marco-MiniLM-L-6-v2`, max_input_chars=2048, repo.repo_id=`institutional_documents`
  - V1: embedding を `intfloat/multilingual-e5-base` に変更、repo.repo_id=`institutional_documents_V1`
  - V2: embedding を `cl-nagoya/ruri-small-v2` に変更、repo.repo_id=`institutional_documents_V2`
  - V3: reranker を `BAAI/bge-reranker-v2-m3` に変更、repo.repo_id=`institutional_documents` (V0 と共有)
  - V4: embedding を `BAAI/bge-m3`、reranker を `BAAI/bge-reranker-v2-m3`、max_input_chars=**8192** を**明示宣言**、repo.repo_id=`institutional_documents_V4`
- [ ] 各 V<N>.yaml が単独で `python -c "from baseline_reporag.config import load_config; print(load_config('configs/_experiments/institutional_V<N>.yaml'))"` で読めることを確認

### Task 2.2: V0 → V3 (reranker swap) 評価 — 逐次必須
- [ ] V0 評価: `python scripts/run_baseline_eval.py --config configs/_experiments/institutional_V0.yaml --repo-id institutional_documents --eval-set data/eval_sets/institutional_static_eval.jsonl --output logs/institutional_V0_$(date +%Y%m%d_%H%M%S).jsonl`
- [ ] V0 完了確認 (116Q 完走、predictions JSONL 生成)
- [ ] V3 評価 (V0 完了後): 上記コマンドの V0 を V3 に置換、reranker config が反映されることを確認
- [ ] 注意: SQLite ファイル lock 競合回避のため V0 と V3 は **必ず逐次** (並列禁止)

### Task 2.3: V1, V2, V4 — index 再 build + 評価
- [ ] V1: `python scripts/build_indexes.py --config configs/_experiments/institutional_V1.yaml --repo-id institutional_documents_V1` (~30 min)
- [ ] V1 評価: 上記コマンドの output で実行
- [ ] V2: 同様 (build ~60 min + 評価 ~60 min)
- [ ] V4: 同様 (build ~30 min + 評価 ~30 min、bge-m3/bge-reranker-v2-m3 で +5GB RAM)
- [ ] 注意: HF cache 同時 download 衝突回避のため V1/V2/V4 も **逐次推奨**

### Task 2.4: aggregator 個別実行
- [ ] V0〜V4 の 5 ファイルそれぞれに対し:
  ```bash
  python scripts/aggregate_institutional_baseline.py \
    --predictions logs/institutional_V<N>_<ts>.jsonl \
    --output - \
    --section overall,category,latency
  ```
- [ ] **禁止**: `--predictions logs/institutional_V*.jsonl` のような glob 一括実行 (5 variant 合算 580Q レポートになり比較不能)

---

## Phase 3: 採否判定 + レポート (~30 min)

### Task 3.1: `reports/institutional_retrieval_ab.md` 作成
- [ ] 設計方針書 5-6 のテンプレートに従い 5 variant 比較表 + 採否判断 + 各 variant aggregator raw output + 運用波及 checkbox を記載
- [ ] チェックボックス checkbox に基づく採否判定:
  - 主指標: V0 比 NC rate -2pt 以上改善 → 採用
  - 改善 < 2pt → 「非採用 (V0 維持)」と明記
  - タイブレーカー (差 ≤ 1pt 時): category 別 NC 悪化数 → p95 latency → memory footprint
- [ ] レポート末尾の運用波及 checkbox を採用結果に応じて埋める (採用反映タスクの抜け漏れチェック)

### Task 3.2: 採否確定 → Phase 4 分岐
- [ ] **非採用 (V0 維持)** → Task 5.1 (commit #4 のみ) に進む
- [ ] **V0/V1/V2/V3 採用** → Task 4.1〜4.3 に進む (4 commit)
- [ ] **V4 採用** → Task 4.1〜4.4 に進む (4 commit、deployment.md L13 + L88、第 3 invariant test 追加)

---

## Phase 4: コード反映 (採用ありの場合のみ、~30 min)

### Task 4.1: `configs/institutional_docs.yaml` 更新 (commit #1)
- [ ] `embedding.model_id` を採用値に置換
- [ ] `reranker.model_id` を採用値に置換
- [ ] `embedding.max_input_chars` を **明示宣言** (現状 fallback。採用 V4 なら 8192、それ以外 2048)
- [ ] **既存 index 強制再 build**: `rm -rf data/indexes/institutional_documents/embedding/` → `python scripts/build_indexes.py --config configs/institutional_docs.yaml --repo-id institutional_documents`
- [ ] サーバ起動疎通確認 (1 query 成功で OK、設計方針書 8 章の品質基準)
- [ ] commit message: `feat(institutional): adopt V<N> embedding/reranker in institutional_docs.yaml (#137)`

### Task 4.2: `tests/test_pipeline_factory_yaml_invariants.py` invariant 活性化 (commit #2)
- [ ] `INSTITUTIONAL_RERANKER_MODEL_ID` (line 36) を採用 reranker model_id に置換
- [ ] `INSTITUTIONAL_EMBEDDING_MODEL_ID` (line 37) を採用 embedding model_id に置換
- [ ] (V4 採用時のみ) `INSTITUTIONAL_EMBEDDING_MAX_INPUT_CHARS = 8192` 定数 + 設計方針書 5-4 の test 関数を追記
- [ ] `python -m pytest tests/test_pipeline_factory_yaml_invariants.py -v` で全 test pass を確認 (V0-V3 採用: 3 件全 active / V4 採用: 4 件全 active)
- [ ] commit message: `test(institutional): activate invariant pinning for V<N> (#137)`

### Task 4.3: `docs/deployment.md` 更新 (commit #3)
- [ ] L88 の reranker model_id 行直下に institutional 専用 2 行を追加 (`retrieval.reranker.model_id (institutional)` + `indexing.embedding.model_id (institutional)`)
- [ ] (V4 採用時のみ) L13 memory 要件を `~5-6 GB (bge-m3 + bge-reranker-v2-m3)` に更新
- [ ] commit message: `docs(deployment): add institutional reranker/embedding row (#137)` (V4 時は `add institutional V4 + memory update`)

### Task 4.4: `reports/institutional_retrieval_ab.md` 完成 (commit #4)
- [ ] Task 3.1 で作成したレポートを最終化 (運用波及 checkbox を採用反映後の状態で確定)
- [ ] commit message: `docs(institutional): add 5-variant retrieval A/B report (#137)`

---

## Phase 5: PR + マージ (~30 min)

### Task 5.1: 品質チェック
- [ ] `python -m pytest` 全 pass (既知 pre-existing 2 件除く)
- [ ] `ruff check .` 警告 0 件
- [ ] `ruff format --check .` 差分なし
- [ ] (採用ありの場合) サーバ起動疎通: institutional プロファイルで 1 query 成功

### Task 5.2: PR 作成
- [ ] `gh pr create` で develop 向け PR 作成
- [ ] PR タイトル: 採用結果に応じて
  - 非採用: `docs(institutional): 5-variant A/B 非採用レポート (#137)`
  - V0-V3 採用: `feat(institutional): 5-variant A/B 採用反映 V<N> (#137)`
  - V4 採用: `feat(institutional): 5-variant A/B 採用反映 V4 bge-m3 (#137)`
- [ ] PR description に Phase B 開始時の base HEAD commit + 並列開発リスク (#135 5.5 日直列) を明記
- [ ] CI 全 pass を確認

### Task 5.3: cleanup (Phase B 後の retention)
- [ ] 未採用 variant の `data/indexes/institutional_documents_V<N>/` を `rm -rf` (ディスク ~600MB-1.2GB 解放)
- [ ] (任意) `~/.cache/huggingface/` の不要 model を削除 (例: 採用が e5-small なら e5-base/ruri/bge-m3/bge-reranker-v2-m3 を削除)
- [ ] predictions JSONL (`logs/institutional_V*_*.jsonl`) は採否根拠として最低 1 ヶ月保持推奨

---

## 品質チェック項目

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| invariant 保護 | `python -m pytest tests/test_pipeline_factory_yaml_invariants.py -v` | 採用前: 1 active + 2 skip / V0-V3 採用後: 3 件全 active / V4 採用後: 4 件全 active |
| 全 test | `python -m pytest` | 全 pass (既知 pre-existing 2 件除く) |
| ruff check | `ruff check .` | 警告 0 件 |
| ruff format | `ruff format --check .` | 差分なし |
| Phase B 完走 | 5 variant × 116Q | 全 variant タイムアウト/OOM なし |
| 採否判断 | `reports/institutional_retrieval_ab.md` 存在 + 5 variant 比較表 + 採否明記 | 必須項目記載 |
| (採用ありの場合) サーバ起動 | institutional プロファイルで 1 query 成功 | dim mismatch ValueError なし |

---

## Definition of Done

- [ ] Phase B 5 variant institutional eval 完走 (116Q)
- [ ] `reports/institutional_retrieval_ab.md` に比較表 + category 別 NC + p95 latency + 採否判断
- [ ] 採用判定: -2pt 以上改善で採用 / 未満で非採用 (V0 維持)
- [ ] `configs/baseline.yaml` の global default 不変 (#114 invariant test で自動保証)
- [ ] (採用ありの場合) 4 commit (config + test + docs + report) 構成で PR 作成
- [ ] (V4 採用時のみ) 第 3 invariant test 追加 + deployment.md L13 memory 更新
- [ ] CI 全 pass + サーバ起動疎通確認
- [ ] cleanup (未採用 variant index 削除、HF cache retention 整理)

---

## リスクと対策

| リスク | 兆候 | 対策 |
|--------|------|------|
| `run_baseline_eval.py --repo-id` が index load 先を変えない誤解 | V1/V2/V4 で意図せず V0 の index を読む (NC が V0 と完全一致) | variant config の `repo.repo_id` 明示 + `--repo-id` を一致 (Task 2.1, 2.3) |
| aggregator 合算誤実行 | 5 variant 比較表ではなく 580Q 単一集計レポートになる | Task 2.4 で variant ごと個別実行 (glob 禁止) |
| eval set 取り違え | NC が突然乖離 (例: 6.7%) | Task 2.2/2.3 で `--eval-set data/eval_sets/institutional_static_eval.jsonl` 明示 |
| embedding 切替後に再 build 忘れ | サーバ起動時に dim mismatch ValueError | Task 4.1 で `rm -rf .../embedding/` + build_indexes 必須 |
| Phase B 期間中の base config 並行更新 | variant config が古い base 由来 | Task 1.2 で HEAD commit 記録 + PR description に明記 |
| V4 採用したのに第 3 invariant 追加忘れ | CI 通過するが drift 検知不能 | Task 3.1 レポート checkbox + Task 4.2 で V4 時の追加を明示 |

---

## 次のアクション

1. **本作業計画の承認確認** (人間 review)
2. **Phase 1 環境チェック実施** (Task 1.1〜1.3)
3. **Phase 2 実機 A/B 開始** (Task 2.1〜2.4、合計 4-5h)
4. **Phase 3 採否判定 + レポート** (Task 3.1〜3.2)
5. **Phase 4 コード反映** (採用ありの場合、Task 4.1〜4.4)
6. **Phase 5 PR + マージ + cleanup** (Task 5.1〜5.3)
7. **#135 本格再学習に institutional 採用 model を組み込む** (本 Issue 完了後、直列実行)
