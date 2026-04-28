# Issue #144 作業計画書

## Issue 概要

**タイトル**: test(retrieval): ruri-small-v2 (V2) build failure follow-up — manual spiece.model download workaround
**Issue 番号**: #144
**サイズ**: M (~1.5-2h、build/eval 計算時間込み)
**優先度**: 🟢 Low (V4 採用判定は既に成立、V2 結果は将来検討用)
**依存 Issue**: #137 (5-variant A/B、CLOSED 済) / #142 (PR、MERGED 済)
**ブランチ**: `feature/issue-144-ruri-v2-build` (既存)
**ベース**: `develop` (最新)

---

## ゴール

V2 (`cl-nagoya/ruri-small-v2`) を fix し、5-variant A/B 比較を完成させる。本格的な V2 vs V4 比較を成立させ、`reports/institutional_retrieval_ab.md` の 5 行目を埋める。

---

## 設計方針書 / レビュー成果物

- 設計方針書: `workspace/design/issue-144-ruri-v2-build-design-policy.md`
- Issue レビュー: `workspace/issues/144/issue-review/summary-report.md`
- 設計レビュー: `workspace/issues/144/multi-stage-design-review/summary-report.md`

---

## タスク分解

### Phase 1: 依存追加 + setup helper 実装 (実装タスク)

#### Task 1.1: `requirements.txt` に `huggingface_hub` 追加

- **成果物**: `requirements.txt` (`# Tokenizer\ntransformers` の直下に `# HF Hub direct file access (Issue #144: spiece.model 個別 download 用)\nhuggingface_hub` を追加)
- **依存**: なし
- **検証**:
  - `pip install -r requirements.txt` 成功
  - `python -m pip check` で resolver 不整合なし (DR4-003 反映)
  - `python -m pip show huggingface_hub transformers sentence-transformers` で resolved version を Issue コメントに記録 (DR4-003 反映)
- **想定工数**: 5 分

#### Task 1.2: `scripts/_setup_ruri_models.py` 新規作成

- **成果物**: `scripts/_setup_ruri_models.py` (設計方針書 §3.1 の関数構成に従う)
  - `fetch_ruri_files() -> None` (huggingface_hub から `spiece.model` を download、log は filename + size のみ、full path 出力なし — DR4-002)
  - `verify_load() -> None` (SentenceTransformer 単独 load 検証、失敗時 sanitized exception summary のみ Issue コメント貼付)
  - `RURI_MODELS = ['cl-nagoya/ruri-small-v2']` (hard-coded、CLI 引数化しない — least privilege)
  - `RURI_FILES = ['spiece.model']` (hard-coded)
- **依存**: Task 1.1 (huggingface_hub install 必要)
- **検証**:
  - `python scripts/_setup_ruri_models.py` 成功 (`fetch_ruri_files()` + `verify_load()` 両方完走)
  - `verify_load()` 成功 log を Issue コメント貼付 (HF cache full path / raw traceback / token を貼らない — DR4-002)
- **想定工数**: 30 分 (実装 + 手動実行)
- **設計判断**: unit test 不要 (`scripts/` は `tests/test_no_scaffolding_in_prod.py:PROD_ROOTS` 対象外、Issue #139)

---

### Phase 2: V2 build/eval 実行

#### Task 2.1: variant config 作成 (`configs/_experiments/institutional_V2.yaml`)

- **成果物**: `configs/_experiments/institutional_V2.yaml` (gitignored、commit 対象外)
  - `mkdir -p configs/_experiments` 後 `cp configs/institutional_docs.yaml configs/_experiments/institutional_V2.yaml`
  - 設計方針書 §3.3 表の 4 行を yq / python -c / 手編集で override:
    - `indexing.embedding.model_id = "cl-nagoya/ruri-small-v2"`
    - `indexing.embedding.max_input_chars = 2048`
    - `retrieval.reranker.model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
    - `repo.repo_id = "institutional_documents_V2"`
- **依存**: Task 1.2 (verify_load 成功で真因確定後)
- **検証**:
  - `cat configs/_experiments/institutional_V2.yaml` で override 4 行が反映済み
  - その他のフィールド (chunking, batch_size, normalize, repo.repo_path, repo.repo_commit) は base と同値
- **想定工数**: 10 分

#### Task 2.2: variant config security validation gate

- **成果物**: 設計方針書 §3.4 Step 2.5 の python script 実行 (assert 全件 PASS)
  - `repo_id == "institutional_documents_V2"`、sentinel commit / path、`embedding.model_id == "cl-nagoya/ruri-small-v2"`、`max_input_chars == 2048`、`batch_size == 32`、`normalize is True`、chunking 800/100、reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` を fail-fast 検証 (DR4-001)
- **依存**: Task 2.1
- **検証**: assert 例外なく完走、fail-fast で typo / tampering を検出
- **想定工数**: 5 分

#### Task 2.3: ingest 再実行

- **成果物**: `data/indexes/institutional_documents_V2/chunks.db` (gitignored)
- **コマンド**:
  ```bash
  python scripts/ingest_repo.py \
    --repo /Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents \
    --commit 9e500539f29555364217b773368305e7f59aa026 \
    --config configs/_experiments/institutional_V2.yaml \
    --repo-id institutional_documents_V2
  ```
- **依存**: Task 2.2 (validation PASS)
- **検証**: chunks.db 生成 (4228 docs 想定)
- **想定工数**: 5-10 分

#### Task 2.4: build_indexes 実行

- **成果物**:
  - `data/indexes/institutional_documents_V2/lexical.pkl`
  - `data/indexes/institutional_documents_V2/embedding/{embeddings.npy,chunk_ids.json,model_id.txt,max_input_chars.txt}`
  - すべて gitignored
- **コマンド**:
  ```bash
  python scripts/build_indexes.py \
    --config configs/_experiments/institutional_V2.yaml \
    --repo-id institutional_documents_V2
  ```
- **依存**: Task 2.3
- **検証**: embedding/model_id.txt の内容が `cl-nagoya/ruri-small-v2` であること
- **想定工数**: 30-60 分 (ruri-small-v2 で 4228 docs を re-embed)

#### Task 2.5: run_baseline_eval 実行

- **成果物**:
  - `logs/institutional_V2_<timestamp>.jsonl` (predictions、gitignored)
  - `logs/bench_variant_*.jsonl` (RunLogger、gitignored)
  - `logs/sessions/session_eval-*.json` (SessionManager、gitignored)
- **コマンド**:
  ```bash
  python scripts/run_baseline_eval.py \
    --config configs/_experiments/institutional_V2.yaml \
    --repo-id institutional_documents_V2 \
    --eval-set data/eval_sets/institutional_static_eval.jsonl \
    --output logs/institutional_V2_$(date +%Y%m%d_%H%M%S).jsonl
  ```
- **依存**: Task 2.4
- **検証**: predictions JSONL に全 eval 行が記録されている、NC rate / latency 集計可能
- **想定工数**: 30 分

---

### Phase 3: レポート + config コメント更新 (commit 対象)

#### Task 3.1: `reports/institutional_retrieval_ab.md` の V2 行を更新

- **成果物**: `reports/institutional_retrieval_ab.md`
  - V2 行に NC rate、category breakdown、latency を反映
  - aggregate 関数 (`scripts/aggregate_institutional_baseline.py` 等) で predictions を集計
- **依存**: Task 2.5 (predictions JSONL 完成)
- **検証**: V0/V1/V2/V3/V4 が表として完成、判定根拠 (V4 採用が V2 と比較しても妥当か) が文章で追記済み
- **想定工数**: 10-15 分

#### Task 3.2: `configs/institutional_docs.yaml:84-87` コメント更新

- **成果物**: `configs/institutional_docs.yaml` (コメント L84-87 のみ、embedding.model_id pin 値は不変 = bge-m3 維持)
  - 例: 「V2 ruri 計測不能」→「V2 ruri-small-v2 NC X.XX%」
- **依存**: Task 3.1 (V2 NC 値確定)
- **検証**:
  - `python -m pytest tests/test_pipeline_factory_yaml_invariants.py -v` 全パス (invariant test は値のみを assert、コメントは不問)
  - `git diff configs/institutional_docs.yaml` でコメント以外の変更がないこと
- **想定工数**: 5 分

#### Task 3.3 (条件付き、低確率分岐): V2 が V4 を上回った場合の対応

- **発火条件**: V2 の NC rate が V4 の NC rate を下回る (V2 が V4 を上回る = より良い)
- **成果物** (発火時のみ):
  - `configs/institutional_docs.yaml`: `embedding.model_id` / `max_input_chars` / `reranker.model_id` を V2 値に更新
  - `tests/test_pipeline_factory_yaml_invariants.py:43-47`: institutional pin 定数を V2 値に更新
  - `docs/deployment.md:13-16`: institutional memory note を V2 値に更新
  - GitHub: Issue #137 reopen + 採用 variant 切替 PR
- **依存**: Task 3.1
- **想定工数**: 1-2h (発火時のみ)

---

### Phase 4: 品質チェック + 受入

#### Task 4.1: テスト全パス確認

```bash
python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v
```

- **基準**: 全テストパス (約 507/509、既知の pre-existing failure 2 件は除外可)
- **特に重要**:
  - `tests/test_pipeline_factory_yaml_invariants.py` PASS (invariant 不変)
  - `tests/test_no_scaffolding_in_prod.py` PASS (scripts/_setup_ruri_models.py は対象外)

#### Task 4.2: lint / format

```bash
ruff check .
ruff format --check .
```

- **基準**: 警告 0 件、差分なし

#### Task 4.3: 手動受入記録

- **成果物**: Issue #144 へのコメント
  - `verify_load()` 成功 log (sanitized — DR4-002)
  - `python -m pip show` resolved version (DR4-003)
  - V2 NC rate / latency 抜粋
  - V4 採用判定の確認文 (V2 が V4 を超えていない場合)

---

## 実行順序とフォールバック分岐

```
Task 1.1 (requirements.txt)
   └── Task 1.2 (scripts/_setup_ruri_models.py + verify_load)
        ├── verify_load 成功 → Task 2.1 へ
        ├── "Couldn't instantiate the backend tokenizer" error → Option B 分岐
        │     └── 専用 venv で旧版 install → Task 1.1 再実行 → Task 1.2 retry
        │     └── `.gitignore` に `.venv-ruri-*/` 追加 (DR3-003)
        └── 上記以外 error → Option C (上流 issue 報告) で blocked エスカレーション
   ↓
Task 2.1 (variant config) → Task 2.2 (security validation gate)
   ↓
Task 2.3 (ingest) → Task 2.4 (build_indexes) → Task 2.5 (run_baseline_eval)
   ↓
Task 3.1 (report 更新) → Task 3.2 (config コメント更新)
   ↓ (条件付き)
Task 3.3 (V2 採用低確率分岐: invariant test + deployment docs 更新)
   ↓
Task 4.1-4.3 (品質チェック + 受入記録)
```

---

## 影響ファイル一覧

### Commit 対象 (PR で push)

- `requirements.txt` (1 行追加)
- `scripts/_setup_ruri_models.py` (新規)
- `reports/institutional_retrieval_ab.md` (V2 行更新)
- `configs/institutional_docs.yaml` (コメント L84-87 のみ)

### Commit 対象外 (gitignored)

- `configs/_experiments/institutional_V2.yaml` (variant config)
- `data/indexes/institutional_documents_V2/` (chunks.db、lexical.pkl、embedding/)
- `logs/institutional_V2_*.jsonl` (predictions)
- `logs/bench_variant_*.jsonl` (RunLogger)
- `logs/sessions/session_eval-*.json` (SessionManager)

### 条件付き (V2 採用判定発火時のみ)

- `configs/institutional_docs.yaml` (embedding.model_id 等の値変更)
- `tests/test_pipeline_factory_yaml_invariants.py:43-47` (institutional pin 定数)
- `docs/deployment.md:13-16` (institutional memory note)

### 条件付き (Option B fallback 採用時のみ)

- `.gitignore` (`.venv-ruri-*/` 追加)

---

## 品質チェック項目

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| ビルド (Python import) | `python -c "import baseline_reporag, photon_mlx, torch_ref"` | エラー 0 件 |
| Lint | `ruff check .` | 警告 0 件 |
| Format | `ruff format --check .` | 差分なし |
| Test | `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | 全パス (約 507/509) |
| invariant test | `python -m pytest tests/test_pipeline_factory_yaml_invariants.py -v` | PASS |
| scaffolding test | `python -m pytest tests/test_no_scaffolding_in_prod.py -v` | PASS |
| pip resolver | `python -m pip check` | "No broken requirements found." |

---

## Definition of Done

- [ ] Task 1.1: requirements.txt に huggingface_hub 追加 commit
- [ ] Task 1.2: `_setup_ruri_models.py` 作成 + verify_load 成功 log を Issue コメント
- [ ] Task 2.1-2.5: V2 ingest + build + eval 完走、predictions JSONL 保存
- [ ] Task 3.1: reports/institutional_retrieval_ab.md V2 行反映
- [ ] Task 3.2: configs/institutional_docs.yaml:84-87 コメント更新
- [ ] Task 4.1: pytest 全パス
- [ ] Task 4.2: ruff check + ruff format --check 全パス
- [ ] Task 4.3: 手動受入記録 (sanitized log + pip show + V2 NC rate / latency) を Issue コメント
- [ ] PR 作成 (`/create-pr`) → develop へマージ
- [ ] V4 採用判定不変の確認 (V2 < V4 の場合)
- [ ] V2 採用低確率分岐 (条件付き、発火時のみ): Task 3.3 完遂

---

## 想定工数 (合計 ~1.5-2h、Option A 成功時)

| Phase | Task | 工数 |
|-------|------|------|
| 1 | Task 1.1 (requirements.txt) | 5 分 |
| 1 | Task 1.2 (`_setup_ruri_models.py`) | 30 分 |
| 2 | Task 2.1 (variant config 作成) | 10 分 |
| 2 | Task 2.2 (security validation) | 5 分 |
| 2 | Task 2.3 (ingest) | 5-10 分 |
| 2 | Task 2.4 (build_indexes) | 30-60 分 |
| 2 | Task 2.5 (run_baseline_eval) | 30 分 |
| 3 | Task 3.1 (report 更新) | 10-15 分 |
| 3 | Task 3.2 (config コメント更新) | 5 分 |
| 4 | Task 4.1-4.3 (品質チェック + 受入) | 15 分 |
| - | **合計** | **~2.5h** (Option A 成功時、Option B/C 発火時は +1-2h) |

---

## リスクと対応

| リスク | 影響 | 対応 |
|--------|------|------|
| Option A 失敗 (互換性問題が真因) | 工数増 ~+1h | Option B (専用 venv) に分岐、設計方針書 §3.4 フォールバック表参照 |
| Option A/B 失敗 | 完了不能 | Option C (上流 PR) に escalation、別 Issue で長期対応 |
| build_indexes が想定 60 分超過 | スケジュール遅延 | M3 Ultra 性能で大丈夫と推定、超過時は中断・再開不可 (一括実行) |
| V2 採用低確率分岐発火 | 工数 +1-2h | Task 3.3 のサブタスク (invariant test + deployment docs) を順次対応 |
| pip resolver conflict (huggingface_hub) | install 失敗 | `pip check` で事前検出、最悪 Option B venv で隔離 |

---

## 次のアクション

作業計画承認後:
1. **ブランチ確認**: `feature/issue-144-ruri-v2-build` (既存)
2. **タスク実行**: `/pm-auto-dev 144` で TDD 自動開発フェーズに進む
3. **進捗報告**: 各 Phase 完了時に Issue コメント
4. **PR 作成**: 全 Task 完了後 `/create-pr` で develop 向け PR

---

## 関連リンク

- Issue: https://github.com/Kewton/photon-mlx/issues/144
- 親 Issue (closed): https://github.com/Kewton/photon-mlx/issues/137
- 親 PR (merged): https://github.com/Kewton/photon-mlx/pull/142
- 設計方針書: `workspace/design/issue-144-ruri-v2-build-design-policy.md`
- Issue review summary: `workspace/issues/144/issue-review/summary-report.md`
- Design review summary: `workspace/issues/144/multi-stage-design-review/summary-report.md`
