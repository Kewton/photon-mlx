# Issue #135 Day 3 進捗報告 — develop merge 完了 + Phase 6 着手前 blocker 報告

**Issue**: feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合
**実行日**: 2026-04-27 (Day 3、Phase 6 着手前)
**ブランチ**: feature/issue-135-photon-retrain (Day 1-3 = **17 commits**)
**ステータス**: develop merge OK、**corpus 生成本番実行は API mismatch のため保留**、ユーザー判断待ち

---

## ステップ 1 結果: develop merge 完了 (commit `be91682`)

### 取り込み内容

| 由来 PR | 内容 | 影響範囲 |
|---------|------|---------|
| #115 | wizard JP system prompt + institutional template | `app/`, `baseline_reporag/generation/prompt.py` |
| #138 / #139 | 学習/推論の tokenizer 統一 + DR4-001 model_id allow-list | `baseline_reporag/photon_pipeline.py`, `_load_hf_tokenizer` |
| #140 | `embedding_random_init_threshold` schema | `torch_ref/config.py`, `photon_mlx/model.py` |
| **#148 Phase A0** | **fail-loud checkpoint loading + security guards** | `baseline_reporag/photon_pipeline.py` |
| #150 | rebaseline (mulmoclaude 600-step checkpoint integration) | `configs/institutional_docs_photon.yaml` |
| #151 | Phase A pre-flight (vocab padding + num_heads schema) | `_load_hf_tokenizer` |
| #152 | Gate 2 v5 + #113 v2 measurements with fixed pipeline | `reports/` |
| #154 / #155 | repo_id retrieval filter + refusal-aware citation grader | `baseline_reporag/citation.py`, `eval/institutional/citation_eval.py` |

### Conflict 解決サマリ (3 ファイル)

| ファイル | 解決 |
|---------|------|
| `baseline_reporag/photon_pipeline.py` | **develop 採用**: #148 Phase A0 の `_resolve_checkpoint_path` (root containment / symlink escape / weights.npz+state.json shape / fail-fast RuntimeError + `PHOTON_ALLOW_RANDOM_INIT=1` opt-out / root-relative log) は私の S7-001 fix (commit `2dbf458`) の上位互換 |
| `baseline_reporag/tests/test_photon_pipeline.py` | **develop 採用** (`TestBuildPhotonDepsRealTokenizer` #138/#139)。私の `TestBuildPhotonDepsCheckpointLoad` は develop の新ファイル `test_photon_pipeline_checkpoint_load.py` に**包含されているため drop**。**DR1-002 boundary test だけ独立に残す** (`test_photon_pipeline_lazy_import.py` 新設) |
| `photon_mlx/tests/test_config.py` | **両方残す** (#135 Phase 4-1 + #140 純粋追加、衝突は中央位置のみ) |

### 削除ファイル
- `baseline_reporag/tests/test_photon_checkpoint_smoke.py` (Day 1 commit `1c920ae`): develop の `test_photon_pipeline_checkpoint_load.py` で完全包含。

### テスト結果

| 結果 | 件数 |
|------|------|
| ✅ pass | **1245** (Day 2 EOD 1114 → +131 develop 由来 / +1 boundary test 再追加 / -2 削除) |
| ❌ fail | 3 (内訳下記) |

#### Failed 内訳

1. **`tests/test_generate_training_corpus.py::TestMain::test_main_cli_tokenizer_id`**: ✅ 既知 pre-existing failure (CLAUDE.md 記載)
2. **`tests/test_generate_training_corpus.py::TestMain::test_main_uses_tokenize_text`**: ✅ 既知 pre-existing failure
3. **`baseline_reporag/tests/test_photon_pipeline.py::TestBuildPhotonDepsRealTokenizer::test_vocab_size_mismatch_raises`**: 🔍 develop side stale test — #148 Phase A が `_load_hf_tokenizer` で「vocab padding を許可」する変更を入れた結果、`actual < expected` ケースで raise しなくなった (旧 test は raise を期待)。**私の merge regression ではない**

#### 私由来の regression: 0

### ruff
- `ruff check`: All checks passed!
- `ruff format --check`: 差分なし

---

## ステップ 2 着手前確認: 重要 blocker 3 件

ステップ 2 (corpus 生成本番実行) を実行しようとした際、以下の **production-readiness gap** が発覚しました。**実機 LLM 呼び出し前にユーザー判断が必要** です。

### Blocker 1: 🔴 generate script の LLM API mismatch

**現状**: 私が Day 2 PM commit `f25022c` で作成した `scripts/generate_institutional_training_corpus.py` の `LLMClient` Protocol は `generate_turns(*, source_md, scenario, n_turns, lang) -> list[str]` を期待しているが、production の `baseline_reporag.eval.institutional.llm_client.LLMClient` は `generate(prompt: str) -> str` (JSON 1 オブジェクト返却) であり、API が一致しない。

そのため、`select_llm_client("qwen")` を呼んでも `build_sessions` 内で `AttributeError: 'QwenMLXAdapter' object has no attribute 'generate_turns'` で即落ちる。**実機実行不可**。

**原因**: Day 2 PM 設計時に既存 `multi_turn.generate_session` のパターン (prompt 構築 + JSON 解析 + retry) を確認せず、独自 Protocol を定義した。Mock LLM tests (12 件) は通っていたが production 互換性は未検証。

**Resolution options**:

| Option | 内容 | 工数 | リスク |
|--------|------|------|------|
| A | `build_sessions` を refactor し `client.generate(prompt)` + JSON 解析 + retry を `multi_turn.generate_session` パターンで実装。turn 構造は CATEGORY_CONFIG / TURN_TEMPLATES から取得 | 1-2 時間 (CPU、TDD) | 低 (既存パターン踏襲) |
| B | 既存の `baseline_reporag.eval.institutional.multi_turn.generate_session` を**そのまま流用**し、私の Phase 2 script は thin wrapper にする (新 script のオリジナル価値は eval リーク検出 / metadata 出力部分のみ) | 30 分 | 中 (eval set と training corpus で同じ session 生成器を使うが session_id 重複防止だけで分離可能) |
| C | Phase 2 script を破棄し、`scripts/generate_institutional_eval_set.py` (既存) に `--mode training` flag を追加して eval / training 両用化 | 1 時間 | 中 (既存 eval gen workflow に影響、回帰防止 test 追加要) |

**推奨**: **Option A** (script 単体で完結、TDD で安全)。

### Blocker 2: 🟢 mulmoclaude EN corpus 場所判明

**現状**: `data/training/mulmoclaude/` 不在。しかし develop worktree (`/Users/maenokota/share/work/github_kewton/photon-mlx-develop/data/processed/`) には:

| ファイル | 行数 | 用途 |
|---------|------|------|
| `train_multi.jsonl` | **7,188** | 既存 mulmoclaude 600-step 学習で使った EN 多リポ corpus |
| `val_multi.jsonl` | (要確認) | 同 val |

**Resolution**: `configs/institutional_docs_photon_retrain.yaml` の `train_corpora_mix` を以下に更新するか、本 worktree 配下に symlink:

```yaml
train_corpora_mix:
  "./data/training/institutional/train_jp.jsonl": 0.5
  "/Users/maenokota/share/work/github_kewton/photon-mlx-develop/data/processed/train_multi.jsonl": 0.5  # 旧 mulmoclaude EN
```

または symlink: `mkdir -p data/training/mulmoclaude && ln -s /Users/maenokota/share/work/github_kewton/photon-mlx-develop/data/processed/train_multi.jsonl data/training/mulmoclaude/train_en.jsonl`

**注意**: train_multi.jsonl は既に tokenized JSONL (`{"tokens": [...]}`) なので、JP 側も同 schema (= `iterate_mixed_batches` 入力形式) で生成する必要あり。私の `_serialise_session` は plain text turns を含む dict を出力するので **token 化ステップが script に欠けている** (Blocker 1 と一体)。

### Blocker 3: 🟢 mulmoclaude 600-step checkpoint 場所判明

**現状**: `checkpoints/` ディレクトリ不在。develop worktree (`/Users/maenokota/share/work/github_kewton/photon-mlx-develop/checkpoints/`) に:

| Checkpoint | step | val_loss |
|-----------|------|----------|
| `step_000100` 〜 `step_001000` | 100-1000 | 学習途上 |
| **`step_000600`** | 600 | **1.6238** (resume 起点候補) |
| `final` | (要確認) | 最終 |

**注意**: state.json の `best_val_loss=1.6238` は roadmap.md の `0.4525` と乖離 (PR #150 でも記録)。設計方針書 §11 リスク表で言及されていた値の差異は実値で確定 = **1.6238**。

**Resolution**: 環境変数 `PHOTON_CHECKPOINT_ROOT=/Users/maenokota/share/work/github_kewton/photon-mlx-develop/checkpoints` を設定 (develop と同じ運用)、または symlink `ln -s /Users/maenokota/share/work/github_kewton/photon-mlx-develop/checkpoints checkpoints`。

retrain yaml の `model.checkpoint_path` は当面 unset (random init は禁止 + #148 Phase A0 が PHOTON_ALLOW_RANDOM_INIT=1 でしか許可しない)。**resume_from** は `photon_mlx.trainer.train(resume_from=PHOTON_CHECKPOINT_ROOT/step_000600)` 形式で trainer 引数に渡す (`retrain.yaml` ではなく CLI / 環境変数で指定)。

---

## 訓練データ仮説 (commandmate eval 結果) について

> PHOTON が訓練 domain で baseline 比 -6.67pt 勝利

これは PR #153 (`measure(eval): re-run commandmate eval after #154 fixes`) で確定済の追加根拠。本 Issue #135 の本格再学習の意義 (制度文書 domain も訓練 domain にする) を補強する。

---

## ステップ 4 (本格学習) ETA

Blocker 解決済の前提で:

| Phase | 内容 | ETA |
|-------|------|------|
| 6.1 | corpus 生成 (qwen mlx local on M3 Ultra、2000 sessions × 6 turns、retry 込み) | **3-6 時間** GPU/Apple Silicon |
| 6.2 | corpus 数値検証 (jp_token_ratio / scenario_distribution / eval_overlap=0) | 5 分 |
| 6.3-6.5 | Phase 6 学習 (max_steps=10600, resume from step_000600, 1K step 毎の保存 + en eval) | **17-33 時間** GPU 連続稼働 |
| 7.1-7.4 | Phase 7 eval (各 checkpoint × 2-run × 6+ run) | 6-8 時間 |
| 8.1-8.7 | Phase 8 ロールアウト (採用 checkpoint 確定 / reports / CLAUDE.md / roadmap 更新) | 2 時間 |

**合計**: 28-49 時間 (3-5 日)、Day 1-2 corpus 生成 + Day 3-4 学習 + Day 5 eval/採用判定 という当初計画と整合。

---

## 次のアクション (ユーザー判断が必要)

1. ❓ **Blocker 1 (script API mismatch) Option A/B/C どれで進めるか?**
   - Option A 推奨 (1-2 時間 CPU、TDD で安全)
   - 完了後に corpus 生成 (Blocker 2/3 のパス更新付き) を起動

2. ❓ **Blocker 2 / 3 (mulmoclaude EN corpus / 600-step ckpt) 参照方法**
   - **a**. develop worktree 直接参照 (絶対パス、最も簡単) ← 推奨
   - **b**. 本 worktree に symlink
   - **c**. 本 worktree にコピー (容量大、推奨せず)

3. ❓ **`reports/institutional_photon_mt_eval_v2.md` 等の出力先確定**
   - 本 worktree か develop か (PR merge 時に整合させるなら本 worktree でよい)

ユーザー指示があるまで実行は保留。本セッションは idle 状態で待機します。

---

## 参考: Day 1-3 累計コミット (17 件)

```
be91682 merge develop into feature/issue-135-photon-retrain (Phase 6 prep)
994ba29 docs(issue-135): Day 2 EOD 進捗報告
344caae feat(scripts): DR4-001 CLI hardening for training corpus generator
34a833a feat(photon_mlx/trainer): dispatch to iterate_mixed_batches when mix set
d0277e3 docs(issue-135): Day 2 PM 進捗報告
f25022c feat(scripts): training corpus generator scaffolding (#135 / Phase 2)
87802fb feat(configs): institutional_docs_photon_retrain.yaml for #135 Phase 4-3
a2a9d5b feat(photon_mlx/checkpoint): integrity.json SHA-256 verification
587930c feat(torch_ref/config): add train_corpora_mix + val_split (Phase 4-1)
397b0bb feat(photon_mlx/data): add iterate_mixed_batches (Phase 3)
ecd1c2a docs(issue-135): Day 2 AM 進捗報告
8f1672d chore: add pre-commit config with detect-secrets (DR4-006)
f17c3a6 test(photon_pipeline): pin DR1-002 boundary in subprocess (Task 1.4)
1c920ae test(photon_pipeline): add Day 1 checkpoint load smoke test (削除済)
2dbf458 fix(photon_pipeline): load checkpoint in _build_photon_deps (S7-001)
57d7742 docs(issue-135): pm-auto-issue2dev 完了報告
ea2fa57 refactor(photon_mlx): extract checkpoint I/O into checkpoint.py
```
