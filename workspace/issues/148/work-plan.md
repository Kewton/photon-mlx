# 作業計画書 — Issue #148

## 1. Issue 概要

| 項目 | 内容 |
|------|------|
| **Issue 番号** | #148 |
| **タイトル** | `test(eval): re-establish true baseline — fixed PHOTON pipeline + new LLM upgrade (Qwen3.5-9B / Gemma4-26B)` |
| **Issue URL** | https://github.com/Kewton/photon-mlx/issues/148 |
| **サイズ** | XL (設計方針書 842 行、累計 66 findings: Issue レビュー 34 + 設計レビュー 32) |
| **優先度** | **High** (#135 Phase 6-8 本格再学習が Phase A0+A 完了待ちでブロック中) |
| **依存 Issue** | #135 (本格再学習 — Phase A0+A 完了で unblock)、#137 (V4 retrieval 採用済 — Phase C 再検証対象)、#143 (eval reproducibility — 未解消の場合 variance 記載が必要) |
| **ブランチ** | `feature/issue-148-rebaseline` (現在の作業ブランチ) |
| **設計方針書** | `workspace/design/issue-148-rebaseline-design-policy.md` |

### 背景と目的

S7-001 (PHOTON random-init) + #138 (tokenizer mismatch) の 2 つの critical bug を修正した PR (#141, #146, #147) により、**過去のすべての PHOTON eval 結果が無効化** された。本 Issue は「真の PHOTON ベースラインを再確立」し、#135 再学習の比較基準を作ることを目的とする。

あわせて、LLM backbone の新候補 2 件 (Qwen3.x 系 / Gemma4 系) の baseline-only 評価を実施し、将来の LLM 戦略決定基盤を構築する。

---

## 2. PR 単位の Phase 分割

| PR | ブランチ | Phase | 主成果物 | #135 への効果 |
|----|---------|-------|---------|-------------|
| **PR #1 (最優先)** | `feature/issue-148-rebaseline-phase-a` | Phase A0 + Phase A | checkpoint loading 実装 + unit test + yaml + reports 2 件 | **merge で #135 Phase 6-8 着手解禁** |
| **PR #2** | `feature/issue-148-rebaseline-phase-b` | Phase B | 新 LLM configs 4 件 + 比較 report | #135 をブロックしない |
| **PR #3** | `feature/issue-148-rebaseline-phase-c` | Phase C | 採用判定文書 + 本番 config 整合更新一式 | main マージ (LLM 採用確定) |

### PR #1 内の commit 分離方針

PR #1 は Phase A0 (code change) と Phase A (data/config change) の責務が異なるため **単一 PR 内で commit 単位を分離** する:

- **commit A0**: `baseline_reporag/photon_pipeline.py` 変更 + unit test 追加 (code change のみ)
- **commit A**: `configs/institutional_docs_photon.yaml` の `checkpoint_path` 設定 + eval 実行 + reports 2 件出力 (yaml + data change)

#135 unblock 解禁条件は「PR #1 (commit A0+A 両方) merge 後」とする。

---

## 3. Phase A0 の詳細タスク (本 worktree での TDD 実装対象)

> **本 worktree で実装するのは Phase A0 の code 部分のみ。** Phase A の eval 実行・report 作成は別途手動で実施する。Phase B/C は別ブランチで進める。

### Task A0-1: テスト先行作成 (RED フェーズ)

- **成果物**: `baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py` (新規)
- **実装内容**: 以下の test 関数を作成する
  - `test_build_photon_deps_loads_checkpoint_when_path_set`
    - `checkpoint_path` 設定時に `load_checkpoint` が呼ばれることを確認
    - `mocker.patch("baseline_reporag.photon_pipeline.load_checkpoint")` で mock
    - `PHOTON_CHECKPOINT_ROOT` を `tmp_path` に設定し root containment を通過させる
    - `weights.npz` + `state.json` を含む tmp checkpoint directory を作成
    - `_validate_tokenizer_id` を mock で bypass (Issue #139 invariant が先行 raise するため必須)
  - `test_build_photon_deps_raises_on_load_failure_by_default`
    - `load_checkpoint` が例外を投げた場合、デフォルトで `RuntimeError` を raise することを確認
    - `PHOTON_ALLOW_RANDOM_INIT` 未設定の状態で検証
  - `test_build_photon_deps_falls_back_when_PHOTON_ALLOW_RANDOM_INIT`
    - `PHOTON_ALLOW_RANDOM_INIT=1` 設定時のみ fail-soft (WARNING ログのみ、RuntimeError なし) であることを確認
  - `test_checkpoint_path_root_containment_validation`
    - `checkpoint_path` が `PHOTON_CHECKPOINT_ROOT` 外の場合、`load_checkpoint` 呼び出し前に `RuntimeError` を raise することを確認
    - `resolve(strict=True)` 後の root containment 検証の動作確認
  - `test_model_id_repo_id_allowlist`
    - `model.model_id` / `tokenizer.tokenizer_id` に URL (`http://...`)、local path (`./local`)、traversal (`../../x`)、先頭 dot (`.cache/x`)、制御文字、改行を渡した場合に拒否することを確認
    - 正常 slug (`mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`) は通過することを確認
  - `test_build_photon_deps_warns_when_no_checkpoint`
    - `checkpoint_path` 未設定時に WARNING ログが出ることを確認
  - `test_build_photon_deps_rejects_invalid_checkpoint_shape`
    - `weights.npz` または `state.json` が欠けた directory を渡した場合に `RuntimeError` (fail-fast) を確認
  - `test_build_photon_deps_rejects_checkpoint_outside_allowed_root`
    - 許可 root 外 directory を渡した場合に `RuntimeError` を確認
- **依存**: なし (先行実装)

### Task A0-2: `_build_photon_deps` への checkpoint load 実装 (GREEN フェーズ)

- **成果物**: `baseline_reporag/photon_pipeline.py` 編集
- **実装内容**:
  1. **module top-level import 追加**:
     ```python
     import os
     from pathlib import Path
     from photon_mlx.trainer import load_checkpoint
     ```
  2. **`_allowed_checkpoint_roots()` ヘルパ追加**:
     - `Path("checkpoints").resolve()` を base root とする
     - `PHOTON_CHECKPOINT_ROOT` 環境変数が設定されている場合は追加 root として追記
  3. **`_validate_checkpoint_dir(raw_path: str) -> Path` ヘルパ追加**:
     - `Path(raw_path).expanduser().resolve(strict=True)` で実パス解決
     - `is_relative_to(root)` で root containment 検証 (symlink escape 拒否)
     - `weights.npz` + `state.json` の存在確認
     - 不正時は `RuntimeError` で fail-fast
  4. **`_validate_model_id(model_id: str)` ヘルパ追加** (DR4-002):
     - HF repo-id allowlist: `re.fullmatch(r'[A-Za-z0-9._-]+/[A-Za-z0-9._-]+', model_id)`
     - URL (`http://`, `https://`)、local path (`./`, `/`, `~`)、traversal (`..`)、制御文字・改行を拒否
     - 不正時は `ValueError` を raise
  5. **`_build_photon_deps` 内に checkpoint loading 経路追加**:
     - `PhotonModel(photon_cfg)` で random-init 後
     - `checkpoint_path = cfg.model.get('checkpoint_path', None)` で取得
     - `checkpoint_path` が設定されている場合:
       - `_validate_checkpoint_dir(checkpoint_path)` で形状・root 検証
       - `load_checkpoint(model, ckpt_dir)` を呼び出し
       - 成功時: `_logger.info("PhotonModel: checkpoint loaded from %s", ckpt_dir.name)`
       - 失敗時:
         - `PHOTON_ALLOW_RANDOM_INIT=1` の場合: WARNING ログのみで続行 (test/CI 用途限定)
         - それ以外: `RuntimeError` を raise (fail-fast 原則)
     - `checkpoint_path` 未設定の場合: WARNING ログ (`"checkpoint_path not set. Model will run with random-init weights."`)
  6. **ログのセキュリティ** (DR4-004):
     - ログ・例外メッセージには raw absolute path を出さず、`ckpt_dir.name` (basename) または許可 root からの相対 path のみ出力
- **依存**: Task A0-1 (test が RED で失敗することを確認してから実装開始)

### Task A0-3: random-init 検出 WARNING 統合確認

- **成果物**: なし (既存実装の確認のみ)
- **確認内容**:
  - `photon_mlx/inference.py:_check_weight_initialization` が checkpoint load 後の weight 状態を **診断・警告** する責務のみを持ち、checkpoint load を実行しないことを確認
  - Phase A0 で正しく checkpoint が load された場合、weight σ が下がり `_check_weight_initialization` の WARNING が出なくなることを確認
  - WARNING 文言 `"Check model.checkpoint_path and load result"` が `_build_photon_deps` での load 実装後も適切な案内であることを確認 (必要に応じて文言を明確化)
- **依存**: Task A0-2

### Task A0-4: invariant test 追加 (DR-2 設計判断 #2)

- **成果物**: `tests/test_pipeline_factory_yaml_invariants.py` 編集
- **実装内容**:
  - **LLM model_id invariant**: 設計判断 #2 (§5.2 / §10.3) により **(B) LLM model_id は invariant 化しない** と決定済み。本 Task では invariant 化しないことを test ファイルのコメントに明記する
  - **セキュリティ regression test** (§10.4): `model.model_id` / `tokenizer.tokenizer_id` の unsafe 形状を拒否する table-driven test を追加
    - テスト対象: `http://...`, `../../x`, `.cache/x`, `org/..`, 改行入り の各パターン
    - 期待動作: `ValueError` を raise
    - 正常 slug (`mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`) は通過
  - **`photon_` prefix 命名規則テスト**: Phase B で新規作成する yaml が `photon_` prefix を持たないことを検証する test を追加 (Phase B yaml 作成後に有効化)
- **依存**: Task A0-2

### Task A0-5: yaml 編集

- **成果物**: `configs/institutional_docs_photon.yaml` 編集
- **実装内容**:
  - `model.checkpoint_path` を env var 参照形式で追記:
    ```yaml
    model:
      checkpoint_path: "${PHOTON_CHECKPOINT_ROOT}/mulmoclaude_600step"
    ```
  - **注意**: mulmoclaude 600-step ckpt の絶対 path は TBD (担当者 kewton 確認待ち)。Phase A0 では placeholder として env var 参照形式を設定し、Phase A 着手前に実際の path に置き換える
  - Qwen2.5 系の `vocab_size: 152064` / `tokenizer_id` は **変更しない** (Phase D まで維持)
  - Phase B-C 期間中の保護コメント追記 (Qwen2.5 系 2 件のみが対象):
    ```yaml
    # NOTE: tokenizer_id and vocab_size are intentionally kept as Qwen2.5 until
    # Phase D (#135) completes vocab reshape. Do not change during Phase B-C.
    ```
- **依存**: Task A0-2

### Task A0-6: docs 更新

- **成果物**:
  - `docs/deployment.md`: 以下を追記
    - `PHOTON_ALLOW_RANDOM_INIT=1` は **unit test / CI の negative-path 検証専用** であり、Phase A eval・Phase B/C smoke・本番 server 起動では未設定であること
    - `PHOTON_CHECKPOINT_ROOT` 環境変数の説明: checkpoint directory の検索 root を指定。未設定時は repo-local `checkpoints/` 配下が唯一の許可 root
    - checkpoint load 成功の確認方法 (起動ログの `PhotonModel: checkpoint loaded from <basename>` INFO 行)
    - HF cache warm-up 手順 (Phase C merge 前の事前 download 推奨)
  - `docs/troubleshooting.md`: 以下を追記
    - checkpoint load 失敗時の対処:
      - `RuntimeError: PhotonModel: checkpoint load failed` が出た場合の確認手順 (checkpoint directory の存在・形状確認、`weights.npz` / `state.json` の有無)
      - `PHOTON_CHECKPOINT_ROOT` の設定方法と root containment エラーの解消手順
      - `PHOTON_ALLOW_RANDOM_INIT=1` の **test/CI 用途限定** の注意書き (本番・eval では使用禁止)
- **依存**: Task A0-2

### Task A0-7: ruff / pytest 全パス確認

- **確認コマンド**:
  ```bash
  ruff check .
  ruff format --check .
  python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v
  ```
- **基準**: ruff 警告 0 件、format 差分なし、全 test パス (既知 pre-existing failure 2 件: `tests/test_generate_training_corpus.py` は除く)
- **依存**: 全 Task (A0-1〜A0-6)

---

## 4. Phase A の詳細タスク (eval 実行 — Phase A0 完了後に着手)

> Phase A は eval 実行を含むため、本 worktree での code 実装完了後に別途手動で実施する。

### Task A-1: mulmoclaude 600-step ckpt 所在確認・配置 (担当者 kewton 依存)

- **実施内容**: val_loss 0.4525 達成時の checkpoint の保存先 (リポ内 path / 共有サーバ / HF hub URL) を特定する
  - 確認すべき場所: `checkpoints/` 配下、共有 NAS、HF Hub (private)
  - checkpoint directory 形状確認: `weights.npz` + `state.json` の存在必須
  - HF Hub URL / safetensors 単体しかない場合: resolver / converter は本 Issue 範囲外として別 Issue に切り出し
- **成果物**: `configs/institutional_docs_photon.yaml` の `model.checkpoint_path` に実際の path を設定
- **Phase A 着手条件**: **このタスク完了が必須前提**。所在不明の場合は Phase A を開始しない

### Task A-2: smoke test (checkpoint load ログ確認)

- **実施内容**: `configs/institutional_docs_photon.yaml` に `checkpoint_path` 設定後、起動ログで checkpoint load を確認する
  ```bash
  python -m baseline_reporag.cli \
    --config configs/institutional_docs_photon.yaml \
    --repo-id fastapi_fastapi \
    --question "test"
  ```
- **合格条件**:
  - 起動ログに `PhotonModel: checkpoint loaded from <basename>` の INFO 行が出現すること
  - `_check_weight_initialization` の random-init WARNING が出ないこと
  - **ログに INFO 行が欠如した場合は eval 結果を invalid 扱いとする** (設計方針書 §3 DR-1)
  - `PHOTON_ALLOW_RANDOM_INIT` が未設定または `0` であることを確認
- **依存**: Task A-1 完了

### Task A-3: FastAPI MT eval 実行

- **実施内容**: FastAPI MT eval (5 repos × 8 questions × 6 turns) を 2 run 実行
  - 使用 config: `configs/institutional_docs_photon.yaml` (Qwen2.5 + loaded checkpoint)
  - 2 run の variance を記録 (Issue #143 未解消の場合は NC ± std を明示)
- **依存**: Task A-2 合格

### Task A-4: Institutional MT eval 実行

- **実施内容**: Institutional MT eval (30 sessions × 6 turns = 180 turns) を 2 run 実行
  - 使用 config: `configs/institutional_docs_photon.yaml`
  - 2 run の variance を記録
- **依存**: Task A-2 合格

### Task A-5: report 作成

- **成果物**:
  - `reports/gate2_judgment_v5_post_s7001.md` (新規)
    - Gate 2 v4 と同形式で PHOTON 真の数値を更新
    - 比較基準値: Gate 2 v4 final (Static NC PHOTON 20.0%, MT NC 6.7%、出典: `reports/gate2_judgment_v4_final.md`)
    - 「修正前 (random-init) vs 修正後 (loaded checkpoint)」の delta を記載
    - Phase A eval 起動ログの `checkpoint loaded from <basename>` INFO 行を貼付
    - `PHOTON_ALLOW_RANDOM_INIT` が未設定であったことを記録
    - #143 未解消の場合: 各 run の variance と NC ± std の信頼区間を明示
    - PHOTON Drift metrics / Safe RecGen trigger 発火率について「再測定 (follow-up Issue 番号)」または「out-of-scope 判定 + 理由」のいずれかを明記
  - `reports/institutional_photon_mt_eval_v2.md` (新規)
    - #113 と同形式で PHOTON 真の数値を更新
    - 比較基準値: #113 institutional eval (NC 11.39%、出典: `reports/institutional_photon_mt_eval.md`)
    - 同様の delta と variance 記載
- **依存**: Task A-3, A-4 完了

---

## 5. Phase B の詳細タスク (新 LLM eval — 別 PR、Phase A 完了後に着手)

> Phase B は Phase A 完了後に着手 (GPU 競合回避)。HF download のみ Phase A eval 完走待ち中に並列実行可。

### Phase B 着手前必須確認 (Pre-flight)

- [ ] 各 model_id を `huggingface-cli repo info <id>` で存在確認し、正式 slug を確定
  - `mlx-community/Qwen3.5-9B-MLX-8bit` (仮 slug — Issue 記載は非標準呼称の可能性あり)
  - `mlx-community/gemma-4-26b-a4b-4bit` (仮 slug)
  - 不在時は近い alternative に切替えて Issue 本文を更新する (代替 slug 選定基準: param 数 ±20%、同系統 vocab、mlx-community 認証 org、30 日以内 100 downloads 以上)
- [ ] 採用予定 model_id の正式 slug を `_TOKENIZER_ID_PATTERN` で fullmatch 検証
  - `python3 -c "import re; assert re.fullmatch(r'[A-Za-z0-9._-]+/[A-Za-z0-9._-]+', '<slug>'), 'Invalid'"`
- [ ] mlx-lm の現バージョンが Qwen3 / Gemma4 の loader を提供するか確認。未提供なら別 Issue に切り出す
- [ ] 実行環境が Mac Studio M3 Ultra (>=128GB unified memory) であることを確認
- [ ] Gemma4 で `mlx_lm.load(<gemma4-slug>)` の peak RSS 測定: unified memory の 80% 未満であること

### Phase B タスク一覧 (概略)

- **Task B-1**: 新規 baseline yaml 4 件作成
  - `configs/baseline_qwen35.yaml` (新規): `model.provider: "mlx_lm"` + 正式 slug
  - `configs/baseline_gemma4.yaml` (新規): 同上
  - `configs/institutional_docs_qwen35.yaml` (新規)
  - `configs/institutional_docs_gemma4.yaml` (新規)
  - **`photon_` prefix は絶対使用しない** (`_is_photon_profile_yaml` の誤検知防止)
  - **`model.provider` は `"mlx_lm"` を維持** し、`model.model_id` のみ差し替える
- **Task B-2**: HF download + smoke test
  - 各 LLM で `python -m baseline_reporag.cli --config configs/baseline_<llm>.yaml --repo-id fastapi_fastapi --question 'test'`
  - 合格条件: non-empty 応答、tokenizer mismatch なし、<180s、peak RSS 測定 + 記録
- **Task B-3**: 新 LLM 2 件 × 2 dataset × baseline × 2 run = 8 eval runs 実行
  - nondeterminism 検証: 各 LLM × 同質問 × 5 runs で variance 測定
  - Gemma4 MoE で variance 高い場合は 3 runs に増加
- **Task B-4**: grader バイアス検証
  - Qwen3.x 系 LLM 評価時に `qwen3.5:27b` grader との self-preference bias を openai/gpt-4o-mini でクロスチェック
- **Task B-5**: `reports/llm_baseline_comparison_2026q2.md` 作成
  - 3 LLM × 2 dataset の包括比較表 (NC, latency, memory, throughput)
  - Qwen2.5 の true PHOTON 結果と新 LLM baseline-only 結果を区別して表示
  - 新 LLM + PHOTON の本格 eval は Phase D (#135 範囲) に延期であることを明記
  - 代替 slug 採用時は選定根拠 (downloads 数、updated_at、param 数差、vocab 系統) を記載

---

## 6. Phase C の詳細タスク (採用判定 — 別 PR、Phase A+B 完了後)

> Phase C は Phase A + Phase B 完了後に着手。

### Phase C タスク一覧 (概略)

- **Task C-1**: 採用判定文書作成
  - `docs/llm_choice_decision_2026q2.md` (新規)
  - 判定基準: NC rate (overall / Turn 5-6)、latency (p50/p95)、memory peak、tokenizer 互換性
- **Task C-2**: `configs/baseline.yaml` 移行
  - `configs/baseline_qwen25.yaml` を新設 (rollback 用、secret-free 確認)
  - `configs/baseline.yaml` の `model_id` を採用 LLM に更新
- **Task C-3**: `configs/eval.yaml` 更新
  - backbone 差分を明記するコメント追加
  - `report.show_llm_backbone: true` フラグ追加
  - `photon_rag` variant は Qwen2.5 維持 (Phase D まで)
- **Task C-4**: `baseline_reporag/eval/institutional/llm_client.py` 対応
  - `QwenMLXAdapter` の default model 更新方針を決定・実施 (hardcode 更新 or yaml-driven 化)
- **Task C-5**: CI / weekly_eval.yml 対応
  - `timeout-minutes` 見直し (Gemma4 26B MoE 対応)
  - `workflow_dispatch` でドライランを実行し正常完了を確認
- **Task C-6**: cold-start smoke test
  - `baseline_reporag/server.py` / `baseline_reporag/cli.py` の既定 baseline.yaml で cold-start smoke
  - HF cache warm-up 事前実行の確認
- **Task C-7**: docs 7 件更新
  - `docs/deployment.md`, `docs/troubleshooting.md`, `docs/tutorial.md`
  - `workspace/mvp/architecture.md`, `workspace/mvp/app_guide.md`, `workspace/mvp/metrics.md`
  - `README.md`
  - `grep -rn 'Qwen2.5-Coder' docs/ workspace/mvp/ README.md` で全箇所を列挙し更新
- **Task C-8**: CLAUDE.md 更新
  - LLMバックエンド行を採用 LLM に更新
  - 品質チェック表に新 LLM smoke test コマンド行を追加
- **Task C-9**: `bench/run_all.py` + `bench/tests/test_run_all.py` 更新
  - backbone metadata 出力の実装と regression test 追加
- **Task C-10**: #137 V4 retrieval 再検証
  - bge-m3 + bge-reranker-v2-m3 が新 LLM でも正常に機能するか A/B 比較 (追加 ~2-3h)
- **Task C-11**: secret scan
  - `rg -n "HUGGING_FACE_HUB_TOKEN|OPENAI_API_KEY|api[_-]?key|token:|Authorization|Bearer " configs reports workspace/mvp docs README.md`
  - configs / reports に secret 値の混入がないことを確認

---

## 7. 品質チェック (各 PR 共通)

| チェック | コマンド | 基準 |
|---------|---------|------|
| テスト | `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | 全パス (除既知 2 件: `tests/test_generate_training_corpus.py`) |
| ruff lint | `ruff check .` | 警告 0 件 |
| ruff format | `ruff format --check .` | 差分なし |
| Baseline 疎通 | `python -m baseline_reporag.cli --config configs/baseline.yaml --repo-id fastapi_fastapi --question "test"` | 応答あり |
| Phase A smoke | `python -m baseline_reporag.cli --config configs/institutional_docs_photon.yaml --repo-id fastapi_fastapi --question "test"` | 応答あり + ログに `checkpoint loaded from` INFO 行 |
| Security static check | `rg -n "shell=True\|os\\.system\|yaml\\.load\\(\|(^[^.])\\.\\b(eval\|exec)\\(" baseline_reporag photon_mlx bench scripts tests` | 意図しない shell / unsafe yaml / Python code eval なし (`mx.eval` は対象外) |
| Secret scan | `rg -n "HUGGING_FACE_HUB_TOKEN\|OPENAI_API_KEY\|api[_-]?key\|token:\|Authorization\|Bearer " configs reports workspace/mvp docs README.md` | secret 値の commit / artifact 混入なし |

---

## 8. Definition of Done

### PR #1 (Phase A0+A) の Definition of Done

**Phase A0 (code 実装)**:
- [ ] Task A0-1 完了: `baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py` 新規作成、RED フェーズ確認
- [ ] Task A0-2 完了: `baseline_reporag/photon_pipeline.py` の `_build_photon_deps` に checkpoint loading 経路実装、GREEN フェーズ達成
- [ ] Task A0-3 完了: `photon_mlx/inference.py` の random-init WARNING 統合確認
- [ ] Task A0-4 完了: `tests/test_pipeline_factory_yaml_invariants.py` に invariant test 追加 + LLM model_id 非 invariant 化の方針明記
- [ ] Task A0-5 完了: `configs/institutional_docs_photon.yaml` に `checkpoint_path` (env var 参照) 追記 + 保護コメント追記
- [ ] Task A0-6 完了: `docs/deployment.md` + `docs/troubleshooting.md` に環境変数説明・対処手順追記
- [ ] Task A0-7 完了: ruff 警告 0 件・format 差分なし・全 test パス確認

**Phase A (eval 実行)**:
- [ ] Task A-1 完了: mulmoclaude 600-step ckpt の所在特定・`checkpoint_path` に実際の path 設定
- [ ] Task A-2 完了: smoke test 合格 (起動ログに `checkpoint loaded from <basename>` 確認)
- [ ] Task A-3 完了: FastAPI MT eval 2 run 完走
- [ ] Task A-4 完了: Institutional MT eval 2 run 完走
- [ ] Task A-5 完了: `reports/gate2_judgment_v5_post_s7001.md` + `reports/institutional_photon_mt_eval_v2.md` 出力
  - Gate 2 v4 / #113 との delta 記載
  - 起動ログの `checkpoint loaded from` INFO 行貼付
  - `PHOTON_ALLOW_RANDOM_INIT` 未設定の記録
  - Drift metrics / Safe RecGen 指標の follow-up 化判定 (再測定 or out-of-scope) 明記
  - #143 未解消の場合: NC ± std の信頼区間明示

**共通**:
- [ ] secret scan + backbone redaction PR レビュー OK
- [ ] `PHOTON_ALLOW_RANDOM_INIT` が Phase A eval 時に未設定であることを確認
- [ ] PR レビュー承認
- [ ] develop マージ
- [ ] #135 Phase 6-8 着手解禁を Issue / PR description に明記

### PR #2 (Phase B) の Definition of Done

- [ ] Phase B 着手前 Pre-flight チェック全完了 (slug 確認・loader 確認・環境確認・peak RSS 測定)
- [ ] 新規 yaml 4 件が `model.provider: "mlx_lm"` を維持し、`photon_` prefix なし
- [ ] 2 LLM × 2 dataset × baseline × 2 run = 8 eval runs 完走
- [ ] nondeterminism 検証 (各 LLM × 5 runs) 完了
- [ ] Qwen3.x 系の grader bias 検証 (cross-check) 完了
- [ ] `reports/llm_baseline_comparison_2026q2.md` 出力 (3 LLM 比較表、baseline-only 結果明示)
- [ ] develop マージ

### PR #3 (Phase C) の Definition of Done

- [ ] 採用 LLM 確定 + `docs/llm_choice_decision_2026q2.md` 出力
- [ ] CLAUDE.md / `configs/baseline.yaml` / `configs/eval.yaml` / `configs/baseline_qwen25.yaml` 整合更新
- [ ] `baseline_reporag/eval/institutional/llm_client.py` の `QwenMLXAdapter` 方針実施
- [ ] `.github/workflows/weekly_eval.yml` ドライラン合格
- [ ] cold-start smoke test 合格 + HF cache warm-up + rollback 手順 docs 記載
- [ ] docs 7 件 + CLAUDE.md 更新 (`grep -rn 'Qwen2.5-Coder'` で全箇所確認)
- [ ] bench / invariant 更新方針実施
- [ ] #137 V4 retrieval 再検証 A/B 完了
- [ ] secret scan 合格
- [ ] develop マージ → main マージ

---

## 9. リスクと緩和策

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| **S5-001 silent bug 再発**: Phase A0 実装で checkpoint load 経路が誤実装される | PHOTON が random-init のまま eval → 結果 invalid | fail-fast 設計 + Phase A eval 前に起動ログ `checkpoint loaded from <basename>` を必須確認。欠如時は eval 結果を invalid 扱い |
| **mulmoclaude 600-step ckpt の所在不明** (H5 Partially Confirmed) | Phase A 開始不能 | Phase A0 最初の subtask として担当者 (kewton) に確認。HF Hub URL / safetensors しかない場合は resolver / converter を別 Issue に切り出し。未特定なら Phase A を開始しない |
| **Qwen3.x / Gemma4 slug が HF 上に存在しない** (H9 Unverifiable) | Phase B 中断 | Phase B 着手前に `huggingface-cli repo info` で存在確認。不在時は near-alternative に切替えて Issue 更新 |
| **mlx-lm が新 LLM の loader を未提供** | Phase B scope 外化 | Phase B 最初の subtask として確認。未提供なら別 Issue 切り出し |
| **Gemma4 26B MoE の peak memory が OOM** | Phase B 実行不能 (64GB マシン) | Mac Studio M3 Ultra (>=128GB) 必須。Phase B 開始前に peak RSS 測定し 80% 未満を確認 |
| **photon profile yaml の vocab_size が新 LLM と不整合** | #138 invariant で ValueError → PHOTON pipeline 全停止 | Phase B-C 期間中は photon profile yaml を Qwen2.5 系のまま維持 (保護コメント明記)。更新は Phase D (#135 範囲) |
| **configs/baseline.yaml 更新で weekly_eval.yml が silent migration → CI timeout / 誤検知** | CI 安定性低下 | Phase C 受入条件に `workflow_dispatch` ドライラン必須化、timeout / threshold 事前評価 |
| **grader と被評価 LLM が同系列で self-preference bias 発生** | eval 信頼性低下 | Phase B 受入条件に cross-check grader (openai/gpt-4o-mini 等) による bias 検証を追加 |
| **#143 (Qwen nondeterminism) 未解消で 2 runs 平均の信頼区間が不明** | NC 数値の再現性低下 | 未解消のまま進める場合は各 run の variance を report に併記し NC ± std を明示 |
| **PHOTON Drift metrics / Safe RecGen 指標が invalid のまま残る** | 後続 Issue での誤参照 | Phase A reports 内で「再測定 (follow-up Issue 番号)」または「out-of-scope 判定 + 理由」を必ず明記 |
| **checkpoint_path が許可 root 外へ抜ける** | 任意 local path の checkpoint 読み込み / absolute path のログ露出 | repo-local `checkpoints/` または `PHOTON_CHECKPOINT_ROOT` 配下に限定し `resolve(strict=True)` 後の root containment で拒否。ログは basename のみ出力 |
| **model_id が URL / local path / traversal 形状になる** | 意図しない artifact load / supply chain リスク | HF repo-id allowlist を適用し URL / path / dot segment / 制御文字を拒否。HF revision を report に記録 |

---

## 10. 作業順序

### PR #1 の作業順序 (Phase A0 → Phase A)

```
[TDD サイクル]
Task A0-1 (test 先行作成 / RED)
  ↓
Task A0-2 (_build_photon_deps 実装 / GREEN)
  ↓
Task A0-3 (inference.py WARNING 統合確認)
  ↓
Task A0-4 (invariant test 追加)
  ↓
Task A0-5 (yaml 編集: checkpoint_path placeholder 設定)
  ↓
Task A0-6 (docs 更新: 環境変数・対処手順)
  ↓
Task A0-7 (ruff / pytest 全パス確認)
  ↓
[commit A0 作成]
  ↓
Task A-1 (ckpt 所在確認・checkpoint_path 実 path 設定) ← 担当者 (kewton) 確認必須
  ↓
Task A-2 (smoke test: ログで checkpoint load 確認)
  ↓
Task A-3 (FastAPI MT eval 2 run)
Task A-4 (Institutional MT eval 2 run)  ※ A-3 完走後
  ↓
Task A-5 (report 2 件作成)
  ↓
[commit A 作成]
  ↓
PR #1 作成 (Phase A0+A) → develop マージ → #135 解禁
```

### PR #2 の作業順序 (Phase B — PR #1 merge 後)

```
Phase B Pre-flight (slug 確認・loader 確認・環境確認)
  ↓
Task B-1 (yaml 4 件作成)
  ↓
Task B-2 (HF download + smoke test) ※ Phase A eval 完走中に HF download のみ並列可
  ↓
Task B-3 (8 eval runs + nondeterminism 検証 5 runs)
  ↓
Task B-4 (grader bias 検証)
  ↓
Task B-5 (比較 report 作成)
  ↓
PR #2 作成 (Phase B) → develop マージ
```

### PR #3 の作業順序 (Phase C — PR #2 merge 後)

```
Task C-1 (採用判定文書)
  ↓
Task C-2 〜 C-9 (config / docs / CI / bench 更新)
  ↓
Task C-10 (#137 V4 retrieval 再検証 A/B)
  ↓
Task C-11 (secret scan)
  ↓
PR #3 作成 (Phase C) → develop マージ → main マージ
```

---

## 11. 参考リソース

| 種別 | パス / URL |
|------|-----------|
| Issue 本文 (最終版) | `workspace/issues/148/design/latest-issue-body.md` |
| 設計方針書 | `workspace/design/issue-148-rebaseline-design-policy.md` |
| Issue レビューサマリー | `workspace/issues/148/issue-review/summary-report.md` |
| 設計レビューサマリー | `workspace/issues/148/multi-stage-design-review/summary-report.md` |
| 比較基準 (Gate 2 v4) | `reports/gate2_judgment_v4_final.md` |
| 比較基準 (#113 institutional) | `reports/institutional_photon_mt_eval.md` |
| 実装対象 | `baseline_reporag/photon_pipeline.py` |
| 新規 test ファイル | `baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py` |
| checkpoint load API | `photon_mlx/trainer.py:load_checkpoint` |
| invariant test | `tests/test_pipeline_factory_yaml_invariants.py` |
| yaml (Phase A0) | `configs/institutional_docs_photon.yaml` |
| GitHub Issue | https://github.com/Kewton/photon-mlx/issues/148 |
| ブロック対象 | #135 (Phase 6-8 本格再学習) — PR #1 merge で解禁 |
