# Code Review Checklist

このドキュメントは PHOTON-RepoRAG プロジェクトの PR / コードレビュー時に確認する項目をまとめた **Single Source of Truth** です。設計方針書 / Issue / 個別の skill 定義からこの checklist を参照します。

> Issue #140 (S7-001 follow-up) で新設。S7-001 (PHOTON eval が random-init weight で動作) のような **silent bug** を発見するための運用ガイド。

## 1. 命名規則チェック (production code path)

PR レビュー時に以下を確認:

- [ ] `_Stub` で始まるシンボルが production import path にない (例: `_StubTokenizer`, `_StubEncoder`)
- [ ] `_Mock`, `_Dummy`, `_Placeholder` で始まるシンボルも同様
- [ ] `stub_`, `mock_`, `dummy_` で始まる関数名が production にない
- [ ] `# TODO: replace with real ...`, `# placeholder for production`, `# scaffolding` のコメントが残っていない
- [ ] config field の `getattr(cfg, "X", default)` で default が production で使われる場合は **fail-loud (raise) に変更**

### 適用範囲

- 対象: `baseline_reporag/`, `photon_mlx/`, `torch_ref/` の production import path
- **除外** (許容): `*/tests/**`, `bench/**`, `scripts/dev/**`, `demo/**`, `conftest.py`

### CI grep 例 (運用 PR レビュー時の手動コマンド)

```bash
git grep -nE '(_Stub|_Mock|_Dummy|_Placeholder)' baseline_reporag/ photon_mlx/ torch_ref/ ':!*/tests/**'
```

production code path に scaffolding が残ると **silent bug** (S7-001 型) を生む。CI 自動化 (pytest 経由の境界 test) は **#139 Task 1** で実装する (本 Issue #140 では docs 整備のみ)。

> 注: `baseline_reporag/photon_pipeline.py::_StubTokenizer` は #138 修正後も fallback として残存している。本 checklist では「**新規追加禁止**」のルールを適用し、既存分は #139 で扱う。

## 2. silent failure 検出ガイド (S7-001 型 bug 防止)

### 2.1 起動時 sanity check

新規モジュール / pipeline が「random-init モデルで動作する」可能性のある場合、起動時に sanity check を行う:

- 例: `PhotonInference.__init__` の `_check_weight_initialization` (Issue #140 Task 4)
- ログ出力は **スカラー値のみ** (σ, threshold 等)。tensor 自体・サンプル要素・weight matrix 内容はログに出さない (Issue #58 CB-002 / #64 CB-003 と同方針)。
- 初期実装は **WARNING のみ** (raise / exit 1 への昇格は段階的に行う)

### 2.2 Codex クロスレビュー (multi-stage review)

opus 単独レビューは「設計の論理整合性」を確認できるが、「実装と設計のギャップ」を発見しにくい。**Codex 担当 Stage は必須**:
- `/multi-stage-design-review`: Stage 3 (影響分析) / Stage 4 (セキュリティ) — `--agent codex` 経由
- `/multi-stage-issue-review`: Stage 5-8 (2回目イテレーション) — 同上
- skip 時は WARNING + completion report 記録 (Issue #140 / S7-001 follow-up)

### 2.3 reviewer フィールド検証

`/pm-auto-issue2dev` Phase 1/3 / `/pm-auto-design2dev` Phase 2 の完了判定で、Codex 担当 Stage の結果 JSON が `reviewer="codex"` を持つことを確認 (Claude による不正な上書きを検出)。

## 3. セキュリティ checklist

### 3.1 入力検証

- [ ] 外部 (yaml, env, CLI 引数, attacker-controllable) から入る数値 field は型・範囲を `__post_init__` で検証
  - 例: `embedding_random_init_threshold` は bool reject + finite + 非負を強制 (Issue #140 / DR4-002)
- [ ] path-like 入力は数値 ID / 許可文字に制限してから path に展開 (path traversal 防止)

### 3.2 ログ出力

- [ ] 例外メッセージにユーザー入力 / tensor 内容 / API key / token / file path などが含まれないことを確認
- [ ] 例外のログは `type(exc).__name__` のみを出すパターンが推奨 (Issue #58 CB-002 / #64 CB-003)
- [ ] reviewer 値などの string 値はログ出力前に制御文字を置換 (log injection 防止 — Issue #140 / DR4-003)

### 3.3 shell snippet / command injection

- [ ] PM コマンド Markdown 内の bash snippet で変数を path / Python code 文字列に直接展開しない
- [ ] 数値変数は `case "$X" in *[!0-9]*) ... esac` 等で事前検証
- [ ] Python script への path 受け渡しは `sys.argv[N]` 経由、code 文字列への埋め込みは禁止

## 4. テスト品質チェック

- [ ] caplog アサーションが既存の WARNING ログと衝突しないか確認
- [ ] random-init モデルを使う test では起動時 sanity check の WARNING が pytest 出力を汚さないよう、test 用 cfg で閾値を抑制 (例: `embedding_random_init_threshold = 1e9` の有限大値、`float('inf')` は production validation との衝突回避のため不可 — Issue #140 / DR4-002)
- [ ] string-existence test は具体フレーズ (例: 「Codex 担当 Stage は必須」) で完全一致 assert (既存類似文字列との衝突を避ける)
- [ ] **private API (`_build_photon_deps` / `_resolve_checkpoint_path` / `_load_photon_checkpoint`) の signature を変更する場合**、`tests/integration/test_photon_real_weights.py` (Issue #145) と `baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py` の両方が追従更新されているか確認。前者は real PhotonModel + 実 checkpoint で load path 全体を pin、後者は MagicMock 境界で code path を pin する補完関係 (DR3-001)

## 6. Prompt 中立性チェック (Issue #178 follow-up)

production prompt template (`baseline_reporag/generation/prompt.py`) を変更する PR では:

- [ ] corpus type に依存する語 (`code chunks` / `コードチャンク` / `code` / `repository` 等) を generic container として使っていない
- [ ] specific な対象を「コード」「リポジトリ」と呼ぶのは可、generic container は `documents` / `ドキュメント` で統一
- [ ] LLM-visible prompt の standalone `chunk(s)` (regex `\bchunks?\b`) / `根拠 chunk` を generic container として使っていない (内部 Python identifier `Chunk` / `chunk_id` / `pack.chunks` は除外)
- [ ] system role が `code repository analysis` 固定になっていない (`evidence-grounded ... document analysis` 系の neutral 表現)
- [ ] `[C:N]` citation marker は `resolve_citations` (`baseline_reporag/citation.py`) との契約のため改名しない
- [ ] `ABSTAIN_MARKER = "根拠が不足しています"` (`prompt.py`) を変更しない (grader 互換)
- [ ] ja / en 両 prompt (`_SYSTEM` + `_JP_INSTITUTIONAL_HINT`) を同期更新
- [ ] 過去 Issue 設計書 (`workspace/design/issue-N-*-design-policy.md`) は履歴保全のため prompt 中立化と同期しない
- [ ] PHOTON checkpoint 互換性: training input に `_SYSTEM` が含まれていないことを `configs/institutional_docs_photon_retrain.yaml` / `photon_mlx/data.py` / `scripts/generate_institutional_training_corpus.py` の grep で確認 (詳細は Issue #178 設計書 Section 8.1)。**grep 結果は PHOTON checkpoint 再訓練不要の根拠に限られる**
- [ ] **institutional 軽量 eval (50 sample) は原則実施** (Issue #178 設計書 Section 8.3)。generation prompt distribution は変わるため、grep 結果のみで skip しない
  - 実施する場合: NC rate ≤ 2% / no-citation rate +3pt 以内 / reasoning leak 0% を満たす
  - skip する場合は **例外として** 以下を PR description に記録: (i) skip 理由 (時間/環境制約等)、(ii) Section 8.1 grep 結果、(iii) unit test (`baseline_reporag/tests/test_prompt.py::TestPromptDomainNeutrality`) + smoke check 結果

回帰防止: `baseline_reporag/tests/test_prompt.py::TestPromptDomainNeutrality` が banned-substring assert を pin する SSoT。詳細禁止 literal 一覧は Issue #178 設計書 Section 8.0 を参照。

## 5. 関連リンク

- 設計方針書テンプレート: `workspace/design/issue-{N}-*-design-policy.md`
- skill 定義:
  - `/multi-stage-issue-review`: `.claude/commands/multi-stage-issue-review.md`
  - `/multi-stage-design-review`: `.claude/commands/multi-stage-design-review.md`
  - `/pm-auto-issue2dev`: `.claude/commands/pm-auto-issue2dev.md`
  - `/pm-auto-design2dev`: `.claude/commands/pm-auto-design2dev.md`
- 関連 Issue: #140 (本 checklist), #139 (CI grep 自動化), #138 (tokenizer mismatch), #135 (PHOTON 再学習), #178 (prompt 中立性)
