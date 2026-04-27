# Issue #115 PM Auto Dev 進捗報告 (Iteration 1)

実行日: 2026-04-26 〜 2026-04-27
ブランチ: `feature/issue-115-wizard-jp-prompt`

## 実行フェーズサマリー

| Phase | 内容 | ステータス | 備考 |
|-------|------|-----------|------|
| 1 | Issue 情報収集 | 完了 | gh issue view 115 |
| 2 | TDD 実装 | 完了 | 7 タスク全完了、355/355 tests pass |
| 2.5 | Codex コードレビュー (Round 1) | 完了 | CB-001 (high) 検出 → 修正 |
| 2.5 | Codex コードレビュー (Round 2) | 完了 | CB-001 解消、CB-002 (medium) → Phase 4 で対応 |
| 3 | 受入テスト | 完了 | 14/14 acceptance criteria pass |
| 4 | リファクタリング (simplify) | 完了 | 6 修正適用 (DRY/Security/Comment/Note/Selectbox/Cache) |
| 5 | ドキュメント最新化 | 完了 (不要) | CLAUDE.md 更新不要 (既存ファイル編集のみ) |
| 6 | 進捗報告 | 完了 | 本レポート |

## 実装内容

### D-2: Streamlit wizard に institutional_docs domain template

- `app/components/wizard.py`:
  - `_DOMAIN_TEMPLATES` 定数 (pre-#137 確定形、3 キー: embedding.model_id / symbol_graph.enabled / reranker.model_id)
  - `generate_yaml_from_wizard()` に template merge (base_profile=`institutional_docs` 限定)
  - `validate_generated_repo_id()` helper (DR3-001、`baseline_reporag/config.py:validate_repo_id` 再利用で path traversal 防御)
- `app/photon_app.py`:
  - selectbox options を `_BASE_PROFILE_OPTIONS + list(_wizard._DOMAIN_TEMPLATES.keys())` で動的構築 (Quality #5)
  - 保存処理直前に `validate_generated_repo_id()` 呼び出し
  - `_resolve_active_config_path(proj)` / `_pipeline_cache_key(name, path)` helper 追加 (CB-001 修正)
  - `_run_query()` で `proj.photon_config_path or proj.config_path` を優先
  - pipeline cache eviction (CB-002 対応): cache miss 時に古い `pipeline_<name>_*` entry を session_state から削除
  - `_launch_eval_job()` も `_resolve_active_config_path()` 使用に統一 (DRY)

### D-3: 多言語 prompt 調整

- `baseline_reporag/generation/prompt.py`:
  - `detect_language(question) -> Literal["ja", "en", "other"]` (1 ループ集計、ja 30%/en 50% 閾値、edge case 対応)
  - `_JP_INSTITUTIONAL_HINT` 定数 (制度文書条件付きヒント 3 行)
  - `_resolve_system_prompt(question)` helper (DR1-005 SRP split)
  - `build_messages()` 内で `_resolve_system_prompt(question)` 呼び出し (シグネチャ不変)

## テスト

### 新規追加 test (40+ tests)

| ファイル | test class | 件数 | カバレッジ |
|---------|-----------|------|-----------|
| `test_prompt.py` | `TestDetectLanguage` | 15 | ja/en/edge case 全分岐 |
| `test_prompt.py` | `TestResolveSystemPrompt` | 3 | 英語/日本語/空文字 |
| `test_prompt.py` | `TestBuildMessagesLanguageBranching` | 3 | golden snapshot + substring |
| `test_photon_pipeline.py` | `TestBuildMessagesJapaneseFollowUp` | 1 | include_few_shot=False で日本語ヒント維持 |
| `test_photon_app_components.py` | `TestDomainTemplatesInstitutional` | 3 | merge / no-op / yaml source-of-truth guard |
| `test_photon_app_components.py` | `TestValidateGeneratedRepoId` | 7 | DR3-001 + path traversal/shell metachar 防御 |
| `test_photon_app_helpers.py` | `TestResolveActiveConfigPath` | 2 | photon_config_path 優先 / fallback |
| `test_photon_app_helpers.py` | `TestPipelineCacheKey` | 2 | 構造 / 異 path → 異 key |
| `test_photon_app_helpers.py` | `TestRunQueryUsesPhotonConfigPath` | 3 | wizard YAML 優先 / fallback / cache eviction |

### 全体テスト結果

```
623 passed, 2 skipped, 2 failed (pre-existing in test_generate_training_corpus.py per CLAUDE.md)
```

### 既存 test 不変確認

- `test_prompt.py` 既存 3 class (`TestBuildMessages`, `TestFormatHintFewShot`, `TestFlattenMessagesForPlainLM`): 改変なしで pass
- `test_photon_pipeline.py` の `TestBuildMessagesSessionSummary`: 改変なしで pass
- T-C6 既存 5 test: 改変なしで pass

## 品質チェック

| チェック項目 | コマンド | 結果 |
|-------------|----------|------|
| Lint | `ruff check .` | All checks passed (0 件) |
| Format | `ruff format --check .` | 147 files already formatted (差分なし) |
| Test | `pytest baseline_reporag/tests/ tests/` | 623 passed (2 pre-existing fail, scope 外) |

## Codex クロスレビュー結果

| Round | 指摘数 (critical/high/medium/low) | verdict | 対応 |
|-------|-----------------------------------|---------|------|
| 1 | 0/1/0/0 | needs_fix | CB-001 (`_run_query` config_path 不整合) を修正 |
| 2 | 0/0/1/0 | pass | CB-002 (pipeline cache leak) を Phase 4 リファクタリングで対応 |

## simplify レビュー結果

3 つの並列レビュー (Reuse / Quality / Efficiency) を実施し、6 修正を適用:

1. **DRY**: `_launch_eval_job()` を `_resolve_active_config_path()` 使用に統一
2. **Security**: `validate_generated_repo_id()` で `validate_repo_id()` 再利用 (path traversal 防御)
3. **Quality**: 不要なタスク参照コメント (Issue #115 / DR-XXX / CB-001 narrative) 削除
4. **Quality**: `detect_language` Note 段落を 2 文に短縮
5. **Quality**: selectbox options を `_DOMAIN_TEMPLATES.keys()` から動的構築
6. **Efficiency (CB-002)**: pipeline cache eviction で stale pipeline メモリリーク防止

スキップ項目 (skip 理由):
- `_resolve_system_prompt` の inline 化 (設計 DR1-005 維持)
- `_DOMAIN_TEMPLATES` を `apply_best_practice` と merge (設計の責務分離維持)
- YAML safe_load helper 抽出 (3 箇所のみで coupling 増加)
- テスト copy-paste 削減 (細かい改善で工数対効果薄)
- `_SYSTEM_JA` 事前計算 (マイクロ最適化)

## 変更ファイル統計

```
 app/components/wizard.py                       |  69 ++++++++++
 app/photon_app.py                              |  64 ++++++++-
 baseline_reporag/generation/prompt.py          |  85 +++++++++++-
 baseline_reporag/tests/test_photon_pipeline.py |  29 ++++
 baseline_reporag/tests/test_prompt.py          | 132 ++++++++++++++++++
 tests/test_photon_app_components.py            | 164 ++++++++++++++++++++++
 tests/test_photon_app_helpers.py               | 184 +++++++++++++++++++++++++
 7 files changed, 719 insertions(+), 8 deletions(-)
```

## 受入条件達成 (14/14 pass)

設計方針書 §10 完了条件 14 項目 (Issue 受入条件 13 項目 + DR1-001 CI guard) すべて pass。詳細は `acceptance-result.json` 参照。

## 残課題・フォローアップ

- **手動 UAT** (Streamlit UI): `streamlit run app/photon_app.py` で wizard 経由 institutional_docs 選択 → 生成 YAML 確認 → chat で日本語制度文書質問の動作確認が必要
- **#137 (institutional retrieval A/B)** の merge 後: `_DOMAIN_TEMPLATES["institutional_docs"]` に bge-m3 / bge-reranker-v2-m3 / batch_size=32 / max_input_chars=8192 を追加する follow-up Issue 必要
- **CB-002 残存リスク**: pipeline cache eviction で stale pipeline 削除済だが、同一 path への YAML 上書き時 (rare) は cache key 変わらず古い pipeline 再利用される可能性。content hash 化は将来検討

## 次のアクション

- [ ] 手動 UAT
- [ ] `git add` + commit (Conventional Commits 準拠)
- [ ] `/create-pr` で PR 自動作成
