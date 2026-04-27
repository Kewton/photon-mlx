# Issue #115 作業計画書

## Issue: feat(app): Streamlit wizard に制度文書 domain template と日本語 prompt 調整
- **Issue番号**: #115
- **サイズ**: M (実装 ~80 行 + テスト ~150 行)
- **優先度**: Medium (Phase 2 polish)
- **依存Issue**: #112 (CLOSED, config 確立)
- **関連Issue**: #133 / #137 (institutional retrieval A/B)、#135 / #139 (PHOTON pipeline 周辺)
- **ブランチ**: `feature/issue-115-wizard-jp-prompt` (作成済)
- **設計方針書**: `workspace/design/issue-115-wizard-jp-prompt-design-policy.md`

---

## Phase 0: 着手前確認 (PR 直前)

### Task 0.1: pre/post #137 確定 (DR1-003 確定運用ルール)
- [ ] `git fetch origin main` 実行
- [ ] `git diff origin/main configs/institutional_docs.yaml tests/test_pipeline_factory_yaml_invariants.py` で #137 採用状態確認
- [ ] **判定**: 
  - **pre-#137** (現在の状態): 3 キー (`embedding.model_id`, `symbol_graph.enabled`, `reranker.model_id`) を採用
  - **post-#137**: 5 キー (上 3 + `embedding.batch_size`, `embedding.max_input_chars`、model_id は bge 系) を採用
- [ ] 確定結果を本作業計画 §0 のチェックボックスに記録、不採用側の記述は削除

### Task 0.2: 並列ブランチの確認
- [ ] `commandmatedev ls` で #135 / #139 の進捗確認
- [ ] `baseline_reporag/photon_pipeline.py` / `baseline_reporag/tests/test_photon_pipeline.py` への変更を grep で確認
- [ ] 衝突しそうなら rebase 順序を事前に決定

---

## Phase 1: 実装 (TDD Red→Green→Refactor)

### Task 1.1: `detect_language()` 実装 (TDD)
- **成果物**: `baseline_reporag/generation/prompt.py` に関数追加
- **依存**: なし
- **手順**:
  1. **Red**: `baseline_reporag/tests/test_prompt.py` に新規 test class `TestDetectLanguage` を追加
     - 5 個の ja 質問 (純日本語、漢字主体、ひらがな主体、カタカナ主体、混在)
     - 5 個の en 質問 (純英語、英記号混在、技術用語多め、長文、短文)
     - edge case: 空文字、空白のみ、記号のみ、絵文字のみ、ローマ字日本語
  2. **Green**: `detect_language(question: str) -> Literal["ja", "en", "other"]` を実装
     - 設計書 §3-2 docstring の Note (DR1-002) に従い 1 ループ集計
     - codepoint range: hiragana U+3040-U+309F / katakana U+30A0-U+30FF / CJK U+4E00-U+9FFF
     - ja 閾値 30% (分母=total)、en 閾値 50% (分母=non_space_len)
     - 非空白 0 → `"other"` (ZeroDivision 回避)
  3. **Refactor**: コード整理、型ヒント、docstring

### Task 1.2: `_resolve_system_prompt()` helper 実装 (DR1-005 / DR3-002)
- **成果物**: `baseline_reporag/generation/prompt.py` に内部 helper 追加
- **依存**: Task 1.1
- **手順**:
  1. **Red**: `TestResolveSystemPrompt` に 3 tests
     - 英語入力 → `_SYSTEM` そのまま
     - 日本語入力 → `_SYSTEM + _JP_INSTITUTIONAL_HINT` 連結
     - 空文字 → `_SYSTEM` そのまま
  2. **Green**: `_JP_INSTITUTIONAL_HINT` constant (制度文書ヒント文面 3 行) と `_resolve_system_prompt(question)` 実装
  3. **Refactor**: 命名整理、private 命名 (`_` prefix) 維持

### Task 1.3: `build_messages()` への helper 統合
- **成果物**: `baseline_reporag/generation/prompt.py:155-175` の `build_messages()` 修正
- **依存**: Task 1.2
- **手順**:
  1. **Red**: `TestBuildMessagesLanguageBranching` (3 tests) を追加
     - 英語質問 → `messages[0]["content"]` が現状 `_SYSTEM` と完全一致 (golden snapshot)
     - 日本語質問 → `messages[0]["content"]` に substring "制度文書を根拠に回答する場合は" を含む
     - 日本語質問 → 条件節「制度文書を根拠に回答する場合は」を含み無条件強制でないこと
  2. **Green**: `system_content = _resolve_system_prompt(question)` の 1 行で参照する形に修正
  3. **Refactor**: 既存 docstring 更新 (シグネチャ不変を明記)
  4. **既存 test pass 確認**: `pytest baseline_reporag/tests/test_prompt.py -v` で既存 3 class + 新規 2 class 全パス

### Task 1.4: PHOTON follow-up 経路の test 追加
- **成果物**: `baseline_reporag/tests/test_photon_pipeline.py` に 1 test 追加
- **依存**: Task 1.3
- **手順**:
  1. **Red**: `TestBuildMessagesJapaneseFollowUp::test_japanese_hint_persists_with_no_few_shot` 追加
     - `build_messages(question="<日本語>", evidence_text="...", session_summary="...", include_few_shot=False)` を呼ぶ
     - シグネチャ変更なしで動作
     - system prompt に「制度文書を根拠に回答する場合は」が含まれる
  2. **Green**: 既に Task 1.3 で実装済のため pass するはず。fail なら build_messages の分岐ロジックを再確認
  3. **既存 `TestBuildMessagesSessionSummary` が変更なしで pass することも確認**

### Task 1.5: `_DOMAIN_TEMPLATES` 定数追加 (D-2 part 1)
- **成果物**: `app/components/wizard.py` に定数追加
- **依存**: Task 0.1 (pre/post #137 確定)
- **手順**:
  1. **Red**: `tests/test_photon_app_components.py` に `test_domain_template_matches_institutional_yaml` 追加
     - `yaml.safe_load(open("configs/institutional_docs.yaml"))` を直接 parse
     - `_DOMAIN_TEMPLATES["institutional_docs"]` の各 path tuple について yaml dict から該当値を抽出
     - 完全一致を assert
  2. **Green**: 設計書 §3-1 の literal を `wizard.py` に追加 (pre-#137 確定形 = 3 キー)
  3. **Refactor**: 既存 `BEST_PRACTICE_KEYS` の真上に配置、コメントで責務分離を明記

### Task 1.6: `generate_yaml_from_wizard()` に template merge ロジック統合 (D-2 part 2)
- **成果物**: `app/components/wizard.py:220-287` の `generate_yaml_from_wizard()` 修正
- **依存**: Task 1.5
- **手順**:
  1. **Red**: `tests/test_photon_app_components.py` の T-C6 に 2 tests 追加
     - `test_institutional_docs_template_merges_3_keys`: `base_profile="institutional_docs"`, `base_yaml_text` に意図的に異なる 3 値を embed → 生成 YAML が `_DOMAIN_TEMPLATES` 値で上書きされる
     - `test_domain_template_no_op_on_other_profiles`: `base_profile="photon_small"` で institutional template が effect しない
  2. **Green**: `_WIZARD_TOGGLE_MAPPING` 適用直後に `template = _DOMAIN_TEMPLATES.get(base_profile)` 追加 (設計書 §3-1 merge ロジック)
  3. **Refactor**: コメントで「base_profile == template_name 限定」の根拠 (DR-S3-002 / 設計判断 #1) を註記

### Task 1.7: Streamlit UI 拡張 (D-2 part 3, DR3-001 含む)
- **成果物**: `app/photon_app.py:1234-1238` の selectbox 修正 + `repo_id` validation 追加
- **依存**: Task 1.6
- **手順**:
  1. selectbox の `options=[...]` 末尾に `"institutional_docs"` を追加 (label `"Config template"` 不変)
  2. `repo_id` validation: 設計書 DR3-001 の検証ロジックを `app/photon_app.py` の保存処理直前に追加
     - generated_yaml を `yaml.safe_load` して `repo.repo_id` を取得
     - Project 選択 `repo_id` と一致しなければ `st.error()` で拒否
  3. **動作確認**: `streamlit run app/photon_app.py` で wizard を実際に操作し、generated YAML を目視確認

### Task 1.8: 動作確認 (UI/E2E)
- **成果物**: 手動動作確認結果メモ
- **依存**: Task 1.7
- **手順**:
  1. Streamlit 起動: `streamlit run app/photon_app.py`
  2. PHOTON settings expander で `Config template = institutional_docs` を選択
  3. 生成 YAML を Save → `data/projects/<id>/config.yaml` に embedding=multilingual-e5-small / reranker=cross-encoder/ms-marco / symbol_graph.enabled=false が反映されることを確認
  4. 既存 `photon_small` 選択時に generated YAML が現状と完全一致することを確認 (regression check)
  5. 日本語質問 → CLI/UI で実行し、回答に条文番号引用が出る (制度文書 chunk 引いた時)、出ない (code repo chunk 引いた時) を目視確認
  6. 英語質問 → 既存挙動と一致を確認

---

## Phase 2: テスト・品質チェック

### Task 2.1: 全テスト実行
```bash
python -m pytest baseline_reporag/tests/test_prompt.py baseline_reporag/tests/test_photon_pipeline.py tests/test_photon_app_components.py -v
```
- [ ] 新規 test 全 pass (TestDetectLanguage 5+ / TestResolveSystemPrompt 3 / TestBuildMessagesLanguageBranching 3 / TestBuildMessagesJapaneseFollowUp 1 / T-C6 拡張 3)
- [ ] 既存 test 全 pass (test_prompt.py 既存 3 class、test_photon_pipeline.py の TestBuildMessagesSessionSummary、T-C6 既存 5 test)

### Task 2.2: 全体テスト
```bash
python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v
```
- [ ] 既存 約 507/509 test の pass 状態を維持 (pre-existing failure 2 件は除外)
- [ ] 新規 test 12+ test の追加で全 ~519/521 test pass

### Task 2.3: Lint/Format
```bash
ruff check .
ruff format --check .
```
- [ ] `ruff check .` で警告 0 件
- [ ] `ruff format --check .` で差分なし

### Task 2.4: Baseline 疎通確認
```bash
python -m baseline_reporag.cli --config configs/institutional_docs.yaml --repo-id institutional_documents --question "第1条には何が規定されていますか"
```
- [ ] 応答が返る
- [ ] 日本語回答 + `[C:N]` citation 形式が含まれる

---

## Phase 3: ドキュメント・コミット

### Task 3.1: ドキュメント更新確認
- [ ] `CLAUDE.md` 更新は **不要** (新規モジュール追加なし、既存ファイル編集のみ)
- [ ] `docs/deployment.md` / `docs/troubleshooting.md` は institutional_docs template 選択方法を追記する必要があれば 2-3 行追記
- [ ] 設計方針書 (`workspace/design/issue-115-wizard-jp-prompt-design-policy.md`) の最終確認

### Task 3.2: コミット (Conventional Commits)
- [ ] `feat(prompt): #115 Phase 1 — detect_language() helper 追加 + 日本語制度文書ヒント分岐`
- [ ] `feat(wizard): #115 Phase 2 — _DOMAIN_TEMPLATES 定数 + institutional_docs template merge`
- [ ] `feat(app): #115 Phase 3 — Streamlit wizard で institutional_docs 選択可能化 + repo_id validation`
- [ ] `test(wizard): #115 — _DOMAIN_TEMPLATES と institutional_docs.yaml の CI guard`
- [ ] 各コミットを Co-Authored-By: Claude で署名

### Task 3.3: PR 作成
- [ ] `/create-pr` で自動 PR 生成
- [ ] PR タイトル: `feat(app): #115 Streamlit wizard に institutional_docs template + 日本語制度文書 prompt`
- [ ] ラベル: `feature`, `phase-2`
- [ ] 本文に設計方針書とレビュー履歴へのリンクを記載

---

## 品質チェック項目 (Definition of Done)

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| ビルド | `python -m pytest` | エラー 0 件 |
| Lint | `ruff check .` | 警告 0 件 |
| テスト | `python -m pytest` | 全テストパス (pre-existing 2 件除く) |
| フォーマット | `ruff format --check .` | 差分なし |
| Baseline 疎通 | `python -m baseline_reporag.cli --config configs/institutional_docs.yaml --question "test"` | 応答あり |
| Streamlit 動作 | `streamlit run app/photon_app.py` で wizard 操作 | institutional_docs template が effect、photon_small 不変 |

## Definition of Done

Issue 受入条件 14 項目 (Issue 本文 + DR1-001 CI guard) を全て満たすこと:

1. [ ] wizard で `institutional_docs` を選択すると適切な YAML が生成される
2. [ ] best-practice merge が domain 別に動作 (base_profile=`institutional_docs` 限定)
3. [ ] `_DOMAIN_TEMPLATES["institutional_docs"]` の値が source of truth と一致
4. [ ] 日本語質問で条件付き prompt hints が付与される
5. [ ] 日本語/英語代表 5 件ずつ + edge case で `detect_language()` が正しく分類
6. [ ] 空白のみ入力で `other` を返し ZeroDivisionError 起こさない
7. [ ] 英語質問で `build_messages()` が現状と完全一致 (golden snapshot)
8. [ ] 日本語の非制度文書質問で条文番号引用を無条件強制しない
9. [ ] PHOTON follow-up (`include_few_shot=False`) でも日本語ヒント維持
10. [ ] `build_messages()` シグネチャ不変
11. [ ] `test_prompt.py` 既存 3 test class 改変なしで pass
12. [ ] T-C6 拡張 test pass
13. [ ] **DR1-001 / DR2-002**: `test_domain_template_matches_institutional_yaml` で yaml 直接 parse 一致 assert
14. [ ] `ruff check` / `pytest` 全パス

---

## リスク管理

| リスク | 緩和策 | 担当 Task |
|--------|--------|----------|
| #137 が implement 中に merge され reranker 値が変わる | Task 0.1 で確認 → DR1-003 確定運用ルールに従い literal を post-#137 5 キー版に書き換え | Task 0.1 |
| #135 / #139 と `test_photon_pipeline.py` が衝突 | Task 0.2 で確認 → rebase 時に `TestBuildMessagesSessionSummary` 周辺へ test を再配置 | Task 0.2 / Task 1.4 |
| 日本語判定の誤分類で英語質問にヒントが付く | golden snapshot test で英語側完全一致 (Task 1.3) | Task 1.3 |
| `_JP_INSTITUTIONAL_HINT` が code-repo 質問に悪影響 | 条件節「制度文書を根拠に回答する場合は」で LLM 判断、Task 1.8 で目視確認 | Task 1.8 |
| Streamlit Project repo_id と generated YAML repo.repo_id 不一致 | DR3-001 validation で保存前拒否 (Task 1.7) | Task 1.7 |

---

## 推定工数

| Phase | タスク数 | 推定時間 |
|-------|---------|---------|
| Phase 0 (着手前確認) | 2 | 0.5h |
| Phase 1 (実装 TDD) | 8 | 6-8h |
| Phase 2 (品質チェック) | 4 | 1-2h |
| Phase 3 (ドキュメント・PR) | 3 | 1h |
| **合計** | **17** | **8-11h** |

## 次のアクション

1. `/pm-auto-dev 115` で TDD 自動開発を開始
2. または個別 task を `/tdd-impl` で順次実行
