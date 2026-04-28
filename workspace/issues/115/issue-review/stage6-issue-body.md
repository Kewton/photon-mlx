## 背景

Phase 2 で制度文書用 config が作られると、ユーザは wizard から「制度文書向け」を選択できる UX が欲しい。また Qwen 14B の prompt は日本語質問にも対応しているが、制度文書特有の表現（「第○条に規定されている通り」等）に合わせた微調整で精度改善の余地あり。

## 変更内容（D-2 + D-3）

### D-2. Streamlit wizard に domain template 追加

base profile 一覧の追加箇所:
- `app/photon_app.py:1234-1238` の `st.selectbox(..., options=["photon_small", "photon_tiny", "photon_long_context"])` ハードコードリストに `"institutional_docs"` を追加する。
- `app/components/wizard.py` 側に新規定数 `_DOMAIN_TEMPLATES: dict[str, dict[str, Any]]` を新設し、template 名 → merge dict のマッピングを保持する。

`_DOMAIN_TEMPLATES["institutional_docs"]` の merge 内容 (3 キー):
- `indexing.embedding.model_id`: 多言語 e5-small (`intfloat/multilingual-e5-small`)
- `indexing.symbol_graph.enabled`: false
- `retrieval.reranker.model_id`: **当面 `cross-encoder/ms-marco-MiniLM-L-6-v2` 据え置き**。bge-reranker-v2-m3 等の多言語 reranker への切り替えは #133 (institutional retrieval A/B) 完了後に follow-up Issue で対応する。

> 註記: `ingestion.chunker_type` のような config キーは codebase に存在しないため merge 対象外。chunker dispatch は ingest 時に拡張子から自動決定される (`baseline_reporag/ingestion/chunker.py:557`) ため wizard 側 merge は不要。institutional_docs は `repo.include: ['**/*.md']` (institutional_docs.yaml:51-52) で `.md` のみを対象とするため自動 dispatch で十分。

best-practice merge の運用方針 (既存 `BEST_PRACTICE_KEYS` との整理):
- 既存 `BEST_PRACTICE_KEYS` (safe_recgen / evidence_pruning / working_memory / photon_generation / two_pass_search) の Apply best-practice checkbox UX (`app/photon_app.py:1307-1317`) とは独立。
- `_DOMAIN_TEMPLATES` は template 選択時に **常時マージ** する (checkbox 不要)。
- 既存 `BEST_PRACTICE_KEYS` (5 キー) と `_DOMAIN_TEMPLATES` (新規 3 キー) は責務が異なるためそのまま共存させる。
- `_DOMAIN_TEMPLATES` の merge 対象は **base_profile が `institutional_docs` の時のみ effect する** 仕様に絞る (S3-002 方針 (b) 採用)。photon_small / photon_tiny / photon_long_context 等の base に institutional_docs template をかぶせるケースは **不可** とし、warning 表示または単純に no-op とする。
  - 理由: photon_small は `configs/photon_small.yaml:53-57` で `sentence-transformers/all-MiniLM-L6-v2` (英語、384 次元) を使用しており、wizard で multilingual-e5-small に切り替わると既存 `data/indexes/<repo_id>/embedding/` 配下の index と embedding model_id 不整合 (model_id.txt persist 値と cfg 値の乖離) を起こすため。
  - 補足: `institutional_docs.yaml` が既に該当 3 キーの正解値を持つため実質 no-op に近いが、将来 `institutional_docs.yaml` の値が乖離した場合の整合性担保ロジックとして wizard 側にも保持する。T-C6 では `base_yaml_text` に意図的に異なる 3 値を入れ、`_DOMAIN_TEMPLATES` によって正解値へ上書きされることを検証する。

### D-3. 多言語 prompt 調整

`baseline_reporag/generation/prompt.py`:

**日本語判定アルゴリズム**:
- 関数シグネチャ: `detect_language(question: str) -> Literal["ja", "en", "other"]`
- 入力は `question` のみ (history_text / evidence_text は含めない)。
- 検知方式: ひらがな (U+3040-U+309F) / カタカナ (U+30A0-U+30FF) / CJK Unified Ideographs (U+4E00-U+9FFF) の codepoint 比率が **質問文字数の 30% 以上** なら `ja` と判定。
- それ以外で ASCII 英字比率が **非空白文字数の 50% 以上** なら `en`、いずれにも当てはまらない場合は `other`。
- 文字数 0 のときは `other` を返す (ZeroDivisionError 回避)。
- `other` は既存 `_SYSTEM` のまま fallback (英語と同じ挙動)。
- テストケースに『記号のみ・空白のみ・絵文字のみ』を含める (制御文字や絵文字 U+1F600 等の edge case 検証)。

**日本語ヒントの注入方式 (few-shot 言語整合性)**:
- few-shot example は **英語のまま据え置き** (test_prompt.py の既存 7 example assertion を破壊しない方針 = 方針 (a))。
- 注入方式: `build_messages()` 内部で `detect_language(question)` を呼び出し、`_SYSTEM` を分岐する (S3-004 方針 (b) 採用)。
  - `build_messages()` のシグネチャは現状維持 (callsite 不変)。
  - `baseline_reporag/photon_pipeline.py:1005-1010` の callsite を含む 2 callsite (`baseline_reporag/pipeline.py:201-205` と `baseline_reporag/photon_pipeline.py:1005-1010`) + `tests/test_photon_pipeline.py:368-393` の `TestBuildMessagesSessionSummary` test class は変更不要。
  - 理由: (1) photon_pipeline.py の callsite に変更不要、(2) test_photon_pipeline.py の既存 test 不変、(3) golden snapshot test は『英語質問の場合は現状一致』の前提を満たしやすい。
- 日本語と判定された場合のみ `_SYSTEM` 末尾に「日本語向けヒント」を追加挿入する。
- `baseline_reporag/generation/prompt.py` は全 profile 共通の prompt 生成関数のため、制度文書ヒントは **制度文書を根拠に回答する場合だけ効く条件付き文面** にする。通常の code repo に対する日本語質問へ条文引用を強制しない。
- 日本語ヒントは `include_few_shot` 引数とは独立に、**日本語判定された全 turn (1st turn / 2nd turn 以降の follow-up 含む) で適用する**。これは PHOTON pipeline (`photon_pipeline.py:1005-1010`) の 2nd turn 以降 (`include_few_shot=False`) でもヒントが維持されることを意味する。
- citation 形式 `[C:N]` は変更せず（既存 post-processing 互換）。

**日本語向けヒント文面 (draft)**:
```
- 制度文書を根拠に回答する場合は、可能な範囲で条文番号 (第◯条/第◯項/第◯号) を引用すること。
- 制度文書に含まれる法令名・文書名は、根拠 chunk に正式名称がある場合は省略せず記載すること。
- 質問が該当条文を求めており、根拠 chunk に該当条文が無い場合は「該当条文なし」と明記すること。
```

## 影響ファイル

- `app/components/wizard.py` (新規 `_DOMAIN_TEMPLATES` 定数追加)
- `app/photon_app.py`（wizard UI 選択肢追加・template merge 呼び出し）
- `baseline_reporag/generation/prompt.py` (`detect_language()` 追加・日本語ヒント注入)
- 単体テスト (`baseline_reporag/tests/test_prompt.py`, `tests/test_photon_app_components.py`)

## 受入条件

- [ ] wizard で `institutional_docs` を選択すると適切な YAML が生成される
- [ ] best-practice merge が domain 別に動作 (`_DOMAIN_TEMPLATES` 経由で常時マージ、base_profile=`institutional_docs` の時のみ effect)
- [ ] 日本語質問で、制度文書以外の repo に条文引用を強制しない条件付き prompt hints が付与される
- [ ] **日本語/英語代表 5 件ずつのテストケースで `detect_language()` が正しく分類できる** (英語の記号混在ケース、および記号のみ・空白のみ・絵文字のみ・空文字 を含む edge case 検証付き)
- [ ] **英語質問入力時に `build_messages()` の output が現状と完全一致 (golden snapshot test) すること**
- [ ] **日本語の非制度文書質問入力時に、system prompt が条文番号引用を無条件に強制しないこと**
- [ ] **`build_messages()` のシグネチャは現状維持 (callsite 不変)** — `baseline_reporag/pipeline.py:201-205` / `baseline_reporag/photon_pipeline.py:1005-1010` 双方の callsite 修正は不要
- [ ] **`baseline_reporag/tests/test_prompt.py` 既存 test class (現行 3 class) が変更なしで pass すること** (token budget 6000 chars / 7 example count を含む)
- [ ] **`tests/test_photon_app_components.py` の T-C6 (`TestGenerateYamlFromWizard`) に `test_institutional_docs_template_merges_3_keys` を新規追加し、`base_profile="institutional_docs"` かつ `base_yaml_text` に意図的に異なる 3 値を入れても、生成 YAML が以下へ上書きされることを assert する**:
  - `indexing.embedding.model_id == intfloat/multilingual-e5-small`
  - `retrieval.reranker.model_id == cross-encoder/ms-marco-MiniLM-L-6-v2`
  - `indexing.symbol_graph.enabled == false`
- [ ] `ruff check` / `pytest` 全パス

## 関連

- Epic: Phase 2 制度文書ドメイン検証
- 依存: #112（config 確立）
- 関連: #133 (institutional retrieval A/B) — reranker 採用判定はここで決定。本 Issue は採用前提を取らず据え置き。
- **#133 同期メンテ**: `embedding.model_id` / `reranker.model_id` は #133 の institutional retrieval A/B 採用後に follow-up Issue で同時更新する。本 Issue の wizard merge 値は `configs/institutional_docs.yaml` の現値 (`intfloat/multilingual-e5-small`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) を反映する。`tests/test_pipeline_factory_yaml_invariants.py:36-37,71-78` の `INSTITUTIONAL_EMBEDDING_MODEL_ID` / `INSTITUTIONAL_RERANKER_MODEL_ID` placeholder pin 確定時に `_DOMAIN_TEMPLATES` も同時更新が必要。

## 並列開発

- **D-3 (prompt 調整)** は他 Issue と完全独立で先行可能。
- **D-2 (wizard template)** は #133 reranker 採用前提で先行する場合は `retrieval.reranker.model_id` を据え置き (`cross-encoder/ms-marco-MiniLM-L-6-v2`) とし、#133 完了後に follow-up Issue で更新する。
- 本 Issue 内で D-2 / D-3 を 1 PR に同梱しても、#133 完了を待つ必要はない (reranker 値据え置きのため)。

## 註記 (将来拡張)

- 将来 user-defined domain template を許可する拡張時は、template_name は `app/photon_app.py:1326-1330` の `_safe_id` 相当の allowlist 検証を通すこと。本 Issue 内の immediate 対応は不要 (現状 template name は固定 enum `institutional_docs` のため immediate risk なし)。

## レビュー履歴

### Stage 1 (通常レビュー, 2026-04-26)
対応した指摘:
- **S1-001** (Must Fix): #114 結論誤認の修正 — bge-reranker-v2-m3 言及を削除、cross-encoder 据え置き + #133 follow-up 方針に変更
- **S1-002** (Must Fix): `_WIZARD_BASE_PROFILES` 不在の修正 — `app/photon_app.py:1234-1238` ハードコードリスト編集 + `_DOMAIN_TEMPLATES` 新設方針に変更
- **S1-003** (Must Fix): 日本語判定アルゴリズム明文化 — codepoint 比率 30% 閾値 + `Literal["ja","en","other"]` 戻り値型 + 受入条件にテストケース追加
- **S1-004** (Should Fix): few-shot 言語整合性 — 方針 (a) 採用 (英語据え置き + system prompt 末尾ヒント挿入)
- **S1-005** (Should Fix): regression 検証方法 — 受入条件に golden snapshot test と既存 4 test class pass を追記
- **S1-006** (Should Fix): best-practice merge 整理 — `_DOMAIN_TEMPLATES` を新設し既存 `BEST_PRACTICE_KEYS` と独立運用に整理
- **S1-007** (Should Fix): D-2 と D-3 の依存整理 — 「並列開発」節を D-3 独立 / D-2 reranker 据え置きで先行可能に修正
- **S1-008** (Nice to Have): 制度文書ヒント文面 draft 3 行追加

### Stage 3 (影響範囲レビュー, 2026-04-26)
対応した指摘:
- **S3-001** (Must Fix): `ingestion.chunker_type` dead key 削除 — merge キーを 4 → 3 キー (embedding.model_id, reranker.model_id 据え置き値, symbol_graph.enabled) に修正、chunker dispatch は ingest 時の拡張子自動判定に委ねる旨を註記
- **S3-002** (Must Fix): embedding index 不整合リスク — 方針 (b) 採用、`_DOMAIN_TEMPLATES` 適用は base_profile が `institutional_docs` の時のみ effect する仕様に絞り、photon_small 等への重ね合わせは不可と明記
- **S3-003** (Must Fix): #133 embedding pin 同期 — 「## 関連」節に embedding.model_id / reranker.model_id の #133 同期メンテ要件を追記
- **S3-004** (Must Fix): `detect_language()` 注入方式 — 方針 (b) 採用、`build_messages()` 内部で呼び出し `_SYSTEM` を分岐、callsite/test class 不変を明記
- **S3-005** (Should Fix): T-C6 wizard test 拡充 — `test_institutional_docs_template_merges_3_keys` 新規追加を受入条件に明記
- **S3-006** (Should Fix): PHOTON 2nd turn 日本語ヒント — `include_few_shot` と独立に全 turn で適用する旨を D-3 に追記
- **S3-007** (Should Fix): `detect_language()` edge case — 入力は question のみ / 文字数 0 は `other` / 記号・空白・絵文字テストケースを D-3 に追記
- **S3-008** (Nice to Have): 将来拡張時のサニタイズ註記 — `_safe_id` 相当の allowlist 検証要件を末尾「註記 (将来拡張)」節に追記

### Stage 5 (通常レビュー 2回目, 2026-04-27)
対応した指摘:
- **S5-001** (Must Fix): 共通 `build_messages()` への制度文書ヒント適用範囲を修正 — 日本語ヒントを制度文書の場合だけ効く条件付き文面に変更し、通常 code repo の日本語質問へ条文引用を強制しない受入条件を追加
- **S5-002** (Should Fix): T-C6 の検証性改善 — `base_yaml_text` に意図的に異なる 3 値を入れ、`_DOMAIN_TEMPLATES` の上書きが実際に効くことを assert する要件に修正
- **S5-003** (Should Fix): `baseline_reporag/tests/test_prompt.py` の既存 test class 数を現行 3 class に修正
- **S5-004** (Should Fix): `detect_language()` の `en` 判定閾値を ASCII 英字比率 50% 以上として明記
