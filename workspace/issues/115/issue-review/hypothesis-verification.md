# Issue #115 仮説検証レポート (Phase 0.5)

検証日: 2026-04-26
対象: Issue #115 - feat(app): Streamlit wizard に制度文書 domain template と日本語 prompt 調整

## 仮説 1: `app/components/wizard.py` の存在と base profile 構成

- **主張**: 既存の base profile に `["photon_small", "photon_tiny", "photon_long_context"]` が含まれている
- **判定**: **Confirmed**
- **根拠**:
  - `app/components/wizard.py:54-60` の `_INTENTIONAL_CONFLICT_PROFILES` で関連 profile 群が定義済
  - `app/photon_app.py:1234-1238` で wizard UI の profile 選択肢として `["photon_small", "photon_tiny", "photon_long_context"]` がハードコード
  - best-practice merge ロジックは `wizard.py:130-197` の `apply_best_practice()` が 5 キー (safe_recgen, evidence_pruning, working_memory 等) を domain-agnostic にマージ
- **補足**: Issue 記載の `_WIZARD_BASE_PROFILES` 定数名は実装と乖離している可能性あり。実際の選択肢リストは `photon_app.py` 側に定義。

## 仮説 2: `app/photon_app.py` の wizard UI 構造

- **主張**: wizard UI に profile 選択肢追加可能な箇所がある
- **判定**: **Confirmed**
- **根拠**:
  - `app/photon_app.py:1224-1317` の expander "PHOTON settings (Wave 2-4 toggles)" で UI 実装
  - 1234-1238 行で `st.selectbox(..., options=["photon_small", "photon_tiny", "photon_long_context"])`
  - 新規 profile 追加箇所: 1236 行の options list と 1361-1369 行の `_wizard.generate_yaml_from_wizard()` 呼び出し

## 仮説 3: `baseline_reporag/generation/prompt.py` の現状

- **主張**: system prompt 構造、日本語/英語判定ロジック、citation `[C:N]` 生成
- **判定**: **Partially Confirmed**
- **根拠**:
  - `baseline_reporag/generation/prompt.py:10-29` で `_SYSTEM` 定義；rule 6 に "Respond in the same language as the question" あり
  - **日本語判定ロジックは実装側に存在しない** - prompt は言語判定を LLM に委ねており、コードに言語検出機能なし
  - Citation 形式: `prompt.py:14-15` でルール記述（`[C:N]` notation）
  - Post-processing: `baseline_reporag/pipeline.py:apply_citation_postprocess()` が no-citation 時に `[C:1]` を自動付与
- **補足**: Issue 記載の「日本語質問検知（簡易ヒューリスティクス or metadata）」は完全な新規追加が必要。

## 仮説 4: #112 で作成された制度文書 config

- **主張**: configs/ 配下に institutional_docs 系 YAML と多言語 e5-small 設定
- **判定**: **Confirmed**
- **根拠**:
  - `configs/institutional_docs.yaml` 存在
  - embedding model: 84 行で `"intfloat/multilingual-e5-small"` 採用済
  - chunker: `language=markdown` を使用（#109 で自動 dispatch）
- **補足**: wizard 側で best-practice merge 値は institutional_docs.yaml と整合させるべき。

## 仮説 5: #114 (reranker) の結果

- **主張**: bge-reranker-v2-m3 が現在の reranker config に反映されている
- **判定**: **Rejected**
- **根拠**:
  - `configs/photon_small.yaml:77` で `model_id: "cross-encoder/ms-marco-MiniLM-L-6-v2"`
  - `configs/institutional_docs.yaml:107` で同じく `"cross-encoder/ms-marco-MiniLM-L-6-v2"`
  - `reports/institutional_baseline_static.md` で handicap (b) は "未実測" 扱い、#114 を別途実施予定
  - **bge-reranker-v2-m3 は未採用**
- **補足**: ただし、Issue #114 は最近 close されている可能性があるため、git log と reports/ の最新を要確認。

## 仮説 6: 既存テスト構造

- **主張**: wizard.py と prompt.py 関連のテストがある
- **判定**: **Confirmed**
- **根拠**:
  - Wizard テスト: `tests/test_photon_app_components.py:51` で wizard component をロード；security test (T-C8) で YAML safe_load を検証
  - Prompt テスト: `baseline_reporag/tests/test_prompt.py` で `build_messages()` と citation format ルール、`flatten_messages_for_plain_lm()` の security contract を検証

---

## Stage 1 レビューへの申し送り事項

**Rejected/Partially Confirmed 項目の影響**：

1. **日本語判定ロジック不在** (仮説 3): prompt.py は言語判定を LLM に委ねており、explicit ロジックなし。Issue #115 の制度文書日本語対応には、問い合わせ言語の自動検出 → 日本語/英語別 system prompt 分岐が必要。現状は汎用 _SYSTEM のみ。
   - 設計時、`detect_language(question: str) -> Literal["ja", "en", "other"]` のようなヘルパーが必要
   - 既存の few-shot example/citation ルールとの干渉に注意（test_prompt.py の token budget 検証あり）

2. **reranker 未更新** (仮説 5): 制度文書ドメインは英語 MS MARCO 学習の cross-encoder を使用中。評価結果からは 5-10pt NC 下振れ可能性指摘（handicap b）。#114 実施まで制度文書ドメインの NC baseline は過度に悲観的。
   - **要再確認**: 最近 #114 関連 PR #132/#136 がマージ済みのため、reranker 切り替えが完了している可能性あり。git log の "rerank" / "bge" を確認すること
   - 仮に未完了なら、wizard merge 値を bge-reranker-v2-m3 にするのは「未検証 model を default にする」リスクあり

3. **`_WIZARD_BASE_PROFILES` 定数の場所** (仮説 1 補足): Issue 記載は wizard.py 内の定数を想定するが、実際は photon_app.py 側のハードコードリスト。設計時、定数化リファクタリングを別タスクとして切り出すか、photon_app.py 側を直接編集するかの方針選択が必要。

4. **citation post-processing 互換性** (仮説 3 補足): Issue は `[C:N]` 形式を変更しないと宣言しているが、日本語 system prompt 追加時に few-shot example の言語整合性（英語 example のまま日本語回答を要求するのか）を検討する必要あり。
