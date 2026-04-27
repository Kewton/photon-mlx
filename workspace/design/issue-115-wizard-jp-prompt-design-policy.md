# Issue #115 設計方針書 — Streamlit wizard 制度文書 domain template + 日本語 prompt 調整

| 項目 | 値 |
|------|-----|
| Issue | [#115](https://github.com/Kewton/photon-mlx/issues/115) |
| 作成日 | 2026-04-27 |
| ブランチ | `feature/issue-115-wizard-jp-prompt` |
| Epic | Phase 2 制度文書ドメイン検証 |
| 依存 | #112 (config 確立) |
| 関連 | #133 / #137 (institutional retrieval A/B、reranker/embedding 採用判定) |

---

## 1. 目的とスコープ

### 目的
- **D-2**: Streamlit wizard に制度文書 domain template (`institutional_docs`) を追加し、ユーザが wizard から「制度文書向け」プロファイルを選択できる UX を提供する。
- **D-3**: 日本語質問への prompt 調整を行い、制度文書を根拠に回答する場合のみ条文番号引用を促す条件付きヒントを注入する。

### スコープ
- 新規追加: 1 関数 (`detect_language()`), 1 定数 (`_DOMAIN_TEMPLATES`), 1 制御フロー (build_messages 内 system prompt 分岐)
- UI 変更: `app/photon_app.py` の base_profile 選択肢に 1 項目追加
- テスト: `test_prompt.py`, `test_photon_pipeline.py`, `test_photon_app_components.py` に新規 test を追加（既存 test 改変なし）

### スコープ外
- `configs/institutional_docs.yaml` の値変更 (#137 で実施)
- `BEST_PRACTICE_KEYS` (Apply best-practice checkbox UX) の改修
- LLM での言語判定（コードベースの explicit codepoint ロジックで実装）
- user-defined domain template の動的拡張

---

## 2. システムアーキテクチャ位置づけ

```
┌──────────────────────────────────────────────────────────┐
│              Streamlit App (app/photon_app.py)            │
│                                                            │
│   ┌─────────────────────────────────────────────────┐    │
│   │ Wizard UI (PHOTON settings expander)             │    │
│   │ base_profile selectbox:                          │    │
│   │   [photon_small, photon_tiny,                    │    │
│   │    photon_long_context,                          │    │
│   │    institutional_docs] ← 追加                    │    │
│   └────────────────┬────────────────────────────────┘    │
│                    │                                       │
│                    ▼                                       │
│   ┌─────────────────────────────────────────────────┐    │
│   │ wizard helpers (app/components/wizard.py)        │    │
│   │   _DOMAIN_TEMPLATES (新規)                       │    │
│   │   generate_yaml_from_wizard()                    │    │
│   │     ↓ 既存 _WIZARD_TOGGLE_MAPPING 適用           │    │
│   │     ↓ NEW: _DOMAIN_TEMPLATES merge               │    │
│   │       (base_profile=="institutional_docs" 限定) │    │
│   └────────────────┬────────────────────────────────┘    │
└────────────────────┼─────────────────────────────────────┘
                     │ 生成 YAML
                     ▼
┌──────────────────────────────────────────────────────────┐
│        Pipeline (baseline_reporag/)                       │
│                                                            │
│   pipeline_factory.py → pipeline.py / photon_pipeline.py │
│                              │                             │
│                              ▼                             │
│   generation/prompt.py                                    │
│     build_messages(question, ...)                         │
│       ├ NEW: detect_language(question)                   │
│       │     → "ja" / "en" / "other"                      │
│       └ NEW: _SYSTEM 分岐                                │
│           "ja" → _SYSTEM + JP_HINT                       │
│           others → _SYSTEM (現状)                         │
└──────────────────────────────────────────────────────────┘
```

### レイヤー責務

| レイヤー | モジュール | 本 Issue での変更 |
|---------|-----------|----------|
| **UI** | `app/photon_app.py` | base_profile 選択肢追加（1行） |
| **Wizard helpers** | `app/components/wizard.py` | `_DOMAIN_TEMPLATES` 定数追加、`generate_yaml_from_wizard()` 内に template merge ロジック追加 |
| **Generation** | `baseline_reporag/generation/prompt.py` | `detect_language()` 追加、`build_messages()` 内で system prompt 分岐 |
| **Tests** | `baseline_reporag/tests/`, `tests/` | 新規 test 追加のみ、既存 test は不変 |

---

## 3. 詳細設計

### 3-1. D-2: `_DOMAIN_TEMPLATES` 定数と merge ロジック

#### データ構造

`app/components/wizard.py` に追加:

```python
# Issue #115: Domain-specific YAML templates merged when the user picks
# a base profile that has a corresponding domain template entry.
#
# Source of truth: configs/institutional_docs.yaml と
# tests/test_pipeline_factory_yaml_invariants.py の institutional pin 値。
# #137 (institutional retrieval A/B) が先 merge された後は採用 model_id
# (BAAI/bge-m3, BAAI/bge-reranker-v2-m3) と batch_size=32 / max_input_chars=8192
# を反映する。
_DOMAIN_TEMPLATES: dict[str, dict[tuple[str, ...], Any]] = {
    "institutional_docs": {
        # pre-#137 (現行 institutional_docs.yaml と一致する 3 キー):
        ("indexing", "embedding", "model_id"): "intfloat/multilingual-e5-small",
        ("indexing", "symbol_graph", "enabled"): False,
        ("retrieval", "reranker", "model_id"): "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
}
```

> **DR2-001 対応**: 上記 literal は **pre-#137 確定形** (本ブランチ rebase 元 main の `configs/institutional_docs.yaml` を確認した結果に基づく)。post-#137 (5 キー化、`BAAI/bge-m3` / `BAAI/bge-reranker-v2-m3` / `batch_size=32` / `max_input_chars=8192`) への切替えは、#137 が本 Issue より先に merge された場合のみ DR1-003 確定運用ルールに従い PR 着手時に literal を 5 キー版に書き換える。同時 / 後続 merge では別 follow-up Issue で対応 (DR2-007 と整合)。

> **重要**: 実装時に `configs/institutional_docs.yaml` と `tests/test_pipeline_factory_yaml_invariants.py` の現値を確認し、#137 が merge 済みなら post-#137 値で初期実装する。
>
> **DR1-003 (YAGNI) 対応 — 確定運用ルール**:
> 1. PR 着手時 (実装直前) に `git fetch origin main && git diff origin/main configs/institutional_docs.yaml tests/test_pipeline_factory_yaml_invariants.py` を実行し pre-#137 か post-#137 を確定する。
> 2. 確定後、本設計書から **不採用側のキー記述を削除して PR を出す**（並列開発を吸収するための暫定二段構えはレビュー前に解消する）。
> 3. 受入条件 §10 の項目 3 / §3-3 の T-C6 期待値も確定形に統一する。

#### DR1-001 (DRY) 対応 — `_DOMAIN_TEMPLATES` と yaml の整合 CI guard

`_DOMAIN_TEMPLATES["institutional_docs"]` の値と `configs/institutional_docs.yaml` の対応キーが drift しないよう、以下を **完了条件 §10 の必須項目 (#13)** に格上げする:

- **配置先 (DR2-002 確定)**: `tests/test_photon_app_components.py` に `test_domain_template_matches_institutional_yaml` を追加する。`tests/test_pipeline_factory_yaml_invariants.py` は `INSTITUTIONAL_*` pin 定数 (#133 採用判断後に有効化される invariant) と `@pytest.mark.skipif` の責務を持つため、wizard 機能 guard はこちらに置くと skipif 思想と衝突する。
- **比較方法 (DR2-002 確定)**: `yaml.safe_load(open("configs/institutional_docs.yaml"))` で yaml を直接 parse し、pin 定数 (`INSTITUTIONAL_EMBEDDING_MODEL_ID` 等) を経由せず yaml dict から該当パスの値を抽出して `_DOMAIN_TEMPLATES["institutional_docs"]` と完全一致を assert する。これにより #133 pin 採用前後どちらでも CI guard が常に有効化される。
- **位置づけ**: yaml が source of truth、wizard merge は「wizard 経由でも source of truth と等価な YAML を生成する」ことの guard。

#### merge ロジック

`generate_yaml_from_wizard()` の末尾 (`_WIZARD_TOGGLE_MAPPING` 適用直後) に追加:

```python
# Issue #115: Apply domain template merge when the chosen base profile
# has a matching entry. Effects only when base_profile is the same as
# the template name (no cross-profile override) to avoid embedding/index
# drift on photon_small etc.
template = _DOMAIN_TEMPLATES.get(base_profile)
if template is not None:
    for path, value in template.items():
        _deep_set(doc, path, value)
```

#### DR3-001 対応 — Project `repo_id` と generated YAML の repo 整合

Streamlit の Project 登録では `Project.repo_id` が eval job の `--repo-id` に渡される一方、`generate_yaml_from_wizard()` が返す YAML 内の `repo.repo_id` は base profile (`configs/institutional_docs.yaml` 等) の値を保持する。`build_pipeline(cfg)` は cfg 側の `repo.repo_id` で index を load し、eval script は `--repo-id` を query 時に渡すため、両者が不一致だと index load と query 対象 repo が食い違う。

本 Issue では `generate_yaml_from_wizard()` のシグネチャは変更せず、`app/photon_app.py` 側で保存直前に generated YAML を `yaml.safe_load()` し、`generated["repo"]["repo_id"]` と Project 選択 `repo_id` の一致を検証する。

```python
generated_doc = yaml.safe_load(generated_yaml) or {}
generated_repo_id = generated_doc.get("repo", {}).get("repo_id")
if generated_repo_id != repo_id:
    st.error(
        "wizard YAML の repo.repo_id が選択中の repo_id と一致しません: "
        f"{generated_repo_id!r} != {repo_id!r}"
    )
    return
```

`institutional_docs` template は `repo_id="institutional_documents"` の index と組み合わせる前提。別 repo に制度文書向け model/profile だけを流用する拡張は、repo key の override 設計を含む別 Issue で扱う。

#### 設計判断 #1: template 適用範囲を base_profile と同名に限定

| 選択肢 | 内容 | 結論 |
|--------|------|------|
| A | template は base_profile と無関係に常時適用 (既存 BEST_PRACTICE_KEYS と同じ動作) | × |
| B | template name == base_profile の時のみ effect、他は no-op | ○ 採用 |

**理由**:
- A の場合、photon_small (英語 all-MiniLM-L6-v2、384 次元) base に institutional_docs template (multilingual-e5-small、prefix 必要) を被せると、`data/indexes/<repo_id>/embedding/model_id.txt` の persist 値と cfg 値が乖離し retrieval が壊れる
- B なら institutional_docs.yaml の現値がそのまま反映され、wizard 経由でも canonical config と等価になる

**トレードオフ**:
- メリット: index 整合性担保、UX も「institutional 専用」で混乱なし
- デメリット: 「photon_tiny + institutional template」のような実験的組合せができない (将来必要なら別 Issue で拡張)

#### UI 変更 (`app/photon_app.py:1234-1238`)

```python
# Before (実コード `app/photon_app.py:1234-1238` のラベルは "Config template")
wiz_base_profile = st.selectbox(
    "Config template",
    options=["photon_small", "photon_tiny", "photon_long_context"],
)

# After (DR2-004 修正: options 末尾に 1 行追加するのみ・ラベル文字列・他引数は不変)
wiz_base_profile = st.selectbox(
    "Config template",
    options=[
        "photon_small",
        "photon_tiny",
        "photon_long_context",
        "institutional_docs",  # Issue #115
    ],
)
```

`configs/institutional_docs.yaml` は既に存在するため、`generate_yaml_from_wizard()` の `cfg_path.read_text()` はそのまま動作する。

---

### 3-2. D-3: `detect_language()` と system prompt 分岐

#### 関数シグネチャ

`baseline_reporag/generation/prompt.py` に追加:

```python
from typing import Literal

# Issue #115: codepoint ranges for Japanese script detection.
_JA_HIRAGANA = (0x3040, 0x309F)
_JA_KATAKANA = (0x30A0, 0x30FF)
_JA_CJK_UNIFIED = (0x4E00, 0x9FFF)
_JA_RATIO_THRESHOLD = 0.30  # 30% of total chars
_EN_RATIO_THRESHOLD = 0.50  # 50% of non-space chars


def detect_language(question: str) -> Literal["ja", "en", "other"]:
    """Detect language of a question for prompt routing.

    Returns:
        "ja"  if Japanese script (hiragana/katakana/CJK) ratio
              >= 30% of total chars.
        "en"  if ASCII alphabetic ratio >= 50% of non-space chars
              (and not "ja").
        "other" for empty / whitespace-only / symbol-only / emoji-only
              inputs and any other case.

    Note (DR1-002): single pass over `question` to compute all counters
    so KISS is preserved. ja の分母は total, en の分母は non_space_len と
    分母が異なるが、これは意図的: 日本語は空白の有無に意味があり全文字に対する
    比率が自然、英語は単語間 space を分母から除いた方が短い質問で安定するため。
    この非対称性の理由はコメントとして実装に残す。
    """
    if not question:
        return "other"

    total = len(question)
    ja_count = 0
    en_count = 0
    non_space_len = 0
    for ch in question:
        cp = ord(ch)
        if (
            _JA_HIRAGANA[0] <= cp <= _JA_HIRAGANA[1]
            or _JA_KATAKANA[0] <= cp <= _JA_KATAKANA[1]
            or _JA_CJK_UNIFIED[0] <= cp <= _JA_CJK_UNIFIED[1]
        ):
            ja_count += 1
        if not ch.isspace():
            non_space_len += 1
            # ASCII 英字 (a-z / A-Z) のみ計上。
            if ch.isascii() and ch.isalpha():
                en_count += 1

    if ja_count / total >= _JA_RATIO_THRESHOLD:
        return "ja"
    if non_space_len == 0:
        return "other"  # whitespace-only — avoid ZeroDivisionError
    if en_count / non_space_len >= _EN_RATIO_THRESHOLD:
        return "en"
    return "other"
```

> **DR1-005 (SRP 改善) 対応 — `_resolve_system_prompt` helper 切り出し**: 言語判定 + システムプロンプト合成は build_messages 本体から `_resolve_system_prompt(question: str) -> str` 内部 helper に切り出す。build_messages は `system_content = _resolve_system_prompt(question)` の 1 行で参照する。これにより test pyramid (helper 単体 → build_messages 統合) が明確化され、golden snapshot test の保守も独立する。

#### システムプロンプト分岐

`build_messages()` 内部のみで適用、シグネチャ不変:

```python
# Issue #115: 日本語向け制度文書ヒント (条件付き)
_JP_INSTITUTIONAL_HINT = """\

Additional rules for Japanese questions about institutional documents:
- 制度文書を根拠に回答する場合は、可能な範囲で条文番号 (第◯条/第◯項/第◯号) を引用すること。
- 制度文書に含まれる法令名・文書名は、根拠 chunk に正式名称がある場合は省略せず記載すること。
- 質問が該当条文を求めており、根拠 chunk に該当条文が無い場合は「該当条文なし」と明記すること。
"""


def _resolve_system_prompt(question: str) -> str:
    """Return the system prompt for ``question`` without changing callsites."""
    if detect_language(question) == "ja":
        return _SYSTEM + _JP_INSTITUTIONAL_HINT
    return _SYSTEM


def build_messages(
    question: str,
    evidence_text: str,
    history_text: str = "",
    session_summary: str = "",
    include_few_shot: bool = True,
) -> list[dict]:
    parts: list[str] = []
    if session_summary:
        parts.append(f"## Session Summary\n{session_summary}")
    if history_text:
        parts.append(f"## Conversation History\n{history_text}")
    parts.append(f"## Code Chunks\n{evidence_text}")
    parts.append(f"## Question\n{question}")
    hint = _FORMAT_HINT if include_few_shot else _FORMAT_HINT_SHORT
    parts.append(f"## Instructions\n{hint}")

    # Issue #115: Japanese question → augment _SYSTEM with conditional
    # institutional-doc hint. Conditional language ("制度文書を根拠に回答する
    # 場合は…") avoids forcing 条文 citation on ordinary code-repo Japanese
    # questions. include_few_shot とは独立に適用 (PHOTON 2nd turn でも維持)。
    system_content = _resolve_system_prompt(question)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
```

#### 設計判断 #2: 条件付き文面 vs 固定文面

| 選択肢 | 内容 | 結論 |
|--------|------|------|
| A | 「全ての日本語質問に対して条文引用を強制する」固定文 | × |
| B | 「制度文書を根拠に回答する場合は」を冒頭に置く条件付き文 | ○ 採用 |

**理由**:
- `baseline_reporag/generation/prompt.py` は **全 profile 共通**。photon_small で日本語コード質問をした場合、条文引用を強制すると意味不明な回答になる
- 条件節を入れることで、制度文書 chunk を引いた時のみ自然に条文引用が促される

#### 設計判断 #3: detect_language の注入点 (build_messages 内部 vs callsite)

| 選択肢 | 内容 | 結論 |
|--------|------|------|
| A | `build_messages(..., language: Literal[...]|None)` で外部注入 | × |
| B | `build_messages()` 内部で `detect_language(question)` を呼び出し | ○ 採用 |

**理由**:
- B なら 2 callsite (`pipeline.py:201`, `photon_pipeline.py:1005`) と既存 test class (`TestBuildMessagesSessionSummary`) が完全に不変
- 言語判定ロジックが prompt 生成と密結合のため module 局所化が自然
- callsite 側で言語判定する必要があるユースケース (異なる profile で異なる言語ヒントを使い分け) は本 Issue 範囲外

**トレードオフ**:
- 将来「言語判定をアプリ層で override したい」要望が出た時は引数追加で拡張可能 (default は internal detect)

---

### 3-3. テスト戦略

#### 新規追加 test

| ファイル | テスト | 検証内容 |
|---------|--------|----------|
| `baseline_reporag/tests/test_prompt.py` | `TestDetectLanguage` (5+ tests) | ja/en/other の代表 5 件ずつ + edge case (空文字、空白のみ、記号のみ、絵文字のみ、英数字混在) |
| `baseline_reporag/tests/test_prompt.py` | `TestBuildMessagesLanguageBranching` (3 tests) | (1) 英語質問で system prompt が `_SYSTEM` と完全一致 (golden snapshot)、(2) 日本語質問で `_JP_INSTITUTIONAL_HINT` が末尾に付与、(3) 「制度文書を根拠に回答する場合は」の条件節を含み無条件強制でないこと |
| `baseline_reporag/tests/test_photon_pipeline.py` | `TestBuildMessagesJapaneseFollowUp` (1 test) | `build_messages(question=<日本語>, include_few_shot=False, session_summary=<sample>)` がシグネチャ変更なしで動き、system prompt に日本語ヒントが残ること |
| `tests/test_photon_app_components.py` | `test_institutional_docs_template_merges_3_keys` (T-C6 拡張) | `base_profile="institutional_docs"`, `base_yaml_text` に意図的な異なる 3 値を渡し、生成 YAML が source of truth へ上書きされる |
| `tests/test_photon_app_components.py` | `test_domain_template_no_op_on_other_profiles` (T-C6 拡張) | `base_profile="photon_small"` で institutional template が effect しないこと |
| `tests/test_photon_app_components.py` | `test_domain_template_matches_institutional_yaml` (DR1-001 / DR2-002) | `_DOMAIN_TEMPLATES["institutional_docs"]` の値が `yaml.safe_load(configs/institutional_docs.yaml)` から抽出した対応キーと完全一致。pin 定数経由ではなく yaml 直接 parse で照合 |
| `tests/test_photon_app_components.py` / `tests/test_photon_app_helpers.py` | `test_wizard_generated_repo_id_must_match_project_repo_id` (DR3-001) | generated YAML の `repo.repo_id` と Project 選択 `repo_id` が不一致の場合に保存せず error になる validation を検証 |
| `baseline_reporag/tests/test_prompt.py` | `TestResolveSystemPrompt` (3 tests, DR2-005) | `_resolve_system_prompt(question)` を直接呼び (1) 英語入力で `_SYSTEM` そのまま、(2) 日本語入力で `_SYSTEM + _JP_INSTITUTIONAL_HINT` 連結、(3) 空文字で `_SYSTEM` そのまま、を独立 assert (test pyramid: helper 単体 → build_messages 統合) |

> **#137 merge 後**: `test_institutional_docs_template_merges_3_keys` を 5 キー版にリネーム/拡張し、`bge-m3` / `bge-reranker-v2-m3` / `batch_size=32` / `max_input_chars=8192` を assert する。

> **DR1-006 (test 参照方針)**: `_JP_INSTITUTIONAL_HINT` は module-private constant のまま維持し、test 側からは import せず特徴的フレーズ「制度文書を根拠に回答する場合は」を substring assert する。これにより文面の微修正で test が壊れず、private 命名も維持できる。

> **DR2-009 (T-C6 拡張 test 実装スタイル)**: `test_institutional_docs_template_merges_3_keys` の `base_yaml_text` には pre-#137 3 キー (`indexing.embedding.model_id` / `indexing.symbol_graph.enabled` / `retrieval.reranker.model_id`) を意図的に異なる値で書いた最小 YAML literal を embed する。assert は `yaml.safe_load(generated_yaml)` 結果から該当パス値を取り出して `_DOMAIN_TEMPLATES["institutional_docs"]` の値と一致を確認。これにより既存 T-C6 5 test の最小化方針 (5-10 行 YAML literal) を維持しつつ overwrite 動作を検証する。

#### 既存 test の不変性

- `baseline_reporag/tests/test_prompt.py` の現行 3 test class (`TestBuildMessages`, `TestFormatHintFewShot`, `TestFlattenMessagesForPlainLM`) は変更しない
  - 特に `test_few_shot_examples_count` (q_count==7) と `test_format_hint_token_budget` (<=6000 chars) は破壊しない
- `baseline_reporag/tests/test_photon_pipeline.py:368-393` の `TestBuildMessagesSessionSummary` は変更しない
- `tests/test_photon_app_components.py` の T-C6 既存 5 test は変更しない (新規 test を追加するのみ)

---

## 4. データ・config の影響

> **DR2-008 用語統一**: 以下用語を本設計書全体で固定する:
> - **source of truth** = `configs/institutional_docs.yaml` の値そのもの (canonical config と同義)
> - **pin 値** = `tests/test_pipeline_factory_yaml_invariants.py` の `INSTITUTIONAL_*` 定数 (#133 採用判断後に有効化される invariant)
> - **canonical config** = source of truth と同義 (旧表現、新規記述では source of truth を優先)

| ファイル | 変更 | 備考 |
|----------|------|------|
| `configs/institutional_docs.yaml` | **不変** | source of truth として参照されるが値は変えない |
| `configs/photon_small.yaml` | **不変** | template 適用範囲外なので影響なし |
| `tests/test_pipeline_factory_yaml_invariants.py` | **不変** | institutional pin 値を `_DOMAIN_TEMPLATES` の source of truth として参照する read-only relationship |
| `data/indexes/institutional_documents/` | **不変** | wizard merge 値が institutional_docs.yaml と一致するため再 ingest 不要 |
| `data/indexes/<other_repos>/` | **不変** | template が他 profile に effect しないため index 不整合なし |

---

## 5. セキュリティ設計

| 脅威 | 対策 |
|------|------|
| YAML injection (template 名がユーザ入力) | 本 Issue では template 名は固定 enum (`institutional_docs`)。将来拡張時は `_safe_id` 相当の allowlist 検証を通すこと（註記済） |
| `_assert_safe_yaml` バイパス | `_DOMAIN_TEMPLATES` の値はコード定数なので Python 文字列リテラル。`_deep_set` 適用後も既存 `_assert_safe_yaml` 経路を通る |
| 言語判定での DoS (超長文での O(n) 走査) | `detect_language()` の入力は `question` のみ (history/evidence は含めない)。質問長は通常 100-500 chars、最悪でも数千 chars で 1ms 未満 |
| Citation post-processing 互換性 | `[C:N]` 形式は変更せず、`apply_citation_postprocess()` の `ABSTAIN_MARKER`「根拠が不足しています」も維持 |
| Plain LM 経由での role spoofing (DR-62-003) | `flatten_messages_for_plain_lm()` は変更せず、system content の長さが増えるだけ。JSON シリアライズ済みなので boundary spoof 耐性は維持 |

---

## 6. パフォーマンス影響

| 項目 | 現状 | Issue #115 後 | 影響 |
|------|------|------|------|
| `build_messages()` per-call latency | ~0.1ms | ~0.2ms (detect_language 追加) | +0.1ms (無視可能) |
| system prompt token count | ~400 tokens | ja の場合 +30 tokens | LLM input 0.1% 増 |
| _FORMAT_HINT token budget | <= 6000 chars | 不変 | 既存 test pass |

---

## 7. 影響範囲サマリー

| ファイル | 変更種別 | 行数 (推定) | 影響度 |
|---------|---------|-----|------|
| `app/components/wizard.py` | 定数 + merge ロジック追加 | +20 行 | 中 |
| `app/photon_app.py` | selectbox options 追加 + generated YAML repo_id validation | +10 行 | 中 |
| `baseline_reporag/generation/prompt.py` | `detect_language` + 分岐 | +60 行 | 中 |
| `baseline_reporag/tests/test_prompt.py` | 新規 test class 2 つ | +80 行 | - |
| `baseline_reporag/tests/test_photon_pipeline.py` | 新規 test 1 つ | +20 行 | - |
| `tests/test_photon_app_components.py` | T-C6 + repo_id validation test 追加 | +70 行 | - |

合計: 実装 ~90 行、テスト ~170 行

---

## 8. リスクと緩和策

| リスク | 確率 | 影響 | 緩和策 |
|--------|------|------|--------|
| #137 merge 順序により wizard merge 値が古い model に巻き戻る | 中 | 高 | source of truth を `configs/institutional_docs.yaml` + `test_pipeline_factory_yaml_invariants.py` pin 値とし、PR 出す前に grep で再確認。CI で `_DOMAIN_TEMPLATES` の値が institutional_docs.yaml の対応キーと一致することを assert する test を追加 |
| generated YAML の `repo.repo_id` と Project 選択 `repo_id` が不一致 | 中 | 高 | `app/photon_app.py` の保存直前に generated YAML を safe_load し、`repo.repo_id == Project.repo_id` を validation。不一致なら保存せず `st.error` |
| 日本語判定の誤分類で英語質問にヒントが付く | 低 | 低 | 30%/50% 閾値で False Positive は限定的。golden snapshot test で英語側完全一致を保証 |
| PHOTON 2nd turn で日本語ヒントが期待通り維持されない | 低 | 中 | `TestBuildMessagesJapaneseFollowUp` で `include_few_shot=False` 経路を直接 assert |
| `_JP_INSTITUTIONAL_HINT` が code-repo 質問に悪影響 | 低 | 低 | 条件節「制度文書を根拠に回答する場合は」で LLM に判断委譲、prompt-eval で確認 |
| #135/#139 との rebase で `test_photon_pipeline.py` 周辺が衝突 | 中 | 低 | #135/#139 は `baseline_reporag/photon_pipeline.py` と `baseline_reporag/tests/test_photon_pipeline.py` を触る。rebase 時は `TestBuildMessagesSessionSummary` 周辺へ `TestBuildMessagesJapaneseFollowUp` を追加し直し、`python -m pytest baseline_reporag/tests/test_photon_pipeline.py -v` を単独実行 |

---

## 9. 品質基準

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| テスト | `python -m pytest baseline_reporag/tests/test_prompt.py baseline_reporag/tests/test_photon_pipeline.py tests/test_photon_app_components.py -v` | 新規 test 全 pass、既存 test 改変なしで pass |
| 全体テスト | `python -m pytest` | 既存 test を破壊しない |
| リント | `ruff check .` | 警告 0 件 |
| フォーマット | `ruff format --check .` | 差分なし |
| Streamlit 動作確認 | `streamlit run app/photon_app.py` | wizard で institutional_docs 選択 → 生成 YAML を目視確認 |

---

## 10. 完了条件

Issue #115 受入条件 13 項目（最終 body 参照）+ 設計レビューで追加された CI guard / repo_id validation 2 項目 = **計 15 項目** を全て満たすこと (DR2-003 / DR3-001 修正):

1. wizard で `institutional_docs` を選択すると適切な YAML が生成される
2. best-practice merge が domain 別に動作 (base_profile=`institutional_docs` 限定)
3. `_DOMAIN_TEMPLATES["institutional_docs"]` の値が source of truth と一致
4. generated YAML の `repo.repo_id` が Project 選択 `repo_id` と一致し、不一致なら保存しない
5. 日本語質問で条件付き prompt hints が付与される
6. 日本語/英語代表 5 件ずつ + edge case で `detect_language()` が正しく分類
7. 空白のみ入力で `other` を返し ZeroDivisionError 起こさない
8. 英語質問で `build_messages()` が現状と完全一致 (golden snapshot)
9. 日本語の非制度文書質問で条文番号引用を無条件強制しない
10. PHOTON follow-up (`include_few_shot=False`) でも日本語ヒント維持
11. `build_messages()` シグネチャ不変
12. `test_prompt.py` 既存 3 test class 改変なしで pass
13. T-C6 拡張 test pass
14. **DR1-001 対応**: `test_domain_template_matches_institutional_yaml` で `_DOMAIN_TEMPLATES["institutional_docs"]` と `configs/institutional_docs.yaml` の値完全一致を CI で保証
15. `ruff check` / `pytest` 全パス

---

## 11. 設計上の留意点

### `_DOMAIN_TEMPLATES` のシリアライズ

`_DOMAIN_TEMPLATES["institutional_docs"]` の値は dict path → 値の dict 形式。`tuple[str, ...]` をキーにすることで既存 `BEST_PRACTICE_KEYS` と同じ構造を踏襲し、`_deep_set()` がそのまま再利用できる。

### `detect_language()` の関数配置

`baseline_reporag/generation/prompt.py` 内に配置し外部 export。`from baseline_reporag.generation.prompt import detect_language` で test からも使える。汎用性は高いが、本 Issue では prompt 生成専用として位置づける。将来 retrieval 層で言語別 reranker 切替が必要になれば、`baseline_reporag/utils/lang.py` に切り出す follow-up Issue で対応。

**DR1-004 (OCP 拡張コスト見える化)**: 戻り値型 `Literal["ja", "en", "other"]` は Closed な型だが、4 言語目 (例: `"zh"`, `"ko"`) 追加時の修正コストは限定的: (1) Literal 拡張、(2) `build_messages` 内分岐 1 行追加、(3) 新言語向け hint constant 追加、(4) 既存 callsite (build_messages のみ) は内部分岐なので変更不要。テスト追加 5 件程度。

### 条件文面の i18n

`_JP_INSTITUTIONAL_HINT` 内の日本語文字列は実コード内に直接置く。i18n フレームワークは導入しない（過剰）。将来他言語ヒントを追加する場合は `_DOMAIN_HINTS: dict[str, str]` のような mapping に拡張する想定だが本 Issue 範囲外。
