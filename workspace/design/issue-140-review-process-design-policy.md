# Issue #140 設計方針書 — Codex multi-stage review 必須化 + scaffolding 禁止 + embedding norm WARNING

> **対象 Issue**: [#140 process(review): Codex multi-stage design review 必須化 + scaffolding 命名禁止 checklist (S7-001 follow-up)](https://github.com/Kewton/photon-mlx/issues/140)
>
> **レビュー反映済み**: 全 29 finding (Must Fix 12 / Should Fix 12 / Nice to Have 5)
>
> **作成日**: 2026-04-26

---

## 1. ゴール (Issue 本文と一致)

> **「必須」の段階的厳格化** (DR1-010): 本 Issue における「必須」は **第 1 段階** を指し、現実装では Codex stage を skip した際に **WARNING + completion report 記録** までを強制する。raise / exit 1 への昇格 (= 完全強制) は次回 Issue で扱う。

1. `/multi-stage-design-review` (Stage 3-4 = Codex) と `/multi-stage-issue-review` (Stage 5-8 = Codex) の **Codex 担当 Stage 必須化** (= 第 1 段階強制力)
2. `/pm-auto-issue2dev` Phase 1/3、`/pm-auto-design2dev` の対応 Phase で **Codex stage 結果ファイル存在 + `reviewer="codex"` 検証** を strict 化 (初期実装は WARNING + completion report 記録)
3. `docs/code_review_checklist.md` 新規作成 (docs のみ、CI grep 自動化 test は **#139 Task 1 へ委譲**)
4. `PhotonInference.__init__` 起動時の **embedding norm WARNING** (閾値 config 化、初期は log only、raise しない)
5. `CLAUDE.md` スラッシュコマンド表に対象 4 skill を掲載

## 2. 技術スコープ

| 領域 | 変更種別 | 影響度 |
|------|---------|-------|
| Markdown skill 定義 (`.claude/commands/*.md`) | 文言追記 + auto-skip 廃止 | 高 (運用フロー変更) |
| 新規 docs (`docs/code_review_checklist.md`) | 新規作成 | 低 |
| `CLAUDE.md` | テーブル行追加 + リンク追加 | 低 |
| `photon_mlx/inference.py` | Python ロジック追加 (norm check) | 中 (29 箇所の test 呼び出しに影響) |
| `torch_ref/config.py` (or alternative) | 新規 config field | 中 (yaml load 互換性) |
| 新規 test (`tests/test_skill_descriptions.py`) | 新規作成 | 低 |
| 既存 test (`photon_mlx/tests/test_inference.py` / `test_generate.py` / `test_session.py` / `test_config.py`) | 影響確認 + WARNING 抑制 + yaml 互換 test | 中 |

## 3. アーキテクチャ判断

### 設計判断 #1: 閾値 config field の設置先 (S3-006 / S7-003)

**選択肢**:
- (a) `torch_ref/config.py::ModelConfig` 拡張 — 静的モデル設定として
- (b) `photon_mlx/session.py::WorkingMemoryConfig` 拡張 — 運用 knob として
- (c) 新規 `PhotonRuntimeChecksConfig` — 起動時 sanity check 専用

**決定**: **(a) ModelConfig 拡張**

**理由**:
- norm check は **モデル構造に紐付く起動時 sanity check** であり、session 単位の運用 knob ではない (working_memory_cfg は per-session なので意味的に不一致)
- `torch_ref/config.py:170-179` の `_set_fields` は unknown field を warning 化するため、新規 field 追加で **既存 5 個の `configs/photon_*.yaml`** (`photon_600m_paper` / `photon_long_context` / `photon_small` / `photon_tiny` / `photon_tiny_recgen`) の load 互換性を保てる (DR2-010)
- 新規 ConfigSection (c) は YAGNI。1 つの float field のために section を作るのは過剰
- (b) を採用すると `WorkingMemoryConfig=None` (`_UNSET`) ケースで閾値解決規則が新たに必要になる (S7-003 のリスク)

**トレードオフ**:
- メリット: 既存 yaml 互換性維持、解決経路が単純 (`cfg.model.embedding_random_init_threshold`)
- デメリット: ModelConfig が runtime-check 用 field を持つことに違和感 (本来は static)
- リスク: 将来 norm check 以外の起動時 check を増やすと ModelConfig が膨らむ → その時点で section 切り出しを検討

**field 名**: `embedding_random_init_threshold: float = 0.3` (デフォルト値、後で trained checkpoint で再校正)

**入力検証方針** (DR4-002): `configs/photon_*.yaml` や外部 config から入る値として扱うため、`torch_ref/config.py::ModelConfig.__post_init__` で **bool を除く数値型 / finite / 0 以上**を検証する。`nan` / `inf` / 負値 / 文字列 / bool は `ValueError` または `TypeError` として fail-loud にする。テスト時 WARNING 抑制にも `float('inf')` は使わず、有限の大値 `TEST_EMBEDDING_RANDOM_INIT_THRESHOLD = 1e9` を用いる。

**S7-003 受入条件の取り扱い宣言** (DR1-002): 本 Issue では (a) ModelConfig 拡張を選択するため、Issue 本文 S7-003 のうち WorkingMemoryConfig 案採用時の閾値解決規則 (`_UNSET` / `None` / 明示 WorkingMemoryConfig 各ケース) と baseline_reporag YAML roundtrip test 要件は **対象外**。代わりに `configs/photon_*.yaml` 5 件 (photon_600m_paper / photon_long_context / photon_small / photon_tiny / photon_tiny_recgen) の **load 互換 test** (新 field 未指定時に default 0.3 で load 成功) を追加する (詳細 §7.4)。

### 設計判断 #2: `_check_weight_initialization` のエラーハンドリング (S3-001)

**選択肢**:
- (a) `__init__` 内で例外を伝播
- (b) `__init__` 内で **silent skip** (try/except + isinstance ガード)

**決定**: **(b) silent skip**

**理由**:
- 既存 test 2 件 (`photon_mlx/tests/test_inference.py:368`、`baseline_reporag/tests/test_photon_pipeline.py:1860`) が `MagicMock()` model を渡しており、属性アクセスで例外が出ても test 意図 (prune_evidence の fail-closed 検証) と無関係
- norm check は **WARNING を出すだけの soft check** であり、init を fail させる正当性はない
- Issue #58 CB-002 の fail-closed pattern と同方針

**SRP トレードオフ判断** (DR1-004):
- `_check_weight_initialization` は責務 (a) 属性取得 (b) σ 計算 (c) WARNING 通知 を兼務する。SOLID-SRP 厳格化の観点では分割すべきだが、いずれも 5 行未満かつ起動時 1 回のみ実行されるため **KISS を優先して分割しない**。
- 将来 (i) 別の起動 check 追加時、または (ii) σ 計算ロジックを test 直接呼び出ししたいケースが出た時点で `_compute_embedding_std()` と `_warn_if_high_variance()` に分離する。

**実装** (Issue 本文 Task 4 と一致):

```python
def _check_weight_initialization(model: PhotonModel, threshold: float) -> None:
    try:
        embed_attr = getattr(model, "token_embed", None)
        if embed_attr is None:
            return
        weight = getattr(embed_attr, "weight", None)
        if not isinstance(weight, mx.array):
            return  # MagicMock や非標準モデルは skip
        norm_std = float(mx.std(weight).item())
    except Exception as exc:
        _logger.debug(
            "skip embedding init check (reason=%s)", type(exc).__name__
        )
        return

    if norm_std > threshold:
        _logger.warning(
            "PHOTON embedding has high variance (σ=%.4f, threshold=%.4f) — "
            "possibly random-init. ... (Issue #135 / S7-001).",
            norm_std, threshold,
        )
```

### 設計判断 #3: テスト時 WARNING 抑制方針 (S3-007 / DR1-001)

**選択肢**:
- (a) `photon_mlx/tests/conftest.py` に **autouse fixture** を新設し、test 中の閾値を `TEST_EMBEDDING_RANDOM_INIT_THRESHOLD = 1e9` に強制
- (b) `photon_mlx/tests/test_inference.py::_tiny_cfg()` 等のテストヘルパー側で `cfg.model.embedding_random_init_threshold = TEST_EMBEDDING_RANDOM_INIT_THRESHOLD` を default 化
- (c) 何もしない (WARNING ログを許容)

**決定**: **(b) `_tiny_cfg` 等テストヘルパーに直接埋め込む**

**理由** (DR1-001 修正):
- 当初案では「`photon_mlx/tests/conftest.py:_make` (= `stub_tokenizer_for_cfg` fixture) を改造」と記述したが、これは **事実誤認**。`_make` は tokenizer factory であって cfg を mutate しない。さらに既存 `test_inference.py` 等は `_tiny_cfg()` というモジュール内ヘルパーで cfg を新規生成しており、conftest.py を経由しない。
- (b) を選ぶと既存テストヘルパー側の改修だけで完結し、tokenizer factory の単一責任を破壊しない (SOLID-SRP)
- (a) を採るには新規 autouse fixture を導入する必要があり、既存 caplog アサーションを持つ test (例: `test_prune_evidence_logs_warning_on_tokenizer_fail`) と干渉するリスクがある
- (c) は CI ログを汚すが test failure は起こさない (logger.warning は warnings.warn ではないため `-W error` は無効) → 暫定的には許容できるが、ヘルパー修正のほうが意図が明示的

**実装方針** (DR1-007 / DR4-002 反映):
- `photon_mlx/tests/test_inference.py` に新規ヘルパー `_photon_cfg(threshold=TEST_EMBEDDING_RANDOM_INIT_THRESHOLD)` を追加し、内部で `_tiny_cfg()` をベースに閾値だけ差し替える wrapper として実装。`float('inf')` は production config validation と衝突し、YAML 経由の不正値を見逃すため使わない。

```python
TEST_EMBEDDING_RANDOM_INIT_THRESHOLD = 1e9

def _photon_cfg(threshold: float = TEST_EMBEDDING_RANDOM_INIT_THRESHOLD) -> PhotonConfig:
    cfg = _tiny_cfg()
    cfg.model.embedding_random_init_threshold = threshold
    return cfg
```

- **`_tiny_cfg` の重複定義と WARNING 抑制対象** (DR2-001 / DR3-001 / DR4-002 反映): `_tiny_cfg` は `photon_mlx/tests/` 配下の少なくとも 4 ファイル (`test_inference.py:39`, `test_optimize.py:31`, `test_generate.py:23`, `test_session.py:40`) で **重複定義** されている。このうち実 `PhotonModel` を渡して `PhotonInference(...)` を生成する経路は `test_inference.py`、`test_generate.py`、`test_session.py` に存在するため、3 ファイルの `_tiny_cfg()` に `cfg.model.embedding_random_init_threshold = TEST_EMBEDDING_RANDOM_INIT_THRESHOLD` を設定する。`test_optimize.py` は現時点で `PhotonInference` 生成経路を持たないため変更不要。4 ヘルパーの共通化 (重複解消) は本 Issue のスコープ外で別 Issue 候補とする。
- **WARNING 抑制対象の経路別整理** (DR2-003 反映):
  - `photon_mlx/tests/test_inference.py` 内 12 件 (実 `PhotonModel` 経路): `_tiny_cfg`/`_photon_cfg` で WARNING 抑制
  - `photon_mlx/tests/test_generate.py` 内の `inference_engine` / `PhotonInference(model, model.cfg, ...)` 経路: `model` fixture が `_tiny_cfg()` 由来のため、同ファイルの `_tiny_cfg` で WARNING 抑制
  - `photon_mlx/tests/test_session.py` 内の `engine` fixture / 個別 `PhotonInference(...)` 経路: 同ファイルの `_tiny_cfg` で WARNING 抑制
  - `baseline_reporag/tests/test_photon_pipeline.py:1860` (MagicMock 経路): `isinstance(weight, mx.array)` ガードで silent skip により無回帰 — 抑制不要
  - `test_optimize.py` など `PhotonInference` を生成しない `_tiny_cfg` 利用: 影響なし
- norm check の挙動 test (3 件、§7.2) は `_photon_cfg(threshold=0.1)` / `_photon_cfg(threshold=10.0)` を呼ぶことで閾値を試験的に切り替える。
- `baseline_reporag/tests/test_photon_pipeline.py` の `_BUILD_PHOTON_DEPS` mock は `PhotonInference` の生成元を mock しているため別途対応不要。

**新規 `baseline_reporag/tests/conftest.py` は作成しない** (S5-004)。

### 設計判断 #4: PM コマンドの reviewer 検証実装形態 (S7-001 / DR1-005)

**選択肢**:
- (a) PM command Markdown に `python -c '...'` snippet を追記し手順として記述
- (b) 新規 helper script (`scripts/verify_codex_review.py` 等) + unit test

**決定**: **(a) Markdown snippet + 文字列存在 test + Markdown snippet 動作 smoke test**

**理由**:
- 本 Issue のスコープは「skill description の strict 化」であり、新規実行可能ロジックを追加しない方が小さく速い
- helper script (b) は YAGNI: `gh issue` の JSON と `jq` で十分。Python script を書くと CI 経路を増やす
- ただし string-existence test 単独では「snippet が実際に動作するか」「reviewer 欠落 / 値ミスマッチで WARNING が出るか」を担保しない (DR1-005) — そこで snippet を一時ファイルに展開して bash で 3 ケース実行する **smoke test** を 1 件追加して構造的カバレッジ不足を埋める

**実装** (DR2-006 反映 — issue review と design review で **2 種類** の snippet を使い分け):

(a) **issue review 用** (`pm-auto-issue2dev.md` Phase 1) — `workspace/issues/$ISSUE/issue-review/` 配下、Stage 5 と 7 が対象:
```bash
# REVIEWER_VERIFICATION_SNIPPET_BEGIN (issue-review)
case "${ISSUE:-}" in
  ''|*[!0-9]*)
    printf 'WARNING: invalid ISSUE=%s\n' "${ISSUE:-}"
    exit 0
    ;;
esac

for stage in 5 7; do
  f="workspace/issues/$ISSUE/issue-review/stage${stage}-review-result.json"
  [ -f "$f" ] || { printf 'WARNING: %s missing\n' "$f"; continue; }
  reviewer=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("reviewer"))' "$f")
  reviewer_log=$(printf '%s' "$reviewer" | LC_ALL=C tr -c '[:alnum:]_.@:-' '?')
  [ "$reviewer" = "codex" ] || printf 'WARNING: %s reviewer=%s (expected codex)\n' "$f" "$reviewer_log"
done
# REVIEWER_VERIFICATION_SNIPPET_END
```

(b) **design review 用** (`pm-auto-issue2dev.md` Phase 3 / `pm-auto-design2dev.md` Phase 2) — `workspace/issues/$ISSUE/multi-stage-design-review/` 配下、Stage 3 と 4 が対象:
```bash
# REVIEWER_VERIFICATION_SNIPPET_BEGIN (design-review)
case "${ISSUE:-}" in
  ''|*[!0-9]*)
    printf 'WARNING: invalid ISSUE=%s\n' "${ISSUE:-}"
    exit 0
    ;;
esac

for stage in 3 4; do
  f="workspace/issues/$ISSUE/multi-stage-design-review/stage${stage}-review-result.json"
  [ -f "$f" ] || { printf 'WARNING: %s missing\n' "$f"; continue; }
  reviewer=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("reviewer"))' "$f")
  reviewer_log=$(printf '%s' "$reviewer" | LC_ALL=C tr -c '[:alnum:]_.@:-' '?')
  [ "$reviewer" = "codex" ] || printf 'WARNING: %s reviewer=%s (expected codex)\n' "$f" "$reviewer_log"
done
# REVIEWER_VERIFICATION_SNIPPET_END
```

`# REVIEWER_VERIFICATION_SNIPPET_BEGIN/END` のマーカーコメントは §7.5 の smoke test (DR2-007) で snippet 抽出を範囲指定するために用いる。

**snippet セキュリティ制約** (DR4-001 / DR4-003):
- `ISSUE` は数値のみ許可し、空文字・path traversal (`../`)・shell metacharacter を含む値は WARNING のみ出して処理を終了する。
- JSON path は Python code へ文字列埋め込みせず、`sys.argv[1]` として渡す。これにより `ISSUE` や path に quote が混入しても Python `-c` のコード注入にならない。
- `reviewer` 値は比較には raw value を使うが、ログ出力前に英数字と限定記号以外を `?` に置換し、改行や制御文字によるログ汚染を防ぐ。

`tests/test_skill_descriptions.py` の test:
- 文字列存在 test: `pm-auto-issue2dev.md` / `pm-auto-design2dev.md` に `reviewer="codex"` snippet が含まれること
- **smoke test** (新規、DR1-005 / DR4-001): snippet を一時ファイルに展開し bash で 3 ケース (reviewer="codex" / reviewer="claude" / ファイル欠落) を実行、WARNING 出力 / 通過の挙動を確認 (issue-review / design-review の 2 経路で parametrize)。加えて invalid ISSUE の path traversal / shell metacharacter / quote 混入を parametrize で確認する。

### 設計判断 #5: auto-skip 廃止の波及範囲 (S5-001 / S7-002 / DR1-008)

**対象** (`.claude/commands/multi-stage-issue-review.md` 内 3 箇所、行番号ではなく **見出し / 引用文字列** で特定):

1. **見出し「2回目イテレーション自動スキップ判定」** を含むセクション全体: **削除または「常に Stage 5-8 を実行する」と書き換え**
2. **Phase Final / サマリーレポート作成** テンプレート内で `5-8 | 2回目イテレーション | X | X | 完了/スキップ` を含む行 (現状 5 列構成: Stage / 内容 / opus / Codex / 状態): **状態列を「完了/スキップ」→「完了」固定にし、状態文言に「(reviewer=codex 検証済)」を併記** (列数を変更せずに統合表現とする — DR2-005 反映)。table header 側に「状態 (reviewer 検証含む)」と短く注記
3. **完了条件** 箇条書き内の「2回目イテレーション自動スキップ」を含む行: **削除**

(行番号は本 Issue 自身で `multi-stage-issue-review.md` を編集するため、編集中にずれる。文字列マッチで対象を特定する方が安全 — DR1-008)

`/multi-stage-design-review` には Stage 5-8 が存在しないので auto-skip 関連の更新は不要。

### 設計判断 #6: failure policy — strict 化 vs WARNING のみ (S5-002)

**決定**: **構造化検証は必ず行うが、初期実装は WARNING + completion report 記録に留め fatal にしない**

**理由**:
- 既存運用が `--skip-stage` を許容しているため、いきなり raise/exit 1 にすると open Issue の再レビューが壊れる
- WARNING を出して completion report に明示することで「不備が起きていることが見えるが運用は止まらない」状態を作る (1 リリース猶予)
- 強制 NG 化は次回 Issue (例: `process: Codex stage skip を fatal 化`) で扱う

**Issue 本文 §後方互換性 / ロールバック** と一致。

## 4. レイヤー別変更マップ

```
┌─────────────────────────────────────────────────┐
│ Layer 1: Process / Skill 定義 (Markdown)         │
│  - .claude/commands/multi-stage-design-review.md │
│  - .claude/commands/multi-stage-issue-review.md  │
│  - .claude/commands/pm-auto-issue2dev.md         │
│  - .claude/commands/pm-auto-design2dev.md        │
├─────────────────────────────────────────────────┤
│ Layer 2: ドキュメント                            │
│  - docs/code_review_checklist.md (新規)          │
│  - CLAUDE.md (リンク追加 + skill 表更新)         │
├─────────────────────────────────────────────────┤
│ Layer 3: Python 実装                             │
│  - photon_mlx/inference.py (norm check)          │
│  - torch_ref/config.py (ModelConfig field 追加)   │
├─────────────────────────────────────────────────┤
│ Layer 4: テスト                                  │
│  - tests/test_skill_descriptions.py (新規)       │
│  - photon_mlx/tests/test_config.py (yaml 互換 + 閾値検証) │
│  - photon_mlx/tests/test_inference.py (3 ケース) │
│  - photon_mlx/tests/test_generate.py (_tiny_cfg) │
│  - photon_mlx/tests/test_session.py (_tiny_cfg)  │
└─────────────────────────────────────────────────┘
```

依存順序: Layer 3 (config field) → Layer 4-pre (yaml 互換 test) → Layer 4-A (test cfg WARNING 抑制) → Layer 3-B (inference.py) → Layer 4-B/C (挙動 test / skill test) → Layer 1 (skill) → Layer 2 (docs)

## 5. セキュリティ設計

| 脅威 | 対策 |
|------|------|
| **WARNING ログから embedding tensor 値が漏洩** | `_check_weight_initialization` のログ出力は σ (スカラー float) と threshold のみ。tensor 自体・サンプル要素はログに出さない (Issue #58 CB-002 / #64 CB-003 と同方針) |
| **CI grep スクリプトが機密ファイル (.env / credentials) を読む** | grep 対象は `baseline_reporag/`, `photon_mlx/`, `torch_ref/` の production import path に限定。`.env` 等は対象外。**詳細な対象/除外パターンは `docs/code_review_checklist.md` を Single Source of Truth とする** (DR1-009) — 本書では要約のみ記述 |
| **新規 yaml field で existing config が破壊** | ModelConfig field にデフォルト値 0.3 を設定、`_set_fields` の warning 経路で field 未指定時も load 成功。field 指定時は `ModelConfig.__post_init__` で bool / 非数値 / 非 finite / 負値を拒否 (DR4-002) |
| **reviewer 検証 snippet の command injection / path traversal** | `ISSUE` は数値のみ許可。JSON path は Python code に埋め込まず `sys.argv[1]` で渡す。`reviewer` はログ出力前に制御文字を置換する (DR4-001 / DR4-003) |
| **test 用 tmp_path 操作の path traversal** | smoke test は `ISSUE` を環境変数で渡し、snippet 側の数値検証を通す。invalid ISSUE (`../140`, `140;touch x`, quote 混入) が `tmp_path/workspace/issues/` 外へアクセスしないことを test する (DR4-001) |

## 6. 後方互換性 / ロールバック

| 項目 | 互換性確保 | ロールバック |
|------|----------|-----------|
| `--skip-stage` フラグ | 保持 (WARNING のみ、強制 NG 化は次回 Issue) | フラグ自体を保持しているため revert 不要 |
| ModelConfig 新 field | デフォルト値で既存 yaml 互換 | field 削除で yaml 側は warning のみ (load 成功) |
| `_check_weight_initialization` 呼び出し | silent skip で既存 test 互換 | 1 行コメントアウトで完全無効化可能 |
| auto-skip 廃止 | iteration 2 が常に実行されるが、既存 issue review の中断はない | 該当セクション再追加で revert 可能 |

## 7. テスト戦略

### 7.1 新規テスト (`tests/test_skill_descriptions.py`)

```python
from pathlib import Path
SKILL_DIR = Path('.claude/commands')

def test_design_review_codex_required():
    """DR2-009 反映: 既存類似文字列との衝突を避けるため本 Issue 追加文言を完全一致 assert"""
    body = (SKILL_DIR / 'multi-stage-design-review.md').read_text(encoding='utf-8')
    assert 'Codex 担当 Stage は必須' in body
    assert 'WARNING' in body and 'completion report' in body

def test_issue_review_codex_required():
    body = (SKILL_DIR / 'multi-stage-issue-review.md').read_text(encoding='utf-8')
    assert 'Codex 担当 Stage は必須' in body
    assert 'WARNING' in body and 'completion report' in body

def test_pm_auto_issue2dev_reviewer_check_snippet():
    body = (SKILL_DIR / 'pm-auto-issue2dev.md').read_text(encoding='utf-8')
    assert 'reviewer="codex"' in body or 'reviewer=\\"codex\\"' in body

def test_claude_md_lists_target_skills():
    body = Path('CLAUDE.md').read_text(encoding='utf-8')
    for skill in ['/multi-stage-issue-review', '/multi-stage-design-review',
                  '/pm-auto-issue2dev', '/pm-auto-design2dev']:
        assert skill in body, f'{skill} not in CLAUDE.md'

def test_code_review_checklist_exists():
    assert Path('docs/code_review_checklist.md').exists()
    body = Path('docs/code_review_checklist.md').read_text(encoding='utf-8')
    assert '_Stub' in body and '_Mock' in body
    assert '*/tests/**' in body  # 除外パターン

def test_auto_skip_removed_from_issue_review():
    body = (SKILL_DIR / 'multi-stage-issue-review.md').read_text(encoding='utf-8')
    # auto-skip 判定セクションが廃止されていること
    assert '2回目イテレーション自動スキップ判定' not in body or '廃止' in body
```

### 7.2 PhotonInference norm check テスト (`photon_mlx/tests/test_inference.py`)

**新規ヘルパー** (DR1-007): test_inference.py モジュール内に `_photon_cfg` を追加 (DR1-001 / 設計判断 #3 の (b) 実装)。`_tiny_cfg()` を呼んで cfg を生成し、`embedding_random_init_threshold` だけ test の意図に合わせて差し替える wrapper。

```python
TEST_EMBEDDING_RANDOM_INIT_THRESHOLD = 1e9

def _photon_cfg(threshold: float = TEST_EMBEDDING_RANDOM_INIT_THRESHOLD) -> PhotonConfig:
    """Tests use this instead of `_tiny_cfg` to control the embedding-norm
    threshold. Defaults to a finite high value so existing tests do not emit WARNING."""
    cfg = _tiny_cfg()
    cfg.model.embedding_random_init_threshold = threshold
    return cfg
```

**テストケース 3 件**:

```python
def test_check_weight_initialization_warns_on_high_variance(caplog):
    cfg = _photon_cfg(threshold=0.1)  # 低 threshold で確実に warn
    model = PhotonModel(cfg)  # random-init (DR2-004: 既存ヘルパー直接呼び出し)
    tokenizer = stub_tokenizer_for_cfg(cfg)  # DR2-004: conftest fixture 名と整合
    with caplog.at_level(logging.WARNING, logger='photon_mlx.inference'):
        PhotonInference(model, cfg, tokenizer)
    assert any('high variance' in r.message for r in caplog.records)

def test_check_weight_initialization_silent_on_low_variance(caplog):
    cfg = _photon_cfg(threshold=10.0)  # 高 threshold で確実に silent
    model = PhotonModel(cfg)
    tokenizer = stub_tokenizer_for_cfg(cfg)
    with caplog.at_level(logging.WARNING, logger='photon_mlx.inference'):
        PhotonInference(model, cfg, tokenizer)
    assert not any('high variance' in r.message for r in caplog.records)

def test_check_weight_initialization_silent_on_magicmock_model(caplog):
    """既存 test (test_inference.py:368 の MagicMock パターン) と同方針"""
    cfg = _photon_cfg()
    with caplog.at_level(logging.WARNING, logger='photon_mlx.inference'):
        PhotonInference(MagicMock(), cfg, _BrokenTokenizer())  # 例外も WARNING も出ない
    assert not any('high variance' in r.message for r in caplog.records)
```

### 7.3 既存テスト無回帰

- `photon_mlx/tests/test_inference.py:368` (MagicMock + _BrokenTokenizer): silent skip により無回帰
- `baseline_reporag/tests/test_photon_pipeline.py:1860` (mock_model): 同上
- 実 `PhotonModel` を使う `PhotonInference(...)` 呼び出し test: `test_inference.py` / `test_generate.py` / `test_session.py` の各 `_tiny_cfg()` で `embedding_random_init_threshold = TEST_EMBEDDING_RANDOM_INIT_THRESHOLD` を設定し WARNING を抑制 (DR1-001 / DR3-001 / DR4-002 反映 — conftest.py を経由しない)
- 既存 caplog 系 security test (`test_generate.py::test_tokenize_chunk_failure_logs_without_exception_text`, `test_session.py::test_tokenize_chunk_warning_contains_only_type_name`) は、embedding norm WARNING が混入しない前提で raw exception body leak の検証を継続できる。

### 7.4 既存 yaml load 互換テスト (DR1-003 / DR2-002 反映)

**配置先**: `photon_mlx/tests/test_config.py` (既に `load_photon_config` / `_set_fields` の config 互換テストを持つため、同じ責務に集約する。DR3-002)

`configs/photon_*.yaml` の 5 件 (`photon_600m_paper`, `photon_long_context`, `photon_small`, `photon_tiny`, `photon_tiny_recgen`) が **新 field 未指定でも** load 成功し default 値 0.3 が適用されることを確認:

```python
import pytest
import math
from pathlib import Path
from torch_ref.config import ModelConfig
from torch_ref.config import load_photon_config

@pytest.mark.parametrize('yml', sorted(Path('configs').glob('photon_*.yaml')))
def test_existing_yaml_loads_without_threshold_field(yml):
    cfg = load_photon_config(yml)  # torch_ref/config.py:215 の関数
    # 新 field を yaml で指定していない場合、default 0.3 が適用される
    assert cfg.model.embedding_random_init_threshold == pytest.approx(0.3)

@pytest.mark.parametrize('bad_threshold', [-0.1, math.nan, math.inf, '0.3', True])
def test_model_config_rejects_invalid_embedding_threshold(bad_threshold):
    with pytest.raises((TypeError, ValueError)):
        ModelConfig(embedding_random_init_threshold=bad_threshold)
```

**注意点** (DR2-002):
- `torch_ref/config.py:215-257` の `load_photon_config` は (1) `_set_fields(cfg.model, raw.get('model', {}))` で yaml の model section を流し込み、(2) その後 `cfg.model.__post_init__()` を **明示再実行** する (line 222)。
- 本 Issue で追加する `embedding_random_init_threshold: float = 0.3` field は **`__post_init__` で bool を除く数値型 / finite / 0 以上を検証する** (DR4-002)。`load_photon_config` は `__post_init__()` を明示再実行するため、YAML で不正値が指定された場合も fail-loud になる。
- `photon_long_context.yaml` / `photon_600m_paper.yaml` は `training` セクションを持つため `_validate_cross_config` (line 189-212) が走る。本 field 追加は `_validate_cross_config` の対象外であり、既存 cross-validation を破壊しないことを暗黙的に保証する。
- 各 `configs/photon_*.yaml` には `embedding_random_init_threshold:` を **書き込まない** (既存 yaml の互換性を保つことが受入条件)。

### 7.5 PM コマンド reviewer 検証 snippet 動作 smoke test (DR1-005 / DR2-007 反映)

snippet を Markdown 内の **マーカーコメント範囲** (`# REVIEWER_VERIFICATION_SNIPPET_BEGIN ... END`) で抽出。これにより複数の `bash` フェンスがあっても誤抽出しない (DR2-007):

```python
import os, re, subprocess, json, pytest
from pathlib import Path

# DR2-007 対策: マーカー範囲で限定 + 最短マッチ。issue-review / design-review の両方の snippet を抽出可能
SNIPPET_RE = re.compile(
    r'#\s*REVIEWER_VERIFICATION_SNIPPET_BEGIN\s*\((?P<kind>issue-review|design-review)\)\s*\n'
    r'(?P<body>.*?)\n'
    r'#\s*REVIEWER_VERIFICATION_SNIPPET_END',
    re.S,
)

def _extract_snippets(md_path: Path):
    md = md_path.read_text(encoding='utf-8')
    return {m.group('kind'): m.group('body') for m in SNIPPET_RE.finditer(md)}

@pytest.mark.parametrize('kind,md_file', [
    ('issue-review', '.claude/commands/pm-auto-issue2dev.md'),
    ('design-review', '.claude/commands/pm-auto-issue2dev.md'),
    ('design-review', '.claude/commands/pm-auto-design2dev.md'),
])
@pytest.mark.parametrize('reviewer,expect_warning', [
    ('codex', False), ('claude', True), (None, True),
])
def test_reviewer_snippet_smoke(tmp_path, kind, md_file, reviewer, expect_warning):
    snippets = _extract_snippets(Path(md_file))
    snippet = snippets.get(kind)
    assert snippet, f'{kind} snippet not found in {md_file}'
    issue = '140'
    if kind == 'issue-review':
        base = tmp_path / 'workspace' / 'issues' / issue / 'issue-review'
        stages = (5, 7)
    else:
        base = tmp_path / 'workspace' / 'issues' / issue / 'multi-stage-design-review'
        stages = (3, 4)
    base.mkdir(parents=True)
    if reviewer is not None:
        for stage in stages:
            (base / f'stage{stage}-review-result.json').write_text(
                json.dumps({'reviewer': reviewer}),
                encoding='utf-8',
            )

    script = tmp_path / 'snippet.sh'
    script.write_text(f'{snippet}\n', encoding='utf-8')
    env = os.environ.copy()
    env['ISSUE'] = issue
    proc = subprocess.run(
        ['bash', str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    if expect_warning:
        assert 'WARNING' in combined
    else:
        assert 'WARNING' not in combined

@pytest.mark.parametrize('kind,md_file', [
    ('issue-review', '.claude/commands/pm-auto-issue2dev.md'),
    ('design-review', '.claude/commands/pm-auto-issue2dev.md'),
])
@pytest.mark.parametrize('issue', ['../140', '140;touch injected', "140'bad"])
def test_reviewer_snippet_rejects_invalid_issue(tmp_path, kind, md_file, issue):
    snippet = _extract_snippets(Path(md_file))[kind]
    script = tmp_path / 'snippet.sh'
    script.write_text(f'{snippet}\n', encoding='utf-8')
    env = os.environ.copy()
    env['ISSUE'] = issue
    proc = subprocess.run(
        ['bash', str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert 'WARNING: invalid ISSUE=' in combined
    assert not (tmp_path / 'injected').exists()
```

(snippet 抽出 + bash 実行で「reviewer=codex なら無音、claude / 欠落なら WARNING」を確認。issue-review / design-review の 2 経路 × 3 ケースを parametrize。加えて invalid ISSUE の path traversal / shell metacharacter / quote 混入が WARNING のみで終了し、`tmp_path` 配下に意図しないファイルを作らないことを確認する)

## 8. 実装順序 (作業計画立案フェーズへの申し送り、DR1-006 反映)

> **再編成方針**: WARNING 抑制基盤を **先に置いてから** norm check 本体を入れる順に変更。これにより Step N で test 失敗 → Step N-1 設計の見直し → Step N-2 まで遡る逆転を防ぐ。

1. **Layer 3-A**: `torch_ref/config.py::ModelConfig` に `embedding_random_init_threshold: float = 0.3` field 追加
2. **Layer 4-pre** (新規、DR2-008): `configs/photon_*.yaml` 5 件の load 互換 test (§7.4) を追加し、新 field 追加直後に既存 yaml が壊れないことを **早期検証**
3. **Layer 4-A** (DR1-006 / DR3-001 / DR4-002): `photon_mlx/tests/test_inference.py` / `test_generate.py` / `test_session.py` の `_tiny_cfg()` を更新 (`embedding_random_init_threshold = TEST_EMBEDDING_RANDOM_INIT_THRESHOLD` を default 化)。`test_inference.py` には新規ヘルパー `_photon_cfg(threshold=TEST_EMBEDDING_RANDOM_INIT_THRESHOLD)` (§7.2) も追加。**`test_optimize.py` の `_tiny_cfg` は PhotonInference 経路を持たないため変更不要**
4. **Layer 3-B**: `photon_mlx/inference.py` に `_check_weight_initialization` 関数を追加し `PhotonInference.__init__` から呼び出し
5. **検証**: `test_inference.py` / `test_generate.py` / `test_session.py` の実 `PhotonModel` を使う既存 `PhotonInference(...)` test が **無回帰** (WARNING 抑制が効きすべて pass)。MagicMock 経路は silent skip で無回帰 (DR2-003 / DR3-001)
6. **Layer 4-B**: `photon_mlx/tests/test_inference.py` に norm check の挙動 test 3 件 (§7.2) を追加
7. **Layer 4-C**: `tests/test_skill_descriptions.py` 新規作成 — 文字列存在 test 6 件 (§7.1) + reviewer snippet smoke test (§7.5、issue-review/design-review × 3 ケースで parametrize)
8. **Layer 1-A**: `.claude/commands/multi-stage-design-review.md` description 更新 (「Codex 担当 Stage は必須」「skip 時は WARNING + completion report 記録」を含む文言に更新 — DR2-009)
9. **Layer 1-B**: `.claude/commands/multi-stage-issue-review.md` description 更新 + auto-skip 廃止 (設計判断 #5 の 3 箇所、文字列マッチで特定)
10. **Layer 1-C**: `.claude/commands/pm-auto-issue2dev.md` Phase 1 / Phase 3 完了判定 (reviewer snippet 2 種、`# REVIEWER_VERIFICATION_SNIPPET_BEGIN/END` マーカー付き) 追加
11. **Layer 1-D**: `.claude/commands/pm-auto-design2dev.md` Phase 2 完了判定 (design-review 用 reviewer snippet、マーカー付き) 追加
12. **Layer 2-A**: `docs/code_review_checklist.md` 新規作成 (Single Source of Truth)
13. **Layer 2-B**: `CLAUDE.md` スラッシュコマンド表更新 (4 skill 追加) + checklist リンク追加
14. **最終検証**: `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` / `ruff check .` / `ruff format --check .`

## 9. 品質基準 (CLAUDE.md と同一)

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| テスト | `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | 全パス (既知 pre-existing failure 2 件: `tests/test_generate_training_corpus.py` を除く) |
| Lint | `ruff check .` | 警告 0 件 |
| Format | `ruff format --check .` | 差分なし |

## 10. 参考資料

- Issue #140: https://github.com/Kewton/photon-mlx/issues/140
- Issue review summary: `workspace/issues/140/issue-review/summary-report.md`
- 仮説検証: `workspace/issues/140/issue-review/hypothesis-verification.md`
- 関連 Issue: #135 (PHOTON 再学習), #138 (tokenizer mismatch — CLOSED), #139 (Stub/Mock audit)
- 元 finding: S7-001 (commit `2dbf458`)
