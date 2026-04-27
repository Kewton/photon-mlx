# Issue #139 設計方針書

**対象 Issue**: [#139](https://github.com/Kewton/photon-mlx/issues/139) — test(photon): Stub/Mock pattern audit + invariant test (S7-001 follow-up)
**作成日**: 2026-04-26
**ブランチ**: `feature/issue-139-stub-audit`
**前提レビュー**: マルチステージ Issue レビュー完了済 (Must Fix 7 / Should Fix 13 / Nice 5、すべて反映)
**スコープ**: Task 1 (scaffolding 排除) + Task 3 Phase A (yaml invariant 拡張) のみ。Task 2 (real-weight integration test) は **#145** に切出済。

---

## 1. 背景と目的

S7-001 (commit `2dbf458` on `feature/issue-135-photon-retrain`) で発見された「`PhotonModel` が random-init 重みのまま production eval を走らせていた」事象は **個別 bug ではなく、構造的な test ギャップ** から生じた:

- production code path (`baseline_reporag/photon_pipeline._build_photon_deps`) に `_StubTokenizer` という test/dev only シンボルが warning + fallback で生存
- 既存 `baseline_reporag/tests/test_photon_pipeline.py` (4453 行) が MagicMock 中心で、実コード path を mock で覆って通っていた
- 「設計 Must Fix を CI で固定する」 invariant test 機構が未整備

本 Issue は **同型の silent bug の再発を CI レベルで構造的に塞ぐ**:
- (Task 1) production path から scaffolding 命名 (`_Stub*` / `_Mock*` / `_Dummy*` / `_Placeholder*`) を **完全削除**、欠落時は `ValueError` で fail-fast
- (Task 3) PHOTON profile yaml に必須フィールド (`tokenizer.vocab_size` / `tokenizer.tokenizer_id`) が存在することを invariant test で恒久的に固定

---

## 2. 関連アーキテクチャ (本 Issue が触る部分のみ)

```
┌────────────────────────────────────────────────────────────┐
│                  baseline_reporag/cli.py                    │
│                  baseline_reporag/server.py                 │
│                       (CLI / FastAPI)                       │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│        baseline_reporag/pipeline_factory.py:create_pipeline │
│           ─ provider 分岐 (lazy MLX import 含む) ─          │
└─────────┬──────────────────────────────────┬───────────────┘
          │ provider=='mlx_lm'              │ provider=='photon'
          ▼                                  ▼
┌──────────────────┐         ┌──────────────────────────────────┐
│  pipeline.py     │         │  photon_pipeline.py              │
│   (Qwen baseline)│         │  ┌─ _build_photon_deps ────────┐ │ ← 本 Issue 改修対象
│                  │         │  │ if tokenizer_id:            │ │
│                  │         │  │   _load_hf_tokenizer(...)   │ │
│                  │         │  │ else: # 本 Issue で削除      │ │
│                  │         │  │   _logger.warning(...)      │ │
│                  │         │  │   _get_stub_tokenizer(...)  │ │
│                  │         │  │ PhotonModel(photon_cfg)     │ │
│                  │         │  └────────────────────────────┘ │
│                  │         │  ※ 本 Issue 後は else 分岐撤去   │
│                  │         │     + ValueError raise に統一   │
└──────────────────┘         └──────────────────────────────────┘

                    [test 階層]
┌────────────────────────────────────────────────────────────┐
│  baseline_reporag/tests/test_photon_pipeline.py (4453 行)  │ ← migration 必要
│    ・_StubTokenizer import / isinstance (line 521, 563)     │
│    ・test_falls_back_to_stub_when_tokenizer_id_missing      │
│      (line 604-636) ─ 削除し test_raises_when_... に置換   │
├────────────────────────────────────────────────────────────┤
│  tests/test_no_scaffolding_in_prod.py (新規)                │ ← Task 1
│    repo root を __file__.resolve().parents[1] で解決し、     │
│    \b_(?:Stub|Mock|Dummy|Placeholder)\w* を検出             │
├────────────────────────────────────────────────────────────┤
│  tests/test_pipeline_factory_yaml_invariants.py (拡張)      │ ← Task 3
│    既存の reranker.model_id 不変宣言と同形式で              │
│    tokenizer.vocab_size / tokenizer.tokenizer_id を追加     │
└────────────────────────────────────────────────────────────┘
```

**呼び出しチェーン (production)**:
`cli.py / server.py` → `pipeline_factory.create_pipeline(cfg)` → (provider=='photon' なら) `photon_pipeline._build_photon_deps(cfg)` → `AutoTokenizer.from_pretrained(cfg.tokenizer.tokenizer_id)` + `PhotonModel(...)` 構築

**weekly_eval / CLAUDE.md 疎通コマンド**: `configs/baseline.yaml` (provider=mlx_lm) → `pipeline.py` 経路 (本 Issue は関与しない)。

---

## 3. レイヤーごとの責務 (本 Issue 範囲)

| レイヤー | 対象ファイル | 本 Issue での責務 |
|---------|-------------|-----------------|
| **Production (PHOTON)** | `baseline_reporag/photon_pipeline.py` | `_StubTokenizer` 系を完全削除。`_build_photon_deps` で tokenizer 必須化 + load 例外を `ValueError` に正規化 |
| **既存 test** | `baseline_reporag/tests/test_photon_pipeline.py` | stub fallback 期待 test を raise 期待 test に migration |
| **境界 test (新規)** | `tests/test_no_scaffolding_in_prod.py` | scaffolding 命名の production 流入を CI で検出 |
| **invariant test (拡張)** | `tests/test_pipeline_factory_yaml_invariants.py` | PHOTON profile yaml に必須フィールドが宣言されていることを CI で固定 |
| **運用 docs** | `docs/troubleshooting.md` | tokenizer 起動失敗系 (未設定 / load 失敗) の checklist を追記 |
| **設定 (必要なら)** | `configs/photon_*.yaml`, `configs/institutional_docs_photon.yaml` | 必須フィールドが既に揃っていれば変更不要、欠落あれば補完 |

---

## 4. 設計判断とトレードオフ

### 設計判断 #1: `_StubTokenizer` を rename ではなく **完全削除** する

**選択肢**:
- A: `_StubTokenizer` → `_DevTokenizer` 等にリネームして production module に残す
- B: `_StubTokenizer` / `_get_stub_tokenizer` を production module から **完全削除** し、`_build_photon_deps` で fallback 経路自体を撤去 (本 Issue 採用)
- C: HuggingFace tokenizer に常時置換 (= fallback を高機能化)

**決定**: **B (完全削除)**

**理由**:
- S7-001 の本質は「production runtime path に test fixture が到達した」ことそのもの。リネーム (A) では「test fixture を残して名前で警戒する」 だけで、構造的な保護にならない (S1-005 / hypothesis-verification.md Claim 7)。
- C は「fallback 自動化」になり、設定漏れの隠蔽が起きやすい (S7-001 が exactly そのパターン)。
- B は **fail-fast** (`ValueError` raise) で設定漏れを起動時に明示し、CI でも境界 test (`test_no_scaffolding_in_prod.py`) で恒久化できる。

**トレードオフ**:
- メリット: production path が test 互換性層を一切経由しなくなる。境界 test と組み合わせて静的に invariant 化できる。
- デメリット: `tokenizer.tokenizer_id` を欠いた古い yaml は起動できなくなる (= 破壊的変更)。
- リスク緩和: `provider=='photon'` の yaml にしか影響しない (`baseline.yaml` は provider=mlx_lm のため無関係)。`weekly_eval.yml` も baseline.yaml を使用するため runtime 影響なし (S3-010)。

### 設計判断 #2: tokenizer **load 失敗** も `ValueError` に正規化する (S5-002 / DR1-002 反映)

**選択肢**:
- A: 「未設定」のみ `ValueError`、load 失敗 (HF Hub 障害 / gated model / cache miss / network) は `transformers` の例外をそのまま伝播
- B: `_build_photon_deps` 境界で load 失敗を捕捉し、対象 `tokenizer_id` を含む message で **`ValueError` に正規化** (本 Issue 採用)

**決定**: **B**

**理由**:
- 運用観点: server / CLI 利用者にとって、(a) yaml に書き忘れた / (b) HF Hub にアクセスできない / (c) gated model にアクセスできない、はいずれも「PHOTON pipeline が立ち上がらない」失敗で、共通の troubleshooting フローに乗せたい (`docs/troubleshooting.md` の photon checklist)。
- test 観点: load 失敗を `_build_photon_deps` 境界で `ValueError` として固定すれば、`AutoTokenizer.from_pretrained` を `OSError` 等で patch するだけで失敗 path をテストできる (S7-001 系の silent bug を再発させない)。

**正規化対象の例外型 (DR1-002 反映)**:

現行 `_load_hf_tokenizer` (`baseline_reporag/photon_pipeline.py:469-510`) には以下の raise が存在し、本 Issue ではそれぞれの扱いを明確化する:

| 既存 raise | 行 | 扱い (本 Issue 後) |
|-----------|-----|-------------------|
| `ImportError` (transformers 不在) | L488 | **そのまま維持** (依存欠落は環境問題で `ValueError` に丸めると診断性低下) |
| `ValueError` (vocab_size mismatch) | L494-498 | **そのまま維持** (Issue #138 由来の既存 invariant) |
| `OSError` / `huggingface_hub.errors.*` (`AutoTokenizer.from_pretrained` 失敗) | L493 | **新規 try/except で sanitized `ValueError` に正規化** |

つまり、try/except は `AutoTokenizer.from_pretrained()` 呼び出しブロック (L493) のみに **限定** し、既存の `ImportError` / `vocab_size ValueError` パスを保護する。

**トレードオフ**:
- メリット: 起動失敗系のうち「load 系」の例外型が単一化され、呼び出し元 (server.py / cli.py) のエラーハンドリングがシンプル。`docs/troubleshooting.md` の checklist も単一の symptom に落とし込める。
- デメリット: 例外チェーン情報がやや見えにくくなるため、`raise ValueError(...) from original_exc` で原因を保持する必要あり。
- リスク: 例外正規化の対象を広げすぎると正常系の例外まで握り潰す → `transformers.AutoTokenizer.from_pretrained` の呼び出しブロックに **限定** する。`ImportError` と vocab mismatch は対象外。
- supply chain 前提 (DR4-002): `huggingface_hub.errors.*` は `transformers` 経由の transitive dependency で version 差分が出やすいため、本 Issue では例外処理のために具体的な `huggingface_hub.errors` class を production code に import しない。`from_pretrained(...)` の局所ブロックで `OSError` とその他 load failure を `Exception` として受け、`raise ValueError(...) from exc` で原因 chain を保持する。
- security invariant (DR4-002): `AutoTokenizer.from_pretrained` は必ず `trust_remote_code=False` を明示する。将来 `True` に変更する場合は本 Issue の範囲外で、別途 security review を必要とする。

### 設計判断 #2-補足: `tokenizer_id` は untrusted yaml input として validate + sanitize する (DR4-001 / DR4-005)

`tokenizer.tokenizer_id` は yaml 由来の入力であり、server / CLI の log、Streamlit error banner、Slack 通知、GitHub issue への貼付に流れる可能性がある。したがって `_load_hf_tokenizer` に渡す前に allowlist validation を行い、公開 message には control character や token-like string をそのまま出さない。

**必須ルール**:
- `tokenizer_id` は `str` かつ空文字不可。
- 許可形式は Hugging Face repo id の allowlist (`^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`) に限定する。
- URL (`://`)、absolute / relative path (`/`, `.`, `~` 開始)、path traversal (`..`)、backslash、control character (`\n`, `\r`, `\t`, `\x00` 等) は `ValueError`。
- production module から `scripts/generate_training_corpus.py` を import しない。既存 `validate_tokenizer_id()` の考え方を参考に、必要なら production 側 helper として同等の validation を実装する。
- 公開 `ValueError` message は sanitized display のみを使う。例外 chain (`from exc`) は保持するが、message 本文に raw `exc` を丸ごと埋め込まない。特に HF token、private path、private model id を含む文字列を Slack / Streamlit / public log に貼らない前提を docs に明記する。

```python
HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _validate_tokenizer_id(raw: object) -> str:
    if not isinstance(raw, str) or raw == "":
        raise ValueError("cfg.tokenizer.tokenizer_id is required for provider=='photon'.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise ValueError("cfg.tokenizer.tokenizer_id contains control characters.")
    if (
        "://" in raw
        or raw.startswith(("/", ".", "~"))
        or ".." in raw
        or "\\" in raw
        or not HF_REPO_ID_RE.fullmatch(raw)
    ):
        raise ValueError(
            "cfg.tokenizer.tokenizer_id must be a Hugging Face repo id like "
            "'org/model', not a URL or local path."
        )
    return raw


def _display_tokenizer_id(tokenizer_id: str) -> str:
    if len(tokenizer_id) > 120:
        tokenizer_id = tokenizer_id[:117] + "..."
    return repr(tokenizer_id)
```

### 設計判断 #3: 境界 test は **repo root 基準 + 単一 regex** とする (S3-003 / S5-001 / S7-001)

**選択肢**:
- A: `Path('baseline_reporag')` 等の cwd 相対 path、root 不在は skip
- B: `Path(__file__).resolve().parents[1]` で repo root を解決し、root 不在は **assert failure** (本 Issue 採用)

**決定**: **B**

**理由**:
- A は pytest を repo root 以外から呼ぶと対象 0 件で **偽 pass** する (S7-001)。CI gate として機能しないと境界 test の意味がない。
- regex は `\b_(?:Stub|Mock|Dummy|Placeholder)\w*` で `_StubTokenizer` 等の識別子全長を捕捉。`r'_Stub\b'` は `\b` が `b|T` 境界に発火しないため `_StubTokenizer` を見逃す (S3-007 検証済)。
- path 除外は `'tests' in f.parts` (**複数形** + tuple membership)。 `'test' in f.parts` は `('baseline_reporag', 'tests', ...)` で False (S3-003)。

### 設計判断 #4: invariant test の対象を **PHOTON profile filter** で絞る (S3-002 / S5-001)

**選択肢**:
- A: 全 `configs/*.yaml` を対象 (Issue 当初案)
- B: `_get_nested(cfg, 'model.provider') == 'photon'` のみで判定
- C: `provider == 'photon'` または **filename ベース** (`photon_*.yaml` / `institutional_docs_photon.yaml`) (本 Issue 採用)

**決定**: **C**

**理由**:
- A は `baseline.yaml` (provider=mlx_lm) と `eval.yaml` (benchmark runner、`model:` block 自体なし) で必ず FAIL し、運用上のメイン config を invariant が壊す (S3-002)。
- B のみだと `configs/photon_tiny.yaml` / `configs/photon_600m_paper.yaml` が `model.provider` 未設定のため取りこぼしされ、PHOTON training profile に必須フィールド欠落があっても CI が検出しない (S5-001)。
- C は実態に即した PHOTON profile の上位集合を捕捉する。将来的に新 photon yaml を追加するときは prefix 命名規約に従えばよく、CI gate を後付けで強化しやすい。

### 設計判断 #5: 既存 test の migration 計画を Issue 受入条件に明記 (S3-004 / DR3-001)

**`baseline_reporag/tests/test_photon_pipeline.py` の直接参照対応**:
- L521 周辺 `from baseline_reporag.photon_pipeline import _StubTokenizer` → 削除
- L563 周辺 `not isinstance(deps["tokenizer"], _StubTokenizer)` → 削除 or `not isinstance(..., MagicMock)` に書換
- L604-636 `test_falls_back_to_stub_when_tokenizer_id_missing` → **削除**
- 新設: `test_raises_when_tokenizer_id_missing` (tokenizer_id 未設定 → `ValueError` 期待)
- 新設: `test_raises_when_tokenizer_load_fails` (S5-002 対応 / `AutoTokenizer.from_pretrained` を `OSError` に patch して `ValueError` 期待)

**`_build_photon_deps` 成功 path fixture の追加 migration (DR3-001)**:

現行 main のコメントは「`tokenizer:` section 以前の ~17 unit tests が fallback で無変更動作する」としており、実測でも direct な `_StubTokenizer` 参照 4 箇所以外に、`_build_photon_deps(cfg)` を呼ぶ成功 path test が多数存在する。fallback 削除後はこれらが `ValueError` で落ちるため、**tokenizer ブロック追加 + `AutoTokenizer.from_pretrained` mock** を migration plan に含める。

実装着手時に以下を再実行して対象を確定する:

```bash
perl -ne 'if(/^\s*def\s+(test_[^(]+)/){$d=$1;$l=$.;} if(/_build_photon_deps\(/){print "$.:$d (def line $l): $_"}' \
  baseline_reporag/tests/test_photon_pipeline.py
```

2026-04-27 時点で、`tokenizer:` ブロックを持たずに `_build_photon_deps` 成功を期待する主な test は以下。これらは minimal yaml に `tokenizer.tokenizer_id` / `tokenizer.vocab_size` を追加し、ネットワーク不要にするため `AutoTokenizer.from_pretrained` を `MagicMock` tokenizer で patch する:

- L472 `test_returns_required_keys`
- L502 `test_safe_recgen_disabled`
- L1927 `test_build_deps_wires_rope_scaling`
- L1959 `test_build_deps_defaults_when_rope_scaling_missing`
- L2010 / L2044 / L2078 / L2108 Safe RecGen 系 4 件
- L2873 / L2904 WorkingMemory 系 2 件
- L3171 / L3208 / L3257 / L3326 / L3396 aggregation 系 5 件
- L3477 `test_build_pipeline_canonical_and_reexport_match` (spy 経由で real `_build_photon_deps` を呼ぶ)
- L4307 / L4372 past-turn pinning working-memory 系 2 件

上記とは別に、L635 `test_falls_back_to_stub_when_tokenizer_id_missing` は成功 path ではなく `ValueError` 期待 test に置換する。これを **「単に削除するだけ」では本来満たすべき仕様 (raise) のテストが欠落する** ため、削除・成功 path fixture 更新・新設 raise tests をセットで Issue 受入条件に書く。

### 設計判断 #6: Task 3 は **Phase A の 2 件のみ**、残り 6 件は別 Issue (S1-006 / S1-007)

**対象**: `tokenizer.vocab_size` + `tokenizer.tokenizer_id`
**対象外 (Phase B / 別 Issue)**: `head_dim` / `max_position_embeddings` / `rope_theta` / `safe_recgen_enabled` / `provider` / `session_memory` / `answering` 系 / (#135 マージ後の) `model.checkpoint_path`

**理由**:
- 一度に 8 件すべてを必須化すると yaml 補完範囲が広がり、本 Issue の核 (`_StubTokenizer` 排除) と無関係な変更で diff が膨らむ。
- 残り 6 件は default の意図 / 用途 / 必須化要否が個別判定であり、設計判断 1 件 1 件に値する。Phase B として別 Issue 化することで 1 PR の review burden を抑える。

---

## 5. 実装計画 (Task 単位)

### Task 1: `_StubTokenizer` 排除 + 境界 test

#### 1-A: production 改修 (`baseline_reporag/photon_pipeline.py`)

> DR2-001 / DR2-002 / DR2-003 反映: 以下のサンプルは実 main コード (HEAD: 8e677ca) と整合済。実装着手時には行番号 drift がある可能性があるため、`grep -n "_StubTokenizer\|_get_stub_tokenizer\|_logger.warning" baseline_reporag/photon_pipeline.py` で再確認する。

```python
# 変更前 (current main, baseline_reporag/photon_pipeline.py 周辺):
# (a) tokenizer 取得 (L301-306):
tokenizer_section = cfg.get("tokenizer")
tokenizer_id: str | None = (
    getattr(tokenizer_section, "tokenizer_id", None)
    if tokenizer_section is not None
    else None
)

# (b) tokenizer 構築 (L335-343):
if tokenizer_id:
    tokenizer = _load_hf_tokenizer(tokenizer_id, photon_cfg.tokenizer.vocab_size)
else:
    _logger.warning(
        "config.tokenizer.tokenizer_id is unset; falling back to "
        "_StubTokenizer (test/dev only — production photon configs "
        "must set tokenizer_id, Issue #138)."
    )
    tokenizer = _get_stub_tokenizer(photon_cfg.tokenizer.vocab_size)

# (c) `_load_hf_tokenizer` (L469-510):
#     - ImportError (transformers 不在 / L488)
#     - vocab_size mismatch → ValueError (L494-498)
#     - AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False) (L493) は
#       OSError / huggingface_hub.errors.* を素のまま伝播

# (d) `class _StubTokenizer` (L451-) + `def _get_stub_tokenizer` (L465-466)
```

```python
# 変更後 (本 Issue 適用後):

# (a) tokenizer 取得は現行と同じ (None-safe アクセス維持):
tokenizer_section = cfg.get("tokenizer")
tokenizer_id: str | None = (
    getattr(tokenizer_section, "tokenizer_id", None)
    if tokenizer_section is not None
    else None
)

# (b) tokenizer 構築: fallback ブランチ削除 + raise:
if not tokenizer_id:
    raise ValueError(
        "cfg.tokenizer.tokenizer_id is required for provider=='photon'. "
        "Set the `tokenizer:` block with a valid tokenizer_id "
        "(e.g. 'mlx-community/Qwen2.5-Coder-14B-Instruct-4bit') in the yaml config."
    )
tokenizer_id = _validate_tokenizer_id(tokenizer_id)
tokenizer = _load_hf_tokenizer(tokenizer_id, photon_cfg.tokenizer.vocab_size)

# (c) `_load_hf_tokenizer` の AutoTokenizer.from_pretrained 呼び出しブロックのみ
#     try/except で囲い、OSError 等を ValueError に正規化:
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)
except (OSError, Exception) as exc:  # OSError + HfHubHTTPError 等を捕捉
    # ImportError / vocab mismatch ValueError は別ブロックで保持
    raise ValueError(
        f"failed to load tokenizer {_display_tokenizer_id(tokenizer_id)} "
        f"({type(exc).__name__})"
    ) from exc
# ImportError (transformers 不在) と vocab_size mismatch ValueError は現行通り維持

# (d) `class _StubTokenizer` および `def _get_stub_tokenizer` は **完全削除**
```

#### 1-B: 既存 test の migration (`baseline_reporag/tests/test_photon_pipeline.py`) (DR1-005 反映で網羅化)

`grep -n _StubTokenizer baseline_reporag/tests/test_photon_pipeline.py` で確認した **全 4 箇所** を migration 対象とする:

- L521 周辺: `from baseline_reporag.photon_pipeline import _StubTokenizer, _build_photon_deps` のうち `_StubTokenizer` を削除
- L563 周辺: `assert not isinstance(deps["tokenizer"], _StubTokenizer)` を削除 (or `assert not isinstance(..., MagicMock)` に書換)
- L604-636 `test_falls_back_to_stub_when_tokenizer_id_missing` → **関数まるごと削除** (この削除に L611 の import + L636 の `assert isinstance(...)` も内包される)
- 加えて `photon_mlx/tests/conftest.py:5` (DR1-004 反映): module docstring の "used by :class:`baseline_reporag.photon_pipeline._StubTokenizer`" の参照を更新 (例: "test-only stub tokenizer" 等の自己完結した記述に差し替え)。conftest.py 内の `_StubTokenizer` (L15) は別物なので削除しない。

> 注 (DR2-004 反映): 上記行番号は 2026-04-26 時点 main (HEAD: 8e677ca) のスナップショット。実装着手時には `grep -n _StubTokenizer baseline_reporag/tests/test_photon_pipeline.py` を再実行して drift を確認する。

**間接影響の migration (DR3-001)**:

direct `_StubTokenizer` 参照を消すだけでは不足。`_build_photon_deps` を成功 path として呼ぶ既存 test fixture の多くは `tokenizer:` ブロックを持たず、現行 fallback に依存している。実装では以下のどちらかを必ず行う:

1. helper (例: `_fake_tokenizer_block(vocab_size=1000, tokenizer_id="fake-org/fake-tokenizer")`) を用意し、成功 path の minimal yaml 全てに `tokenizer:` ブロックを追加する。
2. 各 success test で `transformers.AutoTokenizer.from_pretrained` を patch し、`vocab_size` / `pad_token_id` / `encode()` を持つ `MagicMock` tokenizer を返す。

対象候補は `perl -ne ... _build_photon_deps` の出力で確認する。2026-04-27 時点では L472 / L502 / L1927 / L1959 / L2010 / L2044 / L2078 / L2108 / L2873 / L2904 / L3171 / L3208 / L3257 / L3326 / L3396 / L3477 / L4307 / L4372 が該当する。
- 新設 (~L604, DR2-005 反映で既存スタイル = yaml 文字列 + tmp_path + load_config に揃える):
  ```python
  def test_raises_when_tokenizer_id_missing(self, tmp_path):
      """`tokenizer.tokenizer_id` 未設定時は `_build_photon_deps` が ValueError を raise する (Issue #139)。"""
      from baseline_reporag.config import load_config
      from baseline_reporag.photon_pipeline import _build_photon_deps

      cfg_file = tmp_path / "photon.yaml"
      # 既存 test_falls_back_to_stub_when_tokenizer_id_missing と同じ minimal yaml
      # (tokenizer: ブロックを持たない photon config)
      cfg_file.write_text(
          "model:\n"
          "  provider: photon\n"
          "  ...  # 既存 fixture と同形\n"
      )
      cfg = load_config(str(cfg_file))
      with pytest.raises(ValueError, match="tokenizer_id is required"):
          _build_photon_deps(cfg)

  def test_raises_when_tokenizer_load_fails(self, tmp_path, monkeypatch):
      """`AutoTokenizer.from_pretrained` 失敗時は ValueError に正規化される (Issue #139 / S5-002)。"""
      from baseline_reporag.config import load_config
      from baseline_reporag.photon_pipeline import _build_photon_deps

      cfg_file = tmp_path / "photon.yaml"
      cfg_file.write_text(
          "model:\n"
          "  provider: photon\n"
          "  ...\n"
          "tokenizer:\n"
          '  tokenizer_id: "non/existent-tokenizer"\n'
          "  vocab_size: 152064\n"
      )
      cfg = load_config(str(cfg_file))

      def _boom(*args, **kwargs):
          raise OSError("HF Hub unreachable")
      monkeypatch.setattr(
          "transformers.AutoTokenizer.from_pretrained", _boom
      )
      with pytest.raises(ValueError, match="failed to load tokenizer 'non/existent-tokenizer'"):
          _build_photon_deps(cfg)
  ```

- 追加 security tests (DR4-001): `tokenizer_id` が URL / local path / `..` / backslash / control character を含む場合は `ValueError`。特に newline を含む入力で公開 error message に raw newline が混入しないことを確認する。
- success path tests は `AutoTokenizer.from_pretrained(..., trust_remote_code=False)` を明示的に assert する (DR4-002)。

#### 1-C: 境界 test 新規 (`tests/test_no_scaffolding_in_prod.py`)

Issue 本文記載のサンプル通り。要点:
- `REPO_ROOT = Path(__file__).resolve().parents[1]` で cwd 非依存
- `PROD_ROOTS = [REPO_ROOT / 'baseline_reporag', ..., REPO_ROOT / 'torch_ref']`
- root 不在は **assert failure** (`assert root.is_dir()`)
- `EXCLUDED_DIR_PARTS = {'tests', '__pycache__'}`
- regex: `re.compile(r'\b_(?:Stub|Mock|Dummy|Placeholder)\w*')`
- symlink / 巨大 file 防御 (DR4-003): `Path.rglob('*.py')` の対象が symlink の場合は follow して読まず violation とする。`f.resolve().is_relative_to(root.resolve())` 相当で root 外へ抜けないことも確認する。想定外に巨大な `.py` (例: 2MB 超) は violation とし、CI runner で無制限に `read_text()` しない。
- `read_text(encoding='utf-8')` は strict のままにし、`UnicodeDecodeError` は skip せず violation として報告する。production module の `.py` は UTF-8 decode 可能であるべきなので、decode 不能 file を silently ignore しない。

> 注 (DR1-007 反映): PROD_ROOTS 配下に `.venv` / `.tox` / `build/` 等の生成物 dir はそもそも存在しないため除外不要。将来 production tree に generated dir が追加された場合のみ EXCLUDED_DIR_PARTS を拡張する。

#### 1-D: `photon_mlx/tests/conftest.py` の test-only stub は維持 (DR3-002)

`photon_mlx/tests/conftest.py` 内の `_StubTokenizer` は production module から import されるものではなく、`photon_mlx` 単体 test 用 fixture。`tests/test_no_scaffolding_in_prod.py` は `tests/` dir を除外するため検出対象外でよい。

影響確認:
- `photon_mlx/tests/test_inference.py` は `stub_tokenizer_for_cfg` fixture を使って `PhotonInference(model, cfg, tokenizer)` を直接組み立てる (L396 / L399 / L408 / L453 / L465 / L493 周辺)。
- `photon_mlx/tests/test_generate.py` / `test_inference.py` / `test_session.py` には各 test 内の小さな byte tokenizer も存在する。

したがって本 Issue で行うのは **docstring 更新のみ**。`photon_mlx/tests/conftest.py` の `_StubTokenizer` class / `stub_tokenizer_for_cfg` fixture は削除・rename しない。

### Task 3: yaml invariant 拡張

#### 3-A: invariant test 拡張 (`tests/test_pipeline_factory_yaml_invariants.py`)

Issue 本文記載のサンプル通り。要点 (DR1-001 反映で既存 helper 再利用に統一):

- 既存 file の `CONFIGS_DIR` 定数 + `from baseline_reporag.config import load_config` を **再利用** する (`tests/test_pipeline_factory_yaml_invariants.py:28,30,51,67,77` と同一様式)
- ad-hoc な `_load_yaml` / `_get_nested` ヘルパは **追加しない** (DRY)
- `_is_photon_profile_yaml(path: Path, cfg) -> bool`:
  - filename 判定 (`path.name.startswith('photon_')` or `path.name == 'institutional_docs_photon.yaml'`) を **main signal**
  - `getattr(cfg.model, 'provider', None) == 'photon'` を **insurance** (将来命名規約が崩れたとき / DR1-006 反映)
  - 上記いずれか満たせば PHOTON profile と判定
- `required_keys = ['tokenizer.vocab_size', 'tokenizer.tokenizer_id']` を attribute access で確認:
  - `getattr(cfg.tokenizer, 'tokenizer_id', None)` 等
- 値が `None` / 空文字なら failure 配列に追加 (yaml で `tokenizer_id: ""` を書いた場合も検出)
- 既存 file 末尾に新規 test 関数として追加し、既存 `test_baseline_yaml_reranker_model_id_unchanged` 等と同形式で並べる
- yaml load は必ず `baseline_reporag.config.load_config` を使う。同 helper は `yaml.safe_load` を使うため、invariant test で ad-hoc に `yaml.load` / arbitrary object deserialize を追加しない (DR4-004)。
- この invariant は CI gate なので、`@pytest.mark.skip` / `skipif` / `xfail` を付けない。既存 file の Issue #133 向け `skipif` は別 invariant の一時措置であり、本 test へ横展開しない (DR4-004)。

**サンプル (修正後)**:

```python
# tests/test_pipeline_factory_yaml_invariants.py に追加 (既存 helper 再利用)
import pytest
from pathlib import Path
from baseline_reporag.config import load_config

# CONFIGS_DIR は既存定義 (line 30) を再利用


def _is_photon_profile_yaml(path: Path, cfg) -> bool:
    if path.name.startswith('photon_') or path.name == 'institutional_docs_photon.yaml':
        return True
    return getattr(getattr(cfg, 'model', None), 'provider', None) == 'photon'


def test_photon_yaml_has_required_tokenizer_fields():
    """PHOTON profile yaml は tokenizer.vocab_size / tokenizer.tokenizer_id を必須とする。"""
    failures: list[tuple[str, str]] = []
    for yaml_path in sorted(CONFIGS_DIR.glob('*.yaml')):
        cfg = load_config(str(yaml_path))
        if not _is_photon_profile_yaml(yaml_path, cfg):
            continue
        tok = getattr(cfg, 'tokenizer', None)
        for key in ('vocab_size', 'tokenizer_id'):
            value = getattr(tok, key, None) if tok is not None else None
            if value in (None, ''):
                failures.append((str(yaml_path), f'tokenizer.{key}'))
    assert not failures, f"photon yaml missing required tokenizer fields: {failures}"
```

#### 3-B: configs 補完 (必要に応じて)

- 現状 `configs/photon_*.yaml` / `configs/institutional_docs_photon.yaml` の `tokenizer.vocab_size` / `tokenizer.tokenizer_id` 設定有無を invariant test 実装後に確認
- 不足があれば yaml に追記 (本 Issue scope に含む)

### docs/troubleshooting.md 追記 (S3-006 / S7-002 / DR1-003 反映)

**追記アンカー** (DR1-003 反映): 実装時に `grep -n 'cfg.model.provider' docs/troubleshooting.md` で該当 section を特定する。現状は **「Streamlit アプリ: drift_metrics が `N/A` のまま表示される (Issue #82)」** section の `cfg.model.provider == "photon"` checklist (item 1 〜 4 を持つ) が直近のアンカー候補。L149 という具体行番号は実装時の現状で再確認 (旧 design draft の暫定値)。

**追記内容**:

- 既存 checklist に以下を追加:
  - `tokenizer.tokenizer_id` 未設定 → `ValueError` で起動失敗 (`_build_photon_deps` 境界 / Issue #139)
  - tokenizer load 失敗系 (HF Hub 障害 / gated model / 未 cache / 未 login / tokenizer_id 誤設定) → `ValueError("failed to load tokenizer '...'")` で起動失敗
  - 確認項目: 事前 `huggingface-cli login` / `hf cache scan` / network 疎通 / 対象 tokenizer_id の存在確認 / yaml の `tokenizer.tokenizer_id` が正しいか
  - security note (DR4-005): HF token / PAT / secret env var を yaml、Issue、Slack、log に平文で貼らない。認証は `huggingface-cli login` または runner secret (`HF_TOKEN` 等) で行い、troubleshooting には token 値そのものを書かない。
  - private model id を含む error は公開 log に転記する前に必要に応じて redaction する。実装側の `ValueError` message も sanitized `tokenizer_id` + exception class を基本とし、raw exception text を公開 message に丸ごと埋め込まない。

---

## 6. テスト戦略

| カテゴリ | 内容 | 期待動作 |
|---------|------|---------|
| 単体 (raise) | `test_raises_when_tokenizer_id_missing` | 空文字 / 未設定で `ValueError` |
| 単体 (raise) | `test_raises_when_tokenizer_load_fails` | `AutoTokenizer.from_pretrained` を `OSError` に patch して `ValueError` 連鎖 |
| 単体 (success) | 既存 success path test (`tokenizer_id` 設定済) | mock を MagicMock 等に置換し継続 pass |
| 境界 (CI gate) | `test_no_scaffolding_naming_in_production` | violations == [] |
| 境界 (CI gate) | `test_photon_yaml_has_required_tokenizer_fields` | failures == [] (PHOTON profile yaml で全件) |
| 既存 regression | `python -m pytest` 全 collect | 既存テスト全パス (CLAUDE.md の pre-existing failure 2 件を除く) |
| lint | `ruff check .` / `ruff format --check .` | 警告 0 件 / 差分なし |

---

## 7. リスクと緩和策

| リスク | 影響度 | 緩和策 |
|--------|--------|--------|
| 古い PHOTON yaml で `tokenizer.tokenizer_id` 欠落 → 起動失敗 | 中 | 本 Issue で `configs/photon_*.yaml` 全件を確認し、欠落あれば補完。invariant test で恒久検出 |
| untrusted yaml の `tokenizer_id` が URL / local path / control character / secret-like 文字列を含む | **高** | `_validate_tokenizer_id` で Hugging Face repo id allowlist に限定し、公開 message は sanitized display のみ使う (DR4-001 / DR4-005) |
| HF Hub 障害時 PHOTON pipeline 全停止 | 中 | `docs/troubleshooting.md` に checklist 追記。server/CLI 起動失敗の例外 message は sanitized `tokenizer_id` と exception class に限定 |
| `huggingface_hub.errors.*` の version 差分で exception handling が drift | 中 | 具体 class import に依存せず、`AutoTokenizer.from_pretrained` の局所 try/except で `ValueError` に正規化。`trust_remote_code=False` を維持 (DR4-002) |
| `_StubTokenizer` を削除すると `test_photon_pipeline.py` の他 test も間接破綻 | 中 | grep `_StubTokenizer` で全 4 件 (L521 / L563 / L611 / L636) 把握済。受入条件で migration を明示 |
| `_build_photon_deps` 成功 path の既存 17-18 test が tokenizer-less fixture のまま `ValueError` で失敗 | **高** | DR3-001 の対象リストを migration plan に含める。成功 path yaml に `tokenizer:` ブロックを追加し、`AutoTokenizer.from_pretrained` を mock して hermetic に保つ |
| 境界 test が将来の偶発命名 (e.g. `_PlaceholderEmbedding`) で誤検出 | 低 | 必要時に `# noqa` ベースで個別豁免。Issue scope では発生しない |
| 境界 test が symlink escape / 巨大 file / decode 不能 file を無制限に読む | 中 | symlink は follow せず violation、root 外 resolve を拒否、size cap、UTF-8 decode error は violation として fail (DR4-003) |
| 新規 invariant test が skip/xfail され CI gate を迂回 | 中 | `test_photon_yaml_has_required_tokenizer_fields` に skip/skipif/xfail を付けない。`load_config` (`yaml.safe_load`) 以外の yaml loader を使わない (DR4-004) |
| #135 ブランチとの merge conflict (確認済) | 中 | PR 戦略を Issue に明記: 本 Issue → main → #135 rebase。手動統合は #135 側で実施 |
| Phase B に回した `getattr default` が実装中に増え、audit 対象が stale になる | 低 | 実装完了前に `rg -n "getattr\\(cfg|cfg\\.get" baseline_reporag photon_mlx -g '*.py' -g '!*/tests/*'` を再実行し、Issue 本文の 8 件から増えたものは Phase B follow-up に追記する (DR3-004) |

---

## 8. リリース順序とマージ戦略

```
[現在] main (8e677ca) ── #138 マージ済
              │
              ├── feature/issue-139-stub-audit  (本 Issue)
              │     ──> PR #N → main (1st: 軽量、即着手可)
              │
              └── feature/issue-135-photon-retrain  (S7-001 fix + 本格再学習)
                    └ 本 Issue マージ後に rebase が必要
                       ──> PR #M → main (2nd: 大物、本格再学習を含む)
                       └ rebase で `_build_photon_deps` の手動統合
```

**併走中の #135 への影響**:
- ブランチ責任者は本 Issue マージ後の `_build_photon_deps` (= raise 化された tokenizer 強制 path) の上に `photon_mlx.checkpoint.load_checkpoint(...)` を再適用
- `_StubTokenizer` への参照は #135 側でも消す必要あり (`baseline_reporag/tests/test_photon_pipeline.py` の test 関連)
- DR3-003 確認: `git diff HEAD..feature/issue-135-photon-retrain -- baseline_reporag/photon_pipeline.py baseline_reporag/tests/test_photon_pipeline.py photon_mlx/tests/conftest.py` では、#135 側が #138 の real tokenizer path (`tokenizer_section` / `_load_hf_tokenizer` / `tokenizer.vocab_size`) を外し、`_get_stub_tokenizer(photon_cfg.tokenizer.vocab_size)` + `model.checkpoint_path` load へ差し替えている。tests 側も real tokenizer / vocab mismatch tests を checkpoint load tests に置換し、`tokenizer:` ブロック無しの fixture を増やしている。
- rebase 方針: #139 側の **real tokenizer / `tokenizer.vocab_size` canonical / fallback 削除** を優先し、その上で #135 側の `photon_mlx.checkpoint.load_checkpoint(model, ckpt_dir)` を `model = PhotonModel(photon_cfg)` 直後に再適用する。#135 側の checkpoint smoke tests も `tokenizer:` ブロック + tokenizer mock を追加してから取り込む。

---

## 9. 影響範囲サマリ

| モジュール | 変更種別 | 影響度 |
|-----------|---------|--------|
| `baseline_reporag/photon_pipeline.py` | 削除 + 改修 | **高** (`_StubTokenizer` / `_get_stub_tokenizer` 削除、`_build_photon_deps` 動作変更) |
| `baseline_reporag/tests/test_photon_pipeline.py` | 削除 + 改修 + 追加 | **高** (direct 4 箇所更新 + tokenizer-less `_build_photon_deps` success path 17-18 件に `tokenizer:` block / tokenizer mock 追加 + 新規 raise test 2 件追加) |
| `photon_mlx/tests/conftest.py` | docstring 更新 | 低 (L5 の production `_StubTokenizer` 参照を test-only 記述に差替。fixture 本体は `photon_mlx/tests/test_inference.py` 等が使うため維持) |
| `tests/test_no_scaffolding_in_prod.py` | 新規 | 中 (CI gate 新設) |
| `tests/test_pipeline_factory_yaml_invariants.py` | 拡張 | 中 (CI gate 強化) |
| `docs/troubleshooting.md` | 追記 | 低 (運用 docs) |
| `configs/photon_*.yaml`, `configs/institutional_docs_photon.yaml` | 補完あり得る | 低 (現状確認後に判断) |
| `bench/` `scripts/` `demo/` | 対象外 | - (Issue で明記) |
| `.github/workflows/weekly_eval.yml` | 影響なし | - (provider=mlx_lm) |

**影響確認メモ (DR3-004 / Stage 3 実測)**:
- `.github/workflows/` には `_StubTokenizer` / `_get_stub_tokenizer` assertion は存在しない。
- `configs/photon_*.yaml` と `configs/institutional_docs_photon.yaml` は 2026-04-27 時点で全て `tokenizer.tokenizer_id` / `tokenizer.vocab_size` を持つため、invariant test 追加だけで CI を破壊する既知欠落はない。
- `bench/issue61_prune_batch.py` は独自 `_StubTokenizer` を持ち、docstring で `baseline_reporag.photon_pipeline._StubTokenizer` を参照するが import はしていない。`bench/` は Issue 対象外だが、stale reference が気になる場合は別 Issue / Nice-to-Have docs cleanup とする。

---

## 10. 品質基準

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| テスト | `python -m pytest` | 全 collected test pass (CLAUDE.md 既知の pre-existing failure 2 件は除外可) |
| 新規 test (raise) | `pytest baseline_reporag/tests/test_photon_pipeline.py -k 'raises_when_tokenizer'` | 2 件 pass |
| 境界 test | `pytest tests/test_no_scaffolding_in_prod.py` | pass |
| invariant test | `pytest tests/test_pipeline_factory_yaml_invariants.py` | pass |
| security invariant (tokenizer) | `pytest baseline_reporag/tests/test_photon_pipeline.py -k 'tokenizer_id or tokenizer_load'` | unsafe tokenizer_id が `ValueError`、`trust_remote_code=False` assert |
| CI gate bypass check | `rg -n "@pytest.mark.(skip|skipif|xfail).*test_photon_yaml_has_required_tokenizer_fields" tests/test_pipeline_factory_yaml_invariants.py` | match 0 件 |
| Lint | `ruff check .` | 警告 0 件 |
| Format | `ruff format --check .` | 差分なし |
| baseline 疎通 | `python -m baseline_reporag.cli --config configs/baseline.yaml --repo-id fastapi_fastapi --question "test"` | 応答あり (provider=mlx_lm のため非影響) |

---

## 11. 完了条件

- [ ] `_StubTokenizer` / `_get_stub_tokenizer` が `baseline_reporag/photon_pipeline.py` から削除されている
- [ ] `_build_photon_deps` が tokenizer_id 未設定 / load 失敗時に `ValueError` を raise する
- [ ] `tokenizer_id` validation が追加され、URL / local path / traversal / control character を拒否し、公開 error message に raw untrusted string を埋め込まない
- [ ] `_load_hf_tokenizer` の `AutoTokenizer.from_pretrained` 呼び出しで `trust_remote_code=False` を維持し、test で assert されている
- [ ] `tests/test_no_scaffolding_in_prod.py` が新規追加され pass する (cwd 非依存 / repo root 基準 / symlink 非 follow / size cap / UTF-8 decode error fail)
- [ ] `tests/test_pipeline_factory_yaml_invariants.py` に PHOTON profile yaml 用必須フィールドチェックが追加され pass する (`load_config` / `yaml.safe_load` 経由、skip/skipif/xfail なし)
- [ ] `baseline_reporag/tests/test_photon_pipeline.py` の migration 完了 (削除 3 箇所、新設 2 test)
- [ ] `_build_photon_deps` 成功 path 既存 tests の tokenizer-less fixture を全て migration 済み (`tokenizer:` ブロック追加 + `AutoTokenizer.from_pretrained` mock)
- [ ] `docs/troubleshooting.md` に tokenizer 起動失敗 checklist と secret/token を平文転記しない注意を追記済
- [ ] `python -m pytest` / `ruff check` / `ruff format --check` すべて pass
- [ ] PR description で本 Issue 受入条件にすべてチェックが付く

---

## 12. 関連資料

- Issue 本文 (最終形): `workspace/issues/139/issue-review/updated-issue-body.md`
- 仮説検証: `workspace/issues/139/issue-review/hypothesis-verification.md`
- レビュー履歴: `workspace/issues/139/issue-review/summary-report.md`
- 切り出し先: [#145](https://github.com/Kewton/photon-mlx/issues/145) (real-weight integration test、#135 マージ後着手)
- 並列ブランチ: `feature/issue-135-photon-retrain` (本格再学習、merge は本 Issue 後)

> 整合性チェック (DR2-007 反映): 実装着手前に `diff <(gh issue view 139 --json body --jq .body) workspace/issues/139/issue-review/updated-issue-body.md` を実行し、差分が末尾改行のみであることを確認する。
