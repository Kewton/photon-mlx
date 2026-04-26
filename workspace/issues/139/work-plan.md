# Issue #139 作業計画書

## Issue: test(photon): Stub/Mock pattern audit + invariant test (S7-001 follow-up)

**Issue番号**: [#139](https://github.com/Kewton/photon-mlx/issues/139)
**サイズ**: M (test migration が広いが production 改修は小さい)
**優先度**: High (構造的 silent bug 防止 + #135 マージ前提)
**依存Issue**: なし (#138 マージ済 / #135 は本 Issue 後に rebase)
**切り出し先**: [#145](https://github.com/Kewton/photon-mlx/issues/145) (Task 2 = real-weight integration test、本 Issue マージ後 / #135 マージ後)

**前提資料**:
- Issue 本文 (最新): `workspace/issues/139/issue-review/updated-issue-body.md` (25 finding 反映済)
- 設計方針書 (最新): `workspace/issues/139/design-policy.md` (23 finding 反映済)
- Issue review summary: `workspace/issues/139/issue-review/summary-report.md`
- 設計 review summary: `workspace/issues/139/multi-stage-design-review/summary-report.md`

**累積 review 件数 (Phase 1 + 3)**: Must Fix 9 / Should Fix 30 / Nice to Have 9 = **48 finding** すべて反映済。

---

## 着手前チェック

実装着手時に **必ず実行** (DR2-004 / DR2-007 / DR3-004 反映):

```bash
# 1. Issue 本文と updated-issue-body.md の整合 (DR2-007)
diff <(gh issue view 139 --json body --jq .body) workspace/issues/139/issue-review/updated-issue-body.md
# 差分は末尾改行のみであるべき

# 2. 行番号 drift の確認 (DR2-004)
grep -n "_StubTokenizer\|_get_stub_tokenizer\|_logger.warning\|_load_hf_tokenizer" baseline_reporag/photon_pipeline.py
grep -n "_StubTokenizer" baseline_reporag/tests/test_photon_pipeline.py

# 3. Phase B 対象 getattr default の最新確認 (DR3-004)
rg -n "getattr\(cfg|cfg\.get" baseline_reporag photon_mlx -g '*.py' -g '!*/tests/*'

# 4. PHOTON yaml の必須フィールド存在確認 (DR3 確認済だが念押し)
for f in configs/photon_*.yaml configs/institutional_docs_photon.yaml; do
  echo "=== $f ==="; grep -n "tokenizer_id\|vocab_size" "$f" || echo "(none)"
done

# 5. #135 ブランチとの想定 conflict 範囲確認
git diff main..feature/issue-135-photon-retrain -- baseline_reporag/photon_pipeline.py baseline_reporag/tests/test_photon_pipeline.py photon_mlx/tests/conftest.py | wc -l
```

---

## 詳細タスク分解

### Phase 1 — Production 改修

#### Task 1.1: tokenizer_id validation/sanitization helper を追加 (DR4-001 反映)

**成果物**: `baseline_reporag/photon_pipeline.py` に新規 private helper 2 件

```python
# allowlist regex: HF repo id (org/name) のみ許可。URL / path / control char / backslash / 空白 を拒否
_TOKENIZER_ID_PATTERN = re.compile(r'^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$')

def _validate_tokenizer_id(tokenizer_id: str) -> str:
    """untrusted yaml input から取得した tokenizer_id を allowlist で validate する。"""
    if not isinstance(tokenizer_id, str) or not tokenizer_id:
        raise ValueError("cfg.tokenizer.tokenizer_id must be a non-empty string")
    if not _TOKENIZER_ID_PATTERN.fullmatch(tokenizer_id):
        raise ValueError(
            f"cfg.tokenizer.tokenizer_id has unsafe form (expected '<org>/<name>')"
        )
    return tokenizer_id

def _display_tokenizer_id(tokenizer_id: str) -> str:
    """log / ValueError message に出すときの sanitized 表示 (制御文字除去)。"""
    return repr(tokenizer_id)  # repr で control char を escape 表示
```

**依存**: なし
**TDD**: Task 2.6 (validation の unit test) を先に書く

#### Task 1.2: `_load_hf_tokenizer` の例外正規化 (DR1-002 / DR4-002 反映)

**成果物**: `baseline_reporag/photon_pipeline.py:469-510` 周辺修正

- `AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)` 呼び出しブロックを `try/except OSError, Exception` で囲む
- 例外を `ValueError(f"failed to load tokenizer {_display_tokenizer_id(tokenizer_id)}: <sanitized class name>") from exc` に正規化
- 既存の `ImportError` (transformers 不在) と `ValueError` (vocab mismatch) は変更しない

**依存**: Task 1.1
**TDD**: Task 2.5 (load failure → ValueError) を先に書く

#### Task 1.3: `_build_photon_deps` の fallback 削除 + raise 化 (S1-005 / DR2-002 反映)

**成果物**: `baseline_reporag/photon_pipeline.py:301-306, 335-343` 周辺修正

- `tokenizer_section` / `tokenizer_id` 取得は現行維持 (None-safe)
- `if not tokenizer_id: raise ValueError(...)` (`_build_photon_deps` 境界)
- raise 後に `_validate_tokenizer_id(tokenizer_id)` を呼ぶ
- `_load_hf_tokenizer(...)` 呼び出しは現行維持 (内部で Task 1.2 の正規化が効く)
- `else: _logger.warning(...) + _get_stub_tokenizer(...)` 分岐を **完全削除**

**依存**: Task 1.1, 1.2
**TDD**: Task 2.4 (missing tokenizer_id → ValueError) を先に書く

#### Task 1.4: `_StubTokenizer` クラスと `_get_stub_tokenizer` 関数の削除

**成果物**: `baseline_reporag/photon_pipeline.py:451-466` を **削除**

**依存**: Task 1.3 完了 (production path から呼び出されなくなったことが前提)
**注意**: Task 2 の test migration が完了するまで一時的に **失敗** する。Phase 2 で速やかに対応すること。

---

### Phase 2 — テスト改修

#### Task 2.1: `_StubTokenizer` direct 参照 4 箇所の migration (S3-004 / DR1-005 反映)

**成果物**: `baseline_reporag/tests/test_photon_pipeline.py` の以下 4 箇所:

- L521 周辺: `from baseline_reporag.photon_pipeline import _StubTokenizer, _build_photon_deps` の `_StubTokenizer` を削除
- L563 周辺: `assert not isinstance(deps["tokenizer"], _StubTokenizer)` を削除 or `not isinstance(..., MagicMock)` に書換
- L604-636 `test_falls_back_to_stub_when_tokenizer_id_missing` を **関数ごと削除** (L611 import + L636 assertion を内包)

**依存**: Task 1.4 完了

#### Task 2.2: `test_raises_when_tokenizer_id_missing` を新設 (S3-004 / DR2-005 反映)

**成果物**: `baseline_reporag/tests/test_photon_pipeline.py` の旧 `test_falls_back_to_stub_when_tokenizer_id_missing` 跡地に追加

- 既存スタイル (yaml 文字列 + `tmp_path` + `load_config(str(cfg_file))`) で書く
- `tokenizer:` ブロックを持たない photon config で `_build_photon_deps(cfg)` 呼び出し
- `pytest.raises(ValueError, match="tokenizer_id is required")`

**依存**: Task 1.3, 2.1

#### Task 2.3: `test_raises_when_tokenizer_load_fails` を新設 (S5-002 / DR1-002 反映)

**成果物**: `baseline_reporag/tests/test_photon_pipeline.py` に追加

- `tokenizer.tokenizer_id: "non/existent-tokenizer"` を yaml で設定
- `monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: (_ for _ in ()).throw(OSError("HF Hub unreachable")))`
- `pytest.raises(ValueError, match="failed to load tokenizer 'non/existent-tokenizer'")`

**依存**: Task 1.2

#### Task 2.4: `test_rejects_unsafe_tokenizer_id` を新設 (DR4-001 反映)

**成果物**: `baseline_reporag/tests/test_photon_pipeline.py` に追加

各 unsafe 値で `pytest.raises(ValueError)` をパラメトライズ:
- `"http://evil.example/etc/passwd"` (URL)
- `"../../etc/passwd"` (path traversal)
- `"a\\b"` (backslash)
- `"a\nb"` (newline / log injection)
- `"a"` (no slash)
- `""` (empty)

**依存**: Task 1.1

#### Task 2.5: 17-18 件 success-path test の tokenizer-less fixture migration (DR3-001 反映 / **重要**)

**成果物**: `baseline_reporag/tests/test_photon_pipeline.py` の以下 18 箇所周辺の test 関数:

L472, L502, L1927, L1959, L2010, L2044, L2078, L2108, L2873, L2904, L3171, L3208, L3257, L3326, L3396, L3477, L4307, L4372

各々について:
1. yaml fixture に `tokenizer:` ブロックを追加 (`tokenizer_id: "fake-org/fake-tokenizer"`, `vocab_size: <既存値>`)
2. `_build_photon_deps` 呼び出しを `with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer)` で囲む (既存 `test_loads_real_tokenizer_when_tokenizer_id_set` のスタイル参照)
3. `fake_tokenizer.vocab_size = <既存 vocab_size>`, `fake_tokenizer.pad_token_id = 0` を設定

**依存**: Task 1.3 完了

**手順** (実装着手時):
```bash
# 対象行を再確認
grep -n "_build_photon_deps\|_get_stub_tokenizer" baseline_reporag/tests/test_photon_pipeline.py
# 各行を一つずつ migration、既存 success-path style に合わせる
```

#### Task 2.6: `photon_mlx/tests/conftest.py` の docstring 更新 (DR1-004 反映)

**成果物**: `photon_mlx/tests/conftest.py:5` の docstring 修正

- 旧: `"used by :class:\`baseline_reporag.photon_pipeline._StubTokenizer\` so..."`
- 新: 自己完結した記述 (e.g. `"test-only stub tokenizer for PHOTON inference unit tests."`)
- conftest.py 内の `_StubTokenizer` (L15) は別物なので **削除しない**

**依存**: なし (production 削除と独立)

---

### Phase 3 — 新規 test 追加

#### Task 3.1: `tests/test_no_scaffolding_in_prod.py` 新規追加 (S3-003 / DR1-007 / DR4-003 反映)

**成果物**: `tests/test_no_scaffolding_in_prod.py` (新規ファイル)

要件:
- `REPO_ROOT = Path(__file__).resolve().parents[1]` で cwd 非依存
- `PROD_ROOTS = [REPO_ROOT / 'baseline_reporag', REPO_ROOT / 'photon_mlx', REPO_ROOT / 'torch_ref']`
- root 不在は `assert root.is_dir()` で **failure** (S7-001 偽 pass 防止)
- `EXCLUDED_DIR_PARTS = {'tests', '__pycache__'}` (tuple membership 完全一致)
- regex: `re.compile(r'\b_(?:Stub|Mock|Dummy|Placeholder)\w*')`
- **Hardening (DR4-003)**:
  - `Path.rglob` 結果のうち `f.is_symlink()` は violation として扱う (root 外 leak 防止)
  - `f.resolve()` で root 配下に確実に居ることを確認 (`f.resolve().is_relative_to(REPO_ROOT)`)
  - file size cap: 1MB を超える .py は violation (異常 file 検出)
  - `read_text(encoding='utf-8')` で `UnicodeDecodeError` も violation
- `assert not violations`

**依存**: Task 1.4 完了 (production から `_StubTokenizer` が削除されていること)

#### Task 3.2: `tests/test_pipeline_factory_yaml_invariants.py` 拡張 (S5-001 / DR1-001 / DR4-004 反映)

**成果物**: 既存 `tests/test_pipeline_factory_yaml_invariants.py` 末尾に追加

要件:
- 既存 `from baseline_reporag.config import load_config` と `CONFIGS_DIR` を **再利用**
- 新ヘルパ `_is_photon_profile_yaml(path: Path, cfg) -> bool`:
  - filename 判定 main: `path.name.startswith('photon_') or path.name == 'institutional_docs_photon.yaml'`
  - provider 判定 insurance: `getattr(getattr(cfg, 'model', None), 'provider', None) == 'photon'`
- 新 test 関数 `test_photon_yaml_has_required_tokenizer_fields()`:
  - `for yaml_path in sorted(CONFIGS_DIR.glob('*.yaml')):`
  - `if not _is_photon_profile_yaml(...): continue`
  - `tok = getattr(cfg, 'tokenizer', None)` から `tokenizer_id` / `vocab_size` を attribute access
  - `None` / 空文字なら failures 配列に追加
  - `assert not failures`
- **`@pytest.mark.skip` / `@pytest.mark.skipif` / `@pytest.mark.xfail` を付けない** (DR4-004)

**依存**: なし (yaml 改修と独立)

#### Task 3.3: `configs/photon_*.yaml` / `configs/institutional_docs_photon.yaml` の必須フィールド補完 (必要時)

**成果物**: `configs/*.yaml` (PHOTON profile) のうち `tokenizer.vocab_size` / `tokenizer.tokenizer_id` が未設定のものを補完

**手順**:
```bash
# 対象 yaml の現状確認 (実装着手時)
for f in configs/photon_*.yaml configs/institutional_docs_photon.yaml; do
  echo "=== $f ==="
  grep -E "^tokenizer:|^  tokenizer_id:|^  vocab_size:" "$f" || echo "(missing)"
done
# 欠落があれば test_photon_yaml_has_required_tokenizer_fields が pass するように補完
```

**依存**: Task 3.2 完了 (test を先に走らせて欠落を特定)

---

### Phase 4 — ドキュメント

#### Task 4.1: `docs/troubleshooting.md` 追記 (S3-006 / S7-002 / DR1-003 / DR4-005 反映)

**成果物**: `docs/troubleshooting.md` の `cfg.model.provider == "photon"` checklist (Issue #82 drift_metrics N/A section 内) に追記

**実装手順**:
```bash
grep -n "cfg.model.provider" docs/troubleshooting.md
```
で該当 section を特定し、checklist (item 1〜4) の末尾に追記:

- `tokenizer.tokenizer_id` 未設定 → `_build_photon_deps` 境界で `ValueError` (Issue #139)
- tokenizer load 失敗系 (HF Hub 障害 / gated model / 未 cache / 未 login / tokenizer_id 誤設定) → `ValueError("failed to load tokenizer '...'")`
- 確認項目: `huggingface-cli login` 状態 / `hf cache scan` / network 疎通 / yaml の `tokenizer.tokenizer_id` 値

**注意 (DR4-005 反映 / 機密情報の扱い)**:
- HF token / PAT / secret env var を **平文で yaml / Issue / Slack / log に貼らない**
- 認証は `huggingface-cli login` または CI runner secret で実施
- private model id は公開 log 転記前に redaction (e.g. mask 後半部)
- raw exception text を直接 paste せず、sanitized message に絞る

**依存**: Task 1.3 完了 (新 ValueError message が確定していること)

---

### Phase 5 — 品質チェック

#### Task 5.1: 個別 test 実行

```bash
# 新規 raise / validation tests
python -m pytest baseline_reporag/tests/test_photon_pipeline.py::TestBuildPhotonDepsRealTokenizer -v
python -m pytest baseline_reporag/tests/test_photon_pipeline.py -k "raises_when_tokenizer or rejects_unsafe" -v

# 境界 test
python -m pytest tests/test_no_scaffolding_in_prod.py -v

# invariant test
python -m pytest tests/test_pipeline_factory_yaml_invariants.py -v
```

#### Task 5.2: 全体 regression test

```bash
python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v
# CLAUDE.md 既知の pre-existing failure 2 件 (test_generate_training_corpus.py) は除外可
```

#### Task 5.3: lint / format

```bash
ruff check . --fix
ruff format .
ruff check .
ruff format --check .
```

#### Task 5.4: baseline 疎通 (provider=mlx_lm の影響なし確認)

```bash
python -m baseline_reporag.cli --config configs/baseline.yaml --repo-id fastapi_fastapi --question "test"
```

---

## 品質チェック項目 (Definition of Done)

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| 個別 test (raise) | `pytest -k "raises_when_tokenizer or rejects_unsafe"` | 新規 4 test 全 pass |
| 境界 test | `pytest tests/test_no_scaffolding_in_prod.py` | violations == [] |
| invariant test | `pytest tests/test_pipeline_factory_yaml_invariants.py` | failures == [] |
| 全 regression | `python -m pytest` | 既知 2 件 pre-existing failure 以外全 pass |
| Lint | `ruff check .` | 警告 0 件 |
| Format | `ruff format --check .` | 差分なし |
| baseline 疎通 | `python -m baseline_reporag.cli --config configs/baseline.yaml ...` | 応答あり |
| Issue 整合 | `diff <(gh issue view 139 --json body --jq .body) workspace/issues/139/issue-review/updated-issue-body.md` | 末尾改行のみ |

---

## Definition of Done

- [ ] Task 1.1 - 1.4 (Production 改修) 完了
- [ ] Task 2.1 - 2.6 (test migration) 完了
- [ ] Task 3.1 - 3.3 (新規 test + yaml 補完) 完了
- [ ] Task 4.1 (docs/troubleshooting.md 追記) 完了
- [ ] Task 5.1 - 5.4 (品質チェック) 全パス
- [ ] PR description で Issue #139 受入条件 (8 項目) すべてチェック付け
- [ ] `_StubTokenizer` / `_get_stub_tokenizer` が production 配下に存在しないこと (`grep` で確認)
- [ ] `_build_photon_deps` で tokenizer_id 未設定 / unsafe / load 失敗が **すべて `ValueError`** で raise されること

---

## TDD 実装順序 (推奨)

Red-Green-Refactor を以下順序で進める:

1. **Red**: Task 2.4 (`test_rejects_unsafe_tokenizer_id`) を書く → 既存 `_validate_tokenizer_id` 不在で fail
2. **Green**: Task 1.1 (`_validate_tokenizer_id`) 実装 → Task 2.4 pass
3. **Red**: Task 2.2 (`test_raises_when_tokenizer_id_missing`) を書く → 既存 fallback で warning + 通過 (= 期待と異なる pass)
4. **Green**: Task 1.3 (`_build_photon_deps` raise 化) 実装 → Task 2.2 pass
5. **Red**: Task 2.3 (`test_raises_when_tokenizer_load_fails`) を書く → 既存 OSError 素通りで fail
6. **Green**: Task 1.2 (`_load_hf_tokenizer` 例外正規化) 実装 → Task 2.3 pass
7. **Red**: Task 3.1 (`test_no_scaffolding_in_prod.py`) を書く → 現 main の `_StubTokenizer` で fail
8. **Green**: Task 1.4 (`_StubTokenizer` 削除) 実装 → Task 3.1 pass
9. **同時**: Task 2.1, 2.5 (既存 test migration) を実施 → 全 regression 復活
10. **Red**: Task 3.2 (`test_photon_yaml_has_required_tokenizer_fields`) を書く → 欠落 yaml で fail
11. **Green**: Task 3.3 (yaml 補完) → Task 3.2 pass
12. **Refactor**: Task 2.6 (conftest.py docstring), Task 4.1 (docs)
13. **Final**: Task 5.x (品質チェック)

---

## 次のアクション

- 作業計画承認後、TDD 実装開始 (`/pm-auto-dev 139` または `/tdd-impl`)
- 実装完了後、PR 作成 (`/create-pr`)、CI チェック、レビュー、マージ
- マージ後、`feature/issue-135-photon-retrain` ブランチで rebase 作業 (#135 担当者)
