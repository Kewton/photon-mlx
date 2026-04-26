# Issue #139 開発進捗レポート (iteration-1)

**Issue**: [#139](https://github.com/Kewton/photon-mlx/issues/139) test(photon): Stub/Mock pattern audit + invariant test (S7-001 follow-up)
**ブランチ**: `feature/issue-139-stub-audit`
**HEAD (main 比)**: 8e677ca
**実施日**: 2026-04-26 〜 2026-04-27
**スキル**: `/pm-auto-issue2dev 139` (Issue review → Design → Design review → Work plan → TDD)
**スコープ**: Task 1 (scaffolding 排除) + Task 3 Phase A (yaml invariant 拡張) + Codex 受入指摘 (CB-001 + CB-002) を一括反映。Task 2 (real-weight integration test) は **#145** に切出済。

---

## 累積 review カウント

| Phase | レビュアー | Must / Should / Nice | 適用 |
|-------|-----------|---------------------|------|
| Phase 1 (Issue review × 8 stages) | claude-opus + codex | 7 / 13 / 5 | 全件 |
| Phase 3 (Design review × 4 stages) | claude-opus + codex | 2 / 17 / 4 | 全件 |
| Phase 5.5 (Codex code review) | codex | 1 high / 1 medium | 全件 (CB-001/CB-002) |
| **総計** | | **9 + 1 high + 1 medium / 30 / 9** | **48 + 2 = 50** |

---

## 変更ファイル

### Production (1)

- **`baseline_reporag/photon_pipeline.py`**
  - `_StubTokenizer` クラス + `_get_stub_tokenizer` 関数を **削除** (Issue #139 Task 1)
  - `_validate_tokenizer_id` 新設: HF repo-id allowlist `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$` + component-level validation (total/per-component length cap、leading `/.~` 拒否、`./..` component 拒否、leading-dot component 拒否、`..` substring 拒否) (DR4-001 + Codex CB-001)
  - `_display_tokenizer_id` 新設: log/error message 用 sanitized 表示 (`repr()` で control char escape) (DR4-001)
  - `_load_hf_tokenizer` を改修: `AutoTokenizer.from_pretrained` ブロックを try/except で囲い、`OSError` 等を `ValueError` に正規化。**`raise ... from None`** で raw HF exception が `__cause__` 経由で漏洩しないよう変更 (Codex CB-002)。診断用に WARNING level で内部 log に exception class 名 + sanitized id を残す。
  - `_build_photon_deps` の fallback 経路を撤去: `tokenizer_id` 未設定で `ValueError` を raise、`_validate_tokenizer_id` 経由で値を検証、`_load_hf_tokenizer` を呼ぶ (S1-005 / DR2-002)

### Test (3)

- **`baseline_reporag/tests/test_photon_pipeline.py`**
  - module-level autouse `_stub_hf_loader` fixture 追加: `_load_hf_tokenizer` を mock してテスト全体を hermetic に保つ (DR3-001 で発見した 17-18 success-path test の崩壊を一括解消)。`@pytest.mark.real_hf_loader` で opt-out 可。
  - 既存 success-path 17 件の yaml fixture に `tokenizer:\n  tokenizer_id: "fake-org/fake-tokenizer"\n` を追加 (replace_all で一括反映)
  - 2 件の `_yaml.safe_dump(merged)` 系 test の base_cfg dict に `"tokenizer": {"tokenizer_id": "fake-org/fake-tokenizer"}` を追加
  - `TestBuildPhotonDepsRealTokenizer` class に `@pytest.mark.real_hf_loader` を付与 (autouse stub を opt-out)
  - `_StubTokenizer` import 4 箇所削除 / `isinstance(_StubTokenizer)` 2 箇所削除 (S3-004 / DR1-005)
  - 旧 `test_falls_back_to_stub_when_tokenizer_id_missing` を削除 (Issue #139 で削除する fallback の test だった)
  - 新規 test 追加 (3 件):
    - `test_raises_when_tokenizer_id_missing` (S1-005)
    - `test_raises_when_tokenizer_load_fails` (S5-002 / DR1-002 / Codex CB-002)
    - `test_rejects_unsafe_tokenizer_id` (DR4-001 / Codex CB-001) — 17 unsafe id case の parametrize
- **`tests/test_no_scaffolding_in_prod.py`** (新規 / 137 行)
  - `\b_(?:Stub|Mock|Dummy|Placeholder)\w*` regex で production 配下を CI gate (S3-003 / DR1-005)
  - repo-root anchor (S7-001), `'tests' in f.parts` (複数形 / S3-003), `__pycache__` 除外
  - DR4-003 hardening: symlink 拒否 / repo-root 外 resolve 拒否 / 1MiB size cap / UnicodeDecodeError 拒否
- **`tests/test_pipeline_factory_yaml_invariants.py`** (拡張)
  - `_is_photon_profile_yaml(path, cfg)`: filename main + `provider == "photon"` insurance (DR1-006 / S5-001)
  - `test_photon_yaml_has_required_tokenizer_fields`: PHOTON profile yaml に `tokenizer.vocab_size` / `tokenizer.tokenizer_id` 必須 invariant (S3-001 / S3-002)
  - 既存 `load_config()` helper 再利用 (DR1-001) / `@pytest.mark.skip` 系を付けない方針コメント (DR4-004)

### Docs (2)

- **`docs/troubleshooting.md`** (拡張)
  - `cfg.model.provider == "photon"` checklist に **5. tokenizer.tokenizer_id 未設定 → ValueError** と **6. tokenizer load 失敗 → ValueError** を追加 (S3-006 / S7-002 / DR1-003 / DR4-005)
  - DR4-005: HF token / private model id の平文転記禁止、redaction 推奨を明記
- **`photon_mlx/tests/conftest.py`** (docstring 更新)
  - production `_StubTokenizer` 参照を削除し、test-only stub であることを明記 (DR1-004)

---

## 品質チェック結果

| チェック | コマンド | 結果 |
|---------|----------|------|
| Lint | `ruff check .` | ✅ 警告 0 件 |
| Format | `ruff format --check .` | ✅ 差分なし (148 files) |
| 個別 (raise/unsafe) | `pytest -k "rejects_unsafe or raises_when_tokenizer or load_fails"` | ✅ 19 件 pass |
| 個別 (boundary) | `pytest tests/test_no_scaffolding_in_prod.py` | ✅ 1 件 pass |
| 個別 (invariant) | `pytest tests/test_pipeline_factory_yaml_invariants.py` | ✅ 3 件 pass (2 #133 skip) |
| 全体 | `python -m pytest` | ✅ **1100 passed**, 2 pre-existing failure (CLAUDE.md 既知の `test_generate_training_corpus.py`), 2 skipped |
| baseline 疎通 | provider=mlx_lm 経路は本 Issue 影響なし (S3-010) | -  |

---

## 受入条件チェック (Issue #139 から)

| # | 条件 | 状態 |
|---|------|------|
| 1 | scaffolding 命名が production 配下に存在しない | ✅ test_no_scaffolding_in_prod.py pass |
| 2 | test_no_scaffolding_in_prod.py が cwd 非依存 + repo root anchor + violation の symlink/oversize/non-utf8 を拒否 | ✅ DR4-003 反映済 |
| 3 | tokenizer_id 未設定で ValueError raise | ✅ test_raises_when_tokenizer_id_missing pass |
| 4 | tokenizer load 失敗で ValueError raise | ✅ test_raises_when_tokenizer_load_fails pass |
| 5 | test_photon_pipeline.py の `_StubTokenizer` 参照を削除 / migration 完了 | ✅ 4 箇所削除、新 3 test 追加 |
| 6 | tests/test_pipeline_factory_yaml_invariants.py に tokenizer.vocab_size + tokenizer.tokenizer_id 存在チェックを追加し PHOTON profile yaml で全件 pass | ✅ |
| 7 | 既存テスト + 新規テスト全パス (CLAUDE.md 既知 2 件は除外可) | ✅ 1100 passed |
| 8 | ruff check / ruff format 警告 0 件 | ✅ |
| 9 | (out-of-scope) Task 2 = #145 で追跡 | ✅ |

---

## 累積成果物

```
workspace/issues/139/
├── design-policy.md                                                     ← Phase 2/3
├── work-plan.md                                                          ← Phase 4
├── issue-review/
│   ├── original-issue.json
│   ├── hypothesis-verification.md
│   ├── stage{1..8}-*.json
│   ├── stage{1,3,5,7}-review-summary.md (一部)
│   ├── updated-issue-body.md (= 最終 Issue #139 本文)
│   ├── followup-issue-body.md (= #145 本文)
│   └── summary-report.md
├── multi-stage-design-review/
│   ├── stage{1..4}-review-result.json
│   ├── stage{1..4}-apply-result.json
│   └── summary-report.md
└── pm-auto-dev/iteration-1/
    ├── codex-review-prompt.md
    ├── codex-review-result.json (verdict: needs_fix, CB-001/CB-002)
    ├── codex-review-fix-result.json (verdict_after_fix: pass)
    └── progress-report.md (this file)
```

---

## 次のアクション

- [ ] **Phase 6**: 最終検証 (本 progress report) — 完了
- [ ] PR 作成: `/create-pr` で Issue #139 PR を develop ブランチに作成
- [ ] CI 確認 → main マージ
- [ ] **#145 (real-weight integration test) は #135 マージ後に着手**
- [ ] **#135 ブランチの rebase**: 本 Issue マージ後、#135 担当者が `_build_photon_deps` の手動統合を実施 (DR3-003 で確認した実 conflict 範囲: `baseline_reporag/photon_pipeline.py` の `_load_hf_tokenizer` 削除 / `_get_stub_tokenizer` 復活方針との統合)
- [ ] (Phase B 別 Issue): 残り 6 件の `getattr(cfg, ..., default)` invariant 化 (`head_dim`, `max_position_embeddings`, `rope_theta`, `safe_recgen_enabled`, `provider`, `session_memory`/`answering` 系) — 設計判断 #6 / S1-006

---

## 補足: 副次的成果

- **Issue #139 本文**: 25 finding (Phase 1) を反映した最終形 = `updated-issue-body.md`、GitHub Issue 同期済
- **#145 (新 Issue) 作成**: real-weight integration test の切り出し
- **#135 との実 conflict 範囲確定** (DR3-003 / S7-003): rebase 時の手動統合方針も work-plan / design-policy に明記済
- **設計判断 6 件** を design-policy.md に記録: `_StubTokenizer` 完全削除 vs rename / 例外正規化対象範囲 / 境界 test scope / invariant test scope / test migration 計画 / Phase A vs Phase B
