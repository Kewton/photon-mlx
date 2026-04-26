# Issue #139 マルチステージ設計レビュー完了報告

**対象**: `workspace/issues/139/design-policy.md`
**実施日**: 2026-04-26 〜 2026-04-27
**ブランチ**: `feature/issue-139-stub-audit`

---

## ステージ別結果

| Stage | レビュー種別 | レビュアー | Must / Should / Nice | 反映 |
|-------|------------|----------|---------------------|------|
| DR1 | 通常 (設計原則: SOLID/KISS/YAGNI/DRY/fail-fast) | claude-opus (inline) | **0 / 5 / 2** = 7 | 全件適用 |
| DR2 | 整合性 (design vs current code) | claude-opus (inline) | **0 / 5 / 2** = 7 | 全件適用 |
| DR3 | 影響分析 | codex (cross) | **1 / 3 / 0** = 4 | 全件適用 ✓ reviewer=codex |
| DR4 | セキュリティ | codex (cross) | **1 / 4 / 0** = 5 | 全件適用 ✓ reviewer=codex |

**合計**: 2 Must Fix / 17 Should Fix / 4 Nice to Have = **23 finding** がすべて design doc に反映。

> 注: DR1 + DR2 (Stage 1-2) は Claude opus 配下 subagent quota に達したため、Claude main agent (opus) が **inline** で実施。`reviewer` フィールドに明記済。Stage 3-4 はスキル要件に従い Codex (commandmatedev `--agent codex`) で実施。

---

## 主要 finding ハイライト

### Must Fix (2 件)

- **DR3-001**: design doc の test migration plan は `_StubTokenizer` direct 参照 4 件のみだったが、現コードには `_build_photon_deps` の **success-path test 17-18 件** が `_get_stub_tokenizer` fallback に依存している (test_photon_pipeline.py:472, 502, 1927, 1959, 2010, 2044, 2078, 2108, 2873, 2904, 3171, 3208, 3257, 3326, 3396, 3477, 4307, 4372)。本 Issue で fallback を削除すると **これら全件が破綻**。design doc に migration plan 拡張、`tokenizer:` ブロック追加 + `AutoTokenizer.from_pretrained` mock 方針を追記済。

- **DR4-001**: `tokenizer.tokenizer_id` は **untrusted yaml input**。Stage 4 前の設計では (a) HF repo id allowlist validation なし、(b) ValueError メッセージに raw `tokenizer_id` + 元例外文字列を埋め込み → log injection / control char / private model id 露出 / token leakage の余地あり。design doc に `_validate_tokenizer_id` (allowlist regex `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`) と `_display_tokenizer_id` (sanitization) サンプル、unsafe tokenizer_id unit test 要件、sanitized ValueError 方針を追記済。

### Should Fix (17 件)

- DR1-001 (DRY): invariant test サンプルを既存 `load_config()` helper 再利用へ統一
- DR1-002 (fail-fast): tokenizer load 失敗の例外正規化を `AutoTokenizer.from_pretrained` 呼び出しブロックのみに限定 (`ImportError` / vocab mismatch ValueError を保護)
- DR1-003 (docs): `docs/troubleshooting.md` のアンカーを `Issue #82 drift_metrics N/A section` に修正
- DR1-004 (test): `photon_mlx/tests/conftest.py:5` の docstring 更新を migration plan に追加
- DR1-005 (test): `_StubTokenizer` 参照を 4 箇所 (L521, 563, 611, 636) で網羅化
- DR2-001-005 (整合性): design サンプルコード (`before`/`after`) を実 main コード (`_logger.warning` / None-safe アクセス / `trust_remote_code=False`) と一致化
- DR3-002 (test): `photon_mlx/tests/conftest.py` の test-only stub の利用 test 群明示
- DR3-003 (merge): #135 ブランチの実 diff を確認し、rebase 方針を具体化 (`_load_hf_tokenizer` 削除 / `_get_stub_tokenizer` 復活 / checkpoint_path 追加)
- DR3-004 (運用): Phase B 対象が stale 化するリスクと完了前 audit 再実行の手順
- DR4-002 (supply chain): `huggingface_hub.errors.*` への concrete import 非依存 + `trust_remote_code=False` invariant
- DR4-003 (test hardening): 境界 test の symlink 非 follow / size cap / UTF-8 decode error violation
- DR4-004 (CI gate): 新規 invariant に `skip`/`skipif`/`xfail` 禁止 + `yaml.safe_load` 経由限定
- DR4-005 (info leak): docs/troubleshooting.md 追記に HF token / private model id 平文転記禁止

### Nice to Have (4 件)

- DR1-006 (KISS): `_is_photon_profile_yaml` filename main / provider insurance の整理
- DR1-007 (DRY): 境界 test の `EXCLUDED_DIR_PARTS` 将来拡張ノート
- DR2-006 (整合): §2 ASCII 図に `if/else` 分岐 + fallback 削除予定を反映
- DR2-007 (整合): §12 末尾に GitHub Issue body と updated-issue-body.md の diff チェック手順

---

## 完了条件

- [x] 全 4 ステージ実施
- [x] 反映: 全 23 findings (2 Must Fix + 17 Should Fix + 4 Nice) 適用済
- [x] design-policy.md が最新状態
- [x] Codex の result file 4 件 (`stage3-review-result.json`, `stage3-apply-result.json`, `stage4-review-result.json`, `stage4-apply-result.json`) で `reviewer == "codex"` を確認

---

## 累積 review カウント (Phase 1 Issue review + Phase 3 design review)

| Phase | Must Fix | Should Fix | Nice to Have | 合計 |
|-------|---------|-----------|-------------|------|
| Phase 1 (Issue review × 8 stages) | 7 | 13 | 5 | 25 |
| Phase 3 (Design review × 4 stages) | 2 | 17 | 4 | 23 |
| **総計** | **9** | **30** | **9** | **48** |

---

## 次のアクション

- [ ] Phase 4: 作業計画立案 (`/work-plan 139`) — 23 finding 反映後の design doc に基づく具体的タスク分解
- [ ] Phase 5: TDD 自動開発
- [ ] Phase 6: 最終検証 + 完了報告
