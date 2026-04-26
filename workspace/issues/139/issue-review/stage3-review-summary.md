# Stage 3 影響範囲レビュー（1回目） — Issue #139

- 対象: Issue #139 (post Stage-2 反映後の本文)
- レビュー観点: 影響範囲・破壊的変更・後方互換・merge 衝突・docs / CI 影響
- 反復: 1 回目
- レビュアー: claude-opus

## サマリ

| 重要度 | 件数 |
|--------|------|
| Must Fix | 4 |
| Should Fix | 3 |
| Nice to Have | 3 |
| **合計** | **10** |

## 総合評価

実装スコープと既存資産との整合性に **Must Fix 4 件** (S3-001/002/003/004) を発見。Issue の方向性 (scaffolding 排除 + invariant 強化) 自体は妥当だが、Step 3 のサンプルコードに **semantic bug が複数あり**、そのまま実装すると test が即 FAIL する。実装前に Issue を修正すべき。

---

## Must Fix (4 件)

### S3-001: invariant test の必須フィールド `model.vocab_size` は全 yaml で未設定

- **問題**: Issue が必須化対象とする `model.vocab_size` は `configs/*.yaml` **全 9 件で None**。実際の canonical key は `tokenizer.vocab_size` (`baseline_reporag/photon_pipeline.py:295-314` のコメントで明示)。
- **証拠**: `configs/photon_small.yaml:147-149` で `tokenizer:` ブロックに `vocab_size: 152064`、`model:` ブロックに記載なし。Python で 9 件確認、全件 `model.vocab_size` は None。
- **対応**: 受入条件 124 行目および Step 3 サンプル (107-116行) の `model.vocab_size` を `tokenizer.vocab_size` に書き換える。

### S3-002: `tokenizer.tokenizer_id` 必須化は `configs/baseline.yaml` + `configs/eval.yaml` で破綻

- **問題**: `baseline.yaml` は `tokenizer:` ブロックを持たず `model.tokenizer_id` (line 248、空文字) のみ。`eval.yaml` はベンチマーク runner config で model/tokenizer 両方なし。Issue の `Path('configs').glob('*.yaml')` 全件走査で必ず FAIL。
- **証拠**: `configs/baseline.yaml:245-248` 確認。`.github/workflows/weekly_eval.yml` および CLAUDE.md 疎通コマンドが `baseline.yaml` を参照する運用上のメイン config。
- **対応案**: (a) invariant test の対象を `cfg.model.provider == 'photon'` に絞る (推奨)、(b) baseline は `model.tokenizer_id`、photon は `tokenizer.tokenizer_id` の二系統チェック、(c) `eval.yaml` を明示除外。受入条件 124 行目の '全 configs/*.yaml で pass' を '`provider == \"photon\"` の yaml で pass' に書き直す。

### S3-003: 境界 test の path-exclusion + regex の semantic bug

- **問題1**: `if 'test' in f.parts` は tuple **完全一致** check。本リポの test dir は `tests` (複数形) のため除外されず、29 件の test file が誤って scan 対象に。
- **問題2**: `r'_Stub\b'` は word boundary のため `_StubTokenizer` を検出しない (`re.search(r'_Stub\b', '_StubTokenizer')` is None)。
- **証拠**: `'test' in ('a', 'tests')` → False (Python 検証)。`re.search(r'_Stub\b', 'class _StubTokenizer:')` → None (Python 検証)。Stage 1 S1-004 修正は構文上の bug は直したが word boundary 自体の意味を再検討していない。
- **対応**: (1) 除外を `if 'tests' in f.parts` または `any(p == 'tests' or p.startswith('test_') for p in f.parts)` に修正。(2) regex を `r'_Stub'` または `r'\b_Stub'` または `r'\b_(?:Stub|Mock|Dummy|Placeholder)\w*'` に変更。

### S3-004: `baseline_reporag/tests/test_photon_pipeline.py` で `_StubTokenizer` 削除に伴う既存 3 テストが破綻

- **問題**: 既存 test の以下 3 ヶ所が ImportError / AssertionError になる:
  - line 521 `from baseline_reporag.photon_pipeline import _StubTokenizer` → ImportError
  - line 563 `assert not isinstance(deps[\"tokenizer\"], _StubTokenizer)` → NameError
  - line 604-636 `test_falls_back_to_stub_when_tokenizer_id_missing` → 仕様変更で AssertionError、本テスト自体が削除対象
- **証拠**: `grep -rn "_StubTokenizer" baseline_reporag/tests/` で 5 ヒット (line 514/521/563/611/636)。
- **対応**: Issue '影響ファイル' に '`test_photon_pipeline.py:521,563` の参照を削除、`test_falls_back_to_stub_when_tokenizer_id_missing` (604-636) は削除し `test_raises_when_tokenizer_id_missing` で置換' を明記。

---

## Should Fix (3 件)

### S3-005: #135 と `_build_photon_deps` で近接行を編集 — merge 順序の事前合意要

- `feature/issue-135-photon-retrain` は `_build_photon_deps` で line 306 後に checkpoint load の ~22 行を追加。本 Issue は line 335-343 の stub fallback を raise に書き換え。両者 8-15 行離れており auto-merge は通る可能性が高いが、`_get_stub_tokenizer` 行が #139 で消えるため後 merge 側に意味的ずれが残る。
- **対応**: '関連' 節に 'merge 順は #139 → #135 (#135 を後で rebase)' を追記。

### S3-006: tokenizer_id 必須化の新 failure mode が docs に未反映

- CLAUDE.md は `docs/troubleshooting.md` を operational doc と定義。新 raise の failure mode は docs に追加すべき。
- **対応**: '影響ファイル' に '`docs/troubleshooting.md`: photon 起動失敗 checklist に tokenizer_id 未設定 → ValueError の節を追加' を追記。または明示的に out-of-scope 宣言。

### S3-007: `_Stub\b` regex は意図対象を検出しない (S1-004 の見直し不足)

- S3-003 と関連。`\b` は `_Stub` と `T` の間にマッチしないため `_StubTokenizer` を検出できない。
- **対応**: regex を `r'_Stub'` / `r'\b_Stub'` / `r'\b_(?:Stub|Mock|Dummy|Placeholder)\w*'` のいずれかに変更し、S1-004 メモを更新。

---

## Nice to Have (3 件)

### S3-008: bench/scripts/demo の `_StubTokenizer` / `_Mock*` 対象外を明示

- `bench/issue61_prune_batch.py:51` に自己完結 `_StubTokenizer` あり。`bench/tests/test_run_all.py:24,33,39` に `_MockLatency`/`_MockMemory`/`_MockQueryResult`。新 test の scope (baseline_reporag/photon_mlx/torch_ref) は除外しているが、根拠 (research harness で production runtime 外) を Issue に明記しておくと良い。

### S3-009: 新規 test 2 件の CI コストは無視可能 (<1s)

- yaml 9 件 + py 約 100 件 walk で 100ms オーダー。受入条件への記載は不要、注記程度。

### S3-010: weekly_eval.yml + CLAUDE.md 疎通コマンドは provider=mlx_lm のため raise 化の影響なし

- `_build_photon_deps` は provider=photon のみ呼ばれる。`baseline.yaml` (provider=mlx_lm) は影響範囲外。これは肯定的所見だが '影響ファイル' に注記しておくと判断材料が揃う。

---

## 関連ファイル

- 結果 JSON: `workspace/issues/139/issue-review/stage3-review-result.json`
- Issue 本文 (Stage 2 反映後): `workspace/issues/139/issue-review/updated-issue-body.md`
- 仮説検証: `workspace/issues/139/issue-review/hypothesis-verification.md`
- 主な検証対象:
  - `configs/baseline.yaml:245-248` (model.tokenizer_id 配置)
  - `configs/photon_small.yaml:147-149` (tokenizer.vocab_size 配置)
  - `configs/eval.yaml` 全文 (model/tokenizer block 不在)
  - `baseline_reporag/photon_pipeline.py:295-343` (現行 tokenizer 解決 + stub fallback)
  - `baseline_reporag/tests/test_photon_pipeline.py:514-636` (既存 stub-依存 test)
  - `tests/test_pipeline_factory_yaml_invariants.py` (既存 invariant pattern)
  - `.github/workflows/weekly_eval.yml`
  - `docs/troubleshooting.md:149`
