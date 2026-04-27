# Issue #139 Stage 1 通常レビュー (1 回目) サマリー

- **対象**: Issue #139 "test(photon): Stub/Mock pattern audit + real-weight integration test (S7-001 follow-up)"
- **レビュー観点**: 整合性 / 正確性 / 実装可能性 / 粒度 / 依存関係
- **レビュー日**: 2026-04-26
- **ブランチ**: feature/issue-139-stub-audit (HEAD: 8e677ca)
- **件数**: Must Fix 3 / Should Fix 5 / Nice to Have 2

---

## 指摘一覧

| ID | 重要度 | 区分 | タイトル |
|----|--------|------|----------|
| S1-001 | Must Fix | 依存関係 | 前提となる S7-001 fix (commit 2dbf458) は main にマージされていない |
| S1-002 | Must Fix | 実装可能性 | Task 2 の real-weight integration test に必要な checkpoint 取得手段が未定義 |
| S1-003 | Must Fix | 正確性 | test_photon_pipeline.py の行数 1715 は誤り (実際 4453) |
| S1-004 | Should Fix | 実装可能性 | Task 1 境界 test 例の正規表現 `\\b` が word boundary にならない |
| S1-005 | Should Fix | 粒度 | Task 1 で扱う『Stub の rename or 廃止』方針が未確定 (3 択並記) |
| S1-006 | Should Fix | 実装可能性 | Task 3 getattr default audit 8 件の個別判定方針が Issue に欠落 |
| S1-007 | Should Fix | 整合性 | Task 3 invariant test の `model.checkpoint_path` 必須化は main に該当項目が無い |
| S1-008 | Should Fix | 粒度 | Task 1 / 2 / 3 を 1 PR か分割するか方針が未定義 |
| S1-009 | Nice to Have | 正確性 | 受入条件の『既存 1114 tests』テスト数が現状 (1050) と不一致 |
| S1-010 | Nice to Have | 正確性 | audit grep コマンドの引用符エスケープが BRE で alternation にならない |

---

## 詳細

### S1-001 [Must Fix / 依存関係] 前提となる S7-001 fix (commit 2dbf458) は main にマージされていない

- **現状**: commit `2dbf458` は `feature/issue-135-photon-retrain` ブランチ上のみ存在し、main (HEAD=8e677ca) には未マージ。
- **影響**: 現行 `baseline_reporag/photon_pipeline.py` には `checkpoint_path` 設定も `load_checkpoint` 呼び出しもない (grep checkpoint → コメント line 331/474 のみ)。Task 2 (real-weight integration test) は『S7-001 fix で導入された checkpoint loading』を前提とするため、現状ではテスト対象自体が main に存在しない。
- **修正案**: Issue 本文に明示する: (1) #135 (S7-001 fix) は未マージであること、(2) 本 Issue は『#135 マージ後に着手』する依存を持つか、または『#135 と独立に Task 1/3 を先行、Task 2 は #135 マージ後』のいずれかを Issue で確定する。受入条件にも『#135 マージ済み』を加える。

### S1-002 [Must Fix / 実装可能性] Task 2 の checkpoint 取得手段が未定義

- **現状**: Task 2 擬似コード `download_or_use_existing_smallest_photon_ckpt()` は実体不在。`./checkpoints/` ディレクトリは repo に含まれず、`configs/*.yaml` にも `checkpoint_path` 設定なし。
- **影響**: nightly CI で実行と書かれていても、CI 環境で checkpoint をどう供給するかが決まらないと test 実装自体が始められない。
- **修正案**: Issue Task 2 に checkpoint 用意方針を明記する。代替: hypothesis-verification.md 提案『最小 PhotonConfig (vocab=256, hidden=64, layers=2) で 1 step 学習 → tmp_path に save → load して 1 query』というセルフホスト型 e2e にすれば repo 完結。受入条件に『test が外部リソース不要で完結する』と明記。

### S1-003 [Must Fix / 正確性] test_photon_pipeline.py の行数誤り

- **現状**: Issue 背景で『1715 行』、実測 `wc -l baseline_reporag/tests/test_photon_pipeline.py` → **4453 行** (約 2.6× 過小)。
- **修正案**: Issue 本文の『1715 行』を『4453 行』(または『約 4.5K 行』) に修正。

### S1-004 [Should Fix / 実装可能性] 境界 test の正規表現バグ

- **現状**: Task 1 Step 3 サンプルの `forbidden_patterns = [r'_Stub\\b', r'_Mock\\b', r'_Dummy\\b']` は r-string + 二重バックスラッシュにより regex 上は『リテラル \b』となり word boundary にならない。実装者がコピペすると `_StubTokenizer` がマッチせず test が無意味になる。
- **検証**: `re.search(r'_Stub\\b', 'class _StubTokenizer:')` → `None`。正しくは `r'_Stub\b'` (単一バックスラッシュ)。
- **修正案**: Issue 例コードを `r'_Stub\b'` / `r'_Mock\b'` / `r'_Dummy\b'` に直す。

### S1-005 [Should Fix / 粒度] Stub rename / 廃止方針の 3 択が未確定

- **現状**: Task 1 Step 2 は『rename to `_DevTokenizer`』『HuggingFace tokenizer 置換』『削除 or test 移動』の 3 択並記。判断が実装者に委ねられている。現行 #138 で fallback 設計 (`baseline_reporag/photon_pipeline.py:335-343`) が採用済み。
- **修正案**: 推奨案『tokenizer_id 必須化 + 欠落時 raise + `_StubTokenizer`/`_get_stub_tokenizer` を production module から完全削除 (test fixture へ移設)』を採用条件として Issue 本文に明記する。S1-004 の境界 test と整合する。

### S1-006 [Should Fix / 実装可能性] getattr default 8 件の個別判定が欠落

- **現状**: hypothesis-verification.md Claim 5 が抽出した 8 件 (`head_dim`, `max_position_embeddings`, `rope_theta`, `safe_recgen_enabled`, `provider`, `answering` 系×2, `session_memory`) はそれぞれ性格が異なる。`vocab_size, default=1000` (`baseline_reporag/photon_pipeline.py:314`) は明らかに事故、`safe_recgen_enabled, default=True` は intent ありそう、`session_memory` は無くてよい、など。Issue は方針を一切示しておらず、実装者が 8 件すべての設計判断を負う。
- **修正案**: Issue Task 3 に『必須化対象』を列挙。例: 最小スコープとして『`model.vocab_size` と (S1-001 解消後は) `model.checkpoint_path` の 2 件のみ invariant test で必須化、他は Nice-to-Have として別 Issue 化』。

### S1-007 [Should Fix / 整合性] invariant test の `model.checkpoint_path` 必須化が main の状態と不整合

- **現状**: Task 3 Step 3 サンプルで `required = ['model.vocab_size', 'tokenizer.tokenizer_id', 'model.checkpoint_path']` だが、(a) main の `photon_pipeline.py` は `checkpoint_path` を読まない、(b) `configs/*.yaml` 全部に未設定。
- **修正案**: サンプルから一旦 `model.checkpoint_path` を外す、または『#135 マージ後に追加』と注記。Phase 分割でも可 (Phase A: `vocab_size` + `tokenizer_id`, Phase B: `checkpoint_path`)。

### S1-008 [Should Fix / 粒度] Task 1/2/3 を 1 PR か分割するか不明

- **現状**: 3 つの独立性の高いタスクが束ねられている。Task 2 は S1-001/S1-002 解消待ちだが Task 1/Task 3 は即着手可能。1 PR にすると Task 2 のブロッキングで全体が遅延。
- **修正案**: PR 分割方針を Issue に明記。推奨: PR 1 = Task 1 + Task 3 (即着手), PR 2 = Task 2 (#135 マージ後)。Issue 自体は分割不要、受入条件に書けば十分。

### S1-009 [Nice to Have / 正確性] 受入条件の『1114 tests』数値ずれ

- **現状**: `pytest --collect-only -q torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/` → **1050**、全体 default collect → **1084**。Issue 数値 1114 は出所不明。
- **修正案**: 『既存全テストパス』など数値依存を消す表現に置換。

### S1-010 [Nice to Have / 正確性] audit grep コマンドの BRE alternation

- **現状**: Task 1 Step 1 の grep が `_Stub\|_Mock\|_Dummy\|...` を使うが、`grep -E` を付けないと alternation にならない (basic regex では `|` はリテラル)。
- **修正案**: `grep -rEn '_Stub|_Mock|_Dummy|_Placeholder|stub_|mock_|dummy_' baseline_reporag/ photon_mlx/ torch_ref/ --include='*.py'` に書き換える。

---

## 全体所感

本 Issue は問題意識 (#135 S7-001 で発覚した構造的 test ギャップを systematic に閉じる) として妥当で、3 タスクの分解も方向性は良い。一方で **最大の前提『#135 / S7-001 fix が main に取り込まれている』が成立していない** (S1-001) ため、現状のままでは Task 2 が技術的に着手不能。さらに Task 2 用 checkpoint 取得手段も未定義 (S1-002) で、これは設計判断を要する。

Task 1 / Task 3 は #138 (b19e8db) マージ済みの今すぐ着手可能。実務的には S1-001 / S1-002 / S1-008 を解消して Task 1+3 と Task 2 を Phase 分けし、Task 1 では S1-005 で Stub の最終形 (削除 + 必須化 + raise) を Issue 本文に確定するのが望ましい。S1-003 (行数) と S1-004 (regex バグ) は実装直前に必ず直しておくべき正確性問題。

Stage 2 (整合性レビュー) では「Issue 本文の改訂が #135 ブランチや既存 invariant test とどう整合するか」をフォローすると良い。
