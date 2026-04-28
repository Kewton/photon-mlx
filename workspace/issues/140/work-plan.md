# Issue #140 作業計画書

## Issue 概要

| 項目 | 内容 |
|------|------|
| **Issue 番号** | #140 |
| **タイトル** | process(review): Codex multi-stage design review 必須化 + scaffolding 命名禁止 checklist (S7-001 follow-up) |
| **ラベル** | enhancement |
| **サイズ** | M (Markdown 4 + Python 1 + new tests + docs) |
| **優先度** | Medium (S7-001 再発防止のプロセス強化) |
| **依存 Issue** | #139 (CI grep 自動化テストは #139 Task 1 に委譲、本 Issue Task 3 は docs のみ) |
| **並列可能 Issue** | #135 (PHOTON 再学習 — 閾値再校正は別 Issue で実施) |
| **設計方針書** | `workspace/design/issue-140-review-process-design-policy.md` |
| **ブランチ** | `feature/issue-140-review-process` (作業ブランチ既存) |

## ゴール (設計方針書 §1 と一致)

> **「必須」の段階的厳格化 (DR1-010)**: 本 Issue の「必須」は第 1 段階 = WARNING + completion report 記録までを強制。raise / exit 1 への昇格は次回 Issue。

1. `/multi-stage-design-review` (Stage 3-4) と `/multi-stage-issue-review` (Stage 5-8) の **Codex 担当 Stage 必須化**
2. `/pm-auto-issue2dev` Phase 1/3、`/pm-auto-design2dev` の対応 Phase で **Codex stage 結果ファイル存在 + `reviewer="codex"` 検証** を WARNING ベースで strict 化
3. `docs/code_review_checklist.md` 新規作成 (docs のみ、CI grep 自動化 test は #139 へ委譲)
4. `PhotonInference.__init__` 起動時の **embedding norm WARNING** (閾値 config 化、log only)
5. `CLAUDE.md` スラッシュコマンド表に対象 4 skill を掲載

## 詳細タスク分解

### Phase 1: 実装 (設計方針書 §8 実装順序の 14 ステップを忠実に再現)

> **依存順序の鍵** (DR1-006 / DR2-008): WARNING 抑制基盤 → 本体実装 → 既存テスト無回帰 → 新規テスト → skill 更新 → docs 更新の順で進める。

#### Task 1.1: ModelConfig field 追加 + 型/範囲検証 (Layer 3-A)
- **成果物**: `torch_ref/config.py::ModelConfig`
- **内容**:
  - `embedding_random_init_threshold: float = 0.3` field を追加
  - `__post_init__` で **型・範囲検証** を実装 (DR4-002):
    - `bool` を reject (Python では bool が int の subclass)
    - 数値型のみ許可
    - `nan` / `inf` / 負値を reject
    - 失敗時は `ValueError`
- **依存**: なし
- **行数目安**: ~15 行

#### Task 1.2: yaml load 互換 test (Layer 4-pre、DR2-008 で前倒し / DR3-002 で配置先確定)
- **成果物**: `photon_mlx/tests/test_config.py`
- **内容** (設計方針書 §7.4):
  ```python
  @pytest.mark.parametrize('yml', sorted(Path('configs').glob('photon_*.yaml')))
  def test_existing_yaml_loads_without_threshold_field(yml):
      cfg = load_photon_config(yml)
      assert cfg.model.embedding_random_init_threshold == pytest.approx(0.3)
  ```
  - 既存 5 件 (`photon_600m_paper`, `photon_long_context`, `photon_small`, `photon_tiny`, `photon_tiny_recgen`) で default 0.3 が適用されることを確認
  - **invalid threshold 拒否 test** も追加 (DR4-002 反映): `nan` / `inf` / 負値 / 文字列 / bool を渡した時に `ValueError`
- **依存**: Task 1.1
- **行数目安**: ~30 行

#### Task 1.3: テストヘルパー更新 (Layer 4-A、DR3-001 反映で 3 ファイル対象)
- **成果物**: 
  - `photon_mlx/tests/test_inference.py`
  - `photon_mlx/tests/test_generate.py`
  - `photon_mlx/tests/test_session.py`
- **内容**:
  - 各ファイルの `_tiny_cfg()` 内で `cfg.model.embedding_random_init_threshold = TEST_EMBEDDING_RANDOM_INIT_THRESHOLD` を設定 (DR4-002 で `float('inf')` から有限値 `1e9` に変更)
  - `test_inference.py` には新規ヘルパー `_photon_cfg(threshold=...)` を追加 (`_tiny_cfg` の wrapper)
  - 共通定数 `TEST_EMBEDDING_RANDOM_INIT_THRESHOLD = 1e9` をどこに置くか決定 (各 test ファイル内 or 共通 conftest) — 推奨: 各 test ファイル内に局所定義 (重複 OK)
  - `test_optimize.py` は `PhotonInference` 生成経路を持たないため変更不要
- **依存**: Task 1.1
- **行数目安**: ~12 行 (3 ファイル × 4 行)

#### Task 1.4: PhotonInference に norm check 実装 (Layer 3-B)
- **成果物**: `photon_mlx/inference.py`
- **内容** (設計方針書 §3 設計判断 #2):
  - `_check_weight_initialization(model, threshold)` 関数追加 (try/except + isinstance ガード付き、silent skip 方針)
  - `PhotonInference.__init__` の末尾から `_check_weight_initialization(model, cfg.model.embedding_random_init_threshold)` を呼び出し
  - WARNING ログは σ + threshold のみ (embedding tensor 値は出さない)
- **依存**: Task 1.1, 1.3
- **行数目安**: ~25 行

#### Task 1.5: 既存テスト無回帰検証
- **内容**:
  - `python -m pytest photon_mlx/tests/test_inference.py photon_mlx/tests/test_generate.py photon_mlx/tests/test_session.py photon_mlx/tests/test_config.py baseline_reporag/tests/test_photon_pipeline.py -v` を実行
  - 既存テスト全件 pass + WARNING 抑制が効いていることを確認
- **依存**: Task 1.1, 1.2, 1.3, 1.4

#### Task 1.6: norm check 挙動 test 追加 (Layer 4-B)
- **成果物**: `photon_mlx/tests/test_inference.py`
- **内容** (設計方針書 §7.2 / 3 件):
  - `test_check_weight_initialization_warns_on_high_variance` (`_photon_cfg(threshold=0.1)` + 実 `PhotonModel` で WARNING 発火確認)
  - `test_check_weight_initialization_silent_on_low_variance` (`_photon_cfg(threshold=10.0)` で silent 確認)
  - `test_check_weight_initialization_silent_on_magicmock_model` (MagicMock + `_BrokenTokenizer` で例外も WARNING も出ない)
  - 閾値の絶対値は assert しない (= 挙動 test)
- **依存**: Task 1.4
- **行数目安**: ~40 行

#### Task 1.7: skill description test (Layer 4-C)
- **成果物**: `tests/test_skill_descriptions.py` (新規)
- **内容** (設計方針書 §7.1 / 6+ 件):
  - `test_design_review_codex_required` ('Codex 担当 Stage は必須' + 'WARNING' + 'completion report')
  - `test_issue_review_codex_required` (同上)
  - `test_pm_auto_issue2dev_reviewer_check_snippet` (snippet 文字列存在)
  - `test_pm_auto_design2dev_reviewer_check_snippet` (同)
  - `test_claude_md_lists_target_skills` (4 skill が表に存在)
  - `test_code_review_checklist_exists` (`docs/code_review_checklist.md` 存在 + 必須キーワード)
  - `test_auto_skip_removed_from_issue_review` ('2回目イテレーション自動スキップ判定' が無い、または '廃止' を含む)
- **依存**: Task 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2 (skill / docs 更新後に test を pass させる必要あり、ただし test ファイル自体は先に書ける)
- **行数目安**: ~80 行

#### Task 1.8: reviewer snippet smoke test (Layer 4-C / §7.5)
- **成果物**: `tests/test_skill_descriptions.py` (Task 1.7 と同ファイル)
- **内容** (設計方針書 §7.5 / DR3-003 で具体化済 / DR4-001 でセキュリティ強化済):
  - `SNIPPET_RE` でマーカー範囲 (`# REVIEWER_VERIFICATION_SNIPPET_BEGIN/END`) で snippet 抽出
  - issue-review (`pm-auto-issue2dev.md`) と design-review (`pm-auto-issue2dev.md` / `pm-auto-design2dev.md`) の 2 経路 × 3 ケース (reviewer="codex" / "claude" / 欠落) を parametrize
  - tmp_path 配下に擬似 `workspace/issues/$ISSUE/...` を構築し snippet を bash 実行
  - reviewer="codex" → 無音 / "claude" or 欠落 → WARNING を assert
  - **invalid ISSUE 検証 test** (DR4-001): `../140` / `140;touch injected` / quote 混入で path traversal が起きないこと
- **依存**: Task 2.3, 2.4 (snippet が PM コマンド md にマーカー付きで存在する必要あり)
- **行数目安**: ~80 行

### Phase 2: skill / process 更新 (Layer 1)

#### Task 2.1: multi-stage-design-review 文言更新
- **成果物**: `.claude/commands/multi-stage-design-review.md`
- **内容**:
  - 冒頭 description に「**Codex 担当 Stage (3-4) は必須**であり、`--skip-stage` で skip した場合は **WARNING** を出し、最終 summary report に skipped 状態と reviewer 検証結果を **completion report に記録** する」を追記
  - skill 完了報告の必須出力項目に「Stage 別 finding 数 (Must/Should/NTH) と reviewer フィールド検証結果」を追加
- **依存**: なし

#### Task 2.2: multi-stage-issue-review 文言更新 + auto-skip 廃止
- **成果物**: `.claude/commands/multi-stage-issue-review.md`
- **内容** (設計方針書 §3 設計判断 #5):
  - 冒頭 description に「**Codex 担当 Stage (5-8) は必須**であり、skip 時は **WARNING + completion report 記録**」を追記
  - **auto-skip 関連 3 箇所の更新** (文字列マッチで対象特定):
    1. 「2回目イテレーション自動スキップ判定」見出しを含むセクション全体: 削除または「常に Stage 5-8 を実行する」に書き換え
    2. summary template の `5-8 | 2回目イテレーション | X | X | 完了/スキップ` 行: 状態を「完了 (reviewer=codex 検証済)」の統合表現に変更 (列数は変えない、DR2-005)
    3. 完了条件内の「2回目イテレーション自動スキップ」を含む箇条書き: 削除
- **依存**: なし

#### Task 2.3: pm-auto-issue2dev に reviewer 検証 snippet 追加
- **成果物**: `.claude/commands/pm-auto-issue2dev.md`
- **内容** (設計方針書 §3 設計判断 #4 / §7.5):
  - **Phase 1 完了判定** (multi-stage-issue-review): issue-review 用 snippet (`# REVIEWER_VERIFICATION_SNIPPET_BEGIN (issue-review)` ... `END`)
    - Stage 5, 7 の `reviewer="codex"` 検証
    - `ISSUE` を数値検証 (DR4-001)
    - Python は `sys.argv[1]` 経由で path 受け渡し
    - reviewer 値は raw 比較、ログ前に制御文字を置換 (DR4-003)
  - **Phase 3 完了判定** (multi-stage-design-review): design-review 用 snippet (`# REVIEWER_VERIFICATION_SNIPPET_BEGIN (design-review)` ... `END`)
    - Stage 3, 4 の `reviewer="codex"` 検証
- **依存**: なし

#### Task 2.4: pm-auto-design2dev に reviewer 検証 snippet 追加
- **成果物**: `.claude/commands/pm-auto-design2dev.md`
- **内容**: Phase 2 (multi-stage-design-review 呼出) に Task 2.3 と同型の design-review snippet を追加
- **依存**: なし

#### Task 2.5: pm-auto-dev (任意)
- **成果物**: `.claude/commands/pm-auto-dev.md`
- **内容**: design review を呼ぶフローがあれば該当箇所に snippet 追加 (なければ skip)
- **依存**: なし

### Phase 3: docs 更新 (Layer 2)

#### Task 3.1: code_review_checklist 新規作成
- **成果物**: `docs/code_review_checklist.md` (新規)
- **内容** (設計方針書 §3 Task 3 と同一、Single Source of Truth):
  - 命名規則チェック (production code path)
  - 適用範囲 / 除外パターン
  - CI grep 例 (運用 PR レビュー時の手動コマンド)
  - `_StubTokenizer` 既存分の取り扱い注記
  - **CI 自動化は #139 Task 1 で実装** と明記
- **依存**: なし

#### Task 3.2: CLAUDE.md スラッシュコマンド表更新 + checklist リンク
- **成果物**: `CLAUDE.md`
- **内容**:
  - スラッシュコマンド表に 4 行追加: `/multi-stage-issue-review`, `/multi-stage-design-review`, `/pm-auto-issue2dev`, `/pm-auto-design2dev`
  - `docs/code_review_checklist.md` へのリンクを追加 (どのセクションに置くかは作業時判断 — 推奨: コーディング規約セクションの末尾)
- **依存**: Task 3.1

### Phase 4: 最終検証

#### Task 4.1: pytest 全件パス
```bash
python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v
```
- 基準: 既知 pre-existing failure 2 件 (`tests/test_generate_training_corpus.py`) を除き全件パス
- **依存**: Task 1.1〜1.8 + Task 2.1〜2.5 + Task 3.1〜3.2

#### Task 4.2: ruff check 警告 0 件
```bash
ruff check .
```
- **依存**: Task 4.1

#### Task 4.3: ruff format 差分 0
```bash
ruff format --check .
```
- **依存**: Task 4.1

## タスク依存グラフ

```
1.1 (config field)
 ├─→ 1.2 (yaml test)
 ├─→ 1.3 (test helpers)
 │    └─→ 1.4 (norm check)
 │         └─→ 1.5 (regression check)
 │              └─→ 1.6 (norm check tests)
 │                   └─→ 1.7/1.8 (skill desc tests) ── depends on Phase 2/3
 │
2.1, 2.2, 2.3, 2.4, 2.5 (skill md updates) — independent
3.1 (checklist md) ── independent
 └─→ 3.2 (CLAUDE.md)
       └─→ 1.7 (test_claude_md_lists_target_skills)

4.1 → 4.2 → 4.3 (final verification)
```

## 品質チェック (CLAUDE.md と一致)

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| テスト | `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | 既知 pre-existing failure 2 件除き全パス |
| Lint | `ruff check .` | 警告 0 件 |
| Format | `ruff format --check .` | 差分なし |

## Definition of Done

- [ ] 全 Task (1.1〜1.8, 2.1〜2.5, 3.1〜3.2, 4.1〜4.3) 完了
- [ ] 既存 28+ 件の `PhotonInference(...)` test 無回帰 (WARNING 抑制が効いている)
- [ ] norm check 挙動 test 3 件 + yaml load 互換 test 5 件 + invalid threshold 拒否 test + skill desc test 7 件 + snippet smoke test (parametrize) すべてパス
- [ ] ruff check 警告 0 件
- [ ] ruff format 差分なし
- [ ] CLAUDE.md / docs/code_review_checklist.md / 4 skill md が想定通り更新されている
- [ ] 設計方針書 §8 実装順序の 14 ステップに対応するコミットが揃っている (or 1 PR で網羅)
- [ ] PR 作成後の CI 全パス

## リスクと緩和策

| リスク | 緩和策 |
|--------|-------|
| `_tiny_cfg` 3 ファイル更新で他 test に副作用 | Task 1.5 で全件 pass を確認、不一致なら ValidationError 経路を確認 |
| invalid threshold validation で既存 yaml が破壊 | Task 1.2 の yaml load 互換 test で早期検出 (DR2-008 で前倒し) |
| reviewer snippet smoke test の bash 実行が CI 環境で動かない | tmp_path / sys.argv 経由で path traversal を排除 (DR4-001)、bash の利用が CI で前提できることを確認 |
| skill string-existence test と既存類似文字列の衝突 | DR2-009 反映で「Codex 担当 Stage は必須」のような新規追加文言を完全一致 assert |
| PHOTON 再学習 (#135) 完了で閾値再校正が必要 | 本 Issue は暫定値 0.3 のまま、再校正は次回 Issue で扱う (設計方針書 §1) |

## 実行順序の推奨

設計方針書 §8 の 14 ステップは以下のように作業計画 Task に対応:

| §8 Step | Task |
|---------|------|
| 1 | Task 1.1 |
| 2 | Task 1.2 |
| 3 | Task 1.3 |
| 4 | Task 1.4 |
| 5 | Task 1.5 |
| 6 | Task 1.6 |
| 7 | Task 1.7 + 1.8 |
| 8 | Task 2.1 |
| 9 | Task 2.2 |
| 10 | Task 2.3 |
| 11 | Task 2.4 |
| 12 | Task 3.1 |
| 13 | Task 3.2 |
| 14 | Task 4.1 + 4.2 + 4.3 |

## 次のアクション

1. `/pm-auto-dev 140` または `/tdd-impl 140` で TDD 実装を開始
2. 進捗は `/progress-report 140` で確認
3. 完了後 `/create-pr` で PR を作成 (`feature/issue-140-review-process` → `develop` への PR)
