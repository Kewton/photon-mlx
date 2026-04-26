## 背景

#135 の S7-001 (PHOTON eval が random-init weight で動作) は **`/multi-stage-issue-review` Stage 7 (Codex 影響範囲レビュー 2回目)** で初めて発見された。Stage 1-4 (opus 単独) では見落とされていた。

> 注: 本リポジトリの finding ID 命名体系 — `S7-001` は `/multi-stage-issue-review` Stage 7 (Codex) の最初の finding、`DRN-NNN` は `/multi-stage-design-review` Stage N の finding を指す。`/multi-stage-design-review` は **4 ステージ構成** (Stage 1-2 = opus、**Stage 3-4 = Codex**) であり、`/multi-stage-issue-review` は **8 ステージ構成** (Stage 1-4 = opus、**Stage 5-8 = Codex**)。issue review の偶数 stage (2, 4, 6, 8) は **apply-only** で `stageN-apply-result.json` のみを出力し、`stageN-review-result.json` は出力しない。同様に design review の Stage 1-4 はそれぞれ review + apply の両方を 1 stage で扱う。

これは個別 bug を超えた **品質保証プロセスの構造課題**:

- opus 単独レビューは「設計の論理整合性」は確認できるが、「実装と設計のギャップ」(実コードに silent bug が残っているか) を発見しにくい
- Codex クロスレビュー (design review Stage 3-4 / issue review Stage 5-8) が補完するこの観点が **オプショナルだったため、Codex stage を実施しないケースで bug が混入** した可能性
- scaffolding 命名 (`_Stub`, `_Mock` 等) が production に残ること自体が **code review checklist で禁止項目化されていなかった**

本 Issue で**プロセス・ドキュメント面**の再発防止を恒久化する。

## ゴール

1. `/multi-stage-design-review` skill の **Codex 担当 Stage (3-4) 必須化** および `/multi-stage-issue-review` skill の **Codex 担当 Stage (5-8) 必須化**
2. `/pm-auto-issue2dev` Phase 1 / Phase 3 / `/pm-auto-design2dev` の対応 Phase で **Codex stage 結果ファイルの存在 + `reviewer="codex"` フィールド検証** を必ず実施する。初期実装では後方互換性のため fatal にはせず、WARNING + completion report 記録に留める。
3. **scaffolding 命名禁止** を code review checklist に明示 (production import path のみ、tests/bench/scripts は適用除外)。**ただし grep 自動化テストの新規実装は #139 Task 1 と完全重複するため、本 Issue では docs 整備のみ**
4. "random-init で silent に動く" 型 bug を発見する **embedding norm WARNING** を `PhotonInference` 起動ログに追加 (閾値は config 化、初期は WARNING のみで raise しない)
5. **CLAUDE.md スラッシュコマンド表に対象 4 skill (`/multi-stage-design-review`, `/multi-stage-issue-review`, `/pm-auto-issue2dev`, `/pm-auto-design2dev`) を掲載** (現状未掲載で運用導線が欠落、S3-003)

## 変更内容

### Task 1: `/multi-stage-design-review` および `/multi-stage-issue-review` skill の Codex 必須化

**現状**:
- `.claude/commands/multi-stage-design-review.md`: 4 ステージ、Stage 1-2 が opus、**Stage 3-4 が Codex**。`--skip-stage=3,4` 等で skip 可能。
- `.claude/commands/multi-stage-issue-review.md`: 8 ステージ、Stage 1-4 が opus、**Stage 5-8 が Codex**。`--skip-stage=5,6,7,8` 等で skip 可能。
- どちらも禁止事項として「Codex を Claude サブエージェントで代替実行禁止」は記載済みだが、運用上 Codex stage を skip する余地が残る。

**変更**:
- 両 skill の冒頭 description に「**Codex 担当 Stage は必須**であり、`--skip-stage` で skip した場合は WARNING を出し、最終 summary report に skipped 状態を明示する」を追記。
- skill 完了報告の必須出力項目に「**Codex Stage 別 finding 数 (Must Fix / Should Fix / Nice to Have) と reviewer フィールド検証結果**」を追加。
- `--skip-stage` フラグ自体は破壊的変更を避けて保持し、「skip 時は WARNING」の段階的厳格化（過渡期）として導入。raise/exit 1 への昇格は次回 Issue で扱う。
- `.claude/commands/multi-stage-issue-review.md` の「1回目 Must Fix 合計が 0 件なら Stage 5-8 を自動スキップする」判定は、Codex Stage 必須化および PM 側の Stage 5/7 結果ファイル確認と矛盾するため廃止する。1回目 Must Fix が 0 件でも Stage 5-8 は実行し、findings 0 件は正当な Codex レビュー結果として扱う。

### Task 2: `/pm-auto-issue2dev` / `/pm-auto-design2dev` Phase 完了判定の検証追加

**現状**: `/pm-auto-issue2dev` Phase 1 (multi-stage-issue-review) / Phase 3 (multi-stage-design-review) はそれぞれ skill を呼ぶのみで、Codex stage の有無や finding 数を後続 Phase が確認していない。`/pm-auto-design2dev` も同様。

**変更**: 各 PM コマンドの該当 Phase 完了判定を以下に再定義 (S1-005 / S3-004 / S3-009 / S5-002 反映)。検証は必ず行うが、初期実装では欠落・`reviewer!="codex"` を即時 exit/raise せず、WARNING と completion report の明示に留める。fatal 化は次回 Issue で扱う。

1. **結果ファイル存在確認** (review stage のみ — apply-only stage の Stage 6/8 は対象外):
   - issue review: `workspace/issues/{issue_number}/issue-review/stage5-review-result.json` および `stage7-review-result.json`
   - design review: `workspace/issues/{issue_number}/multi-stage-design-review/stage3-review-result.json` および `stage4-review-result.json`
2. **`reviewer` フィールドが `"codex"` であること**を JSON パースで確認 (Claude による上書きが起きていないか検証)。`apply-result.json` は対象外。
3. **completion report に Stage 別の Must Fix / Should Fix / Nice to Have 件数表を必ず記載**:
   - Issue review: **Stage 1, 3, 5, 7** (review stage のみ)
   - Design review: **Stage 1, 2, 3, 4** (4 stage すべて review + apply 兼任)
4. Stage 6, 8 (issue review apply-only) は `apply-result.json` の存在のみで完了確認とし、reviewer 検証は実施しない。

**スコープ** (S1-009 / S3-009 反映):
- `.claude/commands/pm-auto-issue2dev.md`:
  - **Phase 1 (multi-stage-issue-review) 完了判定**: Stage 5, 7 の reviewer=codex 検証
  - **Phase 3 (multi-stage-design-review) 完了判定**: Stage 3, 4 の reviewer=codex 検証
- `.claude/commands/pm-auto-design2dev.md`:
  - **Phase 2 (multi-stage-design-review) 完了判定**: Stage 3, 4 の reviewer=codex 検証
- (任意) `.claude/commands/pm-auto-dev.md` 内で design review を呼ぶフローがあれば該当箇所

### Task 3: scaffolding 命名禁止 checklist の追加 (docs のみ — test 実装は #139 Task 1 で実施)

> **スコープ縮小** (S3-002 / S3-008 反映): 当初案では『CI grep スクリプトの自己 test を 1 件追加』も Task 3 に含んでいたが、Issue #139 Task 1 が同等の境界 test (`tests/test_no_scaffolding_in_prod.py` 相当) を既に明示しているため、本 Issue では **docs/code_review_checklist.md と CLAUDE.md リンクの整備のみ** に絞り、自動 test 実装は #139 に委譲する。マージ順序は **#139 → #140**、または #139 と並走させて #140 マージ前に test 実装が #139 PR で先行マージされていることを確認する。

**新規ドキュメント**: `docs/code_review_checklist.md`

```markdown
## 命名規則チェック (production code path)

PR レビュー時に以下を確認:

- [ ] `_Stub` で始まるシンボルが production import path にない
- [ ] `_Mock`, `_Dummy`, `_Placeholder` も同様
- [ ] `stub_`, `mock_` で始まる関数名が production にない
- [ ] `# TODO: replace with real ...`, `# placeholder for production` のコメントが残っていない
- [ ] config field の `getattr(cfg, "X", default)` で default が production で使われる場合は **fail-loud (raise) に変更**

### 適用範囲

- 対象: `baseline_reporag/`, `photon_mlx/`, `torch_ref/` の production import path
- **除外** (許容): `*/tests/**`, `bench/**`, `scripts/dev/**`, `demo/**`, `conftest.py`

### CI grep 例 (運用 PR レビュー時の手動コマンド)

```bash
git grep -nE '(_Stub|_Mock|_Dummy|_Placeholder)' baseline_reporag/ photon_mlx/ torch_ref/ ':!*/tests/**'
```

production code path に scaffolding が残ると **silent bug** (S7-001 型) を生む。CI 自動化 (pytest 経由の境界 test) は **#139 Task 1** で実装する。

> 注: `baseline_reporag/photon_pipeline.py::_StubTokenizer` は #138 修正後も fallback として残存している。本 Issue では「**新規追加禁止**」のルールを適用し、既存分は #139 で扱う。
```

CLAUDE.md からこの checklist を参照できるようにリンク追加。

### Task 4: random-init silent failure を発見する embedding norm WARNING

**変更先**: `photon_mlx/inference.py::PhotonInference.__init__`

```python
import logging
import mlx.core as mx

_logger = logging.getLogger(__name__)


def _check_weight_initialization(model: PhotonModel, threshold: float) -> None:
    """Random-init を検知し WARNING を出す。設計 §5 (S7-001 follow-up)。

    - 属性パスは ``photon_mlx/model.py:120`` の ``self.token_embed = nn.Embedding(...)`` と一致。
    - MagicMock 等 ``token_embed.weight`` が ``mx.array`` でない場合は **silent skip**
      (S3-001: 既存テスト 2 件で MagicMock model が渡される)。
    - 例外発生時もログのみで __init__ 全体を fail させない (silent skip)。
    """
    try:
        embed_attr = getattr(model, "token_embed", None)
        if embed_attr is None:
            return
        weight = getattr(embed_attr, "weight", None)
        if not isinstance(weight, mx.array):
            return  # MagicMock や非標準モデルは skip
        norm_std = float(mx.std(weight).item())
    except Exception as exc:  # pylint: disable=broad-except
        _logger.debug(
            "skip embedding init check (reason=%s)", type(exc).__name__
        )
        return

    # threshold は PhotonConfig で設定可能 (詳細は下記校正方針)
    if norm_std > threshold:
        _logger.warning(
            "PHOTON embedding has high variance (σ=%.4f, threshold=%.4f) — "
            "possibly random-init. If this is unexpected, check "
            "model.checkpoint_path is set and load succeeded "
            "(Issue #135 / S7-001).",
            norm_std,
            threshold,
        )
```

**閾値校正方針** (S1-004 反映):

1. 閾値はハードコードせず config field として設定可能に。**field 設置先は設計フェーズで確定**:
   - 候補 (a) `torch_ref/config.py::ModelConfig` 拡張 (静的モデル設定)
   - 候補 (b) `photon_mlx/session.py::WorkingMemoryConfig` 拡張 (運用 knob)
   - 候補 (c) 新規 `PhotonRuntimeChecksConfig` 導入 (起動時 sanity 専用)
2. 既存の `configs/photon_*.yaml` 全件 (現時点では `photon_600m_paper.yaml`, `photon_long_context.yaml`, `photon_small.yaml`, `photon_tiny.yaml`, `photon_tiny_recgen.yaml`) で field 未指定でも **load が成功する (デフォルト値を持つ)** ことを保証する。
3. 初期暫定値は 0.3 (Glorot init を想定したヒューリスティック) とし、Issue #135 で trained checkpoint が出来次第、実測値で再校正。
4. test は **挙動 test** に留める（高 σ で WARNING / 低 σ で WARNING なし / MagicMock model で例外を出さない）。閾値の絶対値を assert しない。
5. 校正データが揃うまでは **WARNING ログのみ**。`raise` への昇格は別 Issue で扱う。

**セキュリティ方針** (S3-011 反映):

- ログ出力は **σ (スカラー float) と threshold のみ**。embedding tensor 値・サンプル要素・weight matrix の内容はログに出さない。
- Issue #58 CB-002 / #64 CB-003 (`type(exc).__name__` のみログ出力する方針) と同方針。
- code review 時に「WARNING ログに weight tensor 値そのものが書き出されないこと」を確認する。

**テスト時 WARNING 抑制** (S3-007 反映):

- 既存 28 件超の `PhotonInference(...)` 呼び出しテストは多くが random-init 状態であり、Task 4 実装後にすべてで WARNING が pytest 出力に流れる。
- 対策: `photon_mlx/tests/conftest.py` および `baseline_reporag/tests/conftest.py` (現状未作成のため、必要なら新規) で `caplog`/`logging` レベルを `ERROR` に上げる autouse fixture を導入、もしくは test 用 cfg で閾値を `float('inf')` に設定して WARNING を発火させない方針を採る。
- 具体実装は設計フェーズで決定。

### Task 5: CLAUDE.md スラッシュコマンド表更新 (S3-003 反映)

**変更先**: `CLAUDE.md` (line 165-181 のスラッシュコマンド表)

**追加 4 行**:

| コマンド | 説明 |
|----------|------|
| `/multi-stage-issue-review` | Issue記載内容の多段階レビュー（通常→影響範囲）×2回と指摘対応を自動実行 |
| `/multi-stage-design-review` | 設計書の4段階レビュー（通常→整合性→影響分析→セキュリティ）と指摘対応を自動実行 |
| `/pm-auto-issue2dev` | Issueレビューから実装完了まで完全自動化 |
| `/pm-auto-design2dev` | 設計レビューから実装完了まで完全自動化 |

(`/uat`, `/orchestrate` 等の他コマンドは既存表に掲載済み)

## 受入条件

- [ ] **Task 1**: 両 skill (`multi-stage-design-review.md`, `multi-stage-issue-review.md`) の description に「Codex 担当 Stage 必須」明記。`multi-stage-issue-review.md` の Stage 5-8 自動スキップ判定を廃止し、1回目 Must Fix 0 件でも Codex Stage を実行することを明記。test 配置先 = `tests/test_skill_descriptions.py` (新規) で string-existence test 各 1 ケース (S3-005)。
- [ ] **Task 2**: `/pm-auto-issue2dev` / `/pm-auto-design2dev` の completion report テンプレートに Stage 別 finding 数表 (Issue review = Stage 1/3/5/7、Design review = Stage 1-4) が含まれることの test fixture。`reviewer="codex"` 検証ロジックを通る test 1 件以上 (apply-result.json は対象外)。欠落・`reviewer!="codex"` ケースが WARNING + completion report 記録になることも fixture で確認する。
- [ ] **Task 3**: `docs/code_review_checklist.md` 存在 + CLAUDE.md からのリンク存在確認。**CI grep 自動化 test は実装しない (#139 Task 1 へ委譲)**。
- [ ] **Task 4**: `_check_weight_initialization` を `PhotonInference.__init__` から呼び出し、(a) 高 σ (random-init) で WARNING / (b) 低 σ (擬似 trained) で WARNING なし / (c) MagicMock model 渡しで例外を出さず WARNING も出さない の test 3 件 (`photon_mlx/tests/test_inference.py`)。閾値の絶対値は assert しない (S1-004 / S3-001)。
- [ ] **Task 5**: CLAUDE.md スラッシュコマンド表に上記 4 skill が記載されていることの string-existence test (`tests/test_skill_descriptions.py` に追加、S3-003)。
- [ ] **既存テスト無回帰**: `photon_mlx/tests/test_inference.py:368` および `baseline_reporag/tests/test_photon_pipeline.py:1860` の MagicMock を使う既存 test が引き続きパスすること (S3-001)。
- [ ] **品質チェック** (CLAUDE.md 品質チェック表準拠):
  - [ ] `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` 全パス (既知 pre-existing failure 2 件: `tests/test_generate_training_corpus.py` を除く)
  - [ ] `ruff check .` 警告 0 件
  - [ ] `ruff format --check .` 差分なし

## 後方互換性 / ロールバック (S1-008 / S3-010 反映)

- 既存の `--skip-stage` フラグ自体は保持し、Codex stage を skip した場合は WARNING を出す段階を 1 リリース挟む。**強制 NG (raise / exit 1) 化は次回 Issue で扱う**。
- pm-auto-issue2dev / pm-auto-design2dev の Phase 完了判定でも、初期実装は「Codex stage 未実施を WARNING + completion report に明示」とする。
- Task 4 の embedding norm WARNING は **log のみ**。raise への昇格は #135 trained checkpoint で校正完了後に別 Issue で扱う。
- ロールバック手順:
  - skill 文言修正と Phase 完了判定はファイル単位で revert 可能。
  - Task 4 は `_check_weight_initialization` 呼び出しを 1 行コメントアウトで回避可能。
  - 閾値 field を ModelConfig 拡張 (候補 a) に置く場合: yaml で field 未指定時 default が適用、本 Issue revert 後も既存 yaml は load 可能 (`_set_fields` の warning 経路、`torch_ref/config.py:170-179`)。
  - 閾値 field を WorkingMemoryConfig 拡張 (候補 b) に置く場合: strict validator のため revert 後は yaml から該当行を削除する作業が必要。
  - 設計フェーズで field 設置先を確定後、ロールバック手順の最終形を決める。

## 並列性 / 依存関係

- **#139 と並走 (Task 3 は #139 先行が必須)**: 本 Issue Task 3 は docs のみ、CI grep 自動化 test は #139 Task 1 で実装。**マージ順序は #139 → #140**、または #140 PR で `tests/test_no_scaffolding_in_prod.py` の存在を前提条件とする。
- **#135 と並走可能**: Task 4 の閾値再校正は #135 trained checkpoint 完了後に別 Issue で実施するため、本 Issue 単独では暫定値 (0.3) のまま完了可能。
- **#138 は独立で既解消** (PR #138 マージ済み: `b19e8db fix(photon): load real HF tokenizer in _build_photon_deps`)。

## 影響ファイル

- `.claude/commands/multi-stage-design-review.md` (Task 1)
- `.claude/commands/multi-stage-issue-review.md` (Task 1)
- `.claude/commands/pm-auto-issue2dev.md` (Task 2)
- `.claude/commands/pm-auto-design2dev.md` (Task 2)
- (任意) `.claude/commands/pm-auto-dev.md` (Task 2)
- `docs/code_review_checklist.md` (新規、Task 3)
- `CLAUDE.md` (Task 3 リンク追加 + Task 5 スラッシュコマンド表更新)
- `photon_mlx/inference.py` (Task 4 norm check)
- `photon_mlx/tests/test_inference.py` (Task 4 norm check の test、S3-001 MagicMock 既存 test 影響確認)
- `photon_mlx/tests/conftest.py` (Task 4 WARNING 抑制 fixture、S3-007)
- `baseline_reporag/tests/test_photon_pipeline.py` (S3-001 MagicMock 既存 test 影響確認)
- `baseline_reporag/tests/conftest.py` (新規、必要時。Task 4 WARNING 抑制 fixture、S3-007 / S5-004)
- `torch_ref/config.py` または `photon_mlx/session.py` 等 (閾値 field 追加先 — 設計フェーズで確定、3 候補)
- `tests/test_skill_descriptions.py` (新規、Task 1 / Task 5 用 string-existence test、S3-005)

## 関連

- 元: S7-001 (#135 commit `2dbf458 fix(photon_pipeline): load checkpoint in _build_photon_deps (#135 / S7-001)`)
- 緊急: #138 (tokenizer mismatch、本 Issue とは独立で **既解消**)
- 並列: #139 (Stub/Mock audit、テスト補強 — **本 Issue Task 3 の test 実装は #139 に委譲**)、#135 (本格再学習 — Task 4 閾値再校正の前提)
