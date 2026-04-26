## 背景

#135 の S7-001 (PHOTON eval が random-init weight で動作) は **`/multi-stage-issue-review` Stage 7 (Codex 影響範囲レビュー 2回目)** で初めて発見された。Stage 1-4 (opus 単独) では見落とされていた。

> 注: 本リポジトリの finding ID 命名体系 — `S7-001` は `/multi-stage-issue-review` Stage 7 (Codex) の最初の finding、`DRN-NNN` は `/multi-stage-design-review` Stage N の finding を指す。`/multi-stage-design-review` は **4 ステージ構成** (Stage 1-2 = opus、**Stage 3-4 = Codex**) であり、`/multi-stage-issue-review` は **8 ステージ構成** (Stage 1-4 = opus、**Stage 5-8 = Codex**)。

これは個別 bug を超えた **品質保証プロセスの構造課題**:

- opus 単独レビューは「設計の論理整合性」は確認できるが、「実装と設計のギャップ」(実コードに silent bug が残っているか) を発見しにくい
- Codex クロスレビュー (design review Stage 3-4 / issue review Stage 5-8) が補完するこの観点が **オプショナルだったため、Codex stage を実施しないケースで bug が混入** した可能性
- scaffolding 命名 (`_Stub`, `_Mock` 等) が production に残ること自体が **code review checklist で禁止項目化されていなかった**

本 Issue で**プロセス・ドキュメント面**の再発防止を恒久化する。

## ゴール

1. `/multi-stage-design-review` skill の **Codex 担当 Stage (3-4) 必須化** および `/multi-stage-issue-review` skill の **Codex 担当 Stage (5-8) 必須化**
2. `/pm-auto-issue2dev` Phase 3 / `/pm-auto-design2dev` の対応 Phase で **Codex stage 結果ファイルの存在 + `reviewer="codex"` フィールド検証** を strict 化
3. **scaffolding 命名禁止** を code review checklist に明示 (production import path のみ、tests/bench/scripts は適用除外)
4. "random-init で silent に動く" 型 bug を発見する **embedding norm WARNING** を `PhotonInference` 起動ログに追加 (閾値は config 化、初期は WARNING のみで raise しない)

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

### Task 2: `/pm-auto-issue2dev` / `/pm-auto-design2dev` Phase 完了判定の strict 化

**現状**: `/pm-auto-issue2dev` Phase 3 は `/multi-stage-design-review {issue_number}` を呼ぶのみで、Codex stage の有無や finding 数を後続 Phase が確認していない。`/pm-auto-design2dev` も同様。

**変更**: 各 PM コマンドの該当 Phase 完了判定を以下に再定義 (S1-005 反映):

1. **結果ファイル存在確認**:
   - design review: `workspace/issues/{issue_number}/multi-stage-design-review/stage3-review-result.json` および `stage4-review-result.json`
   - issue review: `workspace/issues/{issue_number}/issue-review/stage5-review-result.json` および `stage7-review-result.json`
2. **`reviewer` フィールドが `"codex"` であること**を JSON パースで確認 (Claude による上書きが起きていないか検証)
3. **completion report に Stage 別 (Stage 1-4 design / Stage 1-8 issue) の Must Fix / Should Fix / Nice to Have 件数表を必ず記載**

**スコープ** (S1-009 反映):
- `.claude/commands/pm-auto-issue2dev.md` Phase 3
- `.claude/commands/pm-auto-design2dev.md` の対応 Phase
- (任意) `.claude/commands/pm-auto-dev.md` 内で design review を呼ぶフローがあれば該当箇所

### Task 3: scaffolding 命名禁止 checklist の追加

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

### CI grep 例

```bash
git grep -nE '(_Stub|_Mock|_Dummy|_Placeholder)' baseline_reporag/ photon_mlx/ torch_ref/ ':!*/tests/**'
```

production code path に scaffolding が残ると **silent bug** (S7-001 型) を生む。CI で grep ベースの境界 test を回す (#139 Task 1 参照)。

> 注: `baseline_reporag/photon_pipeline.py::_StubTokenizer` は #138 修正後も fallback として残存している。本 Issue では「**新規追加禁止**」のルールを適用し、既存分は別 Issue (#139 など) で扱う。
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

    属性パスは ``photon_mlx/model.py:120`` の ``self.token_embed = nn.Embedding(...)`` と一致。
    """
    embed_weight = model.token_embed.weight
    norm_std = float(mx.std(embed_weight).item())

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

1. 閾値はハードコードせず `PhotonConfig.model.embedding_random_init_threshold` (仮称) として設定可能に。
2. 初期暫定値は 0.3 (Glorot init を想定したヒューリスティック) とし、Issue #135 で trained checkpoint が出来次第、実測値で再校正。
3. test は **挙動 test** に留める（trained checkpoint で WARNING が出ない / 故意の random-init モデルで WARNING が出る）。閾値の絶対値を assert しない。
4. 校正データが揃うまでは **WARNING ログのみ**。`raise` への昇格は別 Issue で扱う。

## 受入条件

- [ ] **Task 1**: 両 skill の description に「Codex 担当 Stage 必須」明記。skill description の必須キーワード存在を確認する pytest または string-existence test (1 ファイルにつき 1 ケース)。
- [ ] **Task 2**: `/pm-auto-issue2dev` / `/pm-auto-design2dev` の completion report テンプレートに Stage 別 finding 数表 (Stage 1-4 / Stage 1-8) が含まれることの test fixture。`reviewer="codex"` 検証ロジックを通る test 1 件以上。
- [ ] **Task 3**: `docs/code_review_checklist.md` 存在 + CLAUDE.md からのリンク存在 + 上記 CI grep スクリプトの自己 test (production path に新規 scaffolding 名がないことの test を 1 件追加)。
- [ ] **Task 4**: `_check_weight_initialization` を `PhotonInference.__init__` から呼び出し、(a) 高 σ (random-init) で WARNING / (b) 低 σ (擬似 trained) で WARNING なし の test 2 件 (`photon_mlx/tests/test_inference.py`)。閾値の絶対値は assert しない。
- [ ] **品質チェック** (CLAUDE.md 品質チェック表準拠):
  - [ ] `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` 全パス (既知 pre-existing failure 2 件: `tests/test_generate_training_corpus.py` を除く)
  - [ ] `ruff check .` 警告 0 件
  - [ ] `ruff format --check .` 差分なし

## 後方互換性 / ロールバック (S1-008 反映)

- 既存の `--skip-stage` フラグ自体は保持し、Codex stage を skip した場合は WARNING を出す段階を 1 リリース挟む。**強制 NG (raise / exit 1) 化は次回 Issue で扱う**。
- pm-auto-issue2dev / pm-auto-design2dev の Phase 完了判定でも、初期実装は「Codex stage 未実施を WARNING + completion report に明示」とする。
- Task 4 の embedding norm WARNING は **log のみ**。raise への昇格は #135 trained checkpoint で校正完了後に別 Issue で扱う。
- ロールバック: skill 文言修正と Phase 完了判定はファイル単位で revert 可能。Task 4 は `_check_weight_initialization` 呼び出しを 1 行コメントアウトで回避可能にする。

## 並列性

#135 / #139 と並列可能 (process 変更 + ドキュメント中心)。#138 は本 Issue とは独立で **既に CLOSED** (PR #138 マージ済み: `b19e8db fix(photon): load real HF tokenizer in _build_photon_deps`)。

## 影響ファイル

- `.claude/commands/multi-stage-design-review.md`
- `.claude/commands/multi-stage-issue-review.md`
- `.claude/commands/pm-auto-issue2dev.md`
- `.claude/commands/pm-auto-design2dev.md`
- (任意) `.claude/commands/pm-auto-dev.md`
- `docs/code_review_checklist.md` (新規)
- `CLAUDE.md` (リンク追加)
- `photon_mlx/inference.py` (norm check 追加)
- `photon_mlx/tests/test_inference.py` (norm check の test)
- `torch_ref/config.py` または `photon_mlx/session.py` 等 (閾値 field 追加先 — 設計フェーズで確定)

## 関連

- 元: S7-001 (#135 commit `2dbf458 fix(photon_pipeline): load checkpoint in _build_photon_deps (#135 / S7-001)`)
- 緊急: #138 (tokenizer mismatch、本 Issue とは独立で **既解消**)
- 並列: #139 (Stub/Mock audit、テスト補強)、#135 (本格再学習)
