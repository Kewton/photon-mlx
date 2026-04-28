# Issue #140 仮説検証レポート

## 検証対象

Issue #140 ("process(review): Codex multi-stage design review 必須化 + scaffolding 命名禁止 checklist (S7-001 follow-up)") に含まれる仮説・前提条件をコードベースと照合する。

## 検証結果サマリー

| # | 仮説/主張 | 判定 | 重要度 |
|---|----------|------|-------|
| H1 | S7-001 は「設計レビュー Stage 7 (Codex クロスレビュー)」で初めて発見された | **Rejected** | High |
| H2 | `/multi-stage-design-review` は Stage 7-8 が Codex クロスレビュー | **Rejected** | High |
| H3 | Stage 7-8 が optional のため、Stage 7 を実施しないケースで bug が混入した | **Partially Rejected** | High |
| H4 | scaffolding 命名 (`_Stub`, `_Mock`) が production code path に残っている | **Confirmed** | High |
| H5 | scaffolding 命名禁止が code review checklist で禁止項目化されていない | **Confirmed** | High |
| H6 | PhotonInference の embedding access path は `model.embed_tokens.weight` | **Rejected** | Medium |
| H7 | `docs/code_review_checklist.md` は未作成 | **Confirmed** | Medium |
| H8 | #138 は本 Issue とは独立で先行解消すべき | **Confirmed** (already CLOSED) | Low |
| H9 | #139 (Stub/Mock audit) と並列可能 | **Confirmed** (still OPEN) | Low |

---

## 詳細検証

### H1: S7-001 は「設計レビュー Stage 7 (Codex クロスレビュー)」で初めて発見された

**Issue 主張**:
> #135 の S7-001 (PHOTON eval が random-init weight で動作) は **設計レビュー Stage 7 (Codex クロスレビュー)** で初めて発見された。

**コードベース照合**:
- `.claude/commands/multi-stage-design-review.md` (line 49-56) は **4 ステージのみ**:
  - Stage 1: 通常レビュー（設計原則）— opus
  - Stage 2: 整合性レビュー — opus
  - Stage 3: 影響分析レビュー — **Codex**
  - Stage 4: セキュリティレビュー — **Codex**
- 設計レビューに Stage 7 は存在しない。
- 一方 `.claude/commands/multi-stage-issue-review.md` (line 60-72 の挙動) は 8 ステージ構成で、Stage 5-8 が Codex（Stage 5 = 通常レビュー 2回目、Stage 7 = 影響範囲レビュー 2回目）。
- commit `2dbf458` メッセージ: "fix(photon_pipeline): load checkpoint in `_build_photon_deps` (#135 / S7-001)" — finding ID `S7-001` の "S7" は **Stage 7 由来の通し番号**である可能性が高いが、それは **issue review** Stage 7 (Codex 影響範囲 2回目) を指している、または独自の番号体系。

**判定**: **Rejected**。S7-001 が設計レビュー Stage 7 で発見されたという主張は、設計レビューが 4 stages しかないため事実と合わない。Issue が `multi-stage-design-review` (4 stages) と `multi-stage-issue-review` (8 stages) を混同している可能性が高い。

**Stage 1 への申し送り**: Issue Task 1 の文言「Stage 7-8 必須化」は **multi-stage-design-review** には適用不可。正しくは **Stage 3-4 (Codex)** の必須化、または **multi-stage-issue-review の Stage 5-8** の必須化のいずれか。**両方の skill が対象である可能性**が高いので、本 Issue では対象 skill を明示的に切り分ける必要がある。

---

### H2: `/multi-stage-design-review` は Stage 7-8 が Codex クロスレビュー

**Issue 主張** (Task 1):
> 現状: Stage 1-6 が opus 単独、Stage 7-8 が Codex クロスレビューだが、運用上 7-8 を skip するケースがある。

**コードベース照合**:
- `.claude/commands/multi-stage-design-review.md` line 53-56:
  ```
  | Stage 1-2（通常・整合性） | Claude opus
  | Stage 3-4（影響分析・セキュリティ） | Codex
  ```
- 全 4 ステージのみ。Stage 5-8 は存在しない。

**判定**: **Rejected**。現行 skill は **Stage 3-4 が Codex**。

**Stage 1 への申し送り**: Task 1 は文言を修正する必要あり。`/multi-stage-design-review` の Stage 3-4 必須化、または `/multi-stage-issue-review` の Stage 5-8 必須化（あるいは両方）として再定義する。

---

### H3: Stage 7-8 が optional のため、Stage 7 を実施しないケースで bug が混入した

**Issue 主張**:
> Codex クロスレビューが補完するこの観点が **オプショナルだったため、Stage 7 を実施しないケースで bug が混入** した可能性

**コードベース照合**:
- `multi-stage-design-review.md` Stage 3-4 (Codex) は line 43-45 で「禁止事項」として強調:
  > Stage 3-4 は必ず `commandmatedev send ... --agent codex` でCodexに委譲すること。Claude サブエージェント（Agent tool）で代替実行してはならない。
- ただし line 16-17 で `--skip-stage=3,4` の使用例が示されており、CLI 上 skip 可能。
- `multi-stage-issue-review.md` も同様に `--skip-stage=5,6,7,8` 例があり、Codex stage を skip 可能 (line 12-13 of skill body)。
- `pm-auto-issue2dev.md` Phase 3 は単に `/multi-stage-design-review {issue_number}` を呼ぶのみ。Codex stage の有無や finding 数を確認するロジックは存在しない (line 91-113)。

**判定**: **Partially Rejected**。「Stage 7 を skip」は文言として誤りだが、「Codex stage を skip 可能でかつ pm-auto-issue2dev が確認していない」という構造課題は **Confirmed**。

**Stage 1 への申し送り**: Task 1 / Task 2 は対象 stage 番号を **3-4 (design review) / 5-8 (issue review)** に修正。Phase 3 strict 化の中身は「Codex finding 数 (Stage 3-4 / Stage 5-8) が completion report に明示される」という運用要件として再定義。

---

### H4: scaffolding 命名 (`_Stub`, `_Mock`) が production code path に残っている

**コードベース照合**:
- `baseline_reporag/photon_pipeline.py:451` に `class _StubTokenizer` が定義され、line 340 でログ警告付きで fallback として使用される。
- `bench/issue61_prune_batch.py:51` に `class _StubTokenizer` (ベンチ専用)。
- `photon_mlx/tests/conftest.py:15` (テスト専用)。

**判定**: **Confirmed**。production import path (`baseline_reporag.photon_pipeline`) に `_StubTokenizer` が現存。Issue #138 でも触れられているが、依然として fallback として残置されている。

---

### H5: scaffolding 命名禁止が code review checklist で禁止項目化されていない

**コードベース照合**:
- `docs/` 直下: `deployment.md`, `troubleshooting.md`, `tutorial.md` のみ。
- `docs/code_review_checklist.md` は存在しない。
- `CLAUDE.md` 内検索: 「scaffolding」「_Stub」「_Mock」のいずれもヒットなし（必要なら別途 grep を回す）。

**判定**: **Confirmed**。

---

### H6: PhotonInference の embedding access path は `model.embed_tokens.weight`

**Issue 主張** (Task 4):
> ```python
> embed_weight = model.embed_tokens.weight  # 例
> ```

**コードベース照合**:
- `photon_mlx/model.py:120`: `self.token_embed = nn.Embedding(t.vocab_size, m.base_embed_dim)`
- `nn.Embedding` の MLX 実装は属性 `.weight` を保持する。
- 実際のアクセスパスは `model.token_embed.weight`、**`embed_tokens` ではない**。

**判定**: **Rejected**。Issue Task 4 のサンプルコードは属性名が誤り。実装時は `model.token_embed.weight` を使用する。

**Stage 1 への申し送り**: Issue 本文を `model.token_embed.weight` に修正。

---

### H7: `docs/code_review_checklist.md` は未作成

**判定**: **Confirmed** (H5 と同根)。

---

### H8: #138 は先行解消すべき

**判定**: **Confirmed**。`gh issue view 138` の `state` は `CLOSED` (PR #138 マージ済 — `b19e8db fix(photon): load real HF tokenizer in _build_photon_deps`)。本 Issue 着手の前提は満たされている。

---

### H9: #139 と並列可能

**判定**: **Confirmed**。`gh issue view 139` の `state` は `OPEN`。#139 は test 補強系、本 Issue は process / docs 系で、ファイル依存の競合は最小限。

---

## Stage 1 へ申し送る要点

1. **Issue Task 1 の対象 stage は誤り**: `multi-stage-design-review` は 4 stages、Codex 担当は **Stage 3-4**。Issue を `multi-stage-design-review` の Stage 3-4 必須化、または `multi-stage-issue-review` の Stage 5-8 必須化として再定義する必要がある。**両 skill 同時対応**にするか、対象を明示するかをレビュアーが判断。
2. **Issue Task 2 の Phase 3 strict 化**: Codex finding 数の確認は **Stage 3-4** (or 5-8 if multi-stage-issue-review) に対応する旨を再定義。
3. **Issue Task 4 のサンプルコードの属性名誤り**: `model.embed_tokens.weight` → `model.token_embed.weight`。
4. **norm 閾値の校正方法**: 「訓練済み weight: norm ≈ 0.02 σ ≈ 0.04、random-init: σ ≈ 0.5」とあるが、PHOTON checkpoint は現状ほぼ全モデルが random-init (Issue #135 が再学習を扱う) のため、校正データが不足。**閾値設定方針は別途明示**が必要。
5. **scaffolding 名禁止の例外定義**: tests/ 配下や bench/ 配下は scaffolding が許容されるべき (Issue では production path のみと記載があるが、CI で grep する場合の除外パターンを明示する必要あり)。
