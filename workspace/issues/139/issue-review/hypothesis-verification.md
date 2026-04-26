# Issue #139 仮説検証レポート

**対象**: Issue #139 "test(photon): Stub/Mock pattern audit + real-weight integration test (S7-001 follow-up)"
**検証日**: 2026-04-26
**ブランチ**: feature/issue-139-stub-audit (HEAD: 8e677ca)

---

## サマリー

| # | Claim | 判定 |
|---|-------|------|
| 1 | `test_photon_pipeline.py` は 1715 行 | **Rejected** |
| 2 | テストは MagicMock 中心で実 PhotonModel + 実 weight を通らない | **Confirmed** |
| 3 | `_StubTokenizer` 等 scaffolding 命名が production path に残存 (#138 後) | **Confirmed** |
| 4 | commit 2dbf458 が S7-001 (random-init weight in production) を修正 | **Confirmed (注意あり)** |
| 5 | `getattr(cfg, "X", default)` で default が silently 効いている箇所がある | **Confirmed** |
| 6 | 「設計 Must Fix を CI で固定する」既存仕組みが無い | **Partially Confirmed** |
| 7 | `baseline_reporag/photon_pipeline.py` に Stub 命名シンボルが残る | **Confirmed** |
| 8 | #138 は既にマージ済みで本 Issue の前提条件を満たす | **Confirmed (注意あり)** |
| 9 | 「smallest available trained checkpoint」が利用可能 | **Unverifiable** |

---

## Claim 1: `test_photon_pipeline.py` は 1715 行

**Verdict**: **Rejected**

**Evidence**:
- `wc -l baseline_reporag/tests/test_photon_pipeline.py` → **4453 行**
- Issue 本文に記載の「1715 行」は実態の約 2.6× 過小

**Notes**:
- Issue 文中の「1715 行」は記載時点の旧スナップショットの可能性が高い。
- レビュー反映時に **正確な行数 (4453)** に更新すべき。
- これは数値の誤りであり、規模感の前提が変わるため、Task 1 の audit 工数見積に影響する可能性がある。

---

## Claim 2: テストは MagicMock 中心で、実 PhotonModel + 実 weight を通る経路が一度も走っていない

**Verdict**: **Confirmed**

**Evidence**:
- `grep -c "MagicMock\|mock\." baseline_reporag/tests/test_photon_pipeline.py` → **54 件**
- `grep "PhotonModel(" baseline_reporag/tests/test_photon_pipeline.py` → **0 件** (PhotonModel の直接インスタンス化なし)
- L5: `from unittest.mock import MagicMock, patch`
- L419-425: TestBuildPhotonDeps fixture が `MagicMock()` で photon_cfg/tokenizer を構築
- 実 checkpoint をロードする test は皆無

**Notes**: Issue の主張の核心となる事実は確認済み。

---

## Claim 3: `_StubTokenizer` 等 scaffolding 命名が production path に残存 (#138 後)

**Verdict**: **Confirmed**

**Evidence** (production code 内で `_Stub*` を持つ箇所):
- `baseline_reporag/photon_pipeline.py:328` — comment に `_StubTokenizer`
- `baseline_reporag/photon_pipeline.py:340` — warning log: `_StubTokenizer (test/dev only...)`
- `baseline_reporag/photon_pipeline.py:451` — `class _StubTokenizer:`
- `baseline_reporag/photon_pipeline.py:465-466` — `def _get_stub_tokenizer(vocab_size: int) -> _StubTokenizer:`
- `baseline_reporag/photon_pipeline.py:474` — docstring に `_StubTokenizer`

**注**: `photon_mlx/tests/conftest.py:15, :30` にも存在するが、これは test 配下のため対象外。

**Notes**:
- #138 (b19e8db) は `_StubTokenizer` を完全に除去せず、**fallback として残し warning を出す** 形で対処した。
- Issue #139 のゴール 1 (production path から排除) は #138 では未達。

---

## Claim 4: commit `2dbf458` が S7-001 (random-init weight in production) を修正

**Verdict**: **Confirmed (重要な注意あり)**

**Evidence**:
- commit message:
  ```
  fix(photon_pipeline): load checkpoint in _build_photon_deps (#135 / S7-001)

  Until #135, _build_photon_deps built a fresh PhotonModel and never
  loaded any trained weights — every PHOTON eval was silently running
  against random-initialised parameters.
  ```
- 当該 commit で `model.checkpoint_path` config 項目を追加し、未設定時は WARNING 出力。

**⚠ 重要**: その後の `b19e8db` (#138 マージ) で **`checkpoint_path` 関連処理が一部 revert または再構成されている可能性** が報告された。
本 Issue が想定する「現在 prod で使われる checkpoint loading path」が S7-001 fix 当時と同じかは Stage 1 で再確認する必要がある。

**Notes**:
- 「S7-001 解消で発見した」という Issue の前提自体は事実。
- ただし「現在の photon_pipeline.py で random-init bug が完全に解消されているか」は Issue 内では断定できておらず、本 Issue Task 2 (real-weight integration test) を実装してはじめて検証可能。

---

## Claim 5: `getattr(cfg, "X", default)` で「常に default が選ばれている」箇所が grep で抽出可能

**Verdict**: **Confirmed**

**Evidence** (production: 8 件):
1. `baseline_reporag/photon_pipeline.py:242` — `getattr(cfg, "session_memory", None)`
2. `baseline_reporag/photon_pipeline.py:283` — `head_dim=getattr(cfg.model, "head_dim", 64)`
3. `baseline_reporag/photon_pipeline.py:284` — `max_position_embeddings=getattr(cfg.model, "max_position_embeddings", 2048)`
4. `baseline_reporag/photon_pipeline.py:285` — `rope_theta=getattr(cfg.model, "rope_theta", 1_000_000.0)`
5. `baseline_reporag/photon_pipeline.py:349` — `safe_recgen_enabled = getattr(cfg.get("inference"), "safe_recgen_enabled", True)`
6. `baseline_reporag/pipeline_factory.py:52` — `provider = getattr(cfg.model, "provider", None) or "baseline"`
7. `baseline_reporag/pipeline.py:212` — `answering_cfg = getattr(cfg, "answering", None)`
8. `baseline_reporag/photon_pipeline.py:1109` — `answering_cfg = getattr(cfg, "answering", None)` (5 と重複に近い)

**Notes**:
- すべて warning なし silent default。
- ただし「常に default が選ばれている」かは grep だけでは断定できない (yaml で明示されているケースもある)。本 Issue Task 3 ではこの 8 件それぞれに対して個別判定が必要。

---

## Claim 6: 「設計 Must Fix が実装に反映されているか」を CI で固定する仕組みが無い

**Verdict**: **Partially Confirmed**

**Evidence**:
- `tests/test_pipeline_factory_yaml_invariants.py` が既に存在: yaml invariant (e.g. `reranker.model_id` 不変宣言、L40-52) を pin している。
- `.github/workflows/` には `weekly_eval.yml` のみ — architectural invariant CI なし。
- 「no stub in prod」「required field 必須化」「no random-init weight」 の test はいずれも未実装。

**Notes**:
- ゼロではないが、本 Issue が言う「scaffolding pattern を CI で固定する仕組み」 自体は存在しない。
- 既存の `test_pipeline_factory_yaml_invariants.py` を **拡張** することは Issue Task 3 の方針と整合。新規ファイル `tests/test_no_scaffolding_in_prod.py` を追加するアプローチも妥当。

---

## Claim 7: `baseline_reporag/photon_pipeline.py` の Stub の rename or 廃止が必要

**Verdict**: **Confirmed**

**Evidence**: Claim 3 と同じ 4 箇所の `_Stub*` シンボルが production file に残存。

**Notes**:
- Issue 提案の「`_StubTokenizer` → `_DevTokenizer`」リネーム案は test/dev only の意図を明示する点で妥当。
- ただし Claim 4 注意点を踏まえると、production 流路に到達することがそもそも事故であり、「リネームして残す」より「production 必須 yaml にして欠落時は raise」のほうが構造的に強い。**Stage 1 で議論する価値あり**。

---

## Claim 8: #138 は既にマージ済みで本 Issue の前提条件を満たす

**Verdict**: **Confirmed (注意あり)**

**Evidence**:
- HEAD: `8e677ca Merge pull request #141 from Kewton/feature/issue-138-tokenizer-mismatch`
- `b19e8db fix(photon): load real HF tokenizer in _build_photon_deps (#138)` が main に取り込まれている。

**注意**: 上述 Claim 4 の通り、#138 が S7-001 fix の checkpoint_path 機構の一部に影響を与えている可能性がある。 **Stage 1 で「現状の checkpoint loading 仕様」を確定** する必要あり。

---

## Claim 9: 「smallest available trained PHOTON checkpoint」が integration test で使える

**Verdict**: **Unverifiable**

**Evidence**:
- `configs/photon_tiny.yaml`, `photon_tiny_recgen.yaml`, `photon_small.yaml`, `photon_600m_paper.yaml`, `photon_long_context.yaml` で `checkpoint_root: "./checkpoints"` 参照。
- `./checkpoints/` ディレクトリ自体はリポジトリに含まれず、download URL も見当たらない。
- 「smallest checkpoint」の所在は Issue 提案の中で曖昧。

**Notes**:
- これは **Stage 1 通常レビューでの Must Fix 候補**。「nightly CI で実行」 と書いてあるが、CI 環境で checkpoint をどう用意するかが未定義。
- 代替案: 「**最小サイズの PhotonConfig (e.g. vocab=256, hidden=64, layers=2) で 1 step 学習 → checkpoint 保存 → load して 1 query」というセルフホスト型 e2e にすれば repo 内で完結する。

---

## Stage 1 への申し送り事項

**Rejected の修正**:
- Claim 1: 行数 1715 → 4453 に修正。

**Confirmed (注意あり) の深掘り依頼**:
- Claim 4 / Claim 8: 「#138 マージ後の現在の checkpoint loading 仕様」を Issue 本文で明示する。 単に「S7-001 で random-init bug を発見した」だけでは現状が読めない。
- Claim 9: 「real-weight integration test で使う checkpoint をどう用意するか」が未定義。本 Issue 受入条件に **checkpoint 用意手段** を明記すべき (Must Fix 候補)。

**Confirmed のうち追加検討点**:
- Claim 7: Stub のリネームより「未設定時 raise」の方が強い。Stage 1 で議論。
- Claim 5: 8 件の getattr default のうちどれが本当に「常に default」か、Stage 1 で個別判定する必要がある。

**新規 Issue 化候補 (本 Issue では扱わない)**:
- 上記 Claim 5 の 8 件のうち、「実は production yaml で明示済みで問題なし」のものは個別 fix から外す。
