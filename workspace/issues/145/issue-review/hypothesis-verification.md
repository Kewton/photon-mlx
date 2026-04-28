# Issue #145 仮説検証レポート

実施日: 2026-04-28
対象ブランチ: feature/issue-145-real-weight-test (main から派生、#135 マージ済み)

---

## 抽出した仮説/前提条件

| # | 主張カテゴリ | 内容 |
|---|------------|------|
| H1 | 前提条件 | "#139 Task 2 は #135 (S7-001 fix, commit `2dbf458` で導入された `model.checkpoint_path` 機構) を前提に設計されている" |
| H2 | 前提条件 | "2026-04-26 時点で #135 は `feature/issue-135-photon-retrain` 上のみ存在し main にマージされていない" |
| H3 | 前提条件 | "現行 main の `baseline_reporag/photon_pipeline.py` には `checkpoint_path` 設定も `load_checkpoint` 呼び出しも存在しない" |
| H4 | 設計仮説 | 候補方針 B: "`_build_photon_deps` 経由で load し query を流す" |
| H5 | 設計仮説 | "random-init detector 例: `pipeline.last_pruning_score_distribution.std() > 0.01`" — random だと std ≈ 0、学習後は非ゼロ |
| H6 | 仮説 | 「random-init weight が production で silently 動いていた」型の構造的事故を CI で検出可能にする |

---

## 検証結果

### H1: `model.checkpoint_path` 機構は #135 で導入されている

**判定**: Confirmed

**根拠**:
- `baseline_reporag/photon_pipeline.py:365` で `getattr(cfg.model, "checkpoint_path", None)` を読む
- `baseline_reporag/photon_pipeline.py:589` `_resolve_checkpoint_path` で `PHOTON_CHECKPOINT_ROOT` 内の path 解決
- `baseline_reporag/photon_pipeline.py:626-633` `load_checkpoint` を `photon_mlx.trainer` から lazy import
- 詳細: `configs/institutional_docs_photon_retrain.yaml:194` で実 ckpt path が指定されている

### H2: #135 は main マージ済み

**判定**: Rejected (Issue 起票時点 2026-04-26 では正しかったが現在は古い情報)

**根拠**:
- `git log main --oneline` 上位コミット:
  - `8c13517 Merge pull request #157 from Kewton/feature/issue-135-photon-retrain`
  - `2b7cc86 feat(training): adopt PHOTON institutional retrain step_003000 — Turn 5-6 NC 0% (#135)`
- 2026-04-28 時点で #135 は main にマージ完了。本 Issue の前提依存は解消済 → **着手可能**

### H3: 現行 main に `checkpoint_path` 設定 / `load_checkpoint` 呼び出しが存在しない

**判定**: Rejected (古い情報)

**根拠**:
- 現行 main の `baseline_reporag/photon_pipeline.py` には:
  - `checkpoint_path` への参照: 11 箇所 (line 355–417)
  - `load_checkpoint` 呼び出し: 1 箇所 (line 633)
- Issue 本文の「現行 main には存在しない」記述は #135 マージ前の状態を指しているため、レビュー反映で **「#135 マージ後の状態を前提とする」と更新する必要** がある (Stage 1 への申し送り)

### H4: `_build_photon_deps` 経由で load する候補方針 B は実現可能

**判定**: Confirmed

**根拠**:
- `baseline_reporag/photon_pipeline.py:256` `def _build_photon_deps(cfg: Config) -> dict[str, Any]:` 実在
- `_build_photon_deps` 内で `_resolve_checkpoint_path` → `_load_checkpoint_into_model` を呼び出すフロー (line 365–417 周辺)
- 候補方針 B は `_build_photon_deps(cfg)` を直接呼ぶ or pipeline factory 経由で wire する形で実装可能

### H5: random-init detector `pipeline.last_pruning_score_distribution.std()` が利用可能

**判定**: Rejected (該当 public API が存在しない)

**根拠**:
- `grep -rn "pruning_score" baseline_reporag/ photon_mlx/` の結果 0 件 (`pruning_score` という名前の属性/メソッドは公開されていない)
- 実コードに存在するのは:
  - `photon_mlx/inference.py:536` `_score_prune_candidates` (private method) — `(index, raw_score)` の tuple list を返す
  - `photon_mlx/inference.py:29` `weighted_hierarchical_score` (関数)
- pipeline 側に `last_pruning_score_distribution` のような **stats を保持する公開属性は存在しない**
- **影響**: Issue 受入条件 (3) の "random-init detector (例: pruning_score 分布の std > 閾値)" は現状コードベースで実装不能。代替案を検討する必要あり。
  - 代替案候補:
    - (a) `DriftMetrics` (`photon_mlx/session.py`) の値を検証 (random-init では特定パターンになる)
    - (b) checkpoint integrity ハッシュ (`photon_mlx.checkpoint` DR4-003 機構) で「load 済みかどうか」を verify
    - (c) 実 query での top-k 選択結果を「想定 chunk が含まれる」で検証 (semantic check)
    - (d) test 内で `_score_prune_candidates` を直接呼び、raw_score の std を検証 (private API 依存)

### H6: random-init silently 動作型の事故を検出できる

**判定**: Partially Confirmed

**根拠**:
- 構造的検出は **可能**: e.g., `_load_checkpoint_into_model` が呼ばれない code path / `checkpoint_path` 未設定で警告ログが出るが pipeline 起動が成功してしまう状態
- 既に `baseline_reporag/photon_pipeline.py:416-417` に warning log は存在 — テストは「warning が出る vs. checkpoint が load された」を判定する形になる
- ただし H5 で示した通り、**weight が実際に load されたか** を assert する一級 API は未整備のため、新規 test 内で何らかの「load 成功シグナル」を仕込む必要あり

---

## Stage 1 レビューへの申し送り事項

1. **Rejected 仮説 H2/H3 を Issue 本文から削除 or 更新**:
   - 「2026-04-26 時点で main にマージされていない」→ 「2026-04-28 時点で main にマージ済み (commit 8c13517)、本 Issue 着手可能」
   - 「現行 main には `checkpoint_path` も `load_checkpoint` も存在しない」→ 「main に `_build_photon_deps` 内で wire 済み (line 365-417, 626-633)」

2. **H5 が示す API 不整合を解消する案を Issue に追記**:
   - 受入条件 (3) "random-init detector" の具体策を、実在 API ベースで再定義する必要がある
   - 候補: (a) DriftMetrics 値検証、(b) checkpoint integrity hash、(c) semantic chunk 選択、(d) `_score_prune_candidates` 直接呼出 — どれを採用するか Stage 1-3 で議論

3. **候補方針 B 実装の wire-up 経路** (`_build_photon_deps` 直接呼出 vs pipeline_factory 経由) を Stage 3 影響範囲レビューで確定する

4. **依存関係ステータスの更新**:
   - "必須: #135 が main にマージされていること" → 充足済 (履歴記録のみ残す)
   - "推奨: #139 (Task 1 + Task 3) が先行マージされていること" → 状況確認が必要
