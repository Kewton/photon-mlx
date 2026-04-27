# Phase A 完了後に GitHub Issue 化する follow-up 一覧

**作成日**: 2026-04-27
**ソース**: Issue #148 Phase A 実行中の発見、および Codex review iter2 の deferred 項目

---

## 1. 🔴 [HIGH] CB-005: MLX import abort in fresh test environment

**Codex review**: `workspace/issues/148/pm-auto-dev/iteration-1/codex-review-result-iter2.json` (CB-005)

**Title 案**: `bug(test): test_photon_pipeline_checkpoint_load.py crashes pytest in fresh shell due to MLX top-level import abort (#148 follow-up)`

**問題**:
- `python -m pytest baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py -q` を fresh shell で実行すると、`autouse fixture` の `monkeypatch.setattr("baseline_reporag.photon_pipeline._load_hf_tokenizer", ...)` が `baseline_reporag.photon_pipeline` を import する。
- そこで module top-level の `import mlx.core as mx` が走り、ある環境で **Fatal Python error: Aborted** で Python process がクラッシュ。
- pytest プロセス全体を巻き込んで終了する (graceful skip/fail にならない)。

**Suggestion**:
- (a) `_validate_repo_id` / `_resolve_checkpoint_path` 等の MLX-free helper を別 module (例: `baseline_reporag/photon_pipeline_helpers.py`) に切り出し、test はそちらだけ import する
- (b) test 側で `sys.modules["mlx.core"] = MagicMock()` 等で stub してから photon_pipeline を import
- (c) MLX が abort する環境を検知して pytest を skip する fixture を追加

**ローカル環境では tests pass** しているため緊急度は中だが、CI / fresh dev env で常に再現する潜在問題。

---

## 2. 🟡 [MEDIUM] yaml/loader schema drift: num_attention_heads vs num_heads

**Title 案**: `refactor(photon): unify yaml→PhotonConfig key names (num_attention_heads / num_heads, generation→model section) (#148 follow-up)`

**問題**:
- `baseline_reporag/photon_pipeline.py:284` は `cfg.model.get("num_heads", 4)` で head 数を読む
- 既存 `configs/photon_*.yaml` は **`generation:` ブロックに `num_attention_heads: 16`** (キー名違い + section 違い) と書く
- loader は yaml の値を一切読まず、ずっとデフォルト 4 を使っていた → PhotonModel は `q_proj=(256, 1024)` で random-init era (S7-001) には何も問題なかったが、**実 ckpt 読込で初めて shape 不一致が surface**

**今回 (PR #151) の対応**:
- `model.num_heads: 10` を photon_small.yaml + institutional_docs_photon.yaml に追加 (ckpt 互換のための一時 patch)
- 根本修正は別 issue として扱う

**Suggestion**:
- (a) loader を `cfg.model.num_attention_heads` または `cfg.model.num_heads` 両対応にする (alias)
- (b) yaml 側を全 photon_*.yaml で `model.num_heads` に統一
- (c) `generation:` ブロック下の `num_attention_heads` / `num_key_value_heads` / `head_dim` / `max_position_embeddings` 等が **本当に generation 用なのか model 用なのか** を整理
- (d) invariant test 追加: `cfg.model.num_heads` が yaml で明示されていることを assertion (key 不在で default 4 silently used を検出)

**影響**: 全 `configs/photon_*.yaml` (5 件) + photon_pipeline.py の loader

---

## 3. 🟢 [INFO] roadmap.md の val_loss 0.4525 と現行 ckpt の val_loss 1.6238 の不一致

**Title 案**: `docs(roadmap): clarify val_loss 0.4525 vs candidate-1 ckpt val_loss 1.62 (#148 follow-up)`

**問題**:
- `workspace/mvp/roadmap.md:17` 記述: 「val_loss 0.4525、600 step 学習済」
- 候補 #1 ckpt (`/Users/maenokota/share/work/github_kewton/photon-mlx-develop/checkpoints/step_000600/state.json`) の `best_val_loss`: **1.6238**
- 候補 #2 (`checkpoints_pre_multi_20260418/step_000600/`) の val_loss は未確認

**仮説**:
- (a) roadmap.md が古い (別 training run の値)
- (b) val_losses 配列の最後 (~0.05) と val_loss 1.62 を取り違えていた
- (c) 0.4525 を出した別 ckpt がどこかにある (本セッションの探索では未発見)

**対応案**:
- 該当する ckpt を発見したら roadmap.md を update
- 発見できなければ roadmap.md の数値を candidate-1 の 1.62 に修正 + 「v4 当時の ckpt 不在」を明記

---

## 4. 🟢 [INFO] #143 (Qwen nondeterminism) との連携: Phase A 2nd run

**Title 案**: `eval(reproducibility): execute Phase A 2nd runs to quantify variance for true PHOTON baseline (#148 + #143 follow-up)`

**問題**:
- 本 PR (Phase A 1st run) は Option B (Minimal pilot) で 1 run のみ実施
- Issue #148 受入条件は **2 run 平均** を要求
- variance / 信頼区間が #143 (Qwen nondeterminism) との関連で要分析

**対応案**:
- Phase A.1 baseline / PHOTON × 2nd run 実行
- Phase A.2 baseline / PHOTON × 2nd run 実行
- 1st-2nd 差分 (NC ± std, latency ± std) を report に追記
- #143 完了が前提条件 (deterministic seeding) であれば順序確定

---

## 5. 🟡 [MEDIUM] Phase A0 deployment.md / troubleshooting.md の追加例

**Title 案**: `docs(deployment): add concrete example of mulmoclaude ckpt placement under PHOTON_CHECKPOINT_ROOT (#148 follow-up)`

**問題**:
- 本セッションで `PHOTON_CHECKPOINT_ROOT=/Users/maenokota/share/work/github_kewton/photon-mlx-develop/checkpoints` + `checkpoint_path: "step_000600"` で動作確認済
- `docs/deployment.md` には env var 説明はあるが、**具体的な ckpt 配置例** (どこに weights.npz/state.json を置くか、複数 worktree でどう symlink するか) が未記載
- 新規担当者が Phase A 再現を試みると同じ調査時間を消費する

**対応案**:
- `docs/deployment.md` に "PHOTON checkpoint placement example" セクション追加
- 共有チェックポイントを別 dir に置いて symlink する pattern を例示
- `docs/troubleshooting.md` に "shape mismatch (Issue #148 Phase A pre-flight)" の対応手順追加

---

## 6. 🟢 [LOW] data/indexes / data/processed の管理方針

**問題**:
- 本セッションで `data/indexes` / `data/processed` / `data/raw` を photon-mlx-develop worktree から symlink した
- これは ad-hoc であり、worktree 間の data 共有方針が docs に未記載
- 将来 Phase A 再現時に各 worktree で再 index 作るのは無駄

**対応案**:
- `data/` の per-worktree 共有方針を CLAUDE.md または docs/deployment.md に追記
- 共有 root を env var (例: `PHOTON_DATA_ROOT`) で指定する案検討

---

## 整理表

| # | 重要度 | カテゴリ | Title (案) | 推奨タイミング |
|---|-------|---------|-----------|--------------|
| 1 | HIGH | bug | CB-005 MLX import abort | Phase A 完了直後 |
| 2 | MEDIUM | refactor | num_attention_heads/num_heads schema | Phase B 着手前 |
| 3 | INFO | docs | roadmap.md val_loss 不一致 | Phase A 結果と一緒に |
| 4 | INFO | eval | Phase A 2nd run for variance | #143 との並行 |
| 5 | MEDIUM | docs | deployment.md ckpt 配置例 | Phase A 完了直後 |
| 6 | LOW | docs | data/ worktree 共有方針 | 余裕があれば |
