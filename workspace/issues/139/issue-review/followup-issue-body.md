## 背景

#139 (Stub/Mock pattern audit) のスコープ整理 (Stage 1 review S1-001 / S1-002 / S1-008) の結果、当初 Task 2 として含まれていた **real-weight integration test** を本 Issue に切り出した。

理由:
- #139 Task 2 は #135 (S7-001 fix, commit `2dbf458` で導入された `model.checkpoint_path` 機構) を前提に設計されている
- 2026-04-26 時点で #135 は `feature/issue-135-photon-retrain` 上のみ存在し main にマージされていない
- 現行 main の `baseline_reporag/photon_pipeline.py` には `checkpoint_path` 設定も `load_checkpoint` 呼び出しも存在しない
- 本 Issue 単独では着手不能なため、#135 マージ後の follow-up とする

## 依存

- **必須**: #135 (本格再学習 / S7-001 fix の checkpoint loading 機構) が main にマージされていること
- 推奨: #139 (Task 1 + Task 3 / scaffolding audit + invariant test) が先行マージされていること (production path から `_StubTokenizer` 等が排除済みであれば、本 test がよりクリーンに書ける)

## ゴール

実 PhotonModel + 実 (or 実用最小) checkpoint + 実 tokenizer + 実 retrieval を通る end-to-end query test を 1 件以上追加し、「random-init weight が production で silently 動いていた」型の構造的事故を CI で検出可能にする。

## 変更内容

新規ファイル: `tests/integration/test_photon_real_weights.py`

### 候補方針 A: 既存最小 checkpoint を使用 (#135 マージ後の実 ckpt 利用)

```python
@pytest.mark.integration
@pytest.mark.slow
def test_photon_pipeline_with_real_checkpoint(tmp_path):
    """E2E: 実 checkpoint + 実 tokenizer + 実 retrieval で query が走る"""
    ckpt = path_to_smallest_committed_or_downloadable_ckpt()  # configs/photon_tiny.yaml 等
    cfg_yaml = make_test_config(ckpt_path=ckpt)
    pipeline = build_pipeline(cfg_yaml)
    result = pipeline.query("What is FastAPI?")
    assert result.answer is not None
    assert len(result.evidence) > 0
    assert pipeline.last_pruning_score_distribution.std() > 0.01  # random だと std ≈ 0
```

ただし、(a) checkpoint 取得手段 (HF Hub / S3 / committed binary etc.) を別途確定する必要あり。

### 候補方針 B: セルフホスト型最小 e2e (推奨, repo 完結)

```python
@pytest.mark.integration
def test_photon_pipeline_with_self_trained_minimal_ckpt(tmp_path):
    """最小 PhotonConfig で 1 step 学習 → save → load → query が走る"""
    # 1. 最小 config (vocab=256, hidden=64, layers=2 等) で PhotonModel を構築
    # 2. ダミーデータで 1 step 学習し、tmp_path に checkpoint 保存
    # 3. 保存した checkpoint を _build_photon_deps 経由で load
    # 4. query を流して result.answer の non-emptiness を検証
    # 5. random-init detector: pruning_score の std が学習前 (≈0) と学習後で異なることを assert
```

利点: 外部リソース不要、CI で確実に再現、20-60 秒程度で完了。
欠点: 「実用 size の weight を通す」 e2e ではないため、Issue #135 想定の random-init bug 検出力は方針 A より弱い。

→ **推奨**: 方針 B を本 Issue で実装、方針 A は更に follow-up Issue で段階導入する。

## 受入条件

- [ ] `tests/integration/test_photon_real_weights.py` に real-weight integration test を 1 件以上追加
- [ ] その test 内で `PhotonModel` の実インスタンスが構築され、checkpoint からの weight load が経路として走ること (mock を使わない)
- [ ] random-init detector (例: pruning_score 分布の std > 閾値) で「checkpoint が読まれていない」ケースを検出可能であること
- [ ] CI 実行戦略を明記する (PR を blocking しない nightly か、毎 PR で 60 秒以内完結なら blocking) — どちらかを Issue 着手時に確定
- [ ] 既存テスト + 新規テスト全パス
- [ ] `ruff check .` 警告 0 件

## 影響ファイル

- `tests/integration/test_photon_real_weights.py` (新規)
- 必要なら `configs/photon_tiny.yaml` 等に test 用 minimal config を 1 件追加
- `.github/workflows/` に nightly trigger を追加する場合は別途
- 本 Issue 完了後の `baseline_reporag/photon_pipeline.py` (#135 マージ後の `checkpoint_path` 仕様に応じて test を整合)

## 関連

- 切り出し元: #139 (Stub/Mock pattern audit) — Task 1 + Task 3 を先行
- 必須依存: #135 (本格再学習 / S7-001 fix)
- 緊急修正: #138 (tokenizer mismatch) — マージ済み (#141)

## レビュー履歴

- 切り出し根拠: #139 Stage 1 通常レビュー S1-001 / S1-002 / S1-008
  - workspace/issues/139/issue-review/stage1-review-result.json
- 切り出し時刻: 2026-04-26
