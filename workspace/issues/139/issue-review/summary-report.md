# Issue #139 マルチステージレビュー完了報告

**対象 Issue**: [#139](https://github.com/Kewton/photon-mlx/issues/139) — test(photon): Stub/Mock pattern audit (S7-001 follow-up)
**実施日**: 2026-04-26
**ブランチ**: `feature/issue-139-stub-audit` (HEAD: 8e677ca)

---

## 仮説検証結果（Phase 0.5）

| # | 主張 | 判定 |
|---|------|------|
| 1 | `test_photon_pipeline.py` は 1715 行 | **Rejected** (実際 4453 行) |
| 2 | テストは MagicMock 中心、実 PhotonModel 経路を通らない | **Confirmed** |
| 3 | `_StubTokenizer` 等が production path に残存 | **Confirmed** |
| 4 | commit `2dbf458` が S7-001 を fix | **Confirmed (注意あり)** |
| 5 | `getattr default` パターンが production に多数 | **Confirmed** (8件) |
| 6 | 設計 Must Fix の CI 固定の仕組みが無い | **Partially Confirmed** |
| 7 | `photon_pipeline.py` に Stub 命名残存 | **Confirmed** |
| 8 | #138 はマージ済み | **Confirmed (注意あり)** |
| 9 | smallest checkpoint が利用可能 | **Unverifiable** |

→ 仮説のうち **Rejected 1 / Unverifiable 1** は Stage 1 で Must Fix 化。

---

## ステージ別結果

### 1 回目イテレーション (Claude opus)

| Stage | レビュー種別 | 指摘数 (Must / Should / Nice) | 反映 |
|-------|------------|-------------------------------|------|
| 1 | 通常レビュー | **3 / 5 / 2** = 10 件 | 全件適用 |
| 3 | 影響範囲レビュー | **4 / 3 / 3** = 10 件 | 全件適用 |

#### Stage 1 主要 finding
- S1-001 (Must Fix): #135 が main 未マージ。Task 2 を別 Issue 化必須 → **#145 を作成**
- S1-002 (Must Fix): real-weight test の checkpoint 取得手段未定義 → #145 で「セルフホスト型最小 e2e」推奨
- S1-003 (Must Fix): `test_photon_pipeline.py` 行数 1715 → 4453

#### Stage 3 主要 finding
- S3-001 (Must Fix): invariant test 必須キー名 `model.vocab_size` → `tokenizer.vocab_size` (canonical)
- S3-002 (Must Fix): 全 yaml glob は `baseline.yaml`/`eval.yaml` で破綻 → `provider == "photon"` で絞る
- S3-003 (Must Fix): サンプルコードの semantic bug 2 種 (`'tests'` 複数形 / regex word-boundary)
- S3-004 (Must Fix): 既存 test の migration plan 欠落

### 2 回目イテレーション (Codex via commandmatedev)

| Stage | レビュー種別 | 指摘数 (Must / Should / Nice) | 反映 | reviewer |
|-------|------------|-------------------------------|------|----------|
| 5 | 通常レビュー (cross) | **0 / 2 / 0** = 2 件 | 全件適用 | codex ✓ |
| 7 | 影響範囲レビュー (cross) | **0 / 3 / 0** = 3 件 | 全件適用 | codex ✓ |

#### Stage 5 主要 finding (Codex)
- S5-001: invariant test の対象 yaml 判定が `provider == 'photon'` のみだと `photon_tiny.yaml` 等を取りこぼす → filename 判定併用
- S5-002: tokenizer load failure (HF Hub 障害等) の `ValueError` 正規化が受入条件で固定されない

#### Stage 7 主要 finding (Codex)
- S7-001: no-scaffolding test が pytest cwd 依存で対象 0 件偽 pass し得る → repo root 基準に修正
- S7-002: tokenizer load failure の運用影響 (HF Hub / gated model / cache 障害) を troubleshooting で扱う
- S7-003: #135 ブランチは `994ba29` まで進み `git merge-tree` で実際の conflict marker が出る

---

## 総合カウント

| 種別 | Stage 1 | Stage 3 | Stage 5 | Stage 7 | **合計** |
|------|---------|---------|---------|---------|---------|
| Must Fix | 3 | 4 | 0 | 0 | **7** |
| Should Fix | 5 | 3 | 2 | 3 | **13** |
| Nice to Have | 2 | 3 | 0 | 0 | **5** |
| 合計 | 10 | 10 | 2 | 3 | **25** |

**全 Must Fix / Should Fix / Nice to Have の 25 件が Issue 本文に反映済み**。

---

## 副次的成果物

- **新 Issue 作成**: #145 (real-weight integration test、#135 マージ後着手)
- **Issue #139 Body 全面リライト**: 25 finding を反映した最終形は `workspace/issues/139/issue-review/updated-issue-body.md` と GitHub Issue body で同期済み
- **#135 との具体衝突情報**: `git merge-tree` ベースで `baseline_reporag/photon_pipeline.py` / `baseline_reporag/tests/test_photon_pipeline.py` の手動統合が必要と確定 (Stage 7-S7-003)

---

## 完了条件チェック

- [x] Phase 0.5 仮説検証完了
- [x] Stage 1 通常レビュー (1st) 完了
- [x] Stage 2 反映完了 (Issue body 更新)
- [x] Stage 3 影響範囲レビュー (1st) 完了
- [x] Stage 4 反映完了
- [x] 2回目スキップ判定: Must Fix 7件 > 0 → Stage 5-8 実行
- [x] Stage 5 通常レビュー (Codex) 完了 — `reviewer == "codex"` 確認済
- [x] Stage 6 反映完了
- [x] Stage 7 影響範囲レビュー (Codex) 完了 — `reviewer == "codex"` 確認済
- [x] Stage 8 反映完了
- [x] GitHub Issue 更新済み

---

## 次のアクション

- [ ] **Phase 2**: 設計方針書作成 (`/design-policy 139`) — 本 Issue 反映後の Task 1 + Task 3 スコープに対する設計
- [ ] Phase 3: マルチステージ設計レビュー
- [ ] Phase 4: 作業計画立案
- [ ] Phase 5: TDD 自動開発
- [ ] (別途) #145 の対応は #135 マージ後に着手
