## 背景

Phase 2 完了時に、FastAPI ドメインで得られた PHOTON の効果（MT NC -2-4pp, Turn 5-6 NC 0%, follow-up -34%）が制度文書ドメインで **どの程度再現**するかを定量比較するレポート。

MVP 判定基準「FastAPI 改善の再現率 70%+」の達成可否を確定させる成果物。

> **2026-04-28 更新**: Phase 2 は #135 (institutional retrain Phase 8) の採用で Turn 5-6 NC 0.00% (refusal-aware) を達成し、`workspace/mvp/roadmap.md` で完了宣言済。本 Issue は完了レポートをコードベースの確定値で形式化することが目的。

## 成果物

`reports/phase2_cross_domain_validation.md`（新規）:

### 内容

1. **メトリクス比較表**（確定値ベース）

| metric | FastAPI (Phase 1 / Gate 2 v4) | 制度文書 (Phase 2 / #135 retrain step_003000) | 再現判定 |
|--------|------------------------------|---------------------------------------------|---------|
| MT NC baseline | 6.94% (15.6% in Gate 2 v4 single-run) | 6.94% (#113 baseline avg, 2-run) | — |
| MT NC PHOTON | 6.7% (Gate 2 v4 final) | 8.33% raw / **0.00% refusal-aware** (#135) | — |
| **Turn 5-6 NC PHOTON** | **0.0%** | **0.00% (refusal-aware)** / 6.67% raw | ✅ 完全再現 |
| **Follow-up p50 改善率** | **-34% to -35%** | **-37.7%** (12,092 ms vs baseline 19,426 ms) | ✅ 110% 再現 |
| Val_loss (採用 checkpoint) | 0.4525 (mulmoclaude 600-step) | 0.4777 (institutional retrain 3K) | (参考) |

エビデンス:
- FastAPI 側: `reports/gate2_judgment_v4_final.md`、`workspace/mvp/metrics.md` Phase 1 セクション
- 制度文書側: `reports/institutional_photon_mt_eval_v2_3k.md`、`reports/institutional_photon_mt_eval_v2_3k_bug_check.md`

2. **Phase 2 成功条件の判定**（測定指標を確定）

- [ ] 制度文書 MT NC < 15% （表面値 8.33% / refusal-aware 0.00% で達成）
- [ ] 制度文書 follow-up latency baseline -20%+ （-37.7% で達成）
- [ ] FastAPI 改善の 70%+ 再現
  - **再現率の定義**: 以下 2 指標とも、制度文書ドメインで FastAPI 改善幅の 70%+ を達成
    - Indicator A: Turn 5-6 NC = 0.0% を再現（FastAPI で 0% → 制度文書で < 1% を 70%+ 再現と判定）
    - Indicator B: Follow-up p50 改善率を再現（FastAPI -34% → 制度文書で -23.8%+ を 70%+ 再現と判定）
  - 両指標達成: ✅ 100% 再現として MVP Go
  - 片方達成: ⚠ 50% 再現として勧告内に明記
  - 両方未達: ❌ Phase 2 pivot 検討

3. **PHOTON 再学習の経緯と結論**

- (a) #113 で『仮説 C（本格再学習が必要）』判定（mulmoclaude 600-step では Turn 5-6 NC 10.83%）
- (b) #135 で institutional retrain 実装（4,228 docs JP:0.7/EN:0.3 mix、累計 3,000 step）
- (c) val_loss 1.6238 → 0.4777 (-70.6%)、Turn 5-6 NC 10.83% → 6.67% raw / **0.00% refusal-aware** (Issue #154 Bug 2 観点で再判定)
- (d) Phase 8 ロールアウト完了（`configs/institutional_docs_photon.yaml` checkpoint 昇格、PR #157 merged 2026-04-28）

4. **Phase 3（MVP 配布）進行可否の勧告**

`reports/gate2_judgment_v4_final.md` のフォーマットに準拠し、以下 3 区分のいずれかを勧告:
- **Go**: Phase 3（pip install + HuggingFace 公開）に着手
- **Conditional Go**: 残課題（例: Issue #156 計測 bug）を condition として Phase 3 着手
- **Pivot**: Phase 2 結果に問題があり Phase 3 設計を見直す必要あり

## 影響ファイル

- `reports/phase2_cross_domain_validation.md`（新規）
- `workspace/mvp/metrics.md`（必要に応じて Phase 2 完了状態セクションの整合性微修正、本 Issue 時点で 2026-04-28 更新済）
- `workspace/mvp/roadmap.md`（既に 2026-04-28 で「Phase 2 完了」反映済、追記不要の見込み）
- **Phase 3 進行 Go 判定の場合**: Phase 3 着手 Epic Issue を新規起票し roadmap.md の Phase 3 セクションに着手予定日を加筆

## 受入条件

- [ ] 全 Phase 2 関連 Issue (#109-#115, Epic #117, #135 retrain, #148 institutional baseline 補強, #156 計測 bug follow-up) の結果を反映
- [ ] FastAPI 改善の 70%+ 再現を 上記 Indicator A / B の二指標で測定し、再現率を確定数値で明記
- [ ] Phase 3 進行可否の勧告（Go / Conditional Go / Pivot のいずれか + 必要条件）を `reports/gate2_judgment_v4_final.md` 準拠フォーマットで記述

## 関連

- Epic: #117 Phase 2 制度文書ドメイン検証
- 主要 retrain: **#135** (institutional retrain Phase 8 採用、PR #157 merged 2026-04-28)
- baseline 補強: #148 (institutional baseline 7.78% NC 確定、PHOTON random-init bug 修正)
- 計測 bug follow-up: #156 (`scripts/run_multi_turn_eval.py` の `is_refusal` 出力欠落)
- Phase 2 構成 Issue: #109-#115 (全 closed)
- **最終タスク**: 他の Phase 2 Issue 全完了後（→ 完了済、本 Issue が Phase 2 close 直前）

## 並列開発

完全に直列（他の全 Issue 完了後）。
