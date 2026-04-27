# Issue #137 マルチステージ設計レビュー 完了報告

**Issue**: #137 feat(retrieval): institutional 多言語 embedding/reranker 5-variant 実機 A/B (#133 Phase B)
**完了日**: 2026-04-26
**設計方針書**: `workspace/design/issue-137-institutional-multilingual-ab-design-policy.md`

## ステージ別結果

| Stage | レビュー種別 | レビュアー | 指摘数 | 反映数 | ステータス |
|-------|------------|----------|--------|--------|----------|
| 1 | 通常レビュー（設計原則 SOLID/KISS/YAGNI/DRY） | **claude-opus** | 7 (MF:0, SF:4, NH:3) | 7/7 | 完了 |
| 2 | 整合性レビュー（設計書 vs Issue/実装/関連doc） | **claude-opus** | 8 (MF:0, SF:4, NH:4) | 6/8 (NH 2件 skip) | 完了 |
| 3 | 影響分析レビュー | **codex** | 2 (MF:0, SF:2, NH:0) | 2/2 | 完了 |
| 4 | セキュリティレビュー | **codex** | 4 (MF:0, SF:4, NH:0) | 4/4 | 完了 |

**合計**: 21 findings (0 Must Fix, 14 Should Fix, 7 Nice to Have) → 19 applied, 2 skipped (Nice to Have)

## 主要な改善ポイント

### Stage 1 (claude-opus, 設計原則)
- 設計判断 #1 の DRY 違反許容理由 (base 凍結) を section 10 リスクマップに追加
- V4 採用時 invariant 追加忘れ防止のレポートテンプレート checkbox 化
- device 死フィールド技術的負債を明示し follow-up 候補化
- 設計判断 #6 (LSP) として 5 variant の interface 前提を明記
- section 5-7 commit 粒度推奨 (SRP) を新規追加

### Stage 2 (claude-opus, 整合性)
- section 5-7 commit 構成を採用結果別 3 パターンに書き直し (Issue 本文 PR 戦略表と整合)
- section 5-5 deployment.md 更新条件を V0-V3/V4 で明確に切り分け
- section 5-4 第 3 invariant test の Config interface 注記を実装事実に整合
- section 11 を「touch する全ファイル一覧」に拡充 (4 カテゴリで網羅)

### Stage 3 (codex, 影響分析)
- section 11 に Phase B 完了後の cleanup / retention 手順 (未採用 variant index, predictions JSONL, HF cache の削除方針, du での容量記録) を追加
- section 7 に #135 への業務影響 (5.5 日直列許容、並行可能な準備作業の限定) を明記

### Stage 4 (codex, セキュリティ)
- security table の eval set tracked status 誤記を修正 (institutional_static_eval.jsonl は実は tracked)
- ingest_repo.py / build_indexes.py の `--repo-id` path traversal リスクを security 観点で明示
- HF model 自動 download の revision pin 不在 (供給網リスク) を follow-up 候補として追加
- invariant bypass 対策を CODEOWNERS 不在前提から required CI + reviewer checklist 主体に書き直し

## 結論

Codex クロスレビューにより、claude-opus が見落とした実装/運用上の重要な前提誤り 6 件 (eval set tracked、path traversal リスク、HF revision pin、CODEOWNERS 不在、cache cleanup 手順、業務影響明文化) を補足できた。設計方針書は Phase B 実機実行と採用反映の両フェーズに必要な手順・前提・受入条件を網羅した状態に到達。Must Fix は全 Stage で 0 件。

## 次のアクション

- [x] Phase 3: マルチステージ設計レビュー
- [ ] Phase 4: 作業計画立案 (`/work-plan 137`)
- [ ] Phase 5: TDD自動開発
- [ ] Phase 6: 完了報告
