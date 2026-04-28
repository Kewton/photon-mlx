# Issue #143 マルチステージ設計レビュー完了報告

実施日: 2026-04-28
対象設計方針書: `workspace/design/issue-143-eval-reproducibility-design-policy.md`
ブランチ: `feature/issue-143-eval-reproducibility`

---

## ステージ別結果

| Stage | レビュー種別 | Reviewer | Must Fix | Should Fix | Nice to Have | 対応数 | ステータス |
|-------|------------|----------|---------|----------|------|------|----------|
| 1 | 通常レビュー (設計原則 SOLID/KISS/YAGNI/DRY) | claude-opus | 3 | 7 | 3 | 13/13 | 完了 |
| 2 | 整合性レビュー | claude-opus | 2 | 6 | 2 | 10/10 | 完了 |
| 3 | 影響分析レビュー | **codex** | 3 | 2 | 0 | 5/5 | 完了 (reviewer=codex 検証済) |
| 4 | セキュリティレビュー | **codex** | 2 | 1 | 0 | 3/3 | 完了 (reviewer=codex 検証済) |
| **合計** | - | - | **10** | **16** | **5** | **31/31** | **全反映済** |

---

## Codex reviewer 検証 (Issue #140 / S7-001 follow-up)

- `stage3-review-result.json`: `reviewer="codex"` ✓
- `stage4-review-result.json`: `reviewer="codex"` ✓
- WARNING なし

---

## 主要な発見

### Stage 1 通常レビュー (Claude opus, 設計原則)

**Must Fix**:
- DR1-001 (DRY): `QwenMLXAdapter.generate` (institutional/llm_client.py) と本 Issue で追加する `Generator.generate` の seed 注入ロジック重複 → ADR-11 新設で「層分離 (B 採択)」を明示
- DR1-002 (SRP): `resolve_eval_seed(cfg)` helper の責務肥大 → `_validate_run_block(run_dict)` private 関数で内部分離
- DR1-003 (API): PHOTON path の Qwen fallback 3 箇所 (photon_pipeline.py L1030, L1043, L1394) で multi-call seed 戦略を ADR-1 末尾に明文化

**Should Fix** (7件): KISS/OCP/YAGNI/エラーハンドリング/命名規則の精緻化

### Stage 2 整合性レビュー (Claude opus)

**Must Fix**:
- DR2-001: `tests/test_aggregate_institutional_baseline.py` (誤記) → `tests/test_aggregate_institutional.py` (実在ファイル) に §3/§7/§10 で全箇所修正
- DR2-002: Issue 本文 Task 3 (run_index/run_seed/run_id 3 fields) と ADR-5 (run_index/run_seed 2 fields) の矛盾 → §12 修正 + Issue body 同期タスクを §9 Step 11 に追加

### Stage 3 影響分析レビュー (Codex 独立クロスレビュー)

**Must Fix (silent bug 含む重要発見)**:
- **DR3-001**: opus が漏らした eval/benchmark scripts `scripts/retrieval_grid_search.py` と `scripts/run_stress_eval.py` も `pipeline.query()` を直接呼ぶため seed 伝播対象に追加
- **DR3-002**: §2 architecture 図の `if seed: mx.random.seed(seed)` が **seed=0 で silent bug** → `if seed is not None:` に統一
- **DR3-003**: predictions schema の §3 表記が ADR-5 の 2 fields 決定と不一致 → 統一 + 旧 JSONL 互換 (run_index/run_seed を REQUIRED_FIELDS に入れない、欠落時は単一 run 扱い) を ADR-6/§5 に追加

### Stage 4 セキュリティレビュー (Codex 独立クロスレビュー)

**Must Fix**:
- **DR4-001**: Python の `bool` は `int` の subclass → YAML `run.seed: true` が `True -> 1` として通る。`type(seed) is int` の厳密判定に修正、bool/NaN/float/str を `TypeError` で拒否
- **DR4-002**: `--runs N` が無制限 → self-hosted runner DoS リスク。argparse validator で `1 <= runs <= 20` に制限

**Should Fix**:
- DR4-003: CLI path traversal 境界の明文化 (CLI は trusted operator tool 維持、Streamlit は `make_eval_paths()` confined path のみ)

---

## Codex 独立クロスレビューの価値

opus 単独では発見できなかった以下の重要な設計上の漏れ・silent bug を Codex が発見:

1. **scripts/retrieval_grid_search.py / scripts/run_stress_eval.py** が seed 伝播対象から漏れ → 影響範囲を 3 → 5 scripts に拡大
2. **seed=0 silent bug**: `if seed:` の truthy 判定 (Python では `0` が False) → seed=0 で固定が無効になる潜在 bug を architecture 図段階で発見
3. **YAML true → int 1 silent bug**: `isinstance(seed, int)` で bool subclass を素通し → `type(seed) is int` に厳密化
4. **--runs DoS**: 計算リソース上限を `1 <= runs <= 20` に bounded

これらは実装着手前に修正できたため、PR 段階で逆戻りするコストを回避。

---

## 最終設計方針書状態

- ファイル: `workspace/design/issue-143-eval-reproducibility-design-policy.md`
- 章構成: §1 〜 §14 維持
- ADR: ADR-1 〜 ADR-11 (合計 11 件、ADR-11 は Stage 1 で新規追加)
- 反映 findings: 31/31 (Must Fix 10、Should Fix 16、Nice to Have 5)

---

## 最終検証結果

**Note**: このコマンドは設計方針書のレビューのみ。ソースコード変更・pytest 実行は行わない。

| チェック項目 | 結果 |
|-------------|------|
| 設計方針書が更新されている | ✓ |
| 全 4 ステージのレビュー完了 | ✓ |
| Codex reviewer 検証 (Stage 3/4) | ✓ |
| サマリーレポート作成 | ✓ |

---

## 次のアクション

- [x] 設計方針書の最終確認
- [ ] **/work-plan 143** で作業計画立案 ← 次のフェーズ
- [ ] /pm-auto-dev 143 で TDD 自動開発
