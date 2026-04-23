# MVP Issue 一覧

**更新日**: 2026-04-24

## 完了済み Wave（Epic #65 / Epic #93）

### Wave 1-4（Gate 2 v4）— ✅ 全完了

Gate 2 v4 判定で PHOTON 差分価値確定。詳細は `reports/gate2_judgment_v4_final.md`。

- session state / drift / Safe RecGen / KV cache 等の core 機能が本線実装として merge 済み。

### Wave 5（#78, #79, #80）— ✅ 全完了・Merged

| Issue | 内容 | 状態 |
|-------|------|------|
| #78 | `find_relevant_past_turn()` method | ✅ Closed |
| #79 | 圧縮保存 + `storage_mode` 対応 | ✅ Closed |
| #80 | aggregation mode 選択可能化（weighted/attention/last）| ✅ Closed |

### Wave 6 初回マージ — ❌ 全 Revert（regression 対応）

| Issue | 内容 | 結果 |
|-------|------|------|
| #88 initial | grid search harness | Revert |
| #89 initial | bge-reranker-base | Revert |
| #90 initial | bge-small-en-v1.5 embedding | Revert |
| #91 initial | graph neighborhood 調整 | Revert |
| #92 initial | dynamic aggregation | Revert |

Wave 6 batch merge で MT NC 5.6% → 17.8% に regression、全件 revert（PR #99 / #100）。

### Wave 6 再着手（single-PR + eval-gate）— ✅ 完結

| Issue | 再着手結果 | commit / 判定 |
|-------|----------|--------------|
| #92 dynamic aggregation | ✅ Merged | `a72775e` (PR #101) |
| #88 grid search harness | ✅ Merged | `5c461a6` (PR #102、harness のみ） |
| #89 bge-reranker | ❌ Abort | Static +0.8pp 悪化、empirical rejection |
| #103 past_turn_pinning (pipeline 統合) | ⚠️ Merged（default off）| `dfc228f` (PR #105)、opt-in benefit 未実証 |
| #104 hybrid default 化 A/B | ❌ 非採用 | MT NC +0.8pp 悪化で weighted 維持 |
| Epic #93 | ✅ Closed | main sync PR #106 で `64144ff` |

### Streamlit アプリ強化 — ✅ Merged

| Issue | 内容 | 状態 |
|-------|------|------|
| #82 | Wave 1-4 機能を Streamlit に反映（drift panel / turn history / eval runner / PHOTON wizard）| ✅ Merged (PR #107, commit `ea108a1`) |

---

## Open Issues（MVP 作業として残存）

### Phase 2: 制度文書ドメイン検証（2026-04-24 登録、Epic #117）

| Group | Issue | タイトル | 工数 | 依存 |
|-------|-------|---------|------|------|
| — | **#117** | **[Epic] Phase 2 制度文書ドメイン検証** | — | — |
| G1 | #109 | feat(ingestion): markdown chunker + symbol graph conditional skip | 3 日 | なし |
| G1 | #110 | feat(eval): 制度文書 eval set 自動生成 | 4 日 | なし |
| G1 | #111 | feat(retrieval): query_expansion / noise_patterns 汎用化 | 3 日 | なし |
| G2 | #112 | feat(configs): 制度文書プロファイル + index + baseline Static eval | 3 日 | G1 |
| G3 | **#113** | **measure: 現行 PHOTON で MT eval 実測（学習要否の判定根拠）** | 2 日 | #112 |
| G3 | #114 | feat(retrieval): 多言語 embedding / reranker A/B | 4 日 | #112 |
| G4 | #115 | feat(app): wizard domain template + 日本語 prompt | 2 日 | #112, #114 |
| G4 | #116 | docs: Phase 2 完了レポート（再現率比較） | 1 日 | 全 Issue |

### その他（継続課題）

| Issue | タイトル | 種別 | 優先 |
|-------|---------|------|------|
| #81 | [Epic] Static NC < 15% 達成のための retrieval チューニング | enhancement | 中（#88 harness で grid 実行可、Phase 2 と並走可能）|
| #49 | PHOTON Medium (1B) スケールアップ検討 | research | 低（MVP 後、#113 結果次第では前倒し候補）|

---

## MVP Phase 別 Issue 整理

### Phase 1: 品質保証 — ✅ 実質完了

| 項目 | 状態 |
|------|------|
| Safe RecGen ログ修正 | ✅ 完了済み |
| fallback recall 計測 | ✅ スクリプト化済 |
| Gate 3 判定 | ✅ 条件付き Go |
| test_photon_pipeline 修正 | ✅ 現状 832/834 pass（既知 2 pre-existing failure のみ）|
| 異常入力テスト | ✅ `_safe_id` / YAML safe_load guardrail で Wave 2 に反映 |

### Phase 2: 制度文書ドメイン検証 — 🏃 進行中（Epic #117、2026-04-24 pivot）

**方針変更**: Django / Pydantic での code 汎用化は skip。**日本語制度文書（4,228 ファイル）への pivot**。

| タスク | Issue | 工数 | 状態 |
|-------|-------|------|------|
| Markdown chunker + symbol graph skip | #109 | 3 日 | 未着手 |
| 制度文書 eval set 自動生成 | #110 | 4 日 | 未着手 |
| query_expansion / noise_patterns 汎用化 | #111 | 3 日 | 未着手 |
| 制度文書 config + baseline eval | #112 | 3 日 | 依存待ち |
| **現行 PHOTON で MT eval（学習要否判定）** | **#113** | **2 日** | **クリティカル** |
| 多言語 embedding / reranker A/B | #114 | 4 日 | 依存待ち |
| wizard + 日本語 prompt | #115 | 2 日 | 依存待ち |
| Phase 2 完了レポート | #116 | 1 日 | 最終 |

### Phase 3: 配布（pip） — ❌ 未着手

| タスク | 工数見積 | 状態 |
|-------|--------|------|
| 汎用コーパス拡張（20+ repos）| 3 日 | 未着手 |
| 汎用モデル学習 | 2 日 | 未着手 |
| 未知 repo eval | 2 日 | 未着手 |
| HuggingFace weights 公開 | 1 日 | 未着手 |
| 自動 DL + pip パッケージ化 | 4 日 | 未着手 |

---

## 未起票の内部改善候補

- `find_relevant_past_turn` の閾値 / `max_pinned_chunks` チューニング（#103 follow-up）
- `hybrid_alpha_base` / `hybrid_alpha_per_turn` チューニング（#104 follow-up）
- retrieval grid search 実行（#88 harness を使って #81 を推進）
- Embedding 更新の独立 A/B（Wave 6 で #90 初回 revert、再試験候補）
- PHOTON Small の State management 特化再訓練（mulmoclaude 1000 step → 10K step へ拡張）
