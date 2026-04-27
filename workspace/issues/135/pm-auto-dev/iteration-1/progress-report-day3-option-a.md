# Issue #135 Day 3 進捗報告 (続) — Option A 適用 + 実測 ETA 再評価

**Issue**: feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合
**実行日**: 2026-04-27 → 2026-04-28
**ブランチ**: feature/issue-135-photon-retrain (Day 1-3 = **22 commits**)
**ステータス**: Option A (seed jitter) 適用済 + smoke 検証済。Option B (--sessions 500) ETA 再評価。**ユーザー判断 (本番 run go/no-go) 要請**

---

## ✅ Option A: retry seed jitter (commit `fdf16bc`)

### 修正内容

`baseline_reporag/eval/institutional/multi_turn.generate_session` と `generator.generate_question` の retry loop で `seed=base_seed + attempt` を渡すように変更:

```python
# Before (Day 3 blocker):
for attempt in range(max_retries):
    raw = client.generate(prompt)  # seed=42 固定 → deterministic LLM では同 output × 3 retry

# After:
for attempt in range(max_retries):
    raw = client.generate(prompt, seed=base_seed + attempt)  # 42, 43, 44 → 別 sample
```

### Tests

`tests/test_institutional_multi_turn.py::test_generate_session_varies_seed_per_retry` 新設:
- `_SeedRecordingClient` で全 retry の seed を記録
- 「3 distinct seeds」を assert
- RED → GREEN: 修正前 `[42, 42, 42]` → 修正後 `[42, 43, 44]` ✅

regression: 869 passed (本変更由来 0、develop side stale 1 のみ既知)

---

## 🔬 Smoke 実機検証 (seed=42, 5 sessions, qwen mlx local)

前回 (Day 3 blocker) の seed=42 smoke は **0/2 success** で停止。
Option A 適用後の seed=42 smoke 結果:

| 指標 | 値 |
|------|------|
| n_sessions_requested | 5 |
| **n_sessions_succeeded** | **4** ✅ |
| eval_overlap | 0 ✅ |
| jp_sequence_ratio | 1.0 ✅ |
| scenario_distribution | drill_down 0.5 / cross_reference 0.25 / real_scenario 0.25 |
| total_tokens (train+val) | 1328 + 339 = 1667 |
| Wall-clock | ~6 分 |

### Seed jitter 動作確認 (実機ログ)

```
PDF-163 / drill_down attempt 1: char 1626 fail
PDF-163 / drill_down attempt 2: char 1565 fail   ← 出力位置が変化 = seed jitter 効いている
児童手当法 / cross_reference attempt 1: char 1712
児童手当法 / cross_reference attempt 2: char 1741   ← 別出力
児童手当法 / cross_reference attempt 3: char 1655   ← さらに別出力
```

**結論**: seed jitter は意図通り動作、ただし PDF-163 / 児童手当法のような **特定 doc は内容由来で 3 retry すべて JSON parse fail** (Qwen 14B 4-bit が長文/複雑 doc で JSON 構造を維持できない)。

---

## 📊 実測ベースの ETA 再計算

### 計測値

| 計測 | 1 回あたり |
|------|-----------|
| Successful session (good doc) | ~50 秒 |
| Failed doc (3 retry, exhausted) | ~150-200 秒 |
| 失敗率 (実測 5 docs) | 20% |

### スケジュール推定

| --sessions | 成功 | 失敗 doc skip | wall-clock | 設計方針書 ≥ 2000? |
|-----------|------|--------------|----------|-------------------|
| 300 | 240 | 60 | 250×50 + 60×200 = 24,500s ≈ **6.8 h** | ❌ (15%) |
| 500 | 400 | 100 | 400×50 + 100×200 = 40,000s ≈ **11.1 h** | ❌ (25%) |
| 1000 | 800 | 200 | 800×50 + 200×200 = 80,000s ≈ **22.2 h** | ❌ (50%) |
| 2000 | 1600 | 400 | 1600×50 + 400×200 = 160,000s ≈ **44 h** | ✅ |

**ユーザー想定 "数時間" (3-6h)** に収めるには `--sessions 200-300` が必要。設計方針書の最低 2000 は壁。

---

## 🎯 残選択肢

### Option B': `--sessions 300` で運用 (~7h)
- 設計方針書 §A-1 の 「sessions ≥ 2,000」を **設計緩和** で 300 に下げる
- mix 比率は維持 (JP 50% / EN 50%) — JP token 数が EN より少なくなる事実は metadata に記録
- ETA: ~7 時間 → "数時間" に近い

### Option B'': `--sessions 500` で運用 (~11h)
- 設計緩和 + ETA 中程度
- 一晩寝かせる前提なら現実的

### Option C: OpenAI provider 切替 (~5h、DR3-002 例外承認要)
- gpt-4o-mini で 8-10s/session、JSON 安定性高
- institutional 文書 (制度文書) を OpenAI に送信する設計方針書 §11 違反
- 例外承認が出れば 2000 sessions も 5-6h で達成可能

### Option D: batched mlx_lm 並列化 (~7h、実装 4-6h)
- mlx_lm の batch generation API 活用
- ETA は短いが実装工数 + OOM リスク

### Option E: 待機 + 一晩寝かせ (B'')
- `nohup ... &` で session 切り離して 11h 走らせる
- 翌日結果検証 → Phase 6 学習に進む
- 工数最小、wall-clock のみ受容

---

## ❓ ユーザー判断要請

1. **どの Option で進めるか?**
   - **B' (sessions=300, 7h)** ← 最速の現実解、設計緩和
   - **B'' (sessions=500, 11h)** ← 一晩寝かせ前提
   - **C (OpenAI)** ← DR3-002 例外承認要
   - **D (batched mlx_lm)** ← 工数 4-6h + 実行 7h
2. (B 採用なら) JP corpus 規模が 設計方針書 ≥ 2,000 を下回る件、**受入条件 A-1 を緩和** で OK か?
3. **本セッションでの実行可否**: B'' / D は本セッション (Claude のチャット session) を超える可能性。`nohup ... &` で切り離すか、別セッションで再開か?

---

## 累計 Day 3 commits (Day 1-3 通算 22)

```
fdf16bc fix(eval/institutional): jitter seed per retry so deterministic LLMs can recover (Option A)
4965f27 docs(issue-135): Day 3 続報 — Step 1-2 完了 + Step 3 corpus 生成 ETA blocker
b7057cd feat(configs): use develop worktree absolute paths
3612d8f feat(scripts): align corpus generator with production LLM client + token output
b099960 docs(issue-135): Day 3 進捗報告 — develop merge OK + Phase 6 blocker 報告
be91682 merge develop into feature/issue-135-photon-retrain (Phase 6 prep)
... (Day 1-2 commits 16 件、進捗報告参照)
```

ユーザー指示があるまで実行保留、**idle 状態で待機**します。
