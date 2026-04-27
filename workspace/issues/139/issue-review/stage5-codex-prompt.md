Issue #139 の **2回目通常レビュー (Stage 5) + 指摘反映 (Stage 6)** を実施してください。

## あなたの役割

Claude opus が実施した 1 回目通常レビュー (Stage 1) + 1 回目影響範囲レビュー (Stage 3) の **クロスレビュー** を担当する Codex reviewer。1 回目で見落とされた issue を補完するのが目的です。

## 対象 / 入力ファイル

すべて作業ディレクトリ `/Users/maenokota/share/work/github_kewton/photon-mlx-feature-issue-139-stub-audit/` 直下のパス。

- 現行 Issue 本文 (Stage 1 + Stage 3 反映済): `workspace/issues/139/issue-review/updated-issue-body.md`
  (= `gh issue view 139 --json body` で取得できる最新版と一致)
- 1 回目通常レビュー結果: `workspace/issues/139/issue-review/stage1-review-result.json` (Must Fix 3 / Should Fix 5 / Nice 2)
- 1 回目影響範囲レビュー結果: `workspace/issues/139/issue-review/stage3-review-result.json` (Must Fix 4 / Should Fix 3 / Nice 3)
- 1 回目指摘反映結果: `workspace/issues/139/issue-review/stage2-apply-result.json`, `workspace/issues/139/issue-review/stage4-apply-result.json`
- 仮説検証レポート: `workspace/issues/139/issue-review/hypothesis-verification.md`

## 実施手順

1. `gh issue view 139 --json body` で最新 Issue 本文を取得し、`updated-issue-body.md` と一致することを確認 (差分があれば最新を優先)。
2. Stage 1 / Stage 3 で挙がった全指摘 (合計 20 件) を読み、それぞれ Issue 本文で対応されているかを確認する。未対応や不十分な反映があれば finding として記録 (severity: Must Fix)。
3. **Stage 5 通常レビュー**: 整合性 / 正確性 / 実装可能性 / 粒度 / 依存関係 の観点で Issue 本文を再レビューし、1 回目で見落とされた issue を抽出する。重複指摘は避ける。
   - 影響範囲は Stage 7 で扱うため対象外。
   - サンプルコード (Task 1 Step 3 / Task 3 Step 3) の semantic 検証も含める (Stage 1 での見落としを Stage 3 が補ったため、Stage 5 では更に踏み込んだ実装適合性チェックを期待)。
4. **Stage 6 反映**: Must Fix / Should Fix の指摘があれば `gh issue edit 139 --body-file <new-body-md>` で Issue 本文を更新する。Nice to Have は記録のみ。
5. 結果を JSON で保存。

## 出力ファイル

### 1. `workspace/issues/139/issue-review/stage5-review-result.json`

```json
{
  "stage": 5,
  "stage_name": "通常レビュー（2回目 / Codex）",
  "focus": "通常（2回目）",
  "iteration": 2,
  "reviewer": "codex",
  "findings": [
    {
      "id": "S5-001",
      "severity": "Must Fix|Should Fix|Nice to Have",
      "category": "整合性|正確性|実装可能性|粒度|依存関係",
      "title": "...",
      "description": "...",
      "evidence": "file:line / 引用",
      "suggested_fix": "..."
    }
  ],
  "must_fix_count": 0,
  "should_fix_count": 0,
  "nice_to_have_count": 0,
  "overall_assessment": "1-3 行で総評"
}
```

Finding ID は `S5-NNN` (S5 = stage 5)。**`reviewer` フィールドは必ず `"codex"`** (Claude が上書きしないこと)。

### 2. `workspace/issues/139/issue-review/stage6-apply-result.json`

```json
{
  "stage": 6,
  "stage_name": "指摘事項反映（2回目 / Stage 5）",
  "iteration": 2,
  "reviewer": "codex",
  "applied_findings": [
    {"id": "S5-NNN", "severity": "...", "action": "...", "result": "applied|deferred|rejected"}
  ],
  "must_fix_applied": 0,
  "should_fix_applied": 0,
  "nice_to_have_applied": 0,
  "side_effects": ["..."],
  "issue_url": "https://github.com/Kewton/photon-mlx/issues/139"
}
```

## 重要な制約

- `reviewer` は **必ず `"codex"`** とすること (Claude による上書きは禁止)。
- findings 0 件は正当な結果 — 無理に件数を作らないこと。
- 既に Stage 1 / Stage 3 で指摘されたものを再掲しないこと (Stage 1/3 の finding ID と重複しないよう確認)。
- Issue 本文を更新したら、最終的な本文を `workspace/issues/139/issue-review/updated-issue-body.md` にも保存し直すこと (両者を同期)。
- 完了したら、上記 2 ファイルが存在することと、`reviewer == "codex"` を最後に echo で確認して終わること。
