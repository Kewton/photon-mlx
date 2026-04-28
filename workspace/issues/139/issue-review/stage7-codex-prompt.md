Issue #139 の **2回目影響範囲レビュー (Stage 7) + 指摘反映 (Stage 8)** を実施してください。

## あなたの役割

Claude opus が実施した 1 回目影響範囲レビュー (Stage 3) の **クロスレビュー** を担当する Codex reviewer。Stage 5 で既に通常レビューは完了済み (Should Fix 2 件反映済み)。本ステージでは、本 Issue の変更が他モジュール / 他テスト / CI / 設定 / 後方互換性 に与える影響範囲を再度確認します。

## 対象 / 入力ファイル

すべて作業ディレクトリ `/Users/maenokota/share/work/github_kewton/photon-mlx-feature-issue-139-stub-audit/` 直下のパス。

- 現行 Issue 本文 (Stage 1/3/5 反映済): `workspace/issues/139/issue-review/updated-issue-body.md`
  (= `gh issue view 139 --json body` で取得できる最新版と一致)
- 1 回目影響範囲レビュー結果: `workspace/issues/139/issue-review/stage3-review-result.json` (Must Fix 4 / Should Fix 3 / Nice 3)
- 2 回目通常レビュー結果 (Codex): `workspace/issues/139/issue-review/stage5-review-result.json`
- 全レビュー反映ログ: `workspace/issues/139/issue-review/stage{2,4,6}-apply-result.json`

## 実施手順

1. `gh issue view 139 --json body` で最新 Issue 本文を取得し、`updated-issue-body.md` と一致することを確認。
2. Stage 3 で挙がった影響範囲指摘 10 件 (Must Fix 4 + Should Fix 3 + Nice 3) を読み、Issue 本文で対応されているかを確認する。
3. **Stage 7 影響範囲レビュー**: 以下の観点で Issue 本文を再レビューし、1 回目 Stage 3 で見落とされた影響を抽出する。
   - 破壊的変更 / 後方互換: tokenizer_id 必須化 + raise への切替で他 service / 他 yaml / 他 CI / 他 docs が壊れないか (Stage 3 で baseline.yaml/eval.yaml と troubleshooting.md は対応済み)
   - 既存テスト互換性: 新 invariant test が `tests/test_pipeline_factory_yaml_invariants.py` の既存 test (`reranker.model_id`) と衝突しないか、既存 fixture との関係
   - 新規 test の安定性: `tests/test_no_scaffolding_in_prod.py` が将来の新規 file 追加で誤検出する可能性 (e.g. `_PlaceholderConfig` のような偶発的命名)、CI runner 環境での文字コードや path semantics
   - 並列マージ衝突: #135 (`feature/issue-135-photon-retrain`) との衝突は Stage 3 で記載済みだが、現在 #135 ブランチに新たな commit があるか / S5-001 反映で `_StubTokenizer` 削除と #135 の変更が新しい conflict を生まないか
   - パフォーマンス / 起動時間: tokenizer load 失敗が production 環境で頻発するシナリオ (HF Hub 障害等) で `ValueError` が即座に伝播することの運用影響 (server start failure 等)
   - 重複指摘は避ける。
4. **Stage 8 反映**: Must Fix / Should Fix の指摘があれば `gh issue edit 139 --body-file <new-body-md>` で Issue 本文を更新する。Nice to Have は記録のみ。
5. 結果を JSON で保存。

## 出力ファイル

### 1. `workspace/issues/139/issue-review/stage7-review-result.json`

```json
{
  "stage": 7,
  "stage_name": "影響範囲レビュー（2回目 / Codex）",
  "focus": "影響範囲（2回目）",
  "iteration": 2,
  "reviewer": "codex",
  "findings": [
    {
      "id": "S7-001",
      "severity": "Must Fix|Should Fix|Nice to Have",
      "category": "後方互換|破壊的変更|テスト互換|merge衝突|運用|...",
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

Finding ID は `S7-NNN`。**`reviewer` フィールドは必ず `"codex"`**。

### 2. `workspace/issues/139/issue-review/stage8-apply-result.json`

```json
{
  "stage": 8,
  "stage_name": "指摘事項反映（2回目 / Stage 7）",
  "iteration": 2,
  "reviewer": "codex",
  "applied_findings": [
    {"id": "S7-NNN", "severity": "...", "action": "...", "result": "applied|deferred|rejected"}
  ],
  "must_fix_applied": 0,
  "should_fix_applied": 0,
  "nice_to_have_applied": 0,
  "side_effects": ["..."],
  "issue_url": "https://github.com/Kewton/photon-mlx/issues/139"
}
```

## 重要な制約

- `reviewer` は **必ず `"codex"`**。
- findings 0 件は正当な結果。無理に件数を作らない。
- Stage 1 / 3 / 5 で挙がった ID と重複しない。
- Issue 本文を更新したら、`workspace/issues/139/issue-review/updated-issue-body.md` も同期。
- 完了したら、`stage7-review-result.json` と `stage8-apply-result.json` の存在 + `reviewer == "codex"` を最後に echo で確認して終わること。
