Issue #139 の **設計方針書に対する Stage 3 影響分析レビュー + 反映** を実施してください。

## 対象 / 入力ファイル

すべて作業ディレクトリ `/Users/maenokota/share/work/github_kewton/photon-mlx-feature-issue-139-stub-audit/` 直下のパス。

- 設計方針書: `workspace/issues/139/design-policy.md` (= 本ステージのレビュー対象)
- Stage 1 (設計原則) レビュー結果: `workspace/issues/139/multi-stage-design-review/stage1-review-result.json`
- Stage 1 反映結果: `workspace/issues/139/multi-stage-design-review/stage1-apply-result.json`
- Stage 2 (整合性) レビュー結果: `workspace/issues/139/multi-stage-design-review/stage2-review-result.json`
- Stage 2 反映結果: `workspace/issues/139/multi-stage-design-review/stage2-apply-result.json`
- Issue 本文 (最新 / 既に 25 finding 反映済): `workspace/issues/139/issue-review/updated-issue-body.md`

> Stage 1-2 は Claude opus 配下の subagent quota 制限に達したため、Claude main agent (opus) が **inline** で実施したことを留意。`reviewer` フィールドは `"claude-opus (inline, due to subagent quota)"` および `"claude-opus (inline)"` と記録されている。

## 役割

Claude opus (inline) が実施した Stage 1 (設計原則) + Stage 2 (整合性) のクロスレビューを担当する Codex reviewer。**影響範囲** の観点で設計方針書をレビューし、Stage 1-2 で見落とされた波及効果を補完する。

## Stage 3 のフォーカス: 影響分析 (波及効果)

設計方針書 §5 / §7 / §8 / §9 をベースに、本 Issue の変更が **他モジュール / 他テスト / CI / 他ブランチ / 運用** に及ぼす影響を確認する:

1. **既存 16-17 unit tests への影響**: design doc は `_StubTokenizer` を fallback ごと削除するが、現コード `baseline_reporag/photon_pipeline.py:300` のコメントは「~17 unit tests that pre-date the `tokenizer:` section keep working without modification」 — つまり多くの test fixture が `tokenizer:` ブロック無しで `_get_stub_tokenizer` を使う前提。これら 16-17 件は design doc の影響範囲表に列挙されておらず、削除すれば全部失敗する可能性がある。実 main コードを `grep -rn "_StubTokenizer\|_get_stub_tokenizer" baseline_reporag/tests/ photon_mlx/tests/ tests/` で網羅的に確認し、design doc の migration plan に欠落がないか報告する。
2. **`photon_mlx/tests/conftest.py` の独自 `_StubTokenizer` (L15) を使う test 群**: design doc は conftest.py の docstring 更新のみを記載するが、conftest fixture を import する test 群がどう影響を受けるかを確認する。
3. **#135 ブランチとの conflict 詳細化**: design doc §8 は merge 順序を記載するが、`feature/issue-135-photon-retrain` で `_StubTokenizer` の利用方法が変わっている可能性 (#135 側でも `_StubTokenizer` を使う test 増設等) を確認。`git diff main..feature/issue-135-photon-retrain -- baseline_reporag/photon_pipeline.py baseline_reporag/tests/test_photon_pipeline.py photon_mlx/tests/conftest.py` を実行し、影響を report。
4. **CI への影響**: `.github/workflows/` 配下 (現在 `weekly_eval.yml` のみ) で `_StubTokenizer` の存在を assertion している箇所はないか確認。
5. **bench/scripts/demo の独自 `_StubTokenizer`**: design doc は対象外と明記したが、影響分析として「これらが production module の `_StubTokenizer` を import していないか」 を grep で再確認 (Stage 3 / S3-008 で確認済だが念押し)。
6. **invariant test 拡張の他効果**: 新規 `test_photon_yaml_has_required_tokenizer_fields` で対象 yaml に欠落があれば pytest 全体が fail する。本 Issue を main にマージした瞬間に CI が破壊されないか、`grep -n "tokenizer_id\|vocab_size" configs/photon_*.yaml configs/institutional_docs_photon.yaml` で各 yaml の現状を確認し、欠落があれば design doc に「必ず本 Issue 内で yaml 補完する」ことを明記すべきか報告。
7. **session memory / answering 系 の getattr default**: design は Phase B として 6 件を別 Issue 化する方針だが、その間に新しい getattr default が追加されるリスクの所感。

## 実施手順

1. 上記入力ファイル + `design-policy.md` を読み込む。
2. 上記 1〜7 の観点で **追加 finding** を抽出 (Stage 1-2 と重複しないこと)。
3. **必要なら `design-policy.md` を直接編集** して反映 (Must Fix / Should Fix のみ。Nice to Have は記録のみ)。
4. `workspace/issues/139/multi-stage-design-review/stage3-review-result.json` および `workspace/issues/139/multi-stage-design-review/stage3-apply-result.json` を出力。

## 出力ファイル

### 1. `workspace/issues/139/multi-stage-design-review/stage3-review-result.json`

```json
{
  "stage": 3,
  "stage_name": "影響分析レビュー（Codex）",
  "focus": "影響範囲",
  "reviewer": "codex",
  "findings": [
    {
      "id": "DR3-001",
      "severity": "Must Fix|Should Fix|Nice to Have",
      "category": "既存テスト|merge衝突|CI|運用|...",
      "title": "...",
      "description": "...",
      "evidence": "file:line / 引用",
      "suggested_fix": "..."
    }
  ],
  "must_fix_count": 0,
  "should_fix_count": 0,
  "nice_to_have_count": 0,
  "overall_assessment": "1-3 line 総評"
}
```

Finding ID は `DR3-NNN`。**`reviewer` フィールドは必ず `"codex"`**。

### 2. `workspace/issues/139/multi-stage-design-review/stage3-apply-result.json`

```json
{
  "stage": 3,
  "stage_name": "指摘事項反映（Stage 3 / 影響分析）",
  "reviewer": "codex",
  "applied_findings": [
    {"id": "DR3-NNN", "severity": "...", "action": "...", "result": "applied|deferred|rejected"}
  ],
  "must_fix_applied": 0,
  "should_fix_applied": 0,
  "nice_to_have_applied": 0,
  "side_effects": ["..."],
  "design_policy_path": "workspace/issues/139/design-policy.md"
}
```

## 重要な制約

- **設計方針書 (design-policy.md) のみを更新**。ソースコードは変更しない。
- `reviewer` フィールドは必ず `"codex"`。Claude による上書きは禁止。
- findings 0 件は正当な結果。無理に件数を作らない。
- Stage 1-2 で挙がった ID と重複しない。
- 完了後、`stage3-review-result.json` と `stage3-apply-result.json` の存在 + `reviewer == "codex"` を最後に echo で確認。
