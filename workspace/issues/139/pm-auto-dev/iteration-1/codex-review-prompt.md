Issue #139 のTDD実装コードに対して、潜在バグとセキュリティ脆弱性のコードレビューを実施してください。

## 対象ファイル (本ブランチで変更/新規)

すべて作業ディレクトリ `/Users/maenokota/share/work/github_kewton/photon-mlx-feature-issue-139-stub-audit/` 直下のパス。

- `baseline_reporag/photon_pipeline.py` (改修)
- `baseline_reporag/tests/test_photon_pipeline.py` (改修 + 新規 test 追加)
- `docs/troubleshooting.md` (追記)
- `photon_mlx/tests/conftest.py` (docstring 更新)
- `tests/test_pipeline_factory_yaml_invariants.py` (拡張)
- `tests/test_no_scaffolding_in_prod.py` (新規)

## レビュー観点

### 1. 潜在バグ
- ロジックエラー、エッジケースの未処理 (None / 空文字 / 型違反)
- off-by-one / index out of bounds
- 例外処理の捕捉範囲が広すぎる (`except Exception`) / 狭すぎる
- リソースリーク (file handle / connection)
- 並行性 / スレッドセーフ性 (本 Issue 範囲では希薄だが念のため)
- 既存 invariant (Issue #138 vocab_size / Issue #58 tokenizer 共有 / Issue #138 tokenizer training/inference unification) の保持

### 2. セキュリティ脆弱性
- **untrusted input validation**: yaml 由来の `tokenizer_id` の allowlist (`^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`) が攻撃に対して十分か (URL / path traversal / control char / shell metachar / log injection 等)
- **info leak**: `ValueError` メッセージ・`_logger.warning` 等で raw user input / token / private model id が露出していないか
- **trust_remote_code=False の固定**: `_load_hf_tokenizer` で `trust_remote_code=False` 引数が変更不可になっているか / 将来の patch で False が外される余地はないか
- **境界 test の hardening**: `tests/test_no_scaffolding_in_prod.py` が symlink / oversize file / non-utf8 file / repo root 外 path を violation として扱っているか / 抜け漏れなし
- **CI gate のバイパス**: 新規 invariant test に `@pytest.mark.skip` 等が付いていないか
- **eval / exec / subprocess の使用**: 本 Issue では追加されていないはずだが念のため確認

## 指示

1. 上記 6 ファイルを読み込んでください
2. 各ファイルに対して上記 2 観点でレビュー
3. 結果をJSON形式で `workspace/issues/139/pm-auto-dev/iteration-1/codex-review-result.json` に出力

## 出力フォーマット

```json
{
  "reviewer": "codex",
  "issue_number": 139,
  "review_focus": ["潜在バグ", "セキュリティ脆弱性"],
  "files_reviewed": [
    "baseline_reporag/photon_pipeline.py",
    "baseline_reporag/tests/test_photon_pipeline.py",
    "docs/troubleshooting.md",
    "photon_mlx/tests/conftest.py",
    "tests/test_pipeline_factory_yaml_invariants.py",
    "tests/test_no_scaffolding_in_prod.py"
  ],
  "findings": [
    {
      "id": "CB-001",
      "severity": "critical|high|medium|low",
      "category": "潜在バグ|セキュリティ脆弱性",
      "file": "...",
      "line": 0,
      "title": "...",
      "description": "...",
      "suggestion": "..."
    }
  ],
  "summary": {
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0,
    "total": 0
  },
  "verdict": "pass|needs_fix"
}
```

## 重要な制約

- `reviewer` は必ず `"codex"`。
- findings 0 件は正当な結果。無理に件数を作らない。
- critical/high が 0 件なら `verdict: "pass"`、1 件以上なら `verdict: "needs_fix"`。
- 完了後、`codex-review-result.json` の存在 + `reviewer == "codex"` を最後に echo で確認。
