Issue #139 の **設計方針書に対する Stage 4 セキュリティレビュー + 反映** を実施してください。

## 対象 / 入力ファイル

すべて作業ディレクトリ `/Users/maenokota/share/work/github_kewton/photon-mlx-feature-issue-139-stub-audit/` 直下のパス。

- 設計方針書: `workspace/issues/139/design-policy.md` (= 本ステージのレビュー対象、DR1-DR3 反映済)
- Stage 1-3 レビュー結果: `workspace/issues/139/multi-stage-design-review/stage{1,2,3}-review-result.json`
- Stage 1-3 反映結果: `workspace/issues/139/multi-stage-design-review/stage{1,2,3}-apply-result.json`
- Issue 本文 (最新): `workspace/issues/139/issue-review/updated-issue-body.md`

## 役割

Codex reviewer。Stage 1-3 で残った **セキュリティ観点** を補完する。

## Stage 4 のフォーカス: セキュリティ

設計方針書に対し以下の観点でレビューする:

1. **Tokenizer 経路のセキュリティ**:
   - `_load_hf_tokenizer` は `trust_remote_code=False` を使用しているが、design doc がこの不変条件を保護する記述になっているか
   - tokenizer load 失敗を `ValueError` に正規化する際、エラーメッセージに `tokenizer_id` をそのまま埋め込む。`tokenizer_id` が **untrusted yaml input** から来る場合に log injection / format string injection 等の余地はないか
   - `huggingface_hub` cache path の権限 / ownership に関する暗黙の前提
2. **Yaml load の安全性**:
   - 既存 `tests/test_pipeline_factory_yaml_invariants.py` は `baseline_reporag.config.load_config` を使う前提だが、これが `yaml.safe_load` 系を使っているか / 任意 Python オブジェクト deserialize の余地はないか
   - 新 invariant test が configs/ 配下を全件 load する。悪意ある yaml が混入した場合の影響範囲
3. **境界 test (no_scaffolding) のセキュリティ**:
   - `Path(__file__).resolve().parents[1]` で repo root を解決し、配下全 .py ファイルを `read_text` する。悪意ある巨大 file / シンボリックリンク (e.g. `/etc/passwd` への symlink) があった場合の挙動
   - `read_text(encoding='utf-8')` で UTF-8 デコード失敗時の挙動 (`UnicodeDecodeError` で test 全体が落ちる) と運用影響
4. **Failure mode の情報露出**:
   - `ValueError` メッセージに `tokenizer_id` を埋め込むが、これは公開 log / Slack 通知 / Streamlit error banner で露出する。private model id / API token を含む可能性がある場合の対策が design doc に書かれているか
   - `docs/troubleshooting.md` への追記内容で機密情報の取扱いが明確か (e.g. `huggingface-cli login` の token を平文で書かない、等)
5. **Supply chain / dependency**:
   - `huggingface_hub.errors.*` を catch する設計。`huggingface_hub` バージョン依存があるか
   - `transformers.AutoTokenizer.from_pretrained` 自体のセキュリティ前提 (e.g. `trust_remote_code=False` 維持の重要性) が design doc で守られているか
6. **テストの side effect**:
   - 新規 test は OSError patch で外部 network を使わない設計か
   - configs/ 配下の yaml に test 専用 fixture が混入し、誤って production deploy されるリスクはないか
7. **削除されるコードのセキュリティ**:
   - `_StubTokenizer` 削除に伴い、過去の test/dev で意図的に使っていた箇所がセキュリティ的な fallback だった可能性は無いか (= 念の為確認)
8. **CI gate のバイパス**:
   - 新規 invariant test を将来意図的に skip する誘惑 (e.g. `@pytest.mark.skip`) を防ぐ design 上の防御策はあるか
   - 既存 `tests/test_pipeline_factory_yaml_invariants.py` の skip 設定確認

## 実施手順

1. 上記入力ファイル + `design-policy.md` を読み込む
2. 各観点ごとに finding を抽出
3. **必要なら `design-policy.md` を直接編集** して反映 (Must Fix / Should Fix のみ)
4. `workspace/issues/139/multi-stage-design-review/stage4-review-result.json` および `workspace/issues/139/multi-stage-design-review/stage4-apply-result.json` を出力

## 出力ファイル

### 1. `workspace/issues/139/multi-stage-design-review/stage4-review-result.json`

```json
{
  "stage": 4,
  "stage_name": "セキュリティレビュー（Codex）",
  "focus": "セキュリティ",
  "reviewer": "codex",
  "findings": [
    {
      "id": "DR4-001",
      "severity": "Must Fix|Should Fix|Nice to Have",
      "category": "untrusted input|log injection|supply chain|info leak|...",
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

Finding ID は `DR4-NNN`。**`reviewer` フィールドは必ず `"codex"`**。

### 2. `workspace/issues/139/multi-stage-design-review/stage4-apply-result.json`

```json
{
  "stage": 4,
  "stage_name": "指摘事項反映（Stage 4 / セキュリティ）",
  "reviewer": "codex",
  "applied_findings": [
    {"id": "DR4-NNN", "severity": "...", "action": "...", "result": "applied|deferred|rejected"}
  ],
  "must_fix_applied": 0,
  "should_fix_applied": 0,
  "nice_to_have_applied": 0,
  "side_effects": ["..."],
  "design_policy_path": "workspace/issues/139/design-policy.md"
}
```

## 重要な制約

- **設計方針書 (design-policy.md) のみを更新**。ソースコード変更禁止。
- `reviewer` フィールドは必ず `"codex"`。
- findings 0 件は正当な結果。無理に件数を作らない。
- Stage 1-3 で挙がった ID と重複しない。
- 完了後、`stage4-review-result.json` + `stage4-apply-result.json` の存在 + `reviewer == "codex"` を最後に echo で確認。
