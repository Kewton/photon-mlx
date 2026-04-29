# Issue #175: institutional 用 synonym 辞書の corpus 分析

**Issue**: #175 feat(retrieval): synonym dictionary for institutional domain query expansion (G2)
**作成日**: 2026-04-29
**分析者**: Claude (Issue #175 Phase 5 T1 として)
**ユーザーレビュー**: 未 (本ドキュメント全体が Phase 5 T1 の成果物 — 辞書を確定させるためのレビュー対象)

---

## 1. 分析手法

### 1.1 corpus 概要

| 項目 | 値 |
|-----|---|
| Source | `/Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents/` |
| ファイル数 | 4,228 .md files |
| `repo_id` (3 institutional configs) | `institutional_documents` |
| `repo_commit` | `9e500539f29555364217b773368305e7f59aa026` |
| 主なテーマ | 国土交通省 等の住宅政策 / 補助事業 / 賃貸契約 / 原状回復 / 補助金交付 |

### 1.2 ツール

依存ゼロ (regex のみ、MeCab/Sudachi 不使用) の N-gram sliding window 方式を採用。理由:
- MeCab/Sudachi 等の追加依存を本 Issue scope 外とするため (CLAUDE.md `pyproject.toml` 不変)
- N-gram は形態素境界を見ないが、頻度集計目的では十分な精度

実装スクリプト:
- `scripts/analyze_institutional_corpus_for_synonyms.py` — 全 4,228 docs の N-gram 頻度分析 (top 500 を JSON 出力)
- `scripts/extract_synonym_candidates.py` — seed concepts 別にクラスタリング
- `scripts/measure_specific_phrases.py` — 候補フレーズの正確な DF を測定

### 1.3 フィルタ条件

- **Token 抽出**: 漢字 / ひらがな / カタカナ / 長音 (`[一-鿿぀-ゟ゠-ヿー]+`) の連続部
- **N-gram 長**: 2-8 文字
- **NOISE_TOKENS**: `について`、`における`、`場合`、`もの`、`こと`、助詞・補助動詞・単位 (`年`、`月`、`回`、`号` 等)
- **頻度足切り**: TF >= 10 かつ DF >= 5 (top 500 で出力)
- **ひらがな単独排除**: 全文字がひらがなの phrase を除外 (助詞・接続詞のみのフレーズ排除)

---

## 2. corpus 観察結果

### 2.1 Top phrases (DF 順)

実 DF 計測 (`workspace/issues/175/corpus-analysis/ngram-frequency.json`) より抜粋:

| Phrase | TF | DF | カテゴリ |
|--------|-----|-----|---------|
| 事業 | 210,751 | 2,718 | 事業 (極頻出) |
| 実施 | 89,065 | 2,470 | 実施 |
| 必要 | 55,321 | 2,379 | 必要 |
| 対象 | 43,091 | 2,195 | 対象 |
| 支援 | 83,068 | 2,169 | 補助/支援 |
| 補助 | 53,433 | 1,419 | 補助 |
| 認定 | 44,885 | 1,489 | 認定 |
| 確認 | 25,213 | 1,669 | 確認 |
| 基準 | 36,492 | 1,651 | 条件/基準 |
| 規定 | 36,251 | 1,419 | 条件/基準 |
| 要件 | 19,562 | 1,113 | 条件/基準 |
| 申請 | 34,444 | 1,195 | 申請 |
| 提出 | 16,139 | 1,115 | 提出/報告 |
| 報告 | 15,023 | 1,158 | 報告 |
| 通知 | 12,734 | 1,181 | 通知 |
| 改修 | 26,386 | 1,170 | (改善でヒット) |

corpus は**住宅政策/補助事業/賃貸契約**ドメインで、Issue #175 例示語の系統と一致。

### 2.2 Issue #175 例示語の corpus 実在性

実測した DF は次の通り (詳細: `workspace/issues/175/corpus-analysis/candidate-dict-df.json`):

#### 認定基準系
| Phrase | DF | 採用判定 |
|--------|-----|---------|
| 認定基準 (key) | 81 | ✅ 採用 |
| 対象事業者 (val) | 94 | ✅ 採用 |
| 認定要件 (val) | 17 | ⚠️ 低 DF だが Issue 例示語のため保持 |
| 認定の基準 (val) | 15 | ⚠️ 同上 |
| 認定の条件 (val) | **2** | ❌ **除外** (corpus にほぼ不在、Issue 例示語だが現実不適合) |

#### 申請方法系
| Phrase | DF | 採用判定 |
|--------|-----|---------|
| 申請方法 (key) | 83 | ✅ 採用 |
| 申請手続 (val) | 185 | ✅ 採用 |
| 提出方法 (val) | 74 | ✅ 採用 |
| 申込方法 (val) | 16 | ⚠️ 低 DF |
| 申し込み方法 (val) | 2 | ❌ 除外 |

#### 必要書類系
| Phrase | DF | 採用判定 |
|--------|-----|---------|
| 必要書類 (key) | 117 | ✅ 採用 |
| 添付書類 (val) | 196 | ✅ 採用 |
| 提出書類 (val) | 136 | ✅ 採用 |
| 必要な書類 (val) | 136 | ✅ 採用 |
| 提出資料 (val) | 61 | ✅ 採用 |

#### 補助率系
| Phrase | DF | 採用判定 |
|--------|-----|---------|
| 補助率 (key) | 362 | ✅ 採用 |
| 補助割合 (val) | 38 | ⚠️ 低 DF だが採用 (Issue 例示語) |
| 補助の割合 (val) | **4** | ❌ **除外** (corpus にほぼ不在) |
| 助成率 (val) | **1** | ❌ **除外** (corpus にほぼ不在) |
| 助成割合 (val) | **2** | ❌ **除外** |

#### 申請期限系
| Phrase | DF | 採用判定 |
|--------|-----|---------|
| 申請期限 (key) | 37 | ⚠️ 低 DF だが key として保持 |
| 締切 (val) | 234 | ✅ 採用 (高 DF) |
| 提出期限 (val) | 88 | ✅ 採用 |
| 申請受付期限 (val) | **3** | ❌ **除外** |
| 応募期限 (val) | 1 | ❌ 除外 |

### 2.3 重要な観察

1. **Issue #175 で例示された synonym は実 corpus に出現しないものが含まれる** (助成率 DF=1, 認定の条件 DF=2 等)。これらをそのまま採用すると expand_query で展開語に投入しても hit する文書が存在せず、recall 改善に寄与しない
2. **逆に corpus 頻度が高い実用的 synonym** が Issue で挙げられていないものもある (例: 補助金 DF=736, 給付金 DF=137, 助成金 DF=69, 交付金 DF=459)
3. **超高 DF の語 (DF > 1000)** は precision 衝突リスクが高い。例: 「支援」(DF 2169) を補助系の synonym として展開すると、無関係の chunks も hit する可能性

---

## 3. 衝突チェック (precision risk)

設計方針書 §4.2 / リスクマップ R3 より、synonym 展開は precision 低下リスクがある。各値の使用文脈を spot-check した結果:

| 検討した synonym | 衝突リスク | 判断 |
|-----------------|-----------|------|
| 支援 (DF 2169) | ⚠️ 高 (汎用語、補助金以外の文脈にも頻出) | **キーとしては避け**、値として「補助/助成」と並列のみで採用 |
| 整備 (DF 1672) | ⚠️ 高 (改修/制度整備等多義) | 値として除外 |
| 評価 (DF 1276) | ⚠️ 中 | 「審査」の値としてのみ採用 |
| 改修/修繕/リフォーム | 低 | 全採用 (住宅領域に限定) |
| 締切 (DF 234) | 低 | 「申請期限」「期限」の値として採用 |
| サポート (DF 705) | 中 (カタカナで多義) | 「支援」の値として保留、辞書投入時は除外 |
| 認定の条件 (DF 2) | 低 (corpus にほぼ不在) | **除外** (recall 改善に寄与しないため) |
| 助成率 (DF 1) | 低 | **除外** |

---

## 4. 最終候補辞書 (≥20 entries)

以下が **AC1 完了基準を満たす最終候補**。3 institutional configs (`institutional_docs.yaml`, `_photon.yaml`, `_photon_retrain.yaml`) に **完全同一** で投入する想定 (設計方針書 §4.3、SoT test で保護):

```yaml
retrieval:
  query_expansion:
    enabled: true
    include_symbol_aliases: true
    include_filename_hints: true
    domain_map:
      # === 認定 / 対象 系 ===
      "認定基準": ["対象事業者", "認定要件", "認定の基準"]
      "対象": ["対象者", "対象事業", "対象事業者"]
      "適用": ["対象", "該当", "適用範囲"]

      # === 申請 / 手続 系 ===
      "申請方法": ["申請手続", "提出方法", "申込方法"]
      "申請": ["申込", "応募", "提出"]
      "申込": ["申請", "申し込み", "応募"]
      "手続": ["手続き", "プロセス", "手順"]

      # === 書類 / 様式 系 ===
      "必要書類": ["添付書類", "提出書類", "必要な書類", "提出資料"]
      "書類": ["資料", "様式", "書面"]
      "添付": ["添付書類", "添付資料"]
      "様式": ["書類", "フォーマット"]

      # === 補助 / 助成 系 ===
      "補助金": ["交付金", "給付金", "支援金", "助成金"]
      "補助率": ["補助割合"]
      "補助金額": ["上限額", "補助上限額", "助成額"]
      "助成": ["補助", "支援", "支給", "給付"]

      # === 期限 / 募集 系 ===
      "申請期限": ["締切", "提出期限", "期日"]
      "期限": ["締切", "期日"]
      "公募": ["募集"]

      # === 報告 / 通知 系 ===
      "報告": ["通知", "連絡", "届出", "申告"]
      "通知": ["連絡", "周知", "案内"]
      "届出": ["報告", "提出", "申告"]

      # === 選定 / 審査 系 ===
      "選定": ["決定", "採択", "選考", "選択"]
      "審査": ["評価", "判定", "検討"]

      # === 賃貸 / 契約 系 (corpus theme) ===
      "賃貸": ["賃借", "貸借", "リース"]
      "賃料": ["家賃", "賃借料"]

      # === 改修 / 修繕 系 ===
      "改修": ["修繕", "リフォーム", "補修", "修理"]
      "修繕": ["改修", "リフォーム", "補修"]
```

**カウント**: 26 entries — AC1 の ≥20 entries 要件を満たす。

### 4.1 設計判断 #2 (cap=8 制限) の確認

`expand_query()` は展開語を `cap=8` (max 8 words after dedup) に絞る。上記辞書で **1 クエリあたり同時 match する entries** の典型:

| 想定クエリ | matching keys | 展開語数 (dedup 前) | 展開語数 (cap=8 後) |
|-----------|---------------|---------------------|-------------------|
| 「認定基準について」 | 認定基準 | 3 | 3 |
| 「申請方法と必要書類」 | 申請方法, 必要書類 | 3 + 4 = 7 | 7 |
| 「補助金の申請期限」 | 補助金, 申請, 申請期限, 期限 | 4 + 3 + 3 + 2 = 12 | **8 (cap 効く)** |
| 「補助率と申請方法と必要書類について」 | 補助率, 申請方法, 必要書類 | 1 + 3 + 4 = 8 | 8 |

→ 通常クエリは cap=8 内に収まる。cap 拡張不要 (R7 緩和済)。

### 4.2 双方向性

Issue 例示語の質問 → corpus 表現マッピングは片方向だが、利用者が逆方向 (corpus 表現で質問) する場合に備え、以下のペアで双方向を確保:

- 補助 ↔ 助成 (双方が key として登場)
- 申請 ↔ 申込 (双方が key として登場)
- 改修 ↔ 修繕 (双方が key として登場)
- 報告 ↔ 通知 / 届出 (報告/通知/届出 が相互に key)

`認定基準` 系は corpus 表現 (対象事業者) が key に登場しないが、利用者が「対象事業者」で質問する場合は通常そのまま hit するため (DF=94)、片方向で OK。

---

## 5. AC3/AC4 用 expected hit 例

E2E テストまたは eval で確認すべき期待 hit:

### AC3: 「認定基準」query → 「認定の条件」「認定要件」を含む chunk hit
**実 corpus DF を踏まえた現実的修正**: 「認定の条件」(DF=2) は使わず、「対象事業者」(DF=94) または「認定要件」(DF=17) を含む chunk が hit することを検証。例えば:

```python
query = "認定基準について教えて"
expected_substrings = ["認定基準", "対象事業者", "認定要件", "認定の基準"]
```

### AC4: 「申請期限」query → 「提出期限」「締切」を含む chunks hit
DF が高い「締切」(234)・「提出期限」(88) の chunks が含まれる:

```python
query = "申請期限はいつですか"
expected_substrings = ["申請期限", "締切", "提出期限", "期日"]
```

### 補足検証クエリ (eval / E2E 候補)

```python
EVAL_QUERY_HIT_PAIRS = [
    ("認定基準について",  ["対象事業者", "認定要件"]),
    ("申請方法を教えて",  ["申請手続", "提出方法", "申込方法"]),
    ("必要書類は何",      ["添付書類", "提出書類", "必要な書類"]),
    ("補助率を確認",      ["補助割合"]),
    ("補助金の上限",      ["上限額", "補助上限額"]),
    ("申請期限はいつ",    ["締切", "提出期限", "期日"]),
    ("公募の時期",        ["募集"]),
    ("報告書類",          ["通知", "連絡", "届出"]),
    ("審査基準",          ["評価", "判定", "認定要件"]),
    ("改修工事",          ["修繕", "リフォーム", "補修"]),
    ("家賃の値上げ",      ["賃料", "賃借料"]),
    ("選定方法",          ["決定", "採択", "選考"]),
]
```

これらは eval (T8) でサンプル質問として実行し、retrieval 上位に expected_substrings 含む chunks が混じるかを集計。

---

## 6. ユーザーレビュー要請事項

T1 (corpus 分析) 完了後、Phase 5 T2-T7 (TDD/regression) に進む前に **ユーザー確認を希望** する事項:

1. **辞書 26 entries の最終承認**
   - 各 entry のキー/値が institutional ドメインの利用者目線で自然か
   - 衝突リスク評価 (§3) の判断 (「支援」キー除外、「認定の条件」値除外 等) が妥当か
   - 双方向性 (§4.2) の方針が良いか

2. **追加除外 / 追加採用の希望**
   - 例: 「整備」を改修系に含めるか (precision リスクで除外したが、ユーザーが整備系の質問を想定するなら含める)
   - 例: 賃貸系 entries の数 (corpus に賃貸契約資料が多いが、利用者の主目的が補助金なら賃貸系を縮小可能)

3. **eval 戦略の確認**
   - AC7 eval は本設計方針書 §7 通り、`institutional_docs_photon.yaml` (採用 LLM 必須) + `institutional_docs.yaml` (多言語 reranker 推奨) の 2 config × Before/After で計測予定
   - これで OK か、または採用 LLM の `_photon.yaml` のみで OK か

---

## 7. 関連ファイル

```
docs/issue-175-synonym-corpus-analysis.md          (本ファイル)
workspace/issues/175/corpus-analysis/
├── ngram-frequency.json                            (top-500 N-gram, DF/TF)
├── synonym-clusters.json                           (seed-based clusters)
└── candidate-dict-df.json                          (candidate dict + measured DF)

scripts/
├── analyze_institutional_corpus_for_synonyms.py    (N-gram 全文分析)
├── extract_synonym_candidates.py                   (seed cluster 抽出)
└── measure_specific_phrases.py                     (候補語 DF 直接測定)
```

---

## 8. 次のアクション (T2 以降)

ユーザーレビュー後:

- [ ] T2: TDD Red — `baseline_reporag/tests/test_query_expansion.py` に institutional 用テスト 4 種追加 (fail 確認)
- [ ] T3: TDD Green — 3 institutional configs に上記辞書を投入、`use_expansion_terms: false` 設定 (PHOTON variants)
- [ ] T4: `pipeline.py` / `photon_pipeline.py` に reranker 条件分岐実装
- [ ] T5: `tests/test_pipeline_factory_yaml_invariants.py` に不変保証テスト追加
- [ ] T6: AC3/AC4 E2E ユニットテスト (もしくは T8 eval で代替)
- [ ] T7: 全テスト regression check (`pytest`, `ruff`)
- [ ] T8: AC7 eval 計測 (Before/After × 2 configs、計 2-4 hours、ユーザー手動キック想定)
- [ ] T9: `reports/issue-175-synonym-eval.md` 作成
- [ ] T10: PR draft 準備 (`develop` target)
